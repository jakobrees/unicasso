"""Free-logit candidate pools (--mode pool); shares the nomination/tabu machinery the swarm builds on.

knn-smooth slaves every blend weight to one latent position z (w = softmax(-cdist(z, codebook)/tau)),
which conflates WHICH glyphs may participate (support = z's k-NN) with HOW the weights move (all
coupled through z). Pool mode decouples them: per cell, a small candidate pool (glyph indices) plus
FREE learnable logits over it -- w_c = softmax((l_c + sigma(t)*gumbel) / tau_c(t)). The VAE is demoted
to seeding (warm-start k-NN), dedup metric, and the blend-embedding substrate for the graph losses.

Exploration is annealed Gumbel noise on the logits (the z-noise successor: every candidate gets
stochastic auditions, heavy-tailed enough that low-weight ones occasionally spike; anneal to 0 to
commit) plus gradient-EMA nomination: score the WHOLE codebook per cell with the
render-gradient dot product and rotate wanted outsiders into the pool. Blur protection = small K +
the blend-diversity penalty in asciify (charges for incoherent composites at commit time), replacing
knn-smooth's geometric look-alike guarantee.

Snap = argmax clean logit, so the final render is definitionally consistent with the blend (a snap
by latent distance could ignore render-visible candidates; that can't occur here).
"""
import math

import torch
import torch.nn as nn

GUMBEL_MEAN = 0.5772156649
GUMBEL_STD = math.pi / math.sqrt(6.0)
TABU_SLOTS = 8   # per-cell ring of recently-evicted glyphs barred from re-nomination


class CandidatePool:
    def __init__(self, GH, GW, K, device):
        M = GH * GW
        self.GH, self.GW = GH, GW
        self.M, self.K, self.device = M, K, device
        self.cand = torch.zeros(M, K, dtype=torch.long, device=device)   # (M,K) glyph indices
        self.logits = nn.Parameter(torch.zeros(M, K, device=device))     # free per-slot logits
        self._ar = torch.arange(M, device=device)
        # nomination state: clean-weight EMA (eviction evidence), slot age (grace),
        # tabu ring (evicted glyphs the tangent still likes -> barred so the pool doesn't thrash),
        # and the render-gradient EMA g_ema (M,P) -- the nomination signal.
        self.w_ema = torch.zeros(M, K, device=device)
        self.age = torch.zeros(M, K, dtype=torch.long, device=device)
        self.fail_glyph = torch.full((M, TABU_SLOTS), -1, dtype=torch.long, device=device)
        self.fail_until = torch.zeros(M, TABU_SLOTS, dtype=torch.long, device=device)
        self.fail_cur = torch.zeros(M, dtype=torch.long, device=device)
        self.g_ema = None
        self.space_idx = None
        self.dedup_eps = 0.0                                             # absolute latent distance
        self.density_block = None        # (M,N) bool: nominee ink too far from the cell's target ink
        self.affinity_partner = None     # (M,) long: strongest affinity partner cell (-1 = none)
        # Latent-adjacency proximity kernel P = exp(-D^2/2h^2) over the codebook, for the "latent"
        # channel: nominate glyphs NEAR the pool's current weight mass. This is knn-smooth's travel
        # rebuilt for pools -- the support frontier moves through latent space as weight shifts, so
        # reaching a far glyph requires a CHAIN of adjacent wins (garbage stays quarantined behind
        # a journey the audition won't fund), and relatives of a dominant winner arrive as the
        # special case (within-family refinement).
        self.cb_prox = None              # (N,N)
        # Ports channel state (tile continuity; see tile_ports.py). ports_T (M,N): per-cell TARGET
        # port match (-inf outside the tile whitelist). ports_Ch/Cv (N,N): glyph-glyph compatibility
        # (overrides applied, clamped) for the live-neighbor term. Proposals carry their OWN
        # evidence -- no gradient gate (the linear score is a matched filter; ports exist because
        # it can't see structure).
        self.ports_T = None
        self.ports_Ch = None
        self.ports_Cv = None
        # File channel: externally certified proposals (e.g. probe_swaps accepts), one glyph per
        # cell, proposed until adopted or tabu'd. Also gate-free: their evidence is the probe.
        self.file_prop = None            # (M,) long, -1 = none
        # Cells OPEN to search (nomination + spread cap). Confidently-empty cells are closed: the
        # empty stack (sharp temp / quiet noise / ink penalty) owns them by design -- and the ink
        # penalty acts on cell_img BEFORE assembly, so img.grad (the nomination signal) sees CLIP's
        # background desire UNOPPOSED there; without this gate the grad channel injects thin glyphs
        # into settled white regions and destabilizes CLIP's reference. None = all cells open.
        self.search_mask = None          # (M,) bool

    @torch.no_grad()
    def init_from_z0(self, z0, codebook, empty_safe=None, space_idx=None, empty_thresh=0.5):
        """Pool = top-K codebook NN of the warm-start z0; logits = -distance, so at the same temp the
        iteration-0 blend matches knn-smooth's first frame. The space glyph is force-seeded into
        confidently-empty cells (replacing the worst-ranked slot): space is just a pool member here,
        no logsumexp/cap machinery -- the logits decide, and the snap honors it by construction."""
        d = torch.cdist(z0, codebook)                                    # (M, N)
        vals, idx = torch.topk(d, self.K, dim=1, largest=False)
        self.cand.copy_(idx)
        self.logits.data.copy_(-vals)
        n_seed = 0
        if space_idx is not None and empty_safe is not None:
            need = (empty_safe > empty_thresh) & ~(self.cand == space_idx).any(dim=1)
            if need.any():
                self.cand[need, self.K - 1] = space_idx
                self.logits.data[need, self.K - 1] = -d[need, space_idx]
            n_seed = int(need.sum())
        return n_seed

    def weights(self, tau_c, noise_scale=0.0, noise_gate=None):
        """Blend weights w = softmax((l + sigma*gumbel) / tau_c). tau_c: scalar or (M,) per-cell temp
        (the empty-cell sharpen multiplies here exactly as it multiplied knn-temp). noise_scale sigma:
        annealed Gumbel exploration in LOGIT units (standardized to zero-mean/unit-std so sigma reads
        like z-noise did); heavy right tail = occasional big auditions for buried candidates.
        noise_gate (M,): per-cell sigma multiplier (empty-cell quieting). sigma=0 -> clean weights."""
        l = self.logits
        if noise_scale > 0:
            u = torch.rand_like(l).clamp_(1e-9, 1.0 - 1e-9)
            g = (-torch.log(-torch.log(u)) - GUMBEL_MEAN) / GUMBEL_STD
            if noise_gate is not None:
                g = g * noise_gate[:, None]
            l = l + noise_scale * g
        if torch.is_tensor(tau_c) and tau_c.dim() == 1:
            return torch.softmax(l / tau_c.clamp_min(1e-6)[:, None], dim=1)
        return torch.softmax(l / max(float(tau_c), 1e-6), dim=1)

    @torch.no_grad()
    def snap(self):
        """Hard per-cell glyph = argmax CLEAN logit (noise-free; consistent with the final blend)."""
        return self.cand[self._ar, self.logits.argmax(dim=1)]

    @torch.no_grad()
    def clamp_spread(self, cap):
        """Post-step projection onto {std(l) <= cap} per cell: center the logits (mean shift is
        softmax-invariant) and rescale deviations only if the std exceeds the cap. ONE-SIDED, not
        normalization: cells under the cap keep their own weaker opinions (scaling small spreads UP
        would manufacture confidence in genuinely uncertain cells).

        Rationale: a Gumbel
        audition flips a cell only when sigma*(g_i - g_j) exceeds the LOGIT GAP -- tau cancels (both
        logits and noise are divided by it), so audition reach is gap-vs-sigma. Demotion-driven
        spread growth (~1.7 std within a few hundred iters under held noise) pushes gaps beyond
        sigma=1 reach early, which stalls search while the schedule still says
        explore. Capped at ~0.6-0.8, rivals stay within
        1-1.5 sigma all window; commitment still arrives untouched via the tau anneal (spread*3/tau
        is huge at tau_end)."""
        l = self.logits.data
        mu = l.mean(dim=1, keepdim=True)
        sd = l.std(dim=1, keepdim=True)
        scale = (cap / sd.clamp_min(1e-8)).clamp(max=1.0)
        if self.search_mask is not None:            # closed (empty) cells keep their full space lead
            scale = torch.where(self.search_mask[:, None], scale, torch.ones_like(scale))
        self.logits.data = (l - mu) * scale

    @torch.no_grad()
    def update_stats(self, tau_c, ema=0.95):
        """Per-step bookkeeping for nomination: EMA the CLEAN blend weights (the learned preference,
        the eviction evidence -- not the noisy audition weights) and age every slot."""
        self.w_ema.mul_(ema).add_(self.weights(tau_c), alpha=1.0 - ema)
        self.age += 1

    @torch.no_grad()
    def update_grad_ema(self, g_cells, decay=0.98):
        """EMA of the per-cell render gradient (M,P). Because bm_flat is constant, EMA-then-project
        equals project-then-EMA, so this single buffer scores the whole codebook on demand. The
        instantaneous gradient is crop-noisy (only cells under this step's CLIP crops hear from
        CLIP) -- the EMA is what makes the score a PERSISTENT preference."""
        if self.g_ema is None:
            self.g_ema = g_cells.clone()
        else:
            self.g_ema.mul_(decay).add_(g_cells, alpha=1.0 - decay)

    @torch.no_grad()
    def _modal_neighbor(self, snaps):
        """Most common glyph among a cell's 4 neighbors (torus wrap at edges, as inject.py)."""
        g = snaps.view(self.GH, self.GW)
        nb = torch.stack([torch.roll(g, 1, 0), torch.roll(g, -1, 0),
                          torch.roll(g, 1, 1), torch.roll(g, -1, 1)], -1).view(self.M, 4)
        eq = (nb[:, :, None] == nb[:, None, :]).sum(-1)
        return nb[self._ar, eq.argmax(1)]

    @torch.no_grad()
    def apply_swaps(self, c, slots, glyphs, evicted, it, tabu=500, rank_lo=2, rank_hi=5):
        """Commit selected swaps (from a dry-run nominate, e.g. after live-probe verification):
        evictee -> tabu ring, arrival at a mid-pack logit, fresh age/w_ema."""
        if c.numel() == 0:
            return
        srt = self.logits.data[c].sort(dim=1, descending=True).values
        lo = max(0, min(rank_lo, self.K - 1))
        hi = max(lo + 1, min(rank_hi, self.K))
        r = torch.randint(lo, hi, (c.numel(),), device=self.device)
        arrival = srt[torch.arange(c.numel(), device=self.device), r]
        cur = self.fail_cur[c]
        self.fail_glyph[c, cur] = evicted
        self.fail_until[c, cur] = it + tabu
        self.fail_cur[c] = (cur + 1) % TABU_SLOTS
        self.cand[c, slots] = glyphs
        self.logits.data[c, slots] = arrival
        self.w_ema[c, slots] = 0.0
        self.age[c, slots] = 0

    @torch.no_grad()
    def tabu_add(self, c, glyphs, it, tabu=500):
        """Bar (cell, glyph) pairs from re-nomination -- used for measured live-probe REJECTIONS
        (stronger evidence than an audition loss: the real network scored the hard swap worse)."""
        if c.numel() == 0:
            return
        cur = self.fail_cur[c]
        self.fail_glyph[c, cur] = glyphs
        self.fail_until[c, cur] = it + tabu
        self.fail_cur[c] = (cur + 1) % TABU_SLOTS

    @torch.no_grad()
    def propose_full(self, bm_flat, codebook, it, margin=1.0, grace=100, evict_below=1.0,
                     channels=("grad",), neighbor_margin=0.0, affinity_margin=0.0,
                     latent_margin=0.0, latent_floor=0.1,
                     ports_floor=0.0, ports_nb_weight=1.0, per_chan=1):
        """EVERY channel's gated proposal per cell -- no pick, no mutation. The live-probe path
        measures several channels' nominations per cell and admits the BEST (channel priority is
        then only queue order / tiebreak, not preemption). Returns None if no gradient EMA yet,
        else a dict: props (C,M) glyph per channel per cell (-1 = abstain), chans (C,) channel
        codes, valid (M,C) gate+dedup pass, slot (M,) evictee slot, out_g (M,) evictee glyph,
        ok_cell (M,) evictable+off-cooldown, gate_counts (6,).

        Channel semantics (each proposes <=1 glyph per cell; margins in member-score-STD units,
        None = the channel carries its own evidence): grad = codebook argmin of the linear score
        (blind reach), neighbor = modal 4-neighbor snap (coordination), affinity = strongest
        affinity partner's snap (consolidation -- mostly propagates exploration's wins),
        latent = most latent-adjacent glyph to the pool's weight mass (family travel), ports =
        tile-continuity match, file = external certified list. Validity also requires
        non-member / non-tabu / density-band / >= dedup_eps from every member."""
        if self.g_ema is None:
            return None
        INF = float("inf")
        S = self.g_ema @ bm_flat.t()                                     # (M, N)
        blocked = torch.zeros_like(S, dtype=torch.bool)                  # invalid-as-nominee (M,N)
        blocked.scatter_(1, self.cand, True)                             # members
        act = (self.fail_until > it) & (self.fail_glyph >= 0)            # live tabu entries
        if act.any():
            rows = self._ar[:, None].expand_as(self.fail_glyph)
            blocked[rows[act], self.fail_glyph[act]] = True
        if self.density_block is not None:
            blocked |= self.density_block
        s_mem = S.gather(1, self.cand)                                   # (M,K) member scores
        sigma_c = s_mem.std(dim=1) + 1e-12
        best_mem = s_mem.min(dim=1).values
        # Channel proposals in PRIORITY order (the `channels` tuple). The pick is the FIRST channel
        # whose gate passes, NOT the best score: grad's proposal is the codebook argmin, so a
        # score-pick hands it every cell it gates into, and its plentiful low-conversion proposals
        # would monopolize the one-audition-at-a-time cooldowns.
        # Scarce, high-conversion coordination proposals go first.
        chan_margin = {"grad": margin, "neighbor": neighbor_margin, "affinity": affinity_margin,
                       "latent": latent_margin, "ports": None, "file": None}   # None = no grad gate
        chan_code = {"grad": 0, "neighbor": 1, "affinity": 2, "latent": 3, "ports": 4, "file": 5}
        snaps = self.snap() if any(n in channels for n in ("neighbor", "affinity", "ports")) else None
        props, gates, chan_ids = [], [], []
        seconds = []           # (pg2, gate, chan_code) runner-up block, appended AFTER all firsts
        for name in channels:
            if name == "grad":
                Sg = S.masked_fill(blocked, INF)
                first = Sg.argmin(dim=1)
                props.append(first)
                if per_chan >= 2:
                    Sg2 = Sg.clone(); Sg2[self._ar, first] = INF
                    seconds.append((Sg2.argmin(dim=1), chan_margin[name], chan_code[name]))
            elif name == "neighbor":
                props.append(self._modal_neighbor(snaps))
            elif name == "affinity" and self.affinity_partner is not None:
                props.append(torch.where(self.affinity_partner >= 0,
                                         snaps[self.affinity_partner.clamp_min(0)],
                                         torch.full_like(snaps, -1)))
            elif name == "ports" and self.ports_T is not None:
                # tile continuity: target port match + live-neighbor compatibility. Proposals only
                # from the tile whitelist (ports_T = -inf elsewhere); abstain below the floor.
                g2 = snaps.view(self.GH, self.GW)
                Lg = torch.roll(g2, 1, 1).reshape(-1)
                Rg = torch.roll(g2, -1, 1).reshape(-1)
                Ug = torch.roll(g2, 1, 0).reshape(-1)
                Dg = torch.roll(g2, -1, 0).reshape(-1)
                NB = (self.ports_Ch[Lg] + self.ports_Ch[:, Rg].t()
                      + self.ports_Cv[Ug] + self.ports_Cv[:, Dg].t())          # (M,N)
                Sp = (self.ports_T + ports_nb_weight * NB).masked_fill(blocked, float("-inf"))
                best = Sp.argmax(dim=1)
                pg = torch.where(Sp[self._ar, best] >= ports_floor, best,
                                 torch.full_like(best, -1))
                props.append(pg)
                if per_chan >= 2:
                    Sp2 = Sp.clone(); Sp2[self._ar, best] = float("-inf")
                    b2 = Sp2.argmax(dim=1)
                    pg2 = torch.where(Sp2[self._ar, b2] >= ports_floor, b2,
                                      torch.full_like(b2, -1))
                    seconds.append((pg2, chan_margin[name], chan_code[name]))
            elif name == "file" and self.file_prop is not None:
                props.append(self.file_prop)
            elif name == "latent" and self.cb_prox is not None:
                # adjacency to the pool's WEIGHT MASS: A(c,g) = sum_k w_ema[c,k] * P[cand[c,k], g].
                # argmax over non-blocked glyphs; abstain (pg=-1) where even the best adjacency is
                # below the floor (an isolated pool has no meaningful relatives to propose).
                A = torch.einsum("mk,mkn->mn", self.w_ema, self.cb_prox[self.cand])
                Am = A.masked_fill(blocked, float("-inf"))
                best = Am.argmax(dim=1)
                pg = torch.where(Am[self._ar, best] >= latent_floor, best,
                                 torch.full_like(best, -1))
                props.append(pg)
                if per_chan >= 2:
                    Am2 = Am.clone(); Am2[self._ar, best] = float("-inf")
                    b2 = Am2.argmax(dim=1)
                    pg2 = torch.where(Am2[self._ar, b2] >= latent_floor, b2,
                                      torch.full_like(b2, -1))
                    seconds.append((pg2, chan_margin[name], chan_code[name]))
            else:
                continue
            gates.append(chan_margin[name]); chan_ids.append(chan_code[name])
        # runner-up block AFTER all firsts: the probe's per-cell candidate list then reads
        # "one pass over every channel, then the seconds" (grad/ports/latent only -- the
        # score-matrix channels where a second-best is well-defined)
        for pg2, mg2, cc2 in seconds:
            props.append(pg2); gates.append(mg2); chan_ids.append(cc2)
        if not props:
            return None
        valid = torch.zeros(self.M, len(props), dtype=torch.bool, device=self.device)
        gate_counts = torch.zeros(6, dtype=torch.long, device=self.device)   # by channel CODE
        for j, (pg, mg) in enumerate(zip(props, gates)):                 # per-channel gate
            pgc = pg.clamp_min(0)
            ok_j = (pg >= 0) & ~blocked[self._ar, pgc]
            if mg is not None:                                           # legacy grad-score gate;
                sj = S[self._ar, pgc]                                    # ports/file carry their own
                ok_j &= sj < best_mem - mg * sigma_c                     # evidence (mg=None)
            if self.dedup_eps > 0:                                       # near-duplicate of a member?
                d_pool = (codebook[pgc][:, None, :] - codebook[self.cand]).norm(dim=-1).min(dim=1).values
                ok_j &= d_pool >= self.dedup_eps
            valid[:, j] = ok_j
            gate_counts[chan_ids[j]] = int(ok_j.sum())
        # evictee: weakest LEARNED slot, grace-protected, must be genuinely abandoned by the
        # optimizer (below a uniform weight share), space never evicted (the escape hatch)
        eligible = (self.age >= grace) & (self.w_ema < evict_below / self.K)
        if self.space_idx is not None:
            eligible &= self.cand != self.space_idx
        w_masked = torch.where(eligible, self.w_ema, torch.full_like(self.w_ema, INF))
        slot = w_masked.argmin(dim=1)                                    # (M,)
        out_g = self.cand[self._ar, slot]
        cool = self.age.min(dim=1).values >= grace                       # one audition at a time
        ok_cell = eligible.any(dim=1) & cool
        if self.search_mask is not None:
            ok_cell &= self.search_mask                                  # empty cells: closed to search
        return dict(props=torch.stack(props), chans=torch.tensor(chan_ids, device=self.device),
                    valid=valid, slot=slot, out_g=out_g, ok_cell=ok_cell, gate_counts=gate_counts)

    @torch.no_grad()
    def nominate(self, bm_flat, codebook, it, margin=1.0, grace=100, tabu=500,
                 rank_lo=2, rank_hi=5, evict_below=1.0,
                 channels=("grad",), neighbor_margin=0.0, affinity_margin=0.0,
                 latent_margin=0.0, latent_floor=0.1,
                 ports_floor=0.0, ports_nb_weight=1.0, dry_run=False):
        """Classic one-proposal-per-cell round: the FIRST channel (priority order) whose gate
        passes wins the cell; the swap is applied immediately (unless dry_run). The live-probe
        path uses propose_full() directly instead, measuring several channels per cell and
        admitting the best. Returns (cells, glyph_in, glyph_out, channel, gate_counts, slots)."""
        z0_ = torch.zeros(0, dtype=torch.long, device=self.device)
        empty = (z0_,) * 4 + (torch.zeros(6, dtype=torch.long, device=self.device), z0_)
        P = self.propose_full(bm_flat, codebook, it, margin=margin, grace=grace,
                              evict_below=evict_below, channels=channels,
                              neighbor_margin=neighbor_margin, affinity_margin=affinity_margin,
                              latent_margin=latent_margin, latent_floor=latent_floor,
                              ports_floor=ports_floor, ports_nb_weight=ports_nb_weight)
        if P is None:
            return empty
        valid, props = P["valid"], P["props"]
        gate_counts = P["gate_counts"]
        has_prop = valid.any(dim=1)
        prio = torch.arange(props.shape[0], 0, -1, device=self.device, dtype=torch.float32)
        pick = (valid.float() * prio).argmax(dim=1)      # first passing channel (positional tiebreak)
        nom = props[pick, self._ar].clamp_min(0)                          # (M,)
        chan = P["chans"][pick]                                           # (M,) channel code
        slot, out_g = P["slot"], P["out_g"]
        ok = has_prop & P["ok_cell"]
        if not ok.any():
            return empty[:4] + (gate_counts, empty[0])
        c = ok.nonzero().flatten()
        if not dry_run:
            self.apply_swaps(c, slot[c], nom[c], out_g[c], it, tabu=tabu,
                             rank_lo=rank_lo, rank_hi=rank_hi)
        return c, nom[c], out_g[c], chan[c], gate_counts, slot[c]
