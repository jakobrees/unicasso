"""Traveling-slot particle swarm (--mode swarm): pool mode's successor parameterization.

The setting: the target image is a grid of cells, and each cell must ultimately ship exactly one
glyph. A glyph VAE embeds every glyph of the font as a point in latent space -- the codebook --
and both predecessor modes parameterize a cell's choice around it. knn-smooth gives each cell ONE
learned latent position z and renders the softmax blend of z's k nearest codebook glyphs,
w = softmax(-d(z, codebook)_topk / tau), annealing the temperature tau so the blend sharpens into
a decision. Pool mode (unicasso/engine/pool.py) instead froze candidates at codebook points and
learned free logits over them.

Both hit the same wall, measurable on the codebook geometry: a settled knn-smooth cell's rivals
sit a full nearest-neighbor (NN) spacing (~3.0 latent units) away while the blend gap is
exp(-gap/tau) -- rivals are render-INVISIBLE at any practical temperature the moment a cell
decides, so all rival gradient dies at settlement (staleness), and discrete candidates can't do
within-family refinement (the `===╦` vs `╤≡≡` gap). knn-smooth papered over the invisibility with
z-noise excursions (exploration noise on z: sigma ~ NN/3 -> ~1.5 std to the Voronoi wall, ~40%
audition odds per step, SGLD via --z-noise-commit); the swarm keeps that machinery AND makes
rivals deterministically visible.

Per cell: K slots. Slot k = a LEARNED latent position z_k (moves by gradient, like knn-smooth's z)
plus a learned weight logit W_k (arbitrates hypotheses, like pool logits). Two-level render:

    v = softmax(W / tau_W)                                (across slots; spread-capped -> all visible)
    blend_k = softmax(-d(z_k, codebook)_topk / tau_z,k)   (per-slot knn-smooth in miniature, k_z~4;
                                                           tau_z,k scaled by local codebook density)
    render_c = sum_k v_k * blend_k . bitmaps

Gradient to z_k and W_k both scale with v_k: the W-spread cap keeps every slot render-visible and
gradient-alive, so a 20%-weight challenger REFINES ITSELF while losing (slides `═`->`╤`->`≡`
continuously) and wins on merit -- discrete latent-channel pitches of a moving target are replaced
by continuous navigation. Snap = argmax W -> that slot's nearest glyph.

Birth-death economy (pool mode's live-probe economy, graded). Nomination channels -- heuristic
proposers (gradient, neighbor context, tile continuity, pixel fit, ...) -- pitch outsider glyphs
at cells. A nominee latent-NEAR a live slot BOOSTS that slot's W and nudges its z toward the
nominee (dedup-by-boost: "another chance" + the where-hint) -- free, no measurement. A far nominee
is live-probed (hard-swapped into the current render and its loss delta actually measured) and
admitted as a slot birth -- strongly (measured clearly better) near the top of the W pack, or as a
minority-weight LOTTERY TICKET (measured ~neutral: ahead-of-its-time hypotheses may exist and wait
for their neighbors instead of being penalized for arriving first); clearly-worse -> tabu (barred
from re-nomination in that cell for a while). Co-located slots merge (logsumexp preserves earned
mass; no vote-splitting at snap), freeing respawn budget for births.

Commitment: tau_W collapses EARLY (the expensive cross-family mixture discretizes while the winner
can still slide to compensate), tau_z collapses late (cheap look-alike discretization) -- the
commitment shock paid in two small installments.
"""
import torch
import torch.nn as nn

from unicasso.engine.pool import GUMBEL_MEAN, GUMBEL_STD, TABU_SLOTS

CHAN_CODE = {"grad": 0, "neighbor": 1, "affinity": 2, "latent": 3, "ports": 4, "file": 5,
             "blend": 6, "join": 7, "pixel": 8}


class ParticleSwarm:
    """Per-cell slot state (z, W, colors) plus the birth/boost/merge/tabu bookkeeping for
    --mode swarm; see the module docstring for the design argument. The name is historical:
    this is not classical (Kennedy-Eberhart) particle swarm optimization -- there are no
    velocities or best-position attractors, just gradient-trained latent hypotheses
    arbitrated by learned weights."""

    def __init__(self, GH, GW, K, L, device):
        M = GH * GW
        self.GH, self.GW = GH, GW
        self.M, self.K, self.L, self.device = M, K, L, device
        self.z = nn.Parameter(torch.zeros(M, K, L, device=device))      # slot latent positions
        self.W = nn.Parameter(torch.zeros(M, K, device=device))         # slot weight logits
        # COLOR mode: per-CANDIDATE fg/bg. Each slot owns the pair it would ship, so W arbitrates
        # (shape, color) jointly -- a slot can win on color with a mediocre glyph or vice versa.
        # Closed-form MSE-initialized at seed/birth, then free for CLIP/recon to move.
        self.fg = nn.Parameter(torch.zeros(M, K, 3, device=device))
        self.bg = nn.Parameter(torch.ones(M, K, 3, device=device))
        self.color_on = False            # render returns cell_rgb only when set
        self.tgt_cell_rgb = None         # (M,P,3) target cells -- closed-form fit source
        self.bm_flat = None              # (N,P) glyph bitmaps, for fitting a newborn's colors
        self.pal = None                  # (Q,3) quantization palette (None = continuous)
        self.pal_lab = None
        # FIT mode: fg/bg are not free leaves at all, but the closed-form MSE optimum against
        # the target, recomputed from each slot's OWN ink every forward. See `_fit_colors`.
        self.color_fit = False
        self.color_fit_mc = 0.0          # in-loop legibility floor (= recolor's --min-contrast)
        self.color_fit_ridge = 1e-3
        self.color_fit_detach = False    # True -> colors track the shape but pass no gradient back
        self.color_bg_dist = None        # (N,P) per-glyph bg-distance table (color.glyph_bg_dist)
        self.color_bg_dist_pow = 0.0     # bg votes ~ dist^pow from the glyph's ink; 0 = plain fit
        self.cluster_fg = None           # (M,3) decompose cluster colors -- blended into the fit
        self.cluster_bg = None
        self.cluster_alpha = 0.0         # cluster share of the shipped colors (0 = pure MSE fit)
        # Learned PER-CELL contrast, applied AFTER the min-contrast floor so k=1 reproduces the
        # fitted colors exactly and switching it on mid-run is continuous, not a step.
        self.k_contrast = nn.Parameter(torch.ones(M, device=device))
        self.k_contrast_on = False       # asciify gates this to the run's tail
        self.k_contrast_max = 4.0
        self.free = torch.zeros(M, K, dtype=torch.bool, device=device)  # parked (merged-away) slots
        self._ar = torch.arange(M, device=device)
        self.v_ema = torch.zeros(M, K, device=device)                   # clean across-slot weight EMA
        self.age = torch.zeros(M, K, dtype=torch.long, device=device)
        self.last_boost = torch.full((M, K), -10**9, dtype=torch.long, device=device)
        self.fail_glyph = torch.full((M, TABU_SLOTS), -1, dtype=torch.long, device=device)
        self.fail_until = torch.zeros(M, TABU_SLOTS, dtype=torch.long, device=device)
        self.fail_cur = torch.zeros(M, dtype=torch.long, device=device)
        self.g_ema = None                # (M, P) render-gradient EMA (nomination signal)
        self.gz_ema = None               # (M, K) per-slot z-grad-norm EMA (legibility diagnostic)
        # short-horizon dynamics trackers (recording only; see track_step)
        self.prev_slot_g = None          # (M,K) last step's slot glyphs
        self.flips = torch.zeros(M, K, dtype=torch.long, device=device)   # per-slot basin flips
        self.step_ema = torch.zeros(M, K, device=device)                  # per-slot ||dz||/step EMA
        self._prev_z = None
        self.space_idx = None
        self.latent_h = 1.0              # latent-channel adjacency bandwidth (absolute units)
        self.density_block = None        # (M,N) bool nominee ink band (as pool)
        self.affinity_partner = None     # (M,) long
        self.ports_T = self.ports_Ch = self.ports_Cv = None
        self.port_mass = None            # (N,4) per-glyph L/R/U/D edge-band ink (dangling penalty)
        self.blend_block = None          # (N,) bool: glyphs the blend channel may never propose
        # (letters by default: accent dots pixel-match blur specks)
        self.nom_block = None            # (N,) bool: glyphs NO channel may nominate (blanks by
        # default: space stays reachable only via continuous latent drift, never by proposal)
        # height prior (detached, target-side): tips W ties toward the slot whose ink centroid
        # matches the target cell's -- the anti-grid-bias tie-breaker (the collective can settle
        # on a coherent line at the WRONG height; nothing else coordinates
        # on height). Self-annealing: a fixed W-unit bias divided by the hardening tau_W.
        self.h_bias = None               # (M,K) last computed render-time W bias
        self.tgt_cent = None             # (M,2) target cell ink centroid (yc, xc) in [0,1]
        self.tgt_norm = None             # (M,2) unit normal (ny, nx) to the cell's line direction:
        # centroid error is PROJECTED onto it -- along-line offset is irrelevant (and for
        # verticals, demanding x-match blocks the only available stubs)
        self.cent_gate = None            # (M,) coherence * ink-band gate
        self.glyph_cent = None           # (N,2) glyph bitmap ink centroids (join gate)
        self.pix_fit = None              # (M,N) static full-res target-vs-glyph MSE (pixel channel)
        self.cell_coh = None             # (M,) raw target-cell coherence (pixel regime split)
        self.cell_ink = None             # (M,) target-cell mean ink
        self.tone_gate = None            # (M,) tone-regime gate (incoherent AND dark): the
        # tone W-bias population -- same cells the pixel channel proposes shades for
        self.glyph_ink = None            # (N,) glyph mean ink (tone density matching)
        self.tone_mask = None            # (N,) bool: shade/block family (tone proposals)
        self.boost_ink_ok = None         # (M,) bool: cells where boosts may fire (near-white
        # target cells excluded: free unmeasured boosts can build phantom │-columns there)
        self.file_prop = None
        self.search_mask = None          # (M,) bool: cells open to search (empties closed)

    # ---------------------------------------------------------------- init
    @torch.no_grad()
    def init_from_z0(self, z0, codebook, empty_safe=None, space_idx=None, empty_thresh=0.5,
                     single=False):
        """Slot 0 = the warm-start position itself (knn-smooth's soft off-manifold blend preserved);
        slots 1..K-1 = the 2nd..Kth nearest DISTINCT glyphs' latents (on-manifold rival hypotheses).
        W = -(d_j - d_1) raw gaps; the first clamp_w_spread projects them into capped units, so
        iteration 0 ranks like knn-smooth ranked. Confidently-empty cells: every slot parked on the
        space latent with a big slot-0 lead (they're closed to search and cap-exempt).
        single=True: only slot 0 is live; slots 1..K-1 start FREE (pure birth budget) -- membership
        is earned through measured births instead of seeded k-NN (no init-duplicate merge wave)."""
        d = torch.cdist(z0, codebook)                                    # (M, N)
        vals, idx = torch.topk(d, self.K, dim=1, largest=False)
        self.z.data[:, 0] = z0
        if single:
            for j in range(1, self.K):
                self.z.data[:, j] = z0
            self.W.data.zero_()
            self.free[:, 1:] = True
        else:
            for j in range(1, self.K):
                self.z.data[:, j] = codebook[idx[:, j]]
            self.W.data.copy_(-(vals - vals[:, 0:1]))
        n_seed = 0
        if space_idx is not None and empty_safe is not None:
            em = empty_safe > empty_thresh
            if em.any():
                self.z.data[em] = codebook[space_idx][None, None, :]
                self.W.data[em] = 0.0
                self.W.data[em, 0] = 3.0                                 # decisive space lead
            n_seed = int(em.sum())
        return n_seed

    # ---------------------------------------------------------------- render pieces
    def weights(self, tau_w, noise_scale=0.0, noise_gate=None, bias=None):
        """Across-slot v = softmax((W + bias + sigma*gumbel)/tau_w); free slots masked out."""
        l = self.W if bias is None else self.W + bias
        if noise_scale > 0:
            u = torch.rand_like(l).clamp_(1e-9, 1.0 - 1e-9)
            g = (-torch.log(-torch.log(u)) - GUMBEL_MEAN) / GUMBEL_STD
            if noise_gate is not None:
                g = g * noise_gate[:, None]
            l = l + noise_scale * g
        l = l.masked_fill(self.free, -1e4)
        if torch.is_tensor(tau_w) and tau_w.dim() == 1:
            return torch.softmax(l / tau_w.clamp_min(1e-6)[:, None], dim=1)
        return torch.softmax(l / max(float(tau_w), 1e-6), dim=1)

    def _colors(self):
        """fg/bg as the render will actually use them: clamped to [0,1], and snapped to the
        palette when one is set -- both straight-through, so the optimizer sees the value it
        will ship while the gradient still flows to the raw leaf.

        The palette is the defense against the color shortcut: with fg/bg free and continuous,
        any cell can hit any tone by tinting a full block, which is smooth, free, and strictly
        easier than finding that tone structurally -- the tone channels would never fire again."""
        fg = self.fg + (self.fg.clamp(0, 1) - self.fg).detach()
        bg = self.bg + (self.bg.clamp(0, 1) - self.bg).detach()
        if self.pal is not None:
            from unicasso.engine.color import srgb_to_lab
            fq = self.pal[torch.cdist(srgb_to_lab(fg.detach().reshape(-1, 3)), self.pal_lab)
                          .argmin(1)].reshape(fg.shape)
            bq = self.pal[torch.cdist(srgb_to_lab(bg.detach().reshape(-1, 3)), self.pal_lab)
                          .argmin(1)].reshape(bg.shape)
            fg = fg + (fq - fg).detach()
            bg = bg + (bq - bg).detach()
        return fg, bg

    def set_palette(self, pal):
        """pal (Q,3) RGB or None."""
        from unicasso.engine.color import srgb_to_lab
        self.pal = pal
        self.pal_lab = srgb_to_lab(pal) if pal is not None else None

    def _quantize(self, fg, bg):
        """Straight-through palette snap. Shared by both color modes so a palette behaves
        identically whether the colors were learned or fitted."""
        if self.pal is None:
            return fg, bg
        from unicasso.engine.color import srgb_to_lab
        fq = self.pal[torch.cdist(srgb_to_lab(fg.detach().reshape(-1, 3)), self.pal_lab)
                      .argmin(1)].reshape(fg.shape)
        bq = self.pal[torch.cdist(srgb_to_lab(bg.detach().reshape(-1, 3)), self.pal_lab)
                      .argmin(1)].reshape(bg.shape)
        return fg + (fq - fg).detach(), bg + (bq - bg).detach()

    def _mc_push(self, fg, bg):
        """The legibility floor, applied to whatever shape fg/bg have. Identical arithmetic to
        recolor.py's --min-contrast: pure MSE drives fg->bg wherever the cell is locally smooth,
        which makes the glyph VANISH and the result read as a soft photo rather than as text."""
        if self.color_fit_mc <= 0:
            return fg
        lum = lambda t: 0.299 * t[..., 0] + 0.587 * t[..., 1] + 0.114 * t[..., 2]
        gap = lum(fg) - lum(bg)
        push = torch.where(gap <= 0, -1.0, 1.0) * (self.color_fit_mc - gap.abs()).clamp_min(0)
        return fg + push[..., None]

    def _apply_k(self, fg, bg):
        """Per-cell learned contrast: push fg/bg apart around the cell's own midpoint.

            mid = (fg + bg)/2,   fg <- mid + k*(fg - mid),   bg <- mid + k*(bg - mid)

        The cell's MEAN color is invariant in k, so this cannot shift hue or brightness -- it can
        only trade pixel error for glyph legibility, which is the one axis where MSE and perception
        genuinely disagree. That narrowness is the point: it is why a learned k does NOT reopen the
        color shortcut --color-fit closed. Free fg/bg could hit any tone by tinting a whole block;
        k cannot move the cell's mean at all.

        Broadcasts over whatever slot axes fg carries: (M,K,3) in the loop, (M,3) at emission."""
        if not self.k_contrast_on:
            return fg, bg
        k = self.k_contrast
        k = k + (k.clamp(0.0, self.k_contrast_max) - k).detach()      # STE clamp
        kk = k.view(-1, *([1] * (fg.dim() - 1)))
        mid = 0.5 * (fg + bg)
        return mid + kk * (fg - mid), mid + kk * (bg - mid)

    def _fit_colors(self, slot_img, slot_g=None):
        """Per-slot fg/bg as the CLOSED-FORM MSE optimum against the target cell, recomputed
        from each slot's OWN current ink every forward. In this mode fg/bg are NOT parameters.
        slot_g (M,K): each slot's nearest glyph, for the bg-distance weighting.

        Why fit instead of learn. With free color leaves CLIP picks the colors, and CLIP
        optimizes perception, not pixel error -- pixel error lands several times worse,
        with a visible cast. The run then ships a post-hoc MSE refit, which means the thing that
        was optimized is NOT the thing that ships: every gradient step was spent tuning a render
        that emission later replaces underneath it. Fitting in-loop closes that gap. CLIP sees
        the shipping colors, so its gradient is about the image you actually get, and the final
        refit degrades from a correction into a consistency check.

        It also closes the color shortcut outright rather than taxing it. Free continuous fg/bg
        let any cell hit any tone by tinting a full block -- smooth, free, and strictly easier
        than finding that tone structurally, which is what would eventually silence the tone and
        grating channels. Here the colors are a FUNCTION of the ink, so that move does not exist.

        Closed form, per slot k:  fg_k = mean of c weighted by ink_k,
                                  bg_k = mean of c weighted by (1 - ink_k).
        Differentiable through ink, so moving a glyph's shape moves its optimal colors and the
        loss feels it. Never materializes (M,K,P,3) -- the einsum contracts P directly, keeping
        the same memory discipline as the render path above.
        """
        C = self.tgt_cell_rgb                                        # (M,P,3)
        ink = 1.0 - slot_img                                         # (M,K,P), 1 = glyph ink
        if self.color_fit_detach:
            ink = ink.detach()
        r = self.color_fit_ridge
        P = ink.shape[-1]
        mean = C.mean(1)                                             # (M,3) degenerate-mask fallback
        sw = ink.sum(-1)                                             # (M,K) ink mass
        num_f = torch.einsum("mkp,mpc->mkc", ink, C)                 # (M,K,3)
        fg = (num_f + r * mean[:, None, :]) / (sw + r).clamp_min(1e-6)[:, :, None]
        if self.color_bg_dist is not None and self.color_bg_dist_pow > 0 and slot_g is not None:
            # bg votes weighted by distance from the slot's nearest glyph's ink: stroke-fringe
            # pixels (antialiasing / slight misalignment) barely count. The weights are constants
            # w.r.t. the graph (detached index gather); gradient still flows through ink.
            d = self.color_bg_dist[slot_g].pow(self.color_bg_dist_pow)  # (M,K,P)
            wb = (1.0 - ink) * d
            num_b = torch.einsum("mkp,mpc->mkc", wb, C)
            bg = (num_b + r * mean[:, None, :]) / (wb.sum(-1) + r).clamp_min(1e-6)[:, :, None]
        else:
            bg = ((C.sum(1)[:, None, :] - num_f) + r * mean[:, None, :]) \
                / ((P - sw) + r).clamp_min(1e-6)[:, :, None]
        if self.cluster_fg is not None and self.cluster_alpha > 0:
            # ship (and optimize against) a mix of the decomposition's own cluster colors and
            # the fit: cluster preserves the cell's bimodal contrast and is glyph-independent;
            # the fit keeps the rendered mean honest. Same mix at emission (fit_emit_colors).
            a = self.cluster_alpha
            fg = a * self.cluster_fg[:, None, :] + (1 - a) * fg
            bg = a * self.cluster_bg[:, None, :] + (1 - a) * bg
        fg = self._mc_push(fg, bg)
        fg, bg = self._apply_k(fg, bg)
        # straight-through clamp: ship the clamped value but keep the gradient, since a hard
        # clamp zeroes it exactly where a saturated cell most needs to move
        fg = fg + (fg.clamp(0, 1) - fg).detach()
        bg = bg + (bg.clamp(0, 1) - bg).detach()
        return self._quantize(fg, bg)

    @torch.no_grad()
    def fit_emit_colors(self, mask, gidx=None):
        """fg/bg (M,3) for a HARD placed mask (M,P) -- the emission twin of `_fit_colors`.

        Without this, emission would read the stale `fg`/`bg` leaves (last written at seed or
        birth) while the loop had been optimizing against freshly fitted colors, so the shipped
        .ans would not be the image that was optimized. Same fit, same floor, same palette.
        gidx (M,): placed glyph indices, for the bg-distance weighting."""
        from unicasso.engine.color import fit_fg_bg
        bg_w = None
        if self.color_bg_dist is not None and self.color_bg_dist_pow > 0 and gidx is not None:
            bg_w = self.color_bg_dist[gidx].pow(self.color_bg_dist_pow)
        fg, bg = fit_fg_bg(self.tgt_cell_rgb, mask, self.color_fit_ridge, bg_w=bg_w)
        if self.cluster_fg is not None and self.cluster_alpha > 0:
            a = self.cluster_alpha
            fg = a * self.cluster_fg + (1 - a) * fg
            bg = a * self.cluster_bg + (1 - a) * bg
        fg = self._mc_push(fg, bg)
        fg, bg = self._apply_k(fg, bg)
        fg, bg = fg.clamp(0, 1), bg.clamp(0, 1)
        if self.pal is not None:
            from unicasso.engine.color import snap_to_palette
            fg, _ = snap_to_palette(fg, self.pal)
            bg, _ = snap_to_palette(bg, self.pal)
        return fg, bg

    @torch.no_grad()
    def fit_slot_colors(self, masks, cells=None, slots=None, ridge=1e-3):
        """Closed-form fg/bg for given ink masks -- the init at seed/birth/reseed.
        masks (n,P) ink (1 = glyph ink). cells/slots (n,) index the slots to write;
        None/None means "all slots", masks then (M,K,P)."""
        if self.tgt_cell_rgb is None:
            return
        from unicasso.engine.color import fit_fg_bg
        if cells is None:
            M, K = self.M, self.K
            tgt = self.tgt_cell_rgb[:, None].expand(M, K, -1, 3).reshape(M * K, -1, 3)
            fg, bg = fit_fg_bg(tgt, masks.reshape(M * K, -1), ridge)
            self.fg.data.copy_(fg.view(M, K, 3))
            self.bg.data.copy_(bg.view(M, K, 3))
            return
        fg, bg = fit_fg_bg(self.tgt_cell_rgb[cells], masks, ridge)
        self.fg.data[cells, slots] = fg
        self.bg.data[cells, slots] = bg

    def render(self, z_eff, codebook, bm_flat, bm_sq, kz, tau_z, tau_w,
               temp_lut=None, w_noise=0.0, w_gate=None, height=None, tone_w=0.0):
        """Two-level soft render. Returns dict: cell (M,P), embed (M,L), v (M,K), div (M,),
        slot_g (M,K) each slot's nearest glyph (detached), slot_d1 (M,K) distance to it,
        commit_ref (M,K,L) detached nearest-glyph latents (for the per-slot commit anchor).

        height: optional (weight, ygrid (P,), xgrid (P,)) height-prior spec; with
        self.tgt_cent/cent_gate set, each slot's blend-ink centroid error vs the target
        centroid becomes a detached W bias (stored in self.h_bias, reused by snap)."""
        M, K, L = self.M, self.K, self.L
        d = torch.cdist(z_eff.reshape(M * K, L), codebook)               # (MK, N)
        vals, idxk = torch.topk(d, kz, dim=1, largest=False)             # (MK, kz)
        tz = tau_z[:, None].repeat(1, K).reshape(M * K, 1) if torch.is_tensor(tau_z) and tau_z.dim() == 1 \
            else torch.full((M * K, 1), float(tau_z), device=self.device)
        if temp_lut is not None:                                         # per-slot density scaling:
            tz = tz * temp_lut[idxk[:, 0]][:, None]                      # sparse regions blend WIDER
        bl = torch.softmax(-vals / tz.clamp_min(1e-6), dim=1)            # (MK, kz)
        slot_img = (bl[:, :, None] * bm_flat[idxk]).sum(1).view(M, K, -1)     # (M,K,P)
        slot_emb = (bl[:, :, None] * codebook[idxk]).sum(1).view(M, K, L)     # (M,K,L)
        w_bias = None
        if height is not None and self.tgt_cent is not None:
            hw, yg, xg = height
            with torch.no_grad():
                si = 1.0 - slot_img.detach()                             # ink space (M,K,P)
                mass = si.sum(-1)
                yc = (si * yg).sum(-1) / mass.clamp_min(1e-6)
                xc = (si * xg).sum(-1) / mass.clamp_min(1e-6)
                dy, dx = yc - self.tgt_cent[:, 0:1], xc - self.tgt_cent[:, 1:2]
                if self.tgt_norm is not None:      # error projected onto the line's normal
                    e = (self.tgt_norm[:, 0:1] * dy + self.tgt_norm[:, 1:2] * dx).abs()
                else:
                    e = dy.abs() + dx.abs()
                # near-inkless slot in a line-evidence cell: centroid is undefined; fixed
                # moderate penalty ("no ink where a coherent line demonstrably is")
                e = torch.where(mass < 0.5, torch.full_like(e, 0.5), e)
                w_bias = -hw * self.cent_gate[:, None] * e               # (M,K), W units
        if tone_w > 0 and self.tone_gate is not None and self.cell_ink is not None:
            # tone W-bias: in incoherent-dark cells (structure demonstrably undrawable),
            # density mismatch of the slot's CURRENT blend is penalized -- self-annealing
            # via /tau_w, so it's a nudge in exploration and decisive at commitment. The
            # counterweight to CLIP's overshoot asymmetry (tone slots get admitted,
            # then out-raced by thin bars). Reads current density -> drift-proof; blank
            # slots in dark cells penalized continuously (e = target ink).
            with torch.no_grad():
                rho = 1.0 - slot_img.detach().mean(-1)                   # (M,K) blend ink
                e_t = (rho - self.cell_ink[:, None]).abs()
                tb = -tone_w * self.tone_gate[:, None] * e_t
                w_bias = tb if w_bias is None else w_bias + tb
        if w_bias is not None:
            self.h_bias = w_bias                    # snap() reads the combined bias
        v = self.weights(tau_w, w_noise, w_gate, bias=w_bias)            # (M,K)
        cell = (v[:, :, None] * slot_img).sum(1)                         # (M,P)
        embed = (v[:, :, None] * slot_emb).sum(1)                        # (M,L)
        slot_sq = (bl * bm_sq[idxk]).sum(1).view(M, K)                   # per-slot E||B||^2
        div = (v * slot_sq).sum(1) - cell.pow(2).sum(1)                  # two-level blend variance
        cell_rgb = div_c = None
        if self.color_on:
            # cell_rgb[m,p] = sum_k v_k * ( bg_k + (fg_k - bg_k) * ink_k[p] ),  ink = 1 - slot_img
            # Never materialize slot_rgb (M,K,P,3) -- at M=1800,K=3 that is 42MB/forward and it
            # scales with the grid; the two einsums below give (M,P,3) straight out.
            fg, bg = (self._fit_colors(slot_img, idxk[:, 0].view(M, K)) if self.color_fit
                      else self._colors())                             # (M,K,3) each
            ink = 1.0 - slot_img                                         # (M,K,P)
            dcol = fg - bg                                               # (M,K,3)
            vb = (v[:, :, None] * bg).sum(1)                             # (M,3)
            cell_rgb = vb[:, None, :] + torch.einsum("mkp,mkc->mpc", ink, v[:, :, None] * dcol)
            # COLOR blend variance: soft-phase W mixes RGB across slots, so a red slot and a blue
            # slot at v=.5 render PURPLE -- a color no candidate can ship after the snap. Grayscale
            # never had this (blending shapes is meaningful); here it is a live reward hack. Same
            # form as `div`: E_k||slot_rgb||^2 - ||cell_rgb||^2, expanded so no (M,K,P,3) appears.
            e_sq = (v * bg.pow(2).sum(-1)).sum(1)[:, None] \
                + 2.0 * torch.einsum("mkp,mk->mp", ink, v * (bg * dcol).sum(-1)) \
                + torch.einsum("mkp,mk->mp", ink.pow(2), v * dcol.pow(2).sum(-1))
            div_c = (e_sq - cell_rgb.pow(2).sum(-1)).clamp_min(0).sum(1)  # (M,)
        return dict(cell=cell, embed=embed, v=v, div=div,
                    cell_rgb=cell_rgb, div_c=div_c,
                    slot_img=slot_img,                                   # (M,K,P) per-slot blends
                    slot_g=idxk[:, 0].view(M, K).detach(),
                    slot_d1=vals[:, 0].view(M, K).detach(),
                    commit_ref=codebook[idxk[:, 0]].view(M, K, L).detach())

    @torch.no_grad()
    def slot_glyphs(self, codebook):
        """(M,K) nearest glyph per slot + (M,K) distance (clean z, no noise)."""
        d = torch.cdist(self.z.data.reshape(self.M * self.K, self.L), codebook)
        dv, gi = d.min(dim=1)
        return gi.view(self.M, self.K), dv.view(self.M, self.K)

    @torch.no_grad()
    def snap(self, codebook):
        """argmax clean W (free-masked, height-bias included) -> that slot's nearest glyph."""
        l = self.W.data if self.h_bias is None else self.W.data + self.h_bias
        k = l.masked_fill(self.free, float("-inf")).argmax(dim=1)
        zw = self.z.data[self._ar, k]
        return torch.cdist(zw, codebook).argmin(dim=1)

    # ---------------------------------------------------------------- housekeeping
    @torch.no_grad()
    def clamp_w_spread(self, cap):
        """Project live slots' W onto {std <= cap} per cell (center + one-sided shrink; see
        pool.clamp_spread rationale -- audition reach is gap-vs-sigma, tau cancels). Free slots are
        excluded from the statistics and re-parked below the live pack; closed (empty) cells keep
        their decisive space lead untouched."""
        l = self.W.data
        live = ~self.free
        n = live.sum(dim=1, keepdim=True).clamp_min(1).float()
        mu = (l * live).sum(dim=1, keepdim=True) / n
        var = (((l - mu) * live) ** 2).sum(dim=1, keepdim=True) / n.clamp_min(1)
        sd = var.sqrt()
        scale = (cap / sd.clamp_min(1e-8)).clamp(max=1.0)
        if self.search_mask is not None:
            scale = torch.where(self.search_mask[:, None], scale, torch.ones_like(scale))
        newl = (l - mu) * scale
        lmin = newl.masked_fill(self.free, float("inf")).min(dim=1, keepdim=True).values
        self.W.data = torch.where(live, newl, lmin - 1.0)                # park free slots below

    @torch.no_grad()
    def update_stats(self, tau_w, ema=0.95):
        self.v_ema.mul_(ema).add_(self.weights(tau_w), alpha=1.0 - ema)
        self.age += 1

    @torch.no_grad()
    def update_grad_ema(self, g_cells, decay=0.98):
        if self.g_ema is None:
            self.g_ema = g_cells.clone()
        else:
            self.g_ema.mul_(decay).add_(g_cells, alpha=1.0 - decay)

    @torch.no_grad()
    def track_step(self, codebook, ema=0.98):
        """Per-step dynamics trackers (call AFTER the optimizer step + SGLD commit, so the step
        length is the total per-iter movement): per-slot basin-flip counts (the z-noise reach /
        late-churn diagnostic) and a per-slot step-length EMA (is the walk sized to tau, or
        wandering / frozen). Costs one cdist -- only wired when recording."""
        sg, _ = self.slot_glyphs(codebook)
        if self.prev_slot_g is not None:
            self.flips += (sg != self.prev_slot_g).long()
        self.prev_slot_g = sg
        if self._prev_z is not None:
            self.step_ema.mul_(ema).add_((self.z.data - self._prev_z).norm(dim=-1), alpha=1.0 - ema)
        self._prev_z = self.z.data.clone()

    @torch.no_grad()
    def update_zgrad_ema(self, decay=0.98):
        """Per-slot z-grad-norm EMA: the 'is the gradient legible for slot k' diagnostic."""
        if self.z.grad is None:
            return
        gn = self.z.grad.norm(dim=-1)                                    # (M,K)
        if self.gz_ema is None:
            self.gz_ema = gn.clone()
        else:
            self.gz_ema.mul_(decay).add_(gn, alpha=1.0 - decay)

    @torch.no_grad()
    def tabu_add(self, c, glyphs, it, tabu=500):
        if c.numel() == 0:
            return
        cur = self.fail_cur[c]
        self.fail_glyph[c, cur] = glyphs
        self.fail_until[c, cur] = it + tabu
        self.fail_cur[c] = (cur + 1) % TABU_SLOTS

    @torch.no_grad()
    def _modal_neighbor(self, snaps):
        g = snaps.view(self.GH, self.GW)
        nb = torch.stack([torch.roll(g, 1, 0), torch.roll(g, -1, 0),
                          torch.roll(g, 1, 1), torch.roll(g, -1, 1)], -1).view(self.M, 4)
        eq = (nb[:, :, None] == nb[:, None, :]).sum(-1)
        return nb[self._ar, eq.argmax(1)]

    # ---------------------------------------------------------------- proposals (pool channels)
    @torch.no_grad()
    def propose_full(self, bm_flat, codebook, it, margin=1.0, grace=100, evict_below=1.0,
                     channels=("grad",), neighbor_margin=0.0, affinity_margin=0.0,
                     latent_margin=0.0, latent_floor=0.1,
                     ports_floor=0.0, ports_nb_weight=1.0,
                     blur_T=None, blend_margin=0.75, blend_nb_weight=0.02, join_floor=1.0,
                     blend_resid=0.02, join_dangling=3.0, join_centroid=0.0,
                     join_coord_sigma=0.0, pixel_margin=0.85, pixel_gate=0.25,
                     pixel_tone_ink=0.35, per_chan=1):
        """pool.propose_full adapted to slots: members = each live slot's nearest glyph; the latent
        channel reads adjacency to the ACTUAL slot positions (continuous weight mass, not frozen
        candidates); the birth target is a FREE slot when one exists (merge respawn budget), else
        the weakest abandoned live slot. Same return contract as pool + is_free (M,)."""
        if self.g_ema is None:
            return None
        INF = float("inf")
        slot_g, _ = self.slot_glyphs(codebook)                           # (M,K)
        S = self.g_ema @ bm_flat.t()                                     # (M,N)
        blocked = torch.zeros_like(S, dtype=torch.bool)
        blocked.scatter_(1, slot_g, True)                                # already-represented glyphs
        act = (self.fail_until > it) & (self.fail_glyph >= 0)
        if act.any():
            rows = self._ar[:, None].expand_as(self.fail_glyph)
            blocked[rows[act], self.fail_glyph[act]] = True
        if self.density_block is not None:
            blocked |= self.density_block
        if self.nom_block is not None:
            blocked |= self.nom_block[None]
        s_mem = S.gather(1, slot_g)                                      # (M,K) member scores
        sigma_c = s_mem.std(dim=1) + 1e-12
        best_mem = s_mem.min(dim=1).values
        chan_margin = {"grad": margin, "neighbor": neighbor_margin, "affinity": affinity_margin,
                       "latent": latent_margin, "ports": None, "file": None,
                       "blend": None, "join": None, "pixel": None}  # blend/join/pixel: own evidence
        chan_code = CHAN_CODE
        snaps = self.snap(codebook) if any(
            n in channels for n in ("neighbor", "affinity", "ports", "blend", "join", "pixel")) else None
        Jn = None   # live-neighbor join score (M,N), shared by ports-style channels
        if self.ports_Ch is not None and snaps is not None \
                and any(n in channels for n in ("join", "blend")):
            g2_ = snaps.view(self.GH, self.GW)
            Lg_ = torch.roll(g2_, 1, 1).reshape(-1)
            Rg_ = torch.roll(g2_, -1, 1).reshape(-1)
            Ug_ = torch.roll(g2_, 1, 0).reshape(-1)
            Dg_ = torch.roll(g2_, -1, 0).reshape(-1)
            Jn = (self.ports_Ch[Lg_] + self.ports_Ch[:, Rg_].t()
                  + self.ports_Cv[Ug_] + self.ports_Cv[:, Dg_].t())
            if self.port_mass is not None and join_dangling > 0:
                # DANGLING-PORT penalty: a candidate's port pointing at a neighbor whose FACING
                # edge is blank connects to nothing -- charge it, so a corner beats a T-cross
                # when there's no third stroke to meet (`╕` over `╤` with blank right).
                pm = self.port_mass                                      # (N,4) L/R/U/D band ink
                dang = torch.zeros_like(Jn)
                for ngh_g, opp, side in ((Lg_, 1, 0), (Rg_, 0, 1), (Ug_, 3, 2), (Dg_, 2, 3)):
                    blank_face = (pm[ngh_g, opp] < 0.02).float()         # (M,) neighbor edge empty
                    dang += blank_face[:, None] * pm[:, side][None]      # (M,1)*(1,N)
                Jn = Jn - join_dangling * dang
        props, gates, chan_ids = [], [], []
        seconds = []       # (pg2, gate, code): runner-up block appended AFTER all firsts
        for name in channels:
            if name == "grad":
                Sg = S.masked_fill(blocked, INF)
                first_g = Sg.argmin(dim=1)
                props.append(first_g)
                if per_chan >= 2:
                    Sg2 = Sg.clone(); Sg2[self._ar, first_g] = INF
                    seconds.append((Sg2.argmin(dim=1), chan_margin[name], chan_code[name]))
            elif name == "neighbor":
                props.append(self._modal_neighbor(snaps))
            elif name == "affinity" and self.affinity_partner is not None:
                props.append(torch.where(self.affinity_partner >= 0,
                                         snaps[self.affinity_partner.clamp_min(0)],
                                         torch.full_like(snaps, -1)))
            elif name == "ports" and self.ports_T is not None:
                g2 = snaps.view(self.GH, self.GW)
                Lg = torch.roll(g2, 1, 1).reshape(-1)
                Rg = torch.roll(g2, -1, 1).reshape(-1)
                Ug = torch.roll(g2, 1, 0).reshape(-1)
                Dg = torch.roll(g2, -1, 0).reshape(-1)
                NB = (self.ports_Ch[Lg] + self.ports_Ch[:, Rg].t()
                      + self.ports_Cv[Ug] + self.ports_Cv[:, Dg].t())
                Sp = (self.ports_T + ports_nb_weight * NB).masked_fill(blocked, float("-inf"))
                best = Sp.argmax(dim=1)
                props.append(torch.where(Sp[self._ar, best] >= ports_floor, best,
                                         torch.full_like(best, -1)))
                if per_chan >= 2:
                    Sp2 = Sp.clone(); Sp2[self._ar, best] = float("-inf")
                    b2 = Sp2.argmax(dim=1)
                    seconds.append((torch.where(Sp2[self._ar, b2] >= ports_floor, b2,
                                                torch.full_like(b2, -1)),
                                    chan_margin[name], chan_code[name]))
            elif name == "file" and self.file_prop is not None:
                props.append(self.file_prop)
            elif name == "blend" and blur_T is not None:
                # BLEND: realize what the cell is painting. Fit against the blur's PRESENCE map
                # (binarized -- raw MSE prefers the dominant component over the union glyph and
                # can never propose the `╪` a `═│` mixture sketches); small join tie-break.
                # Abstain unless the best fit beats the INCUMBENT's fit by the margin ratio --
                # a committed cell (blur ≈ its snap) proposes nothing.
                P_ = bm_flat.shape[1]
                Bi = 1.0 - bm_flat                                       # ink space (N,P)
                Fit = (blur_T.pow(2).sum(1, keepdim=True)
                       - 2.0 * blur_T @ Bi.t() + Bi.pow(2).sum(1)[None]) / P_   # (M,N)
                S_b = Fit - (blend_nb_weight * Jn if Jn is not None else 0.0)
                S_b = S_b.masked_fill(blocked, INF)
                if self.blend_block is not None:
                    S_b = S_b.masked_fill(self.blend_block[None], INF)
                best = S_b.argmin(dim=1)
                fit_inc = Fit[self._ar, snaps]                           # incumbent's blur fit
                # RESIDUAL gate: the blur must contain meaningful ink the incumbent does NOT
                # already cover (an `L`+`_` union ≈ `L` has ~zero residual -> nothing to realize)
                resid = (blur_T * (1.0 - Bi[snaps])).mean(1)             # (M,) uncovered blur frac
                ok_b = (Fit[self._ar, best] < blend_margin * fit_inc) & (fit_inc > 0.01) \
                    & (resid > blend_resid)
                props.append(torch.where(ok_b, best, torch.full_like(best, -1)))
                if per_chan >= 2:
                    Sb2 = S_b.clone(); Sb2[self._ar, best] = INF
                    b2 = Sb2.argmin(dim=1)
                    ok_b2 = (Fit[self._ar, b2] < blend_margin * fit_inc) & (fit_inc > 0.01) \
                        & (resid > blend_resid)
                    seconds.append((torch.where(ok_b2, b2, torch.full_like(b2, -1)),
                                    chan_margin[name], chan_code[name]))
            elif name == "join" and Jn is not None:
                # JOIN: pure tile-connection completion, blur-blind (`═▄`/`║` -> `╗`): argmax
                # live-neighbor port score; must clear the floor AND beat the incumbent's score.
                Js = Jn
                if join_coord_sigma > 0 and self.glyph_cent is not None \
                        and self.tgt_cent is not None:
                    # coord-agreement shaping BEFORE the argmax: join redirects to the best
                    # right-position completion instead of proposing a wrong-height one and
                    # losing its round to a veto. Subtractive quadratic in score units (join
                    # scores go negative via the dangling penalty, so a multiplicative factor
                    # would flip meaning); applied to candidates AND incumbent.
                    if getattr(self, "_join_ec", None) is None:
                        dcy = self.glyph_cent[None, :, 0] - self.tgt_cent[:, None, 0]
                        dcx = self.glyph_cent[None, :, 1] - self.tgt_cent[:, None, 1]
                        if self.tgt_norm is not None:
                            ec_ = (self.tgt_norm[:, 0:1] * dcy + self.tgt_norm[:, 1:2] * dcx).abs()
                        else:
                            ec_ = dcy.abs() + dcx.abs()
                        self._join_ec = (ec_ / join_coord_sigma).pow(2) \
                            * self.cent_gate[:, None]                     # (M,N) static
                    Js = Jn - self._join_ec
                Jm = Js.masked_fill(blocked, float("-inf"))
                best = Jm.argmax(dim=1)
                j_inc = Js[self._ar, snaps]
                ok_j2 = (Jm[self._ar, best] >= join_floor) & (Jm[self._ar, best] > j_inc)
                if join_centroid > 0 and self.glyph_cent is not None and self.tgt_cent is not None:
                    # centroid gate: join may complete connections, but not by installing a
                    # line at the wrong height in a coherent-line cell (the grid-bias
                    # defender: join otherwise propagates a wrong-height consensus). Error projected onto
                    # the line normal so along-line offset can't block valid stubs.
                    dc = self.glyph_cent[best] - self.tgt_cent                  # (M,2)
                    if self.tgt_norm is not None:
                        ec = (self.tgt_norm * dc).sum(1).abs()
                    else:
                        ec = dc.abs().sum(1)
                    ok_j2 = ok_j2 & ((self.cent_gate < 0.5) | (ec <= join_centroid))
                props.append(torch.where(ok_j2, best, torch.full_like(best, -1)))
                if per_chan >= 2:
                    Jm2 = Jm.clone(); Jm2[self._ar, best] = float("-inf")
                    jb2 = Jm2.argmax(dim=1)
                    ok_jb2 = (Jm2[self._ar, jb2] >= join_floor) & (Jm2[self._ar, jb2] > j_inc)
                    seconds.append((torch.where(ok_jb2, jb2, torch.full_like(jb2, -1)),
                                    chan_margin[name], chan_code[name]))
            elif name == "pixel" and self.pix_fit is not None and snaps is not None:
                # PIXEL: static target-truth advocate, split by cell coherence. Blend colludes
                # with the render's own consensus ("realize your blur"); this channel keeps
                # re-proposing what the TARGET's pixels say. Coherent cells: best full-res
                # pixel-match glyph (structure: pixel-MSE can rank `▁` first where the
                # collective settles on `─`). Incoherent DARK cells (the grill
                # signature: isotropic texture, ink too dense to draw): the density-matched
                # tone-family glyph -- tone reads better there, but no other channel
                # proposes it. Probe/tabu/density gates veto as usual.
                Fp = self.pix_fit.masked_fill(blocked, INF)
                inc = self.pix_fit[self._ar, snaps]
                coh = self.cell_coh if self.cell_coh is not None \
                    else torch.ones(self.M, device=self.device)
                best_s = Fp.argmin(dim=1)
                ok_s = (coh > pixel_gate) & (Fp[self._ar, best_s] < pixel_margin * inc) \
                    & (inc > 1e-4)
                best = best_s
                okp = ok_s
                if self.tone_mask is not None and self.glyph_ink is not None \
                        and self.cell_ink is not None:
                    dens = (self.glyph_ink[None, :] - self.cell_ink[:, None]).abs()
                    dens = dens.masked_fill(blocked | ~self.tone_mask[None, :], INF)
                    best_t = dens.argmin(dim=1)
                    ok_t = (coh <= pixel_gate) & (self.cell_ink >= pixel_tone_ink) \
                        & torch.isfinite(dens[self._ar, best_t]) \
                        & (Fp[self._ar, best_t] < pixel_margin * inc) & (inc > 1e-4)
                    best = torch.where(ok_t, best_t, best_s)
                    okp = ok_s | ok_t
                props.append(torch.where(okp, best, torch.full_like(best, -1)))
                if per_chan >= 2:
                    Fp2 = Fp.clone(); Fp2[self._ar, best_s] = INF
                    ps2 = Fp2.argmin(dim=1)
                    ok_ps2 = (coh > pixel_gate) & (Fp2[self._ar, ps2] < pixel_margin * inc) \
                        & (inc > 1e-4)
                    seconds.append((torch.where(ok_ps2, ps2, torch.full_like(ps2, -1)),
                                    chan_margin[name], chan_code[name]))
            elif name == "latent":
                # adjacency to the slots' weight mass at their ACTUAL positions:
                # A(c,g) = sum_k v_ema[c,k] * exp(-||z_k - cb_g||^2 / 2h^2)
                d2 = torch.cdist(self.z.data.reshape(self.M * self.K, self.L), codebook).pow(2)
                prox = torch.exp(-d2 / (2 * self.latent_h ** 2)).view(self.M, self.K, -1)
                vm = self.v_ema * (~self.free).float()
                A = torch.einsum("mk,mkn->mn", vm, prox).masked_fill(blocked, float("-inf"))
                best = A.argmax(dim=1)
                props.append(torch.where(A[self._ar, best] >= latent_floor, best,
                                         torch.full_like(best, -1)))
                if per_chan >= 2:
                    A2 = A.clone(); A2[self._ar, best] = float("-inf")
                    ab2 = A2.argmax(dim=1)
                    seconds.append((torch.where(A2[self._ar, ab2] >= latent_floor, ab2,
                                                torch.full_like(ab2, -1)),
                                    chan_margin[name], chan_code[name]))
            else:
                continue
            gates.append(chan_margin[name]); chan_ids.append(chan_code[name])
        for pg2, mg2, cc2 in seconds:      # seconds AFTER the full first pass over channels
            props.append(pg2); gates.append(mg2); chan_ids.append(cc2)
        if not props:
            return None
        valid = torch.zeros(self.M, len(props), dtype=torch.bool, device=self.device)
        gate_counts = torch.zeros(9, dtype=torch.long, device=self.device)
        for j, (pg, mg) in enumerate(zip(props, gates)):
            pgc = pg.clamp_min(0)
            ok_j = (pg >= 0) & ~blocked[self._ar, pgc]
            if mg is not None:
                sj = S[self._ar, pgc]
                ok_j &= sj < best_mem - mg * sigma_c
            valid[:, j] = ok_j
            gate_counts[chan_ids[j]] = int(ok_j.sum())
        # birth slot: FREE first (respawn budget), else weakest abandoned live slot (grace-protected,
        # below a uniform weight share, never a slot currently showing space -- the escape hatch)
        eligible = (self.age >= grace) & (self.v_ema < evict_below / self.K) & ~self.free
        if self.space_idx is not None:
            eligible &= slot_g != self.space_idx
        has_free = self.free.any(dim=1)
        free_slot = self.free.float().argmax(dim=1)
        w_masked = torch.where(eligible, self.v_ema, torch.full_like(self.v_ema, INF))
        evict_slot = w_masked.argmin(dim=1)
        slot = torch.where(has_free, free_slot, evict_slot)
        out_g = torch.where(has_free, torch.full_like(slot, -1), slot_g[self._ar, evict_slot])
        cool = self.age.min(dim=1).values >= grace
        ok_cell = (has_free | eligible.any(dim=1)) & cool
        if self.search_mask is not None:
            ok_cell &= self.search_mask
        return dict(props=torch.stack(props), chans=torch.tensor(chan_ids, device=self.device),
                    valid=valid, slot=slot, out_g=out_g, ok_cell=ok_cell,
                    gate_counts=gate_counts, is_free=has_free, slot_g=slot_g)

    # ---------------------------------------------------------------- boost / birth / merge
    @torch.no_grad()
    def boost_from_proposals(self, P, codebook, it, radius, amount, nudge, cooldown,
                             exclude_chans=()):
        """Dedup-by-boost, vectorized over all open cells: take each cell's FIRST valid proposal
        (channel priority order); if its glyph latent lies within `radius` of a live NON-ARGMAX
        slot, boost that slot's W by `amount` and nudge its z `nudge` of the way toward the
        nominee (the where-hint). Free -- no measurement; per-slot cooldown stops runaway
        re-boosting. Consumed proposals are invalidated in P so the probe path skips them.
        exclude_chans: channel NAMES whose proposals may only take the measured probe path,
        never the free boost path (e.g. 'join' -- it reinforces the neighbor consensus, so it
        doesn't get unaudited reinforcement). Returns (cells, slots, glyphs, stats) -- stats
        counts near-slot proposals BLOCKED (argmax rule / cooldown)."""
        empty = (torch.zeros(0, dtype=torch.long, device=self.device),) * 3 \
            + (dict(near=0, argmax=0, cool=0),)
        valid, props = P["valid"], P["props"]
        if exclude_chans:
            codes = torch.tensor([CHAN_CODE[n] for n in exclude_chans if n in CHAN_CODE],
                                 device=self.device)
            if codes.numel():
                valid = valid & ~torch.isin(P["chans"], codes)[None, :]
        has = valid.any(dim=1)
        if self.search_mask is not None:
            has &= self.search_mask
        if not has.any():
            return empty
        prio = torch.arange(props.shape[0], 0, -1, device=self.device, dtype=torch.float32)
        pick = (valid.float() * prio).argmax(dim=1)
        nom = props[pick, self._ar].clamp_min(0)                         # (M,)
        dz = (self.z.data - codebook[nom][:, None, :]).norm(dim=-1)      # (M,K) slot->nominee dist
        dz = dz.masked_fill(self.free, float("inf"))
        kmin = dz.argmin(dim=1)
        near = has & (dz[self._ar, kmin] < radius)
        argmax_k = self.W.data.masked_fill(self.free, float("-inf")).argmax(dim=1)
        not_arg = kmin != argmax_k
        off_cool = it - self.last_boost[self._ar, kmin] >= cooldown
        # split rule (most near-slot proposals land on the WINNER; blocking them all would
        # kill within-family refinement): near a LOSER slot -> full boost (W bump
        # + nudge, "another chance" + where-hint); near the ARGMAX slot -> z-NUDGE ONLY (the winner
        # doesn't need weight, it needs the direction -- this is the `=`->`╤` where-hint).
        ok_full = near & not_arg & off_cool
        ok_nudge = near & ~not_arg & off_cool
        if self.boost_ink_ok is not None:            # no free boosts where the target is near-white
            ok_full &= self.boost_ink_ok             # (unmeasured boost cascades can build
            ok_nudge &= self.boost_ink_ok            # phantom │-columns in shaded background)
        ok = ok_full | ok_nudge
        stats = dict(near=int(near.sum()), nudge_win=int(ok_nudge.sum()),
                     cool=int((near & ~off_cool).sum()))
        if not ok.any():
            return empty[:3] + (stats,)
        c = ok.nonzero().flatten()
        k = kmin[c]
        g = nom[c]
        full = ok_full[c]
        if full.any():
            self.W.data[c[full], k[full]] += amount
        self.z.data[c, k] += nudge * (codebook[g] - self.z.data[c, k])
        self.last_boost[c, k] = it
        P["valid"][c, pick[c]] = False                                   # consumed: not probe material
        return c, k, g, stats

    @torch.no_grad()
    def birth(self, c, slots, glyphs, evicted, strong, codebook, it, tabu=500, strong_gap=0.1):
        """Measured admissions. strong (bool per birth): W lands strong_gap below the live max
        (a real contender immediately); lottery: at the live min (visible via the cap, must earn).
        Evicted live glyphs -> tabu (free-slot births evict nothing, evicted = -1)."""
        if c.numel() == 0:
            return
        ev = evicted >= 0
        if ev.any():
            self.tabu_add(c[ev], evicted[ev], it, tabu=tabu)
        Wl = self.W.data[c].masked_fill(self.free[c], float("nan"))
        wmax = Wl.nan_to_num(float("-inf")).max(dim=1).values
        wmin = Wl.nan_to_num(float("inf")).min(dim=1).values
        arrival = torch.where(strong, wmax - strong_gap, wmin)
        self.z.data[c, slots] = codebook[glyphs]
        self.W.data[c, slots] = arrival
        self.free[c, slots] = False
        self.v_ema[c, slots] = 0.0
        self.age[c, slots] = 0
        self.last_boost[c, slots] = it
        if self.color_on and self.bm_flat is not None:
            # fit the arrival's OWN colors: inheriting the evictee's pair would judge a new
            # glyph through the old one's palette and lose contenders that only differ in color
            self.fit_slot_colors(1.0 - self.bm_flat[glyphs], c, slots)

    @torch.no_grad()
    def purge_to_winner(self, codebook, it, avoid=None, keep=1, reseed=1):
        """ELITE PURGE at a reheat seam: per open cell, keep the top-`keep` live slots by W (the
        elites); every other live slot is freed = respawn budget. Up to `reseed` freed slots per
        cell are immediately re-seeded at RANDOM allowed codebook glyphs at W = elite − 1 (the cap
        re-projects next step). The random SEED itself rarely wins, but the wildcard
        SLOT is a fresh mobile vehicle that can travel to a winner -- reseeds buy travel, not
        glyphs. The reheat restores the soft regime for the new generation's arbitration."""
        live = ~self.free
        Wm = self.W.data.masked_fill(self.free, float("-inf"))
        k1 = Wm.argmax(dim=1)
        keep_n = max(1, min(keep, self.K))
        topk = Wm.topk(keep_n, dim=1).indices                            # (M, keep)
        open_m = (self.search_mask if self.search_mask is not None
                  else torch.ones(self.M, dtype=torch.bool, device=self.device))
        keep_m = torch.zeros_like(self.free)
        keep_m.scatter_(1, topk, True)
        keep_m &= live                                                   # can't keep a free slot
        newfree = live & ~keep_m & open_m[:, None]
        self.free |= newfree
        self.v_ema[newfree] = 0.0
        n_purged = int(newfree.sum())
        allowed = torch.ones(codebook.shape[0], dtype=torch.bool, device=self.device)
        if avoid is not None:
            allowed &= ~avoid
        pool_ = allowed.nonzero().flatten()
        rank = newfree.float().cumsum(dim=1) * newfree.float()           # 1..n over freed slots
        cs, ks = [], []
        for j in range(max(0, reseed)):
            mask_j = newfree & (rank == j + 1)
            has = mask_j.any(dim=1)
            slot = mask_j.float().argmax(dim=1)
            g = pool_[torch.randint(0, pool_.numel(), (self.M,), device=self.device)]
            c = has.nonzero().flatten()
            if self.density_block is not None and c.numel():             # wildcard still ink-plausible
                c = c[~self.density_block[c, g[c]]]
            if c.numel():
                self.z.data[c, slot[c]] = codebook[g[c]]
                self.W.data[c, slot[c]] = self.W.data[c, k1[c]] - 1.0
                self.free[c, slot[c]] = False
                self.age[c, slot[c]] = 0
                self.v_ema[c, slot[c]] = 0.0
                self.last_boost[c, slot[c]] = it
                if self.color_on and self.bm_flat is not None:
                    self.fit_slot_colors(1.0 - self.bm_flat[g[c]], c, slot[c])
                cs.append(c); ks.append(slot[c])
        if cs:
            return n_purged, torch.cat(cs), torch.cat(ks)
        return n_purged, self._ar[:0], self._ar[:0]

    @torch.no_grad()
    def merge(self, eps, it):
        """One merge per cell per call: the closest live slot pair under eps folds into the higher-
        v_ema member (W = logsumexp -- earned mass preserved, no vote-splitting at snap); the loser
        is parked free = respawn budget for births. Returns (cells, pair_dist, v_ema_folded) --
        the folded v_ema is the merge-eps wrong-level diagnostic: routinely folding slots that
        still held real weight means eps is destroying live diversity, not recycling duplicates."""
        empty = (torch.zeros(0, dtype=torch.long, device=self.device),
                 torch.zeros(0, device=self.device), torch.zeros(0, device=self.device))
        z = self.z.data
        dz = (z[:, :, None, :] - z[:, None, :, :]).norm(dim=-1)          # (M,K,K)
        K = self.K
        iu, ju = torch.triu_indices(K, K, offset=1, device=self.device)
        pd = dz[:, iu, ju]                                               # (M, K*(K-1)/2)
        live = ~self.free
        pair_ok = live[:, iu] & live[:, ju]
        pd = pd.masked_fill(~pair_ok, float("inf"))
        best = pd.argmin(dim=1)
        bd = pd[self._ar, best]
        do = bd < eps
        if self.search_mask is not None:
            do &= self.search_mask
        if not do.any():
            return empty
        c = do.nonzero().flatten()
        i, j = iu[best[c]], ju[best[c]]
        vi, vj = self.v_ema[c, i], self.v_ema[c, j]
        surv = torch.where(vi >= vj, i, j)
        lose = torch.where(vi >= vj, j, i)
        v_folded = self.v_ema[c, lose].clone()
        self.W.data[c, surv] = torch.logaddexp(self.W.data[c, surv], self.W.data[c, lose])
        self.free[c, lose] = True
        self.v_ema[c, lose] = 0.0
        return c, bd[c], v_folded
