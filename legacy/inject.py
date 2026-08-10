"""Candidate injection for knn-smooth (local neighbor coordination).

Idea: a continuity/consistency spring acts on z, which CLIP can't *see* until z has already moved.
Instead, temporarily add a NEIGHBOR cell's current glyph as a **bias-only** candidate in this cell's
softmax blend, so it's RENDERED and CLIP/recon can adopt it in place if it fits. Survival-of-the-
fittest on the existing softmax: the candidate's (learnable) bias grows if it helps, shrinks if not;
prune the losers, snap to whatever has the biggest weight. Spread is *emergent* -- neighbor sampling
reads each cell's LIVE current glyph, so a successful injection becomes a candidate for its neighbors
next cycle (no explicit "viral" code). One slot per cell. Refinement-phase only.

The injected logit is bias-only (decoupled from latent distance): an injected glyph is deliberately
latent-far, so -cdist would start it invisible; the bias alone controls its visibility and CLIP/recon
shape it. A negative bias is simply "bad fit" -> it loses weight -> pruned.
"""
import torch
import torch.nn as nn


class Injector:
    def __init__(self, GH, GW, device, init_rank_lo=2, init_rank_hi=5, start_frac=0.5, rate=50,
                 prune_every=100, grace=100, fail_fade=500, ema=0.95):
        self.GH, self.GW, self.M, self.device = GH, GW, GH * GW, device
        self.init_lo, self.init_hi, self.start_frac, self.rate = init_rank_lo, init_rank_hi, start_frac, rate
        self.prune_every, self.grace, self.fail_fade, self.ema = prune_every, grace, fail_fade, ema
        self.idx = torch.full((self.M,), -1, dtype=torch.long, device=device)   # injected glyph (-1 = empty)
        self.bias = nn.Parameter(torch.zeros(self.M, device=device))            # learnable; goes in the optimizer
        self.age = torch.zeros(self.M, dtype=torch.long, device=device)
        self.w_ema = torch.zeros(self.M, device=device)                         # injected-slot weight EMA
        self.nat_min_ema = torch.ones(self.M, device=device)                    # weakest native top-k weight EMA
        self.fail_glyph = torch.full((self.M,), -1, dtype=torch.long, device=device)
        self.fail_until = torch.zeros(self.M, dtype=torch.long, device=device)
        self._ar = torch.arange(self.M, device=device)

    def started(self, it, iters):
        return it >= int(self.start_frac * iters)

    def extend(self, idxk, blend_logits):
        """idxk (M,k), blend_logits (M,k) -> ext_idx (M,k+1), ext_logits (M,k+1), k_native."""
        k = idxk.shape[1]
        empty = (self.idx < 0)[:, None]
        ext_idx = torch.cat([idxk, self.idx.clamp_min(0)[:, None]], dim=1)      # placeholder glyph for empty
        inj_log = self.bias[:, None].masked_fill(empty, float("-inf"))          # empty slot -> weight 0
        return ext_idx, torch.cat([blend_logits, inj_log], dim=1), k

    @torch.no_grad()
    def record(self, w, k):
        """w (M,k+1) blend weights -> update EMAs + age (call every step while active)."""
        self.w_ema.mul_(self.ema).add_(w[:, k], alpha=1 - self.ema)
        self.nat_min_ema.mul_(self.ema).add_(w[:, :k].min(1).values, alpha=1 - self.ema)
        self.age += (self.idx >= 0).long()

    @torch.no_grad()
    def diag_maps(self, snaps, it):
        """Per-cell injection diagnostics for asciify --clip-diagnostic. snaps (M,) = current hard snap.
        Returns (status, margin) as (GH,GW):
          status: 0 idle | 1 competing (injected, not the snap) | 2 winning (injected glyph IS the snap)
                  | 3 recently failed (pruned, still in fail-fade window).
          margin: (w_ema - weakest-native-EMA) for MATURE active slots (age>=grace) -- the prune decision
                  surface (>0 blue = safe, <0 red = will be pruned); NaN (grey) for in-grace/idle/failed."""
        active = self.idx >= 0
        winning = active & (self.idx == snaps)
        failed = (self.fail_glyph >= 0) & (it < self.fail_until) & ~active
        status = torch.zeros(self.M, device=self.device)
        status[active] = 1.0
        status[winning] = 2.0
        status[failed] = 3.0
        mature = active & (self.age >= self.grace)
        margin = torch.full((self.M,), float("nan"), device=self.device)
        margin[mature] = (self.w_ema - self.nat_min_ema)[mature]
        return status.view(self.GH, self.GW), margin.view(self.GH, self.GW)

    @torch.no_grad()
    def _modal_neighbor(self, snaps):
        g = snaps.view(self.GH, self.GW)
        nb = torch.stack([torch.roll(g, 1, 0), torch.roll(g, -1, 0),
                          torch.roll(g, 1, 1), torch.roll(g, -1, 1)], -1).view(self.M, 4)
        eq = (nb[:, :, None] == nb[:, None, :]).sum(-1)                         # how many of the 4 each matches
        return nb[self._ar, eq.argmax(1)]                                      # most common neighbor glyph

    @torch.no_grad()
    def step(self, it, iters, idxk, snaps, blend_logits):
        """Periodic prune + inject. idxk (M,k) native candidates (to exclude), snaps (M,) live
        glyph/cell, blend_logits (M,k) native logits (to init b_c to a fair top-rank weight)."""
        if not self.started(it, iters):
            return
        if it % self.prune_every == 0:                                          # prune non-competitive (post-grace)
            loser = (self.idx >= 0) & (self.age >= self.grace) & (self.w_ema < self.nat_min_ema)
            self.fail_glyph[loser] = self.idx[loser]
            self.fail_until[loser] = it + self.fail_fade
            self.idx[loser] = -1
            self.w_ema[loser] = 0.0
        if it % self.rate == 0:                                                 # inject modal neighbor into EMPTY slots
            modal = self._modal_neighbor(snaps)
            faded = (self.fail_glyph == modal) & (it < self.fail_until)
            ok = ((self.idx < 0) & (modal >= 0) & (modal != snaps)
                  & ~(idxk == modal[:, None]).any(1) & ~faded)
            # init b_c to a RANDOM native logit of rank [lo,hi) -> mid-pack of the top-k (NOT top 1-2,
            # which would arrive nearly tied with the incumbent); competitive but must earn its way up.
            kk = blend_logits.shape[1]
            lo = max(0, min(self.init_lo, kk - 1)); hi = max(lo + 1, min(self.init_hi, kk))
            jr = torch.randint(lo, hi, (self.M,), device=self.device)
            init_b = blend_logits[self._ar, jr]
            self.idx[ok] = modal[ok]
            self.bias.data[ok] = init_b[ok]
            self.age[ok] = 0
            self.w_ema[ok] = 0.0
