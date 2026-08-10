"""Corruption + softening sampler for the CLIP domain-adaptation trainer.

Operates in ASCII space: corruptions are edits to a .txt glyph grid, chosen through the
VAE codebook geometry and rendered with the font kit -- so every negative is a state the
optimizer could actually produce, never a pixel-space artifact CLIP could shortcut on.

Families (negatives):
  walk   isotropic latent step + snap over a local cell blob. sigma = the step's expected
         NORM in units of the codebook's median NN spacing (nn1). Measured on the sfmono
         codebook:
         norm 1.0 -> ~9% of touched cells flip, 1.5 -> ~40%, 2.5 -> ~80%; flips land on
         latent-NEAR neighbors -- the hardest, most plausible errors.
  fade   directed interpolation toward the SPACE embedding along a detected line run,
         snapped en route. Small t snaps through the light-glyph ramp (| -> . -> space),
         i.e. the corruption walks the real vanishing-line trajectory, not a caricature.
  shift  a line run displaced one cell perpendicular to its direction -- the grid-phase /
         wrong-height-consensus error.
  confetti  isolated light glyphs sprinkled into deep whitespace -- the ADDED-junk error
         observed in real outputs. Training positives can themselves contain real debris,
         so without this negative the metric only learns junk is ascii-normal, never that
         it's WORSE.
  spur   line glyphs replaced by junction glyphs with one EXTRA arm (┃ -> ┣, ─ -> ┬):
         the spurious-branch error. Arms are detected from bitmaps (ink at the four edge
         midpoints); the replacement is the latent-nearest glyph with arms = old ∪ {new},
         so weight class is preserved (heavy picks heavy, light picks light).

Softening (POSITIVE-preserving augmentation, not a corruption): per-cell latent jitter +
knn blend at a sampled temperature. This reproduces the soft gray renders the CLIP loss
sees for ~90% of an optimization run (animation frames are snapped hard renders and do
not show them).
"""
import os

import numpy as np
import torch

from unicasso.substrate import glyphs as G
from unicasso.substrate import orientation as O
from unicasso.substrate.model import GlyphVAE

from unicasso.substrate import raster as train

FAMILIES = ("walk", "fade", "shift", "confetti", "spur", "break")
# Diffusion-style FIELD noisings (ranked pairs, not margin families): smooth spatial
# dose field + one shared eps, two doses t_lo<t_hi of the SAME trajectory. "wfield" =
# additive small-walk on WHITESPACE cells only (snaps space -> its light-glyph latent
# neighbors = deploy-morphology confetti, field-wide, no curated glyph list); "gfield" =
# VP cosine interpolation of ALL cells toward a shell sample (global degradation axis).
FIELD_FAMILIES = ("wfield", "gfield")
# Rankable dose families: fields + "erode" (nested break-site prefixes -- the REMOVAL
# trajectory: d must rise as connections are deleted; the break margin alone gives
# removal no ordered dose supervision)
RANK_FAMILIES = FIELD_FAMILIES + ("erode",)


class CorruptionSampler:
    def __init__(self, vae_ckpt, device="cpu", profile=None, seed=0, knn_k=8):
        self.device = device
        ck = torch.load(vae_ckpt, map_location=device, weights_only=False)
        cfg = ck.get("config") or {}
        ph = cfg.get("pad", 4)
        pw = cfg.get("pad_w", ph)
        # profile: explicit arg > GLYPHVAE_FONT env (load_glyphs reads it) -- sfmono runs
        # need one of the two, same contract as asciify.
        ink, chars = G.load_glyphs(device=device, pad=(ph, pw), profile=profile)
        if list(chars) != list(ck["chars"]):
            raise ValueError("charset mismatch between glyph render and VAE ckpt "
                             f"({len(chars)} vs {len(ck['chars'])} chars)")
        model = GlyphVAE(latent_dim=ck["latent_dim"],
                         char_h=ck["char_h"], char_w=ck["char_w"]).to(device)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        with torch.no_grad():
            self.codebook, _ = model.encode(ink)               # (N, L)
        self.chars = chars
        self.N = len(chars)
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.space = self.char_to_idx[" "]

        # render bitmaps at the true cell size (unpadded), white=1 -- load_glyphs set the fonts
        self.bitmaps = train.create_char_bitmaps().to(device)  # (N, CH, CW)
        self.CH, self.CW = train.CHAR_HEIGHT, train.CHAR_WIDTH

        # codebook geometry: pairwise distances, nn1 calibration, knn table for softening
        d = torch.cdist(self.codebook, self.codebook)
        d.fill_diagonal_(float("inf"))
        self.nn1_med = float(d.min(dim=1).values.median())
        self.knn_d, self.knn_i = d.topk(knn_k, dim=1, largest=False)   # (N, k) excl. self
        d.fill_diagonal_(0.0)
        # per-dim std so that a "sigma_rel" step has EXPECTED NORM sigma_rel*nn1 (in L dims
        # the norm concentrates at std*sqrt(L) -- without this, labels are 4x off at L=16)
        self._step_std = self.nn1_med / float(np.sqrt(self.codebook.shape[1]))

        # per-glyph structure stats (for region targeting): ink fraction + stroke direction
        gink = 1.0 - self.bitmaps                                # ink=1
        self.g_ink = gink.mean(dim=(1, 2))                       # (N,)
        theta, coh, _ = O.glyph_orientations(gink[:, None], sigma=0.0)
        self.g_theta, self.g_coh = theta.to(device), coh.to(device)
        # "liney" = the coord-loss gate (coherent, ink in the line band)
        self.g_liney = (self.g_coh > 0.5) & (self.g_ink > 0.02) & (self.g_ink < 0.5)

        # arm signatures for the spur family: does ink touch each edge midpoint strip?
        # (up, down, left, right) -- ┃ = {up,down}, ┣ = {up,down,right}, ─ = {left,right}
        g = gink  # (N, CH, CW), ink=1
        cy, cx = self.CH // 2, self.CW // 2
        # arm = ink in the outer THIRD of the center cross-band (span-based, not
        # edge-touch: dashed variants like ┊ never reach the cell edge but do span it)
        bv = g[:, :, cx - 2:cx + 3]                                   # center column band
        bh = g[:, cy - 2:cy + 3, :]                                   # center row band
        th, tw = self.CH // 3, self.CW // 3
        arms = torch.stack([
            bv[:, :th].amax(dim=(1, 2)) > 0.5,                        # up
            bv[:, -th:].amax(dim=(1, 2)) > 0.5,                       # down
            bh[:, :, :tw].amax(dim=(1, 2)) > 0.5,                     # left
            bh[:, :, -tw:].amax(dim=(1, 2)) > 0.5,                    # right
        ], dim=1)                                                     # (N,4): up,down,left,right
        # box-likeness: arms only mean something if the glyph's ink LIVES on the cross
        # (letters graze the bands; box/line glyphs are exactly the cross)
        cross = torch.zeros(self.CH, self.CW)
        cross[:, cx - 2:cx + 3] = 1.0
        cross[cy - 2:cy + 3, :] = 1.0
        self.g_boxlike = ((g * cross).sum(dim=(1, 2))
                          / g.sum(dim=(1, 2)).clamp_min(1e-6)) > 0.6
        self.g_arms = arms
        self._arm_groups = {}                     # frozenset(dirs) -> [boxlike gids] (targets)
        for i in range(self.N):
            if not bool(self.g_boxlike[i]):
                continue
            key = frozenset(j for j in range(4) if bool(arms[i, j]))
            self._arm_groups.setdefault(key, []).append(i)

        # soften must not hesitate TOWARD an arm-superset neighbor: a │ blended toward ├
        # IS a faint spur -- the exact pattern the spur negatives penalize (positive vs
        # negative clash)
        nb = self.g_arms[self.knn_i]                                  # (N, k, 4)
        own = self.g_arms[:, None, :]
        sup = (((nb & own).sum(-1) == own.sum(-1))
               & (nb.sum(-1) > own.sum(-1))
               & (own.sum(-1) >= 2))     # only glyphs that could BE spur bases: the
        self.knn_soft_ok = ~sup          # space glyph own=∅ makes EVERY neighbor a
        # "superset", so gating armless glyphs would strip whitespace cells of soft
        # blending entirely -> whitespace transitions go OOD -> deploy confetti

        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ grid I/O
    def load_txt(self, path):
        """.txt glyph grid -> (GH, GW) long index grid. Unknown chars map to space."""
        with open(path, encoding="utf-8") as f:
            lines = [ln.rstrip("\n") for ln in f]
        while lines and not lines[-1]:
            lines.pop()
        GW = max(len(ln) for ln in lines)
        unknown = set()
        grid = torch.full((len(lines), GW), self.space, dtype=torch.long, device=self.device)
        for y, ln in enumerate(lines):
            for x, c in enumerate(ln):
                i = self.char_to_idx.get(c)
                if i is None:
                    unknown.add(c)
                else:
                    grid[y, x] = i
        if unknown:
            print(f"corrupt: {len(unknown)} unknown char(s) mapped to space: "
                  f"{''.join(sorted(unknown))!r}")
        return grid

    def render(self, grid):
        """(GH, GW) index grid -> (GH*CH, GW*CW) float image, white=1."""
        GH, GW = grid.shape
        cells = self.bitmaps[grid.reshape(-1)].view(GH, GW, self.CH, self.CW)
        return cells.permute(0, 2, 1, 3).reshape(GH * self.CH, GW * self.CW)

    def to_text(self, grid):
        return "\n".join("".join(self.chars[i] for i in row) for row in grid.cpu().tolist())

    # ------------------------------------------------------------------ region samplers
    def _cell_dir(self, g):
        """Quantized along-stroke step (dy, dx) for glyph g from its orientation.
        theta is the direction the stroke RUNS, mod pi, in [-pi/2, pi/2], x-axis = 0."""
        th = float(self.g_theta[g])
        if abs(th) < np.pi / 8:
            return 0, 1                       # horizontal
        if abs(th) > 3 * np.pi / 8:
            return 1, 0                       # vertical
        return (1, 1) if th > 0 else (-1, 1)  # diagonals (y down: th>0 = down-right)

    def _line_run(self, grid, max_len):
        """Grow a contiguous run of inky cells from a random liney seed, along the seed
        glyph's stroke direction (both ways). Returns [(y,x), ...] ordered along the run,
        or None if the grid has no liney cells."""
        GH, GW = grid.shape
        liney = self.g_liney[grid]
        ys, xs = torch.nonzero(liney, as_tuple=True)
        if ys.numel() == 0:
            return None
        for _ in range(8):                                   # a few seed retries for short runs
            j = int(self.rng.integers(ys.numel()))
            y0, x0 = int(ys[j]), int(xs[j])
            dy, dx = self._cell_dir(int(grid[y0, x0]))
            run = [(y0, x0)]
            for sgn in (1, -1):                              # grow forward, then backward
                y, x = y0, x0
                while len(run) < max_len:
                    y, x = y + sgn * dy, x + sgn * dx
                    if not (0 <= y < GH and 0 <= x < GW):
                        break
                    if float(self.g_ink[grid[y, x]]) <= 0.02:
                        break
                    run.append((y, x)) if sgn == 1 else run.insert(0, (y, x))
            if len(run) >= 3:
                return run
        return run                                           # last attempt, even if short

    def _blob(self, grid, size):
        """BFS blob of `size` non-space cells from a random inky seed (random frontier
        pops -> irregular, spatially-correlated region like a real local failure)."""
        GH, GW = grid.shape
        inky = self.g_ink[grid] > 0.02
        ys, xs = torch.nonzero(inky, as_tuple=True)
        if ys.numel() == 0:                                  # blank grid: anywhere
            ys = torch.arange(GH, device=grid.device).repeat_interleave(GW)
            xs = torch.arange(GW, device=grid.device).repeat(GH)
        j = int(self.rng.integers(ys.numel()))
        seed = (int(ys[j]), int(xs[j]))
        blob, frontier = {seed}, [seed]
        while frontier and len(blob) < size:
            y, x = frontier.pop(int(self.rng.integers(len(frontier))))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if (0 <= ny < GH and 0 <= nx < GW and (ny, nx) not in blob
                        and bool(inky[ny, nx])):
                    blob.add((ny, nx))
                    frontier.append((ny, nx))
        return sorted(blob)

    # ------------------------------------------------------------------ corruption families
    def _snap(self, z):
        return torch.cdist(z, self.codebook).argmin(dim=1)

    def corrupt(self, grid, family=None):
        """One corruption -> dict(grid, mask (GH,GW) bool, family, desc, params)."""
        if family is None:
            family = FAMILIES[int(self.rng.integers(len(FAMILIES)))]
        fn = getattr(self, "_" + family)
        out = fn(grid.clone())
        if out is None:               # no line run / degenerate draw: walk always produces one
            out = self._walk(grid.clone())
            out["desc"] += f"  (fallback from {family})"
        return out

    # ---------------------------------------------------- diffusion-style field noise
    def _smooth_field(self, GH, GW, uniform_p=0.25):
        """Smooth spatial dose modulation in [0,1]: gaussian-blurred white noise,
        minmax-normalized (peak somewhere is always 1); sometimes uniform."""
        import torch.nn.functional as tF
        if self.rng.random() < uniform_p:
            return torch.ones(GH, GW), "uniform"
        sig = float(self.rng.uniform(1.5, 5.0))
        n = torch.from_numpy(self.rng.standard_normal((1, 1, GH, GW))).float()
        k = int(2 * round(2 * sig) + 1)
        xs = torch.arange(k, dtype=torch.float32) - k // 2
        g1 = torch.exp(-0.5 * (xs / sig) ** 2)
        g1 = g1 / g1.sum()
        n = tF.conv2d(n, g1.view(1, 1, k, 1), padding=(k // 2, 0))
        n = tF.conv2d(n, g1.view(1, 1, 1, k), padding=(0, k // 2))[0, 0]
        lo, hi = float(n.min()), float(n.max())
        return (n - lo) / (hi - lo + 1e-9), f"field σ={sig:.1f}"

    def _field_grid(self, grid, t, field, eps, kind, amp):
        """One dose t of the shared (field, eps) trajectory -> (grid_t, changed mask)."""
        GH, GW = grid.shape
        idx = grid.reshape(-1)
        z0 = self.codebook[idx]
        m = (t * field.reshape(-1)).to(z0)
        if kind == "w":                                    # whitespace-only additive walk
            ws = idx == self.space
            z = z0 + (m * ws * amp)[:, None] * self._step_std * eps
        else:                                              # VP cosine toward shell sample
            th = (np.pi / 2.0) * m.clamp(0, 1)
            shell = eps / eps.norm(dim=1, keepdim=True) * z0.norm(dim=1, keepdim=True)
            z = torch.cos(th)[:, None] * z0 + torch.sin(th)[:, None] * shell
        new = self._snap(z)
        if kind == "w":
            new = torch.where(ws, new, idx)                # ink cells stay untouched
        g2 = new.view(GH, GW)
        return g2, g2 != grid

    def noise_pair(self, grid, family=None, amp=2.0):
        """Ranked dose pair on ONE noise trajectory: same field, same eps, two doses
        t_lo < t_hi ~ U(0,1). The trainer's rank term demands d(grid_lo, parent) <
        d(grid_hi, parent) -- monotone-meter supervision (anti dose-inversion).
        -> dict(family, t_lo/t_hi, grid_lo/grid_hi, mask_lo/mask_hi, field, desc)."""
        if family is None:
            family = FIELD_FAMILIES[int(self.rng.integers(len(FIELD_FAMILIES)))]
        if family == "erode":
            return self.break_pair(grid)
        kind = "w" if family == "wfield" else "g"
        GH, GW = grid.shape
        field, fdesc = self._smooth_field(GH, GW)
        eps = torch.from_numpy(
            self.rng.standard_normal((GH * GW, self.codebook.shape[1]))).to(self.codebook)
        # Snap is a Voronoi quantizer: cells flip at per-cell dose THRESHOLDS, so uniform
        # t-sampling lands both members on the same side of most thresholds (grid_lo ==
        # original or == grid_hi, no rank signal). The perceptual dose is FLIP COUNT --
        # probe a ladder of doses on this trajectory once, then place t_lo/t_hi at target
        # flip fractions of the trajectory's max.
        ladder = np.geomspace(0.06, 1.0, 15)            # log-spaced: resolution where the
        outs = [self._field_grid(grid, float(tv), field, eps, kind, amp) for tv in ladder]
        flips = np.array([int(m.sum()) for _, m in outs])  # sparse doses live
        if flips[-1] == 0 and kind == "w":              # whole field sub-threshold: escalate
            amp *= 2.0
            outs = [self._field_grid(grid, float(tv), field, eps, kind, amp) for tv in ladder]
            flips = np.array([int(m.sum()) for _, m in outs])
        # SUBTLE pairs: low doses, multiplicatively close (n_hi ~ 1.4-3x n_lo), biased
        # sparse (log-uniform f_lo). Wide-gap pairs are trivially ordered and teach a
        # detector; close-gap sparse pairs are where the residual barrier lives.
        f_lo = float(np.exp(self.rng.uniform(np.log(0.02), np.log(0.35))))
        f_hi = min(1.0, f_lo * float(self.rng.uniform(1.4, 3.0)))
        i_lo = int(np.argmax(flips >= f_lo * max(flips[-1], 1)))
        i_hi = int(np.argmax(flips >= f_hi * max(flips[-1], 1)))
        if flips[i_hi] <= flips[i_lo]:                  # flat ladder stretch: force separation
            i_hi = int(np.argmax(flips)) if flips.max() > flips[i_lo] else len(ladder) - 1
        t = (ladder[i_lo], ladder[i_hi])
        (g_lo, m_lo), (g_hi, m_hi) = outs[i_lo], outs[i_hi]
        return dict(family=family, t_lo=float(t[0]), t_hi=float(t[1]),
                    grid_lo=g_lo, grid_hi=g_hi, mask_lo=m_lo, mask_hi=m_hi, field=field,
                    n_lo=int(m_lo.sum()), n_hi=int(m_hi.sum()),
                    desc=(f"t {t[0]:.2f}->{t[1]:.2f}  {fdesc}  "
                          f"flips {int(m_lo.sum())}->{int(m_hi.sum())}"))

    def _wfield(self, grid):
        """Single-draw view (corrupt()/viewer): the HIGH-dose member of a pair."""
        out = self.noise_pair(grid, "wfield")
        return dict(grid=out["grid_hi"], mask=out["mask_hi"], family="wfield",
                    desc=out["desc"], params=dict(t=out["t_hi"]))

    def _gfield(self, grid):
        out = self.noise_pair(grid, "gfield")
        return dict(grid=out["grid_hi"], mask=out["mask_hi"], family="gfield",
                    desc=out["desc"], params=dict(t=out["t_hi"]))

    def _walk(self, grid):
        cells = self._blob(grid, size=int(self.rng.integers(3, 17)))
        sigma_rel = float(np.exp(self.rng.uniform(np.log(1.0), np.log(2.5))))
        idx = torch.tensor([grid[y, x] for y, x in cells], device=self.device)
        for _ in range(4):        # a 0-flip draw would be a mislabeled negative: escalate σ
            z = self.codebook[idx] + sigma_rel * self._step_std * torch.from_numpy(
                self.rng.standard_normal((len(cells), self.codebook.shape[1]))
            ).to(self.codebook)
            new = self._snap(z)
            if (new != idx).any():
                break
            sigma_rel *= 1.6
        mask = torch.zeros_like(grid, dtype=torch.bool)
        flips = 0
        for (y, x), g in zip(cells, new.tolist()):
            if g != int(grid[y, x]):
                grid[y, x] = g
                mask[y, x] = True
                flips += 1
        if flips == 0:                    # escalation is probabilistic; the guarantee isn't:
            y, x = cells[0]               # force the seed cell to its nearest distinct glyph
            grid[y, x] = int(self.knn_i[int(grid[y, x]), 0])
            mask[y, x] = True
            flips = 1
        return dict(grid=grid, mask=mask, family="walk",
                    desc=f"σ={sigma_rel:.2f}·nn1  {flips}/{len(cells)} cells flipped",
                    params=dict(sigma_rel=sigma_rel, n_cells=len(cells), n_flipped=flips))

    def _fade(self, grid):
        run = self._line_run(grid, max_len=int(self.rng.integers(4, 15)))
        if run is None:
            return None
        ramp = bool(self.rng.random() < 0.5)
        if ramp:                                             # line peters out along the run
            t = np.linspace(self.rng.uniform(0.1, 0.4), self.rng.uniform(0.7, 1.0), len(run))
        else:                                                # whole segment fades uniformly
            t = np.full(len(run), self.rng.uniform(0.5, 1.0))
        idx = torch.tensor([grid[y, x] for y, x in run], device=self.device)
        tt = torch.from_numpy(t).to(self.codebook)[:, None]
        z = (1.0 - tt) * self.codebook[idx] + tt * self.codebook[self.space]
        new = self._snap(z)
        mask = torch.zeros_like(grid, dtype=torch.bool)
        flips = 0
        for (y, x), g in zip(run, new.tolist()):
            if g != int(grid[y, x]):
                grid[y, x] = g
                mask[y, x] = True
                flips += 1
        if flips == 0:                    # all-light run that snapped back: not a negative
            return None
        return dict(grid=grid, mask=mask, family="fade",
                    desc=f"{'ramp' if ramp else 'full'} t∈[{t.min():.2f},{t.max():.2f}]  "
                         f"run {len(run)}, {flips} flipped",
                    params=dict(ramp=ramp, t_min=float(t.min()), t_max=float(t.max()),
                                run_len=len(run), n_flipped=flips))

    def _shift(self, grid):
        run = self._line_run(grid, max_len=int(self.rng.integers(4, 15)))
        if run is None:
            return None
        (y0, x0), (y1, x1) = run[0], run[-1]
        dy, dx = np.sign(y1 - y0), np.sign(x1 - x0)          # run direction
        if dy == 0 and dx == 0:
            dy, dx = 0, 1
        pdy, pdx = (int(-dx), int(dy)) if (dx or dy) else (1, 0)   # perpendicular
        if self.rng.random() < 0.5:
            pdy, pdx = -pdy, -pdx
        GH, GW = grid.shape
        mask = torch.zeros_like(grid, dtype=torch.bool)
        src = [(y, x) for y, x in run if 0 <= y + pdy < GH and 0 <= x + pdx < GW]
        vals = [int(grid[y, x]) for y, x in src]
        for y, x in src:                                     # vacate first (dest may overlap run)
            grid[y, x] = self.space
            mask[y, x] = True
        for (y, x), v in zip(src, vals):
            grid[y + pdy, x + pdx] = v
            mask[y + pdy, x + pdx] = True
        return dict(grid=grid, mask=mask, family="shift",
                    desc=f"run {len(src)} shifted ({pdy:+d},{pdx:+d})",
                    params=dict(run_len=len(src), pdy=pdy, pdx=pdx))

    # debris glyphs actually observed in real outputs (curated); glyphs
    # not in the charset are skipped, light-ink band is the fallback pool
    CONFETTI_CHARS = "‚⁃'¯╶·',."

    def _confetti(self, grid):
        """Light glyphs in whitespace, as a LOCAL COLONY grown from a seed -- real
        confetti clusters (e.g. a vertical string at one column edge), it doesn't
        sprinkle uniformly. Seed: half fringe (blank within 2 cells of structure --
        where observed debris lives), half deep. The colony spreads with an
        anisotropic gaussian: isotropic blob, or elongated along x or y."""
        GH, GW = grid.shape
        inky = (self.g_ink[grid] > 0.02).float()[None, None]
        near = torch.nn.functional.max_pool2d(inky, 5, stride=1, padding=2)[0, 0] > 0
        blank = self.g_ink[grid] <= 0.001
        seeds = (near & blank) if self.rng.random() < 0.5 else (~near & blank)
        pool = seeds.nonzero()
        if pool.shape[0] == 0:
            pool = blank.nonzero()
        light = torch.tensor([self.char_to_idx[c] for c in self.CONFETTI_CHARS
                              if c in self.char_to_idx], dtype=torch.long)
        if light.numel() == 0:
            light = ((self.g_ink > 0.005) & (self.g_ink < 0.08)).nonzero().flatten()
        if pool.shape[0] == 0 or light.numel() == 0:
            return None                                      # fully inked grid: walk fallback
        sy, sx = [(1.6, 1.6), (0.8, 3.5), (3.5, 0.8)][int(self.rng.integers(3))]
        y0, x0 = pool[int(self.rng.integers(pool.shape[0]))].tolist()
        n = int(self.rng.integers(3, 11))
        cells = {(y0, x0)}
        for _ in range(80):                                   # rejection-grow the colony
            if len(cells) >= n:
                break
            y = y0 + int(round(self.rng.normal(0, sy)))
            x = x0 + int(round(self.rng.normal(0, sx)))
            if 0 <= y < GH and 0 <= x < GW and bool(blank[y, x]):
                cells.add((y, x))
        cells = sorted(cells)
        mask = torch.zeros_like(grid, dtype=torch.bool)
        for (y, x), g in zip(cells,
                             light[torch.from_numpy(self.rng.integers(0, light.numel(), len(cells)))].tolist()):
            grid[y, x] = g
            mask[y, x] = True
        shape = {(1.6, 1.6): "blob", (0.8, 3.5): "x-string", (3.5, 0.8): "y-string"}[(sy, sx)]
        return dict(grid=grid, mask=mask, family="confetti",
                    desc=f"{len(cells)}-cell {shape} colony @({y0},{x0})",
                    params=dict(n=len(cells), shape=shape))

    def _spur(self, grid):
        """Add a spurious arm to a few line/junction glyphs (┃ -> ┣): pick cells whose
        glyph has 2-3 arms, replace with the latent-nearest glyph having one arm more."""
        ink = self.g_ink[grid]
        n_arms = self.g_arms[grid].sum(-1)
        cand = ((n_arms >= 2) & (n_arms <= 3) & self.g_boxlike[grid]
                & (ink > 0.02) & (ink < 0.5)).nonzero()
        if cand.shape[0] == 0:
            return None
        k = min(int(self.rng.integers(2, 8)), cand.shape[0])
        pick = cand[torch.from_numpy(self.rng.choice(cand.shape[0], k, replace=False))]
        mask = torch.zeros_like(grid, dtype=torch.bool)
        flips = 0
        for y, x in pick.tolist():
            g0 = int(grid[y, x])
            have = {j for j in range(4) if bool(self.g_arms[g0, j])}
            for d in self.rng.permutation([j for j in range(4) if j not in have]):
                group = self._arm_groups.get(frozenset(have | {int(d)}), [])
                if group:
                    gt = torch.tensor(group)
                    z0 = self.codebook[g0]
                    g1 = int(gt[torch.cdist(z0[None], self.codebook[gt]).argmin()])
                    grid[y, x] = g1
                    mask[y, x] = True
                    flips += 1
                    break
        if flips == 0:
            return None                                      # no arm-augmented glyph exists
        return dict(grid=grid, mask=mask, family="spur",
                    desc=f"{flips}/{k} cells grew an extra arm",
                    params=dict(n=flips))

    def _break(self, grid):
        """REMOVE connections (inverse of spur): junction/corner glyphs lose an arm
        (┣ -> ┃, ┌ -> ╶), or a short cut of a line run drops to space. Every other
        structural family only ever ADDED ink/arms; a well-formed box that loses a
        connect was never a trained negative."""
        if self.rng.random() < 0.4:                # hard cut: 1-3 cells of a run -> space
            run = self._line_run(grid, max_len=int(self.rng.integers(1, 4)))
            if run is not None:
                mask = torch.zeros_like(grid, dtype=torch.bool)
                for y, x in run:
                    grid[y, x] = self.space
                    mask[y, x] = True
                return dict(grid=grid, mask=mask, family="break",
                            desc=f"cut a {int(mask.sum())}-cell gap into a line run",
                            params=dict(n=int(mask.sum()), kind="cut"))
        cand = self._break_sites(grid)
        if cand.shape[0] == 0:
            return None
        k = min(int(self.rng.integers(3, 13)), cand.shape[0])
        pick = cand[torch.from_numpy(self.rng.choice(cand.shape[0], k, replace=False))]
        mask = torch.zeros_like(grid, dtype=torch.bool)
        flips = 0
        for y, x in pick.tolist():
            if self._remove_arm(grid, y, x):
                mask[y, x] = True
                flips += 1
        if flips == 0:
            return None
        return dict(grid=grid, mask=mask, family="break",
                    desc=f"{flips}/{k} cells lost an arm",
                    params=dict(n=flips, kind="arm"))

    def _break_sites(self, grid):
        n_arms = self.g_arms[grid].sum(-1)
        return ((n_arms >= 2) & self.g_boxlike[grid]).nonzero()      # (K, 2)

    def _remove_arm(self, grid, y, x):
        """One-site break in place: latent-nearest arm-SUBSET glyph. False if none."""
        g0 = int(grid[y, x])
        have = [j for j in range(4) if bool(self.g_arms[g0, j])]
        for d in self.rng.permutation(have):
            group = [g for g in self._arm_groups.get(frozenset(set(have) - {int(d)}), [])
                     if g != g0]
            if group:
                gt = torch.tensor(group)
                grid[y, x] = int(gt[torch.cdist(self.codebook[g0][None],
                                                self.codebook[gt]).argmin()])
                return True
        return False

    def erode_ladder(self, grid, fracs):
        """ONE shuffled break-site sequence, snapshotted at cumulative fractions ->
        [(grid_k, mask_k), ...] nested (each grid extends the previous deletions).
        None if the grid has too few sites."""
        cand = self._break_sites(grid)
        if cand.shape[0] < 4:
            return None
        order = self.rng.permutation(cand.shape[0])
        g = grid.clone()
        applied = []
        for k in order.tolist():
            y, x = int(cand[k, 0]), int(cand[k, 1])
            if self._remove_arm(g, y, x):
                applied.append((y, x, int(g[y, x])))
        if len(applied) < 2:
            return None
        out = []
        for f in fracs:
            n = max(1, round(f * len(applied)))
            gk = grid.clone()
            for y, x, gid in applied[:n]:                     # replay = exact nesting
                gk[y, x] = gid
            out.append((gk, gk != grid))
        return out

    def break_pair(self, grid):
        """Ranked REMOVAL dose pair (erode): nested prefixes of one break-site ladder,
        subtle multiplicative gap like the field pairs. d(parent) must rise with
        connections lost -- ordered supervision the break margin alone never gives."""
        f_lo = float(np.exp(self.rng.uniform(np.log(0.08), np.log(0.45))))
        f_hi = min(1.0, f_lo * float(self.rng.uniform(1.5, 3.0)))
        out = self.erode_ladder(grid, (f_lo, f_hi))
        if out is None:
            return None
        (g_lo, m_lo), (g_hi, m_hi) = out
        if int(m_hi.sum()) <= int(m_lo.sum()):
            return None
        return dict(family="erode", t_lo=f_lo, t_hi=f_hi,
                    grid_lo=g_lo, grid_hi=g_hi, mask_lo=m_lo, mask_hi=m_hi,
                    n_lo=int(m_lo.sum()), n_hi=int(m_hi.sum()),
                    desc=(f"erode f {f_lo:.2f}->{f_hi:.2f}  "
                          f"connections lost {int(m_lo.sum())}->{int(m_hi.sum())}"))

    # ------------------------------------------------------------------ positive softening
    def soften(self, grid, temp=None, mix_frac=None, empty_scale=None):
        """Label-preserving soft render: per-cell DIRECTED hesitation + knn blend.

        Mid-run softness is z sitting BETWEEN specific candidates (the loss pulls it
        toward a mixture), not isotropic noise: in 16-d a random offset's projection onto
        any one neighbor direction is ~norm/sqrt(16), so the self-vs-neighbor gap barely
        moves and the blend renders indistinguishable from hard. So
        a sampled fraction of cells move t of the way toward one of their 4 nearest
        codebook neighbors (t up to 0.5 = the 50/50 midpoint), then blend at temperature.

        Empty-cell quieting mirrors the real runs (--empty-temp/-noise-scale): space
        cells in mostly-space neighborhoods get t and tau scaled down, so the background
        stays clean while structure hesitates.
        Returns (image (GH*CH, GW*CW), desc str)."""
        if temp is None:
            temp = float(np.exp(self.rng.uniform(np.log(0.15), np.log(0.6))))
        if mix_frac is None:
            mix_frac = float(self.rng.uniform(0.3, 0.9))      # fraction of cells hesitating
        if empty_scale is None:
            empty_scale = float(self.rng.uniform(0.15, 0.4))  # mirrors --empty-*-scale 0.3
        GH, GW = grid.shape
        flat = grid.reshape(-1)
        # graded empty gate (mirrors --empty-window 5 / --empty-gamma 2): q = how deep into
        # whitespace a space cell sits -> scale from 1 (at structure: full softness, the real
        # transient haze) down to empty_scale (deep background: quiet)
        is_space = (grid == self.space).float()[None, None]
        # count-normalized window (zero-padding would read out-of-frame as INK -> noisy borders)
        num = torch.nn.functional.avg_pool2d(is_space, 5, stride=1, padding=2)[0, 0]
        den = torch.nn.functional.avg_pool2d(torch.ones_like(is_space), 5, 1, padding=2)[0, 0]
        neigh = num / den
        q = ((neigh - 0.6) / 0.4).clamp(0, 1).pow(2.0) * (grid == self.space)
        cell_scale = (1.0 - (1.0 - empty_scale) * q).reshape(-1)
        M = flat.numel()
        z0 = self.codebook[flat]
        # directed hesitation: move t of the way toward a random one of the 4 nearest
        # neighbors (this is what collapses the self-vs-neighbor gap; isotropic can't).
        # Arm-superset neighbors are excluded (proto-spur blends -- see knn_soft_ok).
        k4 = min(4, self.knn_i.shape[1])
        ok4 = self.knn_soft_ok[flat][:, :k4].float() + 1e-9
        gen = torch.Generator()
        gen.manual_seed(int(self.rng.integers(2 ** 62)))
        pick = torch.multinomial(ok4, 1, generator=gen)[:, 0]
        no_ok = ok4.gather(1, pick[:, None])[:, 0] < 0.5              # all-superset rows: no mix
        zj = self.codebook[self.knn_i[flat].gather(1, pick[:, None].to(flat.device))[:, 0]]
        # contested cells sit NEAR the midpoint (that's what makes them contested);
        # below t~0.3 the gap is still ~nn1 and the blend renders ~hard
        t = torch.from_numpy(self.rng.uniform(0.25, 0.55, M)).to(self.codebook)
        t = t * torch.from_numpy(self.rng.random(M) < mix_frac).to(t) * cell_scale
        t = t * (~no_ok).to(t)                                        # proto-spur rows: no mix
        z = z0 + t[:, None] * (zj - z0)
        # small residual isotropic jitter for texture (norm ~0.1*nn1: label-safe)
        z = z + (0.1 * self._step_std) * cell_scale[:, None] * torch.from_numpy(
            self.rng.standard_normal((M, self.codebook.shape[1]))
        ).to(self.codebook)
        # blend over the TRUE glyph + its knn (self-distance from the jittered z, not 0)
        cand = torch.cat([flat[:, None], self.knn_i[flat]], dim=1)         # (M, 1+k)
        dz = (z[:, None, :] - self.codebook[cand]).norm(dim=-1)            # (M, 1+k)
        w = torch.softmax(-dz / (temp * cell_scale[:, None]), dim=1)       # quiet cells: sharper
        cells = (w[:, :, None] * self.bitmaps[cand].flatten(2)).sum(1)     # (M, CH*CW)
        img = cells.view(GH, GW, self.CH, self.CW).permute(0, 2, 1, 3)
        return (img.reshape(GH * self.CH, GW * self.CW),
                f"τ={temp:.2f}  mix={mix_frac:.2f}  empty×{empty_scale:.2f}")
