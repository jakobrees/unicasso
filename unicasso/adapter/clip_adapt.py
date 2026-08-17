"""CLIP domain-adaptation trainer: pull ascii renders onto the natural-image manifold.

    GLYPHVAE_FONT=sfmono python -m unicasso.adapter.clip_adapt <txt-root> <img-root> \
        --out runs/clip_adapt_00 [--steps 4000] [--holdout 10]

Dual-path (locked-tower) contract: ascii renders + corrupted siblings go through the
ADAPTED tower (LoRA/FiLM, see unicasso/engine/clip_adapter.py); parent linework goes through the same
tower with adapters DISABLED (bit-exact base CLIP). The target embedding space never
moves; the ascii branch learns to project into it. Deploy wiring is identical.

Three losses (weights --w-*):
  global   symmetric InfoNCE over aligned CROP pairs at the final embedding (through
           frozen layer4+attnpool) -- whole-tower gradient keeps early-layer adaptations
           semantically coherent; trains the deployed 0.1-weight semantic term directly.
  dense    DenseCL-style location-wise InfoNCE on layer2+3 feature maps: each sampled
           location of the ascii crop must match the SAME location of the parent crop
           against all other locations/crops (ink-weighted sampling; corrupted-region
           locations join the negative pool). Trains the spatial resolution the deployed
           per-crop feature-map L2 needs.
  margin   ranking on the DEPLOYED metric itself (raw-map MSE layers 2+3, + fc cosine):
           the true render must beat its corrupted sibling by a relative margin. Never
           asserts a flawed render is perfect -- only that a vanished line is WORSE. A
           passed margin means the optimizer will feel that failure as a loss increase.

Views: hard renders plus an optional fraction of soften() draws (the mid-run soft
domain, --soft-frac); corruptions (unicasso/adapter/corrupt.py) always compare
hard-vs-hard. Split is by PARENT
image (variants of one image never straddle the split). Step-0 eval = base-model
numbers (adapters are zero-init no-ops).
"""
import argparse
import json
import os
import re

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import open_clip
from open_clip import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

from unicasso.adapter.corrupt import CorruptionSampler, FAMILIES, FIELD_FAMILIES, RANK_FAMILIES
from unicasso.engine.clip_adapter import (inject_adapters, adapter_parameters, adapters_disabled,
                           set_enabled, save_adapters)

EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
# Variant suffixes: _w<cols> (column-sized, the original corpus) and _b<cells>
# (glyph-budget-sized, see batch_asciify.VARIANTS). Both strip to the same parent, so a
# corpus may mix the two and every variant of one image still lands on one side of a split.
VARIANT_RE = re.compile(r"_([wb]\d+(_clip05)?)$")
from unicasso.substrate.glyphs import repo_path
VAE_DEFAULT = repo_path("weights/vae_sfmono/model.pt")


# ---------------------------------------------------------------------- data
def find_pairs(txt_root, img_root):
    """Group .txt outputs by parent (variant suffix stripped) and locate the parent image
    (same rel stem, or stem+'_line'). Returns [{name, img, txts}], skips report unmatched."""
    groups = {}
    for dp, _, fns in os.walk(txt_root):
        for fn in sorted(fns):
            if not fn.endswith(".txt"):
                continue
            rel = os.path.relpath(os.path.join(dp, fn), txt_root)
            stem = os.path.splitext(rel)[0]
            groups.setdefault(VARIANT_RE.sub("", stem), []).append(os.path.join(dp, fn))
    pairs, missing = [], []
    for base, txts in sorted(groups.items()):
        img = None
        for suffix in ("", "_line"):
            for ext in EXTS:
                p = os.path.join(img_root, base + suffix + ext)
                if os.path.exists(p):
                    img = p
                    break
            if img:
                break
        (pairs if img else missing).append(dict(name=base, img=img, txts=txts))
    if missing:
        print(f"WARNING: {len(missing)} parent group(s) without a matching image "
              f"(e.g. {missing[0]['name']}) -- skipped")
    return pairs


def split_parents(pairs, holdout, seed):
    idx = np.random.default_rng(seed).permutation(len(pairs))
    hold = set(idx[:holdout].tolist())
    return ([p for i, p in enumerate(pairs) if i not in hold],
            [p for i, p in enumerate(pairs) if i in hold])


class Data:
    """Grid/canvas cache + item sampling on top of CorruptionSampler."""

    def __init__(self, sampler, device):
        self.s = sampler
        self.device = device
        # canvases + variable-shape crops live on CPU under MPS ONLY (its per-shape
        # kernel cache leaks ~12MB/step on random crop sizes; measured). CUDA has no
        # such pathology and server CPUs are slow -- crop on the GPU there.
        self.cdev = "cpu" if str(device).startswith("mps") else device
        self._grid, self._parent = {}, {}

    def grid(self, txt):
        if txt not in self._grid:
            self._grid[txt] = self.s.load_txt(txt)
        return self._grid[txt]

    def parent(self, img_path, dims):
        key = (img_path, dims)
        if key not in self._parent:
            if len(self._parent) > 32:                       # crude cap (canvases are ~4.5MB)
                self._parent.clear()
            im = Image.open(img_path).convert("L").resize((dims[1], dims[0]), Image.BILINEAR)
            # canvases stay on CPU: crop windows have CONTINUOUS random shapes, and MPS
            # caches a compiled kernel per unique shape FOREVER (~12MB/step host leak,
            # measured). Only fixed-shape 224 batches ever reach the device.
            self._parent[key] = torch.from_numpy(
                np.asarray(im, dtype=np.float32) / 255.0).to(self.cdev)
        return self._parent[key]

    def _decorate(self, grid, parent, rng):
        """Whitespace augmentation: stamp a small synthetic box structure into a BLANK
        region of BOTH target and grid, cell-aligned (identity property: the patch's
        target IS its render). Teaches 'whitespace structure must be TARGET-SUPPORTED'
        -- the supported-side twin of the confetti lesson -- and un-gates blank regions
        the ink thresholds otherwise exclude from training. -> (grid, parent) clones,
        or (None, None) if no blank window fits."""
        from unicasso.adapter.synth_identity import make_patch          # deferred: circular import
        s = self.s
        GH, GW = grid.shape
        if GH < 8 or GW < 12:
            return None, None
        gh = int(rng.integers(5, min(12, GH - 2)))
        gw = int(rng.integers(8, min(24, GW - 2)))
        tink = F.avg_pool2d((1.0 - parent)[None, None], (s.CH, s.CW))[0, 0][:GH, :GW]
        # tone-adaptive blank: unwhitened linework backgrounds sit at 0.01-0.02 cell
        # ink (paper tone), never 0 -- gate on "no structure", not "no tone"
        thr = min(max(0.015, float(tink.median()) + 0.01), 0.03)
        blank = (tink < thr).float()
        # all valid placements in one pass (windows whose every cell is blank)
        ok = F.avg_pool2d(blank[None, None], (gh, gw), stride=1)[0, 0] >= 1.0 - 1e-6
        pos = torch.nonzero(ok)
        if pos.shape[0] == 0:
            return None, None
        y0, x0 = (int(v) for v in pos[int(rng.integers(pos.shape[0]))])
        for _ in range(3):                              # sparse path draws: redraw
            patch = make_patch(rng, gh, gw, s)
            if int((patch != s.space).sum()) >= 6:
                break
        else:
            return None, None
        g2 = grid.clone()
        reg = g2[y0:y0 + gh, x0:x0 + gw]
        put = patch != s.space
        reg[put] = patch[put]                           # stray render marks nearby stay
        p2 = parent.clone()
        pr = s.render(patch).to(parent.device)
        ys, xs = y0 * s.CH, x0 * s.CW
        p2[ys:ys + gh * s.CH, xs:xs + gw * s.CW] = torch.minimum(
            p2[ys:ys + gh * s.CH, xs:xs + gw * s.CW], pr)
        return g2, p2

    def sample_item(self, pair, rng, soft_frac, corrupt, decorate_frac=0.0):
        """-> dict(view (H,W), parent (H,W), kind, [c_img, c_mask (GH,GW), family])."""
        txt = pair["txts"][int(rng.integers(len(pair["txts"])))]
        grid = self.grid(txt)
        dims = (grid.shape[0] * self.s.CH, grid.shape[1] * self.s.CW)
        parent = self.parent(pair["img"], dims)
        if decorate_frac > 0 and rng.random() < decorate_frac:
            g2, p2 = self._decorate(grid, parent, rng)
            if g2 is not None:
                grid, parent = g2, p2                        # clones; caches untouched
        soft = bool(rng.random() < soft_frac)
        view = self.s.soften(grid)[0] if soft else self.s.render(grid)
        item = dict(view=view.to(self.cdev), parent=parent,
                    kind="soft" if soft else "hard", grid=grid)
        if corrupt and not soft:                             # margins compare hard-vs-hard only
            out = self.s.corrupt(grid)
            item["c_img"] = self.s.render(out["grid"]).to(self.cdev)
            item["c_mask"] = out["mask"]
            item["family"] = out["family"]
        return item


# ---------------------------------------------------------------------- crops
def sample_window(rng, H, W, scale=(0.4, 0.9), ratio=(0.75, 4.0 / 3.0)):
    """Mirror of clip_loss._sample_dims: log-uniform AREA fraction + ratio jitter."""
    s = float(np.exp(rng.uniform(np.log(scale[0]), np.log(scale[1]))))
    r = ratio[0] * (ratio[1] / ratio[0]) ** float(rng.random())
    area = s * H * W
    ch = max(32, min(H, int(round((area / r) ** 0.5))))
    cw = max(32, min(W, int(round((area * r) ** 0.5))))
    return int(rng.integers(0, H - ch + 1)), int(rng.integers(0, W - cw + 1)), ch, cw


def inky_window(rng, parent, tries=6, min_ink=0.02, scale=(0.4, 0.9)):
    """Reject near-blank parent crops: a blank-vs-blank identification task is only
    solvable through image noise (false negatives) -- see dense_nce docstring."""
    H, W = parent.shape
    for _ in range(tries):
        p = sample_window(rng, H, W, scale=scale)
        if float((1.0 - parent[p[0]:p[0] + p[2], p[1]:p[1] + p[3]]).mean()) >= min_ink:
            return p
    return p


def crop224(img, p, res=224):
    top, left, ch, cw = p
    c = img[None, None, top:top + ch, left:left + cw]
    return F.interpolate(c, size=(res, res), mode="bilinear", align_corners=False)[0, 0]


def window_hits_mask(p, mask, CH, CW):
    top, left, ch, cw = p
    y0, y1 = top // CH, min((top + ch - 1) // CH + 1, mask.shape[0])
    x0, x1 = left // CW, min((left + cw - 1) // CW + 1, mask.shape[1])
    return bool(mask[y0:y1, x0:x1].any())


def mask_window_centered(rng, mask, CH, CW, H, W):
    """Eval helper: a window guaranteed to overlap the corruption (jittered around it)."""
    ys, xs = torch.nonzero(mask, as_tuple=True)
    j = int(rng.integers(ys.numel()))
    cy, cx = int(ys[j]) * CH + CH // 2, int(xs[j]) * CW + CW // 2
    _, _, ch, cw = sample_window(rng, H, W)
    top = int(np.clip(cy - ch // 2 + int(rng.integers(-ch // 4, ch // 4 + 1)), 0, H - ch))
    left = int(np.clip(cx - cw // 2 + int(rng.integers(-cw // 4, cw // 4 + 1)), 0, W - cw))
    return top, left, ch, cw


# ---------------------------------------------------------------------- tower
class Tower:
    def __init__(self, device, model_name="RN101", pretrained="openai", rank=8,
                 layers=(2, 3)):
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.visual = model.visual.eval()
        for p in self.visual.parameters():
            p.requires_grad_(False)
        self.adapters = inject_adapters(self.visual, rank=rank)   # zero-init no-ops
        self.visual.to(device)
        self.device = device
        self.layers = layers
        self._feats = {}
        for l in layers:
            getattr(self.visual, f"layer{l}").register_forward_hook(self._mk_hook(l))
        self.mean = torch.tensor(OPENAI_DATASET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(OPENAI_DATASET_STD, device=device).view(1, 3, 1, 1)

    def _mk_hook(self, l):
        def hook(_m, _i, out):
            self._feats[l] = out
        return hook

    def encode(self, batch1):
        """(B,1,224,224) [0,1] white=1, ANY device -> (fc (B,D), {layer: (B,C,h,w)}).
        Moves to the tower's device here: the single fixed-shape crossing point."""
        x = (batch1.to(self.device).expand(-1, 3, -1, -1) - self.mean) / self.std
        self._feats = {}
        fc = self.visual(x)
        return fc, dict(self._feats)


# ---------------------------------------------------------------------- losses
def nce_global(fa, fp, logit_scale):
    fa, fp = F.normalize(fa, dim=-1), F.normalize(fp, dim=-1)
    logits = logit_scale.exp().clamp(max=100.0) * fa @ fp.t()
    lab = torch.arange(fa.shape[0], device=fa.device)
    return 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.t(), lab))


def _sample_locs(weights, S, rng):
    """(B, HW) weights -> (B, S) location indices (cpu multinomial: tiny + MPS-safe).
    Seeded from the caller's numpy rng -- NOT the global torch RNG -- so a step is
    reproducible from the run seed alone."""
    w = weights.detach().cpu().clamp_min(1e-6)
    g = torch.Generator()
    g.manual_seed(int(rng.integers(2 ** 62)))
    return torch.multinomial(w, min(S, w.shape[1]), replacement=False, generator=g)


def dense_nce(feats_a, feats_p, parent_crops, layers, S, tau, rng, extra_maps=None,
              extra_masks=None, ink_thresh=0.01):
    """Location-wise InfoNCE. extra_maps/extra_masks: corrupted-crop maps whose in-mask
    locations join the key pool as negatives-only.

    Queries come ONLY from ink-bearing locations (pooled parent ink > ink_thresh):
    a whitespace query's positive is visually identical to every other whitespace key,
    so InfoNCE could only separate them by amplifying image noise -- false negatives
    that teach exactly the junk-sensitivity the deployed metric must not have. Blank
    regions simply don't participate; blank-vs-blank similarity is preserved for free."""
    loss = 0.0
    for l in layers:
        A = F.normalize(feats_a[l], dim=1)
        P = F.normalize(feats_p[l], dim=1)
        B, C, h, w = A.shape
        ink = F.adaptive_avg_pool2d(1.0 - parent_crops[:, None], (h, w)).reshape(B, h * w)
        wgt = ink * (ink > ink_thresh)
        ok = wgt.sum(1) > 0                                  # crops with any inky location
        if not bool(ok.any()):
            continue
        wgt = wgt[ok] + 1e-12                                # multinomial needs nonzero rows
        locs = _sample_locs(wgt, S, rng).to(A.device)        # (B',S')
        valid = wgt.to(A.device).gather(1, locs) > 1e-6      # padded draws on sparse crops
        idx = locs[:, None, :].expand(-1, C, locs.shape[1])
        q = A[ok].reshape(-1, C, h * w).gather(2, idx).permute(0, 2, 1)[valid]
        k = P[ok].reshape(-1, C, h * w).gather(2, idx).permute(0, 2, 1)[valid]
        if q.shape[0] == 0:
            continue
        keys = [k]
        if extra_maps is not None and extra_maps[l].shape[0] > 0:
            E = F.normalize(extra_maps[l], dim=1)                        # (Bc,C,h,w)
            m = extra_masks[l].reshape(E.shape[0], -1)                   # (Bc,hw) bool
            sel = E.reshape(E.shape[0], E.shape[1], -1).permute(0, 2, 1)[m]   # (n,C)
            # trim to a power-of-two bucket: a continuously-varying key-matrix width
            # would compile a fresh MPS kernel per unique value (host-memory drip)
            n = min(sel.shape[0], 2048)
            n = 0 if n < 64 else 1 << (n.bit_length() - 1)
            if n:
                keys.append(sel[:n])
        K = torch.cat(keys, dim=0)
        logits = (q @ K.t()) / tau
        lab = torch.arange(q.shape[0], device=q.device)
        loss = loss + F.cross_entropy(logits, lab)
    return loss / len(layers)


def deployed_d(feats_a, feats_p, layers):
    """The metric asciify minimizes: raw feature-map MSE summed over layers -> (B,)."""
    d = 0.0
    for l in layers:
        d = d + (feats_a[l] - feats_p[l]).pow(2).mean(dim=(1, 2, 3))
    return d


def encode_all(tower, crops, chunk):
    """Chunked no-grad encode -> (fc (B,D), {layer: (B,C,h,w)}). Peak memory = one chunk."""
    fc_l, fm_l = [], {l: [] for l in tower.layers}
    with torch.no_grad():
        for i in range(0, len(crops), chunk):
            fc, fm = tower.encode(torch.stack(crops[i:i + chunk])[:, None])
            fc_l.append(fc)
            for l in tower.layers:
                fm_l[l].append(fm[l])
    return torch.cat(fc_l), {l: torch.cat(v) for l, v in fm_l.items()}


# ---------------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(tower, data, hold_pairs, args, seed=1234):
    """Held-out gates. Returns dict; step-0 call gives BASE numbers (adapters no-op)."""
    s = data.s
    saved_rng = s.rng
    s.rng = np.random.default_rng(seed)
    rng = np.random.default_rng(seed)
    ra, pa, sa = [], [], []
    corr = {f: [] for f in FAMILIES}                        # (d_render, d_corrupt) pairs
    for pair in hold_pairs:
        txt = pair["txts"][0]
        grid = data.grid(txt)
        dims = (grid.shape[0] * s.CH, grid.shape[1] * s.CW)
        render = s.render(grid)                              # CPU, like all canvases
        soft = s.soften(grid)[0]
        parent = data.parent(pair["img"], dims)
        for _ in range(args.eval_windows):
            p = inky_window(rng, parent, min_ink=args.min_window_ink)
            ra.append(crop224(render, p))
            pa.append(crop224(parent, p))
            sa.append(crop224(soft, p))
        for fam in FAMILIES:
            for _ in range(args.eval_corruptions):
                out = s.corrupt(grid, family=fam)
                if out["family"] != fam or not out["mask"].any():   # fallback/degenerate: skip
                    continue
                cimg = s.render(out["grid"])
                p = mask_window_centered(rng, out["mask"], s.CH, s.CW, dims[0], dims[1])
                b = torch.stack([crop224(render, p), crop224(cimg, p)])[:, None]
                _, fa = tower.encode(b)
                with adapters_disabled(tower.adapters):
                    _, fp = tower.encode(crop224(parent, p)[None, None])
                fp2 = {l: f.expand(2, -1, -1, -1) for l, f in fp.items()}
                d = deployed_d(fa, fp2, tower.layers)
                corr[fam].append((float(d[0]), float(d[1])))

    # dose-MONOTONICITY on held-out field trajectories: Spearman rho of deployed d vs
    # flip count over {clean + 5 doses} of ONE shared trajectory. Catches the failure
    # mode where every pairwise gate passes while the
    # dose-response INVERTS (rho << 1) above the training density. Seeded rng => the
    # same trajectories every eval, comparable across steps; step-0 call = base rho.
    mono = {}
    for fam in RANK_FAMILIES:
        kind = "w" if fam == "wfield" else "g"
        rhos = []
        for pair in hold_pairs:
            grid = data.grid(pair["txts"][0])
            GH, GW = grid.shape
            dims = (GH * s.CH, GW * s.CW)
            parent = data.parent(pair["img"], dims)
            if fam == "erode":                      # removal trajectory: nested breaks
                steps = s.erode_ladder(grid, np.linspace(0.15, 1.0, 5))
                if steps is None:
                    continue
            else:
                field, _ = s._smooth_field(GH, GW)
                eps = torch.from_numpy(
                    s.rng.standard_normal((GH * GW, s.codebook.shape[1]))).to(s.codebook)
                steps = [s._field_grid(grid, float(t), field, eps, kind, 2.0)
                         for t in np.linspace(0.2, 1.0, 5)]
            flips = np.array([int(m.sum()) for _, m in steps])
            if flips[-1] < 5 or len(np.unique(flips)) < 3:   # dead trajectory: skip
                continue
            p = mask_window_centered(rng, steps[-1][1], s.CH, s.CW, dims[0], dims[1])
            b = torch.stack([crop224(s.render(grid), p)]
                            + [crop224(s.render(g2), p) for g2, _ in steps])[:, None]
            _, fa = tower.encode(b)
            with adapters_disabled(tower.adapters):
                _, fp = tower.encode(crop224(parent, p)[None, None])
            fpK = {l: f.expand(b.shape[0], -1, -1, -1) for l, f in fp.items()}
            d = deployed_d(fa, fpK, tower.layers).cpu().numpy()
            x = np.concatenate([[0], flips]).astype(np.float64)
            rx = np.argsort(np.argsort(x)) - (len(x) - 1) / 2.0
            ry = np.argsort(np.argsort(d)) - (len(d) - 1) / 2.0
            rhos.append(float((rx * ry).sum()
                              / np.sqrt((rx ** 2).sum() * (ry ** 2).sum() + 1e-12)))
        if rhos:
            mono[f"rho_{fam}"] = float(np.mean(rhos))
    s.rng = saved_rng

    def r_at_1(queries):
        fa = F.normalize(encode_all(tower, queries, args.micro_batch)[0], dim=-1)
        with adapters_disabled(tower.adapters):
            fp = F.normalize(encode_all(tower, pa, args.micro_batch)[0], dim=-1)
        return float((torch.argmax(fa @ fp.t(), dim=1)
                      == torch.arange(fa.shape[0], device=fa.device)).float().mean())

    res = dict(r1_hard=r_at_1(ra), r1_soft=r_at_1(sa), **mono)

    # DEPLOY-COMPAT gate: the deployed loss is an ABSOLUTE L2 (adapted render vs base
    # target) -- ordering metrics are blind to feature-space drift, which can inflate
    # the absolute metric by orders of magnitude while every gate stays perfect.
    # Ratio must stay ~1.
    d_ad, d_b = [], []
    for i in range(0, len(ra), args.micro_batch):
        rb = torch.stack(ra[i:i + args.micro_batch])[:, None]
        pb = torch.stack(pa[i:i + args.micro_batch])[:, None]
        _, f_ra = tower.encode(rb)
        with adapters_disabled(tower.adapters):
            _, f_rb = tower.encode(rb)
            _, f_pb = tower.encode(pb)
        d_ad.append(deployed_d(f_ra, f_pb, tower.layers))
        d_b.append(deployed_d(f_rb, f_pb, tower.layers))
    res["deploy_ratio"] = float((torch.cat(d_ad) / torch.cat(d_b).clamp_min(1e-8)).mean())
    for fam, ds in corr.items():
        if ds:
            dr, dc = np.array(ds).T
            res[f"acc_{fam}"] = float((dc > dr).mean())
            res[f"gap_{fam}"] = float(((dc - dr) / np.maximum(dr, 1e-8)).mean())
    return res


# ---------------------------------------------------------------------- main
def parse_args():
    p = argparse.ArgumentParser(description="CLIP domain-adaptation trainer (LoRA/FiLM)")
    p.add_argument("txt_root", help="batch-asciify output tree (.txt grids)")
    p.add_argument("img_root", help="parent linework images (mirrored rel paths)")
    p.add_argument("--out", default="runs/clip_adapt_00")
    p.add_argument("--vae-ckpt", default=VAE_DEFAULT)
    p.add_argument("--model", default="RN101")
    p.add_argument("--pretrained", default="openai")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--batch-parents", type=int, default=6)
    p.add_argument("--windows", type=int, default=4, help="aligned crop windows per parent")
    p.add_argument("--soft-frac", type=float, default=0.0,
                   help="fraction of views drawn as soften() blends. Default OFF: training on "
                        "soft views legitimizes mid-run mush (kills CLIP's implicit sharpening "
                        "pressure). r1_soft stays reported as a diagnostic")
    p.add_argument("--micro-batch", type=int, default=8,
                   help="crops per forward chunk (GradCache): bounds activation memory; "
                        "gradients are identical to full-batch, cost is a 2nd forward pass")
    p.add_argument("--loc-per-crop", type=int, default=48, help="dense-NCE locations sampled per crop")
    p.add_argument("--mini-windows", type=int, default=0,
                   help="EXTRA small high-magnification windows per parent, feeding a dense "
                        "ALIGNMENT-ONLY term (no negatives: at glyph magnification nearly every "
                        "negative is a pixel-identical false negative). No collapse risk (target "
                        "= frozen base on the parent); blindness guarded by the agreement gate")
    p.add_argument("--mini-scale", type=float, nargs=2, default=(0.04, 0.15),
                   help="area-fraction range for mini windows")
    p.add_argument("--w-mini", type=float, default=0.3, help="mini alignment weight")
    p.add_argument("--mini-agree", type=float, default=0.5,
                   help="align a location only if BASE CLIP already agrees render matches "
                        "parent there (cos > this) -- domain-mapping without error-blindness")
    p.add_argument("--min-window-ink", type=float, default=0.02,
                   help="min mean parent-crop ink for a training/eval window (blank-vs-blank "
                        "discrimination = learning image noise)")
    p.add_argument("--dense-ink-thresh", type=float, default=0.01,
                   help="dense-NCE queries only from map locations with pooled ink above this")
    p.add_argument("--tau-dense", type=float, default=0.2)
    p.add_argument("--w-dense", type=float, default=1.0)
    p.add_argument("--w-margin", type=float, default=0.5)
    p.add_argument("--w-anchor", type=float, default=1.0,
                   help="trust-region: adapted features must stay pointwise close to BASE "
                        "features on every ascii input (per-layer normalized MSE + fc cos). "
                        "Every other objective is scale/offset-invariant, so without this the "
                        "adapter drifts off the base manifold and the deployed ABSOLUTE L2 "
                        "becomes garbage")
    p.add_argument("--margin-rel", type=float, default=0.2, help="corrupt must be (1+m)x farther (deployed metric)")
    p.add_argument("--margin-global", type=float, default=0.05, help="fc-cosine margin")
    p.add_argument("--w-rank", type=float, default=1.0,
                   help="dose-rank term on field-noise trajectories (corrupt.noise_pair): "
                        "3-point chain clean < t_lo < t_hi in the deployed metric, same "
                        "windows. The single-dose margin builds a saturating DETECTOR "
                        "(measured: dose-response INVERTS above the training density -- "
                        "dense confetti sits behind a barrier); ranking along one shared "
                        "noise trajectory forces a monotone METER. 0 = margin term only")
    p.add_argument("--rank-parents", type=int, default=3, help="rank trajectories per step")
    p.add_argument("--rank-windows", type=int, default=3, help="windows per trajectory")
    p.add_argument("--rank-margin", type=float, default=0.1,
                   help="each chain link must be (1+this)x farther (log-ratio hinge)")
    p.add_argument("--rank-wfield-frac", type=float, default=0.7,
                   help="of the non-erode rank draws: fraction using whitespace-field "
                        "noise (the deploy-confetti axis); rest = global VP-cosine field")
    p.add_argument("--rank-erode-frac", type=float, default=0.35,
                   help="fraction of rank draws using the ERODE trajectory (nested "
                        "break-site prefixes): ordered removal supervision. The break "
                        "margin alone gives removal no dose ordering, so removal "
                        "accuracy erodes while the addition families train")
    p.add_argument("--decorate-frac", type=float, default=0.25,
                   help="whitespace decoration augmentation: this fraction of training "
                        "items get a small synthetic box structure stamped into a blank "
                        "region of BOTH target and render (identity property, cell-"
                        "aligned). Teaches whitespace structure = target-supported and "
                        "un-gates blank regions from the ink-thresholded losses; kept "
                        "moderate so the empty-margin prior survives. 0 = off")
    p.add_argument("--lr-lora", type=float, default=1e-4)
    p.add_argument("--lr-film", type=float, default=3e-4)
    p.add_argument("--holdout", type=int, default=10)
    p.add_argument("--split-file", default=None,
                   help="split.json from unicasso.adapter.corpus_split ({train:[...], holdout:[...]}); "
                        "overrides --holdout/--seed random split. Parents in the corpus but "
                        "absent from the file are added to TRAIN (holdout stays fixed as the "
                        "corpus grows); parents listed but not found are ignored")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-windows", type=int, default=6, help="retrieval windows per held-out parent")
    p.add_argument("--eval-corruptions", type=int, default=4, help="per family per held-out parent")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--resume", action="store_true",
                   help="continue from <out>/training_state.pt (written at every eval): "
                        "restores adapters, optimizer moments, logit temp, data-rng and "
                        "schedule position. Ctrl-C between evals loses at most eval-every steps")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ("mps" if torch.backends.mps.is_available()
                             else "cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    pairs = find_pairs(args.txt_root, args.img_root)
    if len(pairs) < args.holdout + 2:
        raise SystemExit(f"only {len(pairs)} parent pairs found -- need > holdout ({args.holdout})")
    if args.split_file:
        with open(args.split_file) as f:
            sp = json.load(f)
        hold_names = set(sp["holdout"])
        train_pairs = [p for p in pairs if p["name"] not in hold_names]
        hold_pairs = [p for p in pairs if p["name"] in hold_names]
        extra = len(train_pairs) - len([p for p in train_pairs if p["name"] in set(sp["train"])])
        print(f"split from {args.split_file}"
              + (f" (+{extra} new parent(s) -> train)" if extra else ""))
        if not hold_pairs:
            raise SystemExit("split-file holdout parents not found in this corpus")
    else:
        train_pairs, hold_pairs = split_parents(pairs, args.holdout, args.seed)
    with open(os.path.join(args.out, "split.json"), "w") as f:
        json.dump(dict(train=[p["name"] for p in train_pairs],
                       holdout=[p["name"] for p in hold_pairs]), f, indent=1)
    print(f"{len(pairs)} parents: {len(train_pairs)} train / {len(hold_pairs)} held out "
          f"({sum(len(p['txts']) for p in pairs)} txts) | device {device}")

    sampler = CorruptionSampler(args.vae_ckpt, device="cpu", seed=args.seed)
    data = Data(sampler, device)
    tower = Tower(device, args.model, args.pretrained, rank=args.rank)
    film, lora = adapter_parameters(tower.adapters)
    logit_scale = torch.nn.Parameter(torch.tensor(np.log(1 / 0.15), dtype=torch.float32,
                                                  device=device))
    n_par = sum(p.numel() for p in film) + sum(p.numel() for p in lora)
    print(f"adapters: {len(tower.adapters)} modules, {n_par / 1e6:.2f}M trainable params")
    opt = torch.optim.AdamW([dict(params=film, lr=args.lr_film),
                             dict(params=lora, lr=args.lr_lora),
                             dict(params=[logit_scale], lr=1e-3)], weight_decay=0.0)
    base_lrs = [g["lr"] for g in opt.param_groups]

    hist = dict(it=[], g=[], d=[], m=[])
    evals = []
    start_step, best = 1, -1.0
    state_path = os.path.join(args.out, "training_state.pt")
    if args.resume:
        if not os.path.exists(state_path):
            raise SystemExit(f"--resume: no {state_path}")
        st = torch.load(state_path, map_location=device, weights_only=False)
        for name, m in tower.adapters.items():               # weights from the paired snapshot
            m.load_state_dict(st["adapters"][name], strict=False)
        opt.load_state_dict(st["opt"])
        logit_scale.data.copy_(st["logit_scale"].to(device))
        rng.bit_generator.state = st["rng"]                  # data order continues seamlessly
        hist, evals, best = st["hist"], st["evals"], st["best"]
        start_step = st["step"] + 1
        print(f"resumed from step {st['step']} (best score {best:.3f})")

    def run_eval(step):
        res = evaluate(tower, data, hold_pairs, args)
        evals.append(dict(step=step, **res))
        tag = "BASE (zero-init)" if step == 0 else f"step {step}"
        print(f"\n[eval {tag}] R@1 hard {res['r1_hard']:.3f}  soft {res['r1_soft']:.3f}  "
              f"deploy×{res['deploy_ratio']:.2f}  | "
              + "  ".join(f"{f}: acc {res.get(f'acc_{f}', float('nan')):.2f} "
                          f"gap {res.get(f'gap_{f}', float('nan')):+.2f}" for f in FAMILIES)
              + "  | mono " + " ".join(f"{f[:1]}ρ {res.get(f'rho_{f}', float('nan')):+.2f}"
                                       for f in RANK_FAMILIES))
        with open(os.path.join(args.out, "evals.json"), "w") as f:
            json.dump(evals, f, indent=1)
        return res

    if start_step == 1:
        run_eval(0)
    timing = bool(os.environ.get("CLIP_ADAPT_TIMING"))
    import time as _time

    def _tic(label, t0, acc={}):
        if timing:
            torch.cuda.synchronize() if str(device).startswith("cuda") else None
            acc[label] = acc.get(label, 0.0) + _time.perf_counter() - t0
            acc["_n"] = acc.get("_n", 0)
            if label == "opt" :
                acc["_n"] += 1
                if acc["_n"] % 5 == 0:
                    tot = sum(v for k, v in acc.items() if not k.startswith("_"))
                    print("\n[timing/5steps] " + "  ".join(
                        f"{k} {v:.1f}s" for k, v in acc.items() if not k.startswith("_"))
                        + f"  | total {tot:.1f}s", flush=True)
                    acc.clear()
        return _time.perf_counter()

    pbar = tqdm(range(start_step, args.steps + 1), initial=start_step - 1,
                total=args.steps, desc="adapt")
    for step in pbar:
        # LR schedule: linear warmup -> cosine to 0
        warm = min(1.0, step / max(args.warmup, 1))
        cos = 0.5 * (1 + np.cos(np.pi * step / args.steps))
        for g, lr0 in zip(opt.param_groups, base_lrs):
            g["lr"] = lr0 * warm * cos

        t0 = _time.perf_counter()
        picks = rng.choice(len(train_pairs), size=min(args.batch_parents, len(train_pairs)),
                           replace=False)
        ra_c, pa_c, ca_c, ca_of, ca_masks = [], [], [], [], []   # crops; corrupt->render idx
        mini_r, mini_p = [], []                                  # high-magnification pairs
        for pi in picks:
            item = data.sample_item(train_pairs[pi], rng, args.soft_frac, corrupt=True,
                                    decorate_frac=args.decorate_frac)
            H, W = item["parent"].shape
            for _ in range(args.mini_windows):
                p = inky_window(rng, item["parent"], min_ink=args.min_window_ink,
                                scale=tuple(args.mini_scale))
                mini_r.append(crop224(item["view"], p))
                mini_p.append(crop224(item["parent"], p))
            for _ in range(args.windows):
                p = inky_window(rng, item["parent"], min_ink=args.min_window_ink)
                ra_c.append(crop224(item["view"], p))
                pa_c.append(crop224(item["parent"], p))
                if "c_img" in item and window_hits_mask(p, item["c_mask"], sampler.CH,
                                                        sampler.CW):
                    ca_c.append(crop224(item["c_img"], p))
                    ca_of.append(len(ra_c) - 1)
                    # pixel mask of corrupted cells inside this window, for the key pool
                    pm = torch.zeros(H, W, dtype=torch.bool)
                    ys, xs = torch.nonzero(item["c_mask"], as_tuple=True)
                    for y, x in zip(ys.tolist(), xs.tolist()):
                        pm[y * sampler.CH:(y + 1) * sampler.CH,
                           x * sampler.CW:(x + 1) * sampler.CW] = True
                    top, left, chh, cww = p
                    ca_masks.append(F.interpolate(
                        pm[None, None, top:top + chh, left:left + cww].float(),
                        size=(224, 224), mode="nearest")[0, 0] > 0)

        # Dose-rank trajectories: one shared (field, eps) noising per rank parent, two
        # doses + the clean render on the SAME windows. Clean crops join ra_c/pa_c (valid
        # NCE positives, and the parent features double as the rank reference); lo/hi
        # crops ride at the END of crops_a, so the anchor covers them for free.
        rk_c, rk_of, rk_m = [], [], []                       # lo,hi interleaved; -> ra idx
        if args.w_rank > 0 and args.rank_parents > 0:
            for pi in picks[:args.rank_parents]:
                pair = train_pairs[pi]
                grid = data.grid(pair["txts"][int(rng.integers(len(pair["txts"])))])
                if rng.random() < args.rank_erode_frac:
                    fam = "erode"                            # removal trajectory
                elif rng.random() < args.rank_wfield_frac:
                    fam = FIELD_FAMILIES[0]
                else:
                    fam = FIELD_FAMILIES[1]
                pr = sampler.noise_pair(grid, fam)
                if pr is None:                               # too few break sites: fall back
                    pr = sampler.noise_pair(grid, FIELD_FAMILIES[0])
                if pr is None or not pr["mask_hi"].any():
                    continue
                dims = (grid.shape[0] * sampler.CH, grid.shape[1] * sampler.CW)
                parent = data.parent(pair["img"], dims)
                clean = sampler.render(grid).to(data.cdev)
                lo = sampler.render(pr["grid_lo"]).to(data.cdev)
                hi = sampler.render(pr["grid_hi"]).to(data.cdev)
                # lo->hi margin scaled by dose gap: close pairs (n_hi ~ 1.5 n_lo) demand
                # a proportionally smaller separation -- a fixed margin on subtle pairs
                # would force a super-linear (detector-shaped) response
                mlink = np.log1p(args.rank_margin) * min(
                    1.0, np.log(max(pr["n_hi"], 2) / max(pr["n_lo"], 1)))
                for _ in range(args.rank_windows):
                    p = mask_window_centered(rng, pr["mask_hi"], sampler.CH, sampler.CW,
                                             dims[0], dims[1])
                    ra_c.append(crop224(clean, p))
                    pa_c.append(crop224(parent, p))
                    rk_of.append(len(ra_c) - 1)
                    rk_m.append(mlink)
                    rk_c.append(crop224(lo, p))
                    rk_c.append(crop224(hi, p))

        # GradCache two-pass: (1) no-grad encode of ALL crops -> cached embeddings as
        # leaves; losses + backward on the cache (full in-batch negatives, tiny graph);
        # (2) re-forward in micro-batches, backprop each against its cached gradient.
        # Same gradients as one big batch; activation memory = one micro-batch.
        t0 = _tic("sample+crop", t0)
        n_r = len(ra_c)
        n_rc = n_r + len(ca_c)                               # minis sit AFTER the corrupts
        n_rcm = n_rc + len(mini_r)                           # rank lo/hi pairs sit LAST
        crops_a = ra_c + ca_c + mini_r + rk_c
        fc_a, feats_a = encode_all(tower, crops_a, args.micro_batch)
        fc_a.requires_grad_(True)
        for l in tower.layers:
            feats_a[l].requires_grad_(True)
        t0 = _tic("encode_adapted", t0)
        with adapters_disabled(tower.adapters):
            fc_p, feats_p = encode_all(tower, pa_c, args.micro_batch)
            if mini_p:                                       # frozen parent view of the minis
                _, feats_mp = encode_all(tower, mini_p, args.micro_batch)
            # base-path view of the SAME ascii crops: the anchor targets
            fc_rb, feats_rb = encode_all(tower, crops_a, args.micro_batch)

        t0 = _tic("encode_frozen", t0)
        fa_r = {l: f[:n_r] for l, f in feats_a.items()}
        extra_maps = extra_masks = None
        if ca_c:
            extra_maps = {l: f[n_r:n_rc] for l, f in feats_a.items()}
            extra_masks = {}
            cm = torch.stack(ca_masks)                       # CPU: pooled to fixed map
            for l, f in extra_maps.items():                  # shapes before crossing
                extra_masks[l] = (F.adaptive_max_pool2d(cm[:, None].float(),
                                                        f.shape[-2:])[:, 0] > 0).to(device)

        l_g = nce_global(fc_a[:n_r], fc_p, logit_scale)
        l_d = dense_nce(fa_r, feats_p, torch.stack(pa_c), tower.layers,
                        args.loc_per_crop, args.tau_dense, rng, extra_maps, extra_masks,
                        ink_thresh=args.dense_ink_thresh)
        if ca_c:
            d_r = deployed_d({l: f[ca_of] for l, f in fa_r.items()},
                             {l: f[ca_of] for l, f in feats_p.items()}, tower.layers)
            d_c = deployed_d({l: f[n_r:n_rc] for l, f in feats_a.items()},
                             {l: f[ca_of] for l, f in feats_p.items()}, tower.layers)
            # RATIO margin in log space: log d_c must exceed log d_r by log(1+m).
            # Scale-invariant by construction -- an absolute hinge on raw MSE rewards
            # feature blow-up (FiLM gains compound multiplicatively across ~50 layers;
            # measured m ~ 1e22 at w-margin 2.0) since d_c has no ceiling.
            l_m = F.relu(np.log1p(args.margin_rel)
                         + torch.log(d_r + 1e-8) - torch.log(d_c + 1e-8)).mean()
            cr = F.cosine_similarity(fc_a[ca_of], fc_p[ca_of], dim=-1)
            cc = F.cosine_similarity(fc_a[n_r:n_rc], fc_p[ca_of], dim=-1)
            l_m = l_m + F.relu(cc - cr + args.margin_global).mean()
        else:
            l_m = fc_a.new_zeros(())

        # trust-region anchor: covers corrupt crops too (they're renders as well)
        l_a = (1.0 - F.cosine_similarity(fc_a, fc_rb, dim=-1)).mean()
        for l in tower.layers:
            l_a = l_a + ((feats_a[l] - feats_rb[l]).pow(2).mean()
                         / feats_rb[l].pow(2).mean().clamp_min(1e-8))

        # mini-window ALIGNMENT (no negatives: at glyph magnification nearly every
        # negative is a pixel-identical false negative). Target = frozen parent features
        # (no collapse); agreement gate = align only where BASE already says render
        # matches parent (no error-blindness). Same formula as the deployed dense term.
        l_mi = fc_a.new_zeros(())
        if mini_r:
            n_loc = 0
            for l in tower.layers:
                A = F.normalize(feats_a[l][n_rc:n_rcm], dim=1)
                P = F.normalize(feats_mp[l], dim=1)
                B = F.normalize(feats_rb[l][n_rc:n_rcm], dim=1)
                agree = (B * P).sum(1) > args.mini_agree     # (B',h,w)
                ink = F.adaptive_avg_pool2d(
                    1.0 - torch.stack(mini_p)[:, None], A.shape[-2:])[:, 0] > args.dense_ink_thresh
                mask = agree & ink.to(A.device)
                if bool(mask.any()):
                    l_mi = l_mi + (1.0 - (A * P).sum(1))[mask].sum()
                    n_loc += int(mask.sum())
            l_mi = l_mi / max(n_loc, 1)

        # dose-rank: monotone chain clean < lo < hi on each trajectory's shared windows.
        # Same log-ratio hinge as l_m (scale-invariant, self-limiting); the parent
        # reference features are the SAME pa_c entries the clean crop pairs with.
        l_r = fc_a.new_zeros(())
        if rk_c:
            ridx = torch.tensor(rk_of, device=fc_a.device)
            pf = {l: f[ridx] for l, f in feats_p.items()}
            d_cl = deployed_d({l: f[ridx] for l, f in feats_a.items()}, pf, tower.layers)
            d_lo = deployed_d({l: f[n_rcm::2] for l, f in feats_a.items()}, pf, tower.layers)
            d_hi = deployed_d({l: f[n_rcm + 1::2] for l, f in feats_a.items()}, pf,
                              tower.layers)
            mr = np.log1p(args.rank_margin)
            mr_pair = torch.tensor(rk_m, device=d_lo.device, dtype=d_lo.dtype)
            l_r = 0.5 * (F.relu(mr + torch.log(d_cl + 1e-8) - torch.log(d_lo + 1e-8))
                         + F.relu(mr_pair + torch.log(d_lo + 1e-8) - torch.log(d_hi + 1e-8))
                         ).mean()

        loss = (l_g + args.w_dense * l_d + args.w_margin * l_m + args.w_anchor * l_a
                + args.w_mini * l_mi + args.w_rank * l_r)
        t0 = _tic("losses", t0)
        opt.zero_grad(set_to_none=True)
        loss.backward()                       # grads land on the cached leaves + logit_scale
        for i in range(0, len(crops_a), args.micro_batch):   # pass 2: chunked re-forward
            fc, fm = tower.encode(torch.stack(crops_a[i:i + args.micro_batch])[:, None])
            sur = (fc * fc_a.grad[i:i + args.micro_batch]).sum() if fc_a.grad is not None \
                else fc.new_zeros(())
            for l in tower.layers:
                if feats_a[l].grad is not None:
                    sur = sur + (fm[l] * feats_a[l].grad[i:i + args.micro_batch]).sum()
            sur.backward()                    # accumulates into adapter params
        t0 = _tic("pass2", t0)
        torch.nn.utils.clip_grad_norm_(film + lora + [logit_scale], 1.0)
        opt.step()
        t0 = _tic("opt", t0)
        if device == "mps" and step % 5 == 0:
            torch.mps.empty_cache()           # release the allocator's high-water mark

        hist["it"].append(step)
        hist["g"].append(float(l_g))
        hist["d"].append(float(l_d))
        hist["m"].append(float(l_m))
        hist.setdefault("a", []).append(float(l_a))
        hist.setdefault("mi", []).append(float(l_mi))
        hist.setdefault("r", []).append(float(l_r))
        if step % 10 == 0:
            post = dict(g=f"{float(l_g):.3f}", d=f"{float(l_d):.3f}",
                        m=f"{float(l_m):.3f}", a=f"{float(l_a):.3f}", nc=len(ca_c))
            if rk_c:
                post["r"] = f"{float(l_r):.3f}"
            if mini_r:
                post["mi"] = f"{float(l_mi):.3f}"
            pbar.set_postfix(**post)
        if step % args.eval_every == 0 or step == args.steps:
            res = run_eval(step)
            # selection: retrieval + ordering accuracy. r1_soft is diagnostic-only (soft
            # training off); gaps deliberately excluded (magnitude inflates past useful).
            score = res["r1_hard"] + np.nanmean(
                [res.get(f"acc_{f}", np.nan) for f in FAMILIES])
            if not (0.5 <= res["deploy_ratio"] <= 2.0):      # off the base scale = unshippable
                print(f"  deploy ratio {res['deploy_ratio']:.2f} outside [0.5, 2] -- "
                      "checkpoint disqualified from best")
                score = -1e9
            save_adapters(os.path.join(args.out, "adapters_last.pt"), tower.adapters,
                          args.rank, 1.0, extra=dict(step=step, model=args.model,
                                                     pretrained=args.pretrained))
            # per-eval snapshot (~2.5MB): keeps every mid-run state; the best/last
            # checkpoint files alone get overwritten
            save_adapters(os.path.join(args.out, f"adapters_step{step}.pt"), tower.adapters,
                          args.rank, 1.0, extra=dict(step=step, model=args.model,
                                                     pretrained=args.pretrained))
            # full training state for --resume (optimizer moments, logit temp, data-rng)
            torch.save(dict(adapters={n: {k: v for k, v in m.state_dict().items()
                                          if not k.startswith(("base.", "bn."))}
                                      for n, m in tower.adapters.items()},
                            opt=opt.state_dict(), logit_scale=logit_scale.detach().cpu(),
                            rng=rng.bit_generator.state, hist=hist, evals=evals,
                            best=max(best, score), step=step), state_path)
            if score > best:
                best = score
                save_adapters(os.path.join(args.out, "adapters_best.pt"), tower.adapters,
                              args.rank, 1.0, extra=dict(step=step, score=score,
                                                         model=args.model,
                                                         pretrained=args.pretrained))
                print(f"  new best (score {score:.3f}) -> adapters_best.pt")
            np.savez(os.path.join(args.out, "loss_hist.npz"),
                     **{k: np.array(v) for k, v in hist.items()})

    # loss curve png
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4))
        for k, lab in (("g", "global NCE"), ("d", "dense NCE"), ("m", "margin"),
                       ("a", "anchor")):
            if k in hist:
                ax.plot(hist["it"], hist[k], label=lab, lw=0.8)
        ax.set_yscale("log")
        ax.legend()
        ax.set_xlabel("step")
        fig.savefig(os.path.join(args.out, "loss_curve.png"), dpi=110,
                    bbox_inches="tight")
    except Exception as e:                                   # curve is a nicety, not a gate
        print(f"(loss curve png skipped: {e})")
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
