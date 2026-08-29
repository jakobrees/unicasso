"""CLIPasso-style perceptual loss: match the CLIP embedding of the ASCII render to the target.

Faithful to CLIPasso's config (yael-vinker/CLIPasso):
  - CLIP RN101 (openai weights).
  - GEOMETRIC loss: L2 between intermediate ResNet feature maps. Layer weights "0,0,1,1,0" ->
    layer2 and layer3 outputs (the "intermediate layers constrain geometry" claim).
  - SEMANTIC loss: 1 - cosine(final image embeddings), weight 0.1.
  - 4 augmentations applied IDENTICALLY to render + target, averaged.

CLIPasso augments with RandomPerspective + RandomResizedCrop (applied identically to sketch +
target). TWO augmentation paths here, selected per call:
  * DEFAULT (rotate=shear=0): random-resized-crop only -- slice + bilinear interpolate.
  * WARP (rotate>0 or shear>0, opt-in via --clip-rotate/--clip-shear): adds rotation + shear about
    the image center, folded into the crop as a single MANUAL bilinear gather. RandomPerspective
    uses grid_sample whose BACKWARD is unimplemented on MPS, so we do the resample by hand (its
    backward is scatter, which MPS supports). Out-of-canvas samples fill white (1).

The render is a sketch-like image (white bg = 1, ink = 0), same polarity as CLIPasso sketches, so
it's fed to CLIP as-is (expanded to 3 channels).
"""
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import open_clip
from open_clip import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD


class ViTValidator:
    """Read-only ViT diagnostic (NOT a loss): does the render match the target in ViT-perceptual space?
    Caches the target's global embedding + an early-layer patch-token grid once; each call forwards the
    render (whole image, no augs, no grad) and returns (global cosine, per-patch cosine map). ViT-token
    L2 is unreliable as a LOSS on messy early renders, but as a validation METRIC it's a good fit."""

    def __init__(self, device, model_name="ViT-B-16", pretrained="openai", layer=3):
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.visual = model.visual.to(device).eval()
        for p in self.visual.parameters():
            p.requires_grad_(False)
        self.device, self.layer, self.res = device, layer, 224
        self.mean = torch.tensor(OPENAI_DATASET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(OPENAI_DATASET_STD, device=device).view(1, 3, 1, 1)
        self._tok = None
        self.visual.transformer.resblocks[layer].register_forward_hook(
            lambda _m, _i, out: setattr(self, "_tok", out))
        self.tgt_e = self.tgt_tok = self.grid = None

    @torch.no_grad()
    def _encode(self, img):                                  # (H,W) white=1 -> (embed D,), patch tokens (P,D)
        x = F.interpolate(img[None, None].expand(1, 3, -1, -1), size=(self.res, self.res),
                          mode="bilinear", align_corners=False)
        e = self.visual((x - self.mean) / self.std)[0]       # (D,)
        t = self._tok
        t = t if t.shape[0] == 1 else t.transpose(0, 1)      # -> (1, P+1, D)
        return e, t[0, 1:]                                   # drop CLS -> (P, D)

    @torch.no_grad()
    def set_target(self, target):
        e, tok = self._encode(target)
        self.tgt_e = F.normalize(e, dim=-1); self.tgt_tok = F.normalize(tok, dim=-1)
        p = tok.shape[0]; g = int(round(p ** 0.5)); self.grid = (g, g)

    @torch.no_grad()
    def eval_render(self, render):                           # -> (global_cos float, agreement map (Ph,Pw))
        e, tok = self._encode(render)
        gcos = float(F.cosine_similarity(F.normalize(e, dim=-1), self.tgt_e, dim=0))
        pcos = (F.normalize(tok, dim=-1) * self.tgt_tok).sum(-1)   # per-patch cosine (P,)
        return gcos, pcos.reshape(*self.grid)


def _as3(x):
    """(H,W) white=1 -> (1,3,H,W) by replication; (H,W,3) RGB -> (1,3,H,W) by permute.
    COLOR mode feeds real RGB here instead of a replicated gray plane."""
    if x.dim() == 3 and x.shape[-1] == 3:
        return x.permute(2, 0, 1)[None]
    return x[None, None].expand(1, 3, -1, -1)


class CLIPPerceptualLoss(nn.Module):
    def __init__(self, device, model_name="RN101", pretrained="openai",
                 layer_weights=(0.0, 0.0, 1.0, 1.0, 0.0), fc_weight=0.1,
                 n_aug=4, crop_scale=(0.7, 1.0), res=224, rotate_deg=0.0, shear_deg=0.0,
                 invert_frac=0.0,
                 scale_alpha=1.0, aspect_jitter=(0.75, 4.0 / 3.0),
                 edge_frac=0.0, edge_beta=0.5, edge_auto=False,
                 vit_layers=((7, 1.0),), vit_drop_cls=False, fp16=False,
                 reg_frac=0.0, cell_h=24, cell_w=12, adapter=None, batch_aug=False,
                 microbatch=0, steer=None, steer_weight=0.0):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model
        self.visual = model.visual
        self.device = device
        self.fc_weight = fc_weight
        self.n_aug = n_aug
        self.crop_scale = crop_scale
        self.invert_frac = invert_frac   # fraction of crop pairs judged in INVERTED polarity (both
        # sides flipped pre-normalization: same matching problem, a decorrelated CLIP feature view)
        self.aspect_jitter = aspect_jitter   # CLIPasso-style ratio jitter; base SQUARE (1,1)=no jitter
        self.scale_alpha = scale_alpha   # crop weight = s^alpha; with log-uniform sampling, alpha=1
        self.res = res                   # => uniform-across-scales (var-reduced), alpha<1 => small-scale emphasis
        self.rotate_deg = rotate_deg
        self.shear_deg = shear_deg
        self.warp_on = rotate_deg > 0 or shear_deg > 0   # opt-in second path; default = crop only
        # Edge-bias counter-sampling: a fraction of crops place their top-left from Beta(a,a) (U-shaped,
        # a<1) so they hug the boundaries, counteracting uniform placement's center bias. edge_auto solves
        # the fraction f* = -Cov(u,d)/Var(d) (flattest combined coverage) lazily on the first forward.
        self.edge_frac, self.edge_beta, self.edge_auto = edge_frac, edge_beta, edge_auto
        self._edge_f = None if edge_auto else edge_frac
        # CROP REGISTRATION: before comparing, nudge the TARGET's sampling window by the small
        # bounded shift (+/- reg_frac of a cell) that best aligns its ink field with the render
        # crop's (NCC on half-res blurred ink maps, no gradients through the search). CLIP then
        # forgives placement error up to the bound and punishes beyond it -- grid-phase tolerance
        # (the per-cell misalignment is incoherent, so no global/smooth warp can provide this).
        self.reg_bx = int(round(reg_frac * cell_w))
        self.reg_by = int(round(reg_frac * cell_h))
        self.reg_on = self.reg_bx > 0 or self.reg_by > 0
        # Registration only applies to SMALL crops (<= reg_max_cells per side): a big crop spans
        # hundreds of cells whose phase demands are incoherent (+/-2.4px, corr ~0), so its best
        # single shift is ~0 -- registering it is cost without effect. Pair --clip-reg-frac with a
        # small --clip-crop-scale MIN so some crops actually fall in the registrable regime.
        self.reg_max_h = 6 * cell_h
        self.reg_max_w = 6 * cell_w
        if self.reg_on:
            ax = torch.arange(-4, 5, device=device, dtype=torch.float32)
            k = torch.exp(-(ax ** 2) / (2 * 1.5 ** 2))
            self._reg_k = (k / k.sum()).view(1, 1, 1, -1)
        self.mean = torch.tensor(OPENAI_DATASET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(OPENAI_DATASET_STD, device=device).view(1, 3, 1, 1)
        # fp16: run the (frozen) visual encoder in half precision (~2x on MPS). Normalize in fp32, cast the
        # input to _dtype for the forward; the render's grad still flows back (cast has a backward). mean/std
        # stay fp32. The loss is cast back to float() so it composes with the fp32 recon/other terms.
        self.batch_aug = batch_aug   # one big encode per forward instead of n_aug small ones
        self.microbatch = int(microbatch)  # >0: encode crops in chunks of N with per-chunk
        # backward into a detached render leaf (gradient accumulation) -- peak encoder-activation
        # memory drops from n_aug crops to N crops; exact same gradients via a dot-product surrogate.
        self.fp16 = fp16
        self._dtype = torch.float16 if fp16 else torch.float32
        if fp16:
            self.visual.half()

        # Geometric-feature hooks. RN101: conv layer outputs (maps). ViT: transformer resblock outputs
        # (token sequences) -> the CLIPascene path. Detected from the visual backbone.
        self.is_vit = hasattr(self.visual, "transformer") and hasattr(self.visual.transformer, "resblocks")
        self.is_convnext = hasattr(self.visual, "trunk") and hasattr(self.visual.trunk, "stages")
        # Domain-adaptation adapters (clip_adapt trainer): DUAL PATH -- the render/snap
        # side runs with adapters ENABLED (the ascii-adapted tower), the target side with
        # adapters DISABLED (bit-exact base CLIP), the same wiring the adapters were
        # trained under. Injected AFTER the fp16 cast so LoRA/FiLM params match dtype.
        self._adapters = None
        if adapter:
            if self.is_vit or self.is_convnext:
                raise ValueError("--clip-adapter supports the ModifiedResNet models only")
            from unicasso.engine.clip_adapter import load_adapters
            self._adapters, extra = load_adapters(self.visual, adapter, device)
            for m in self._adapters.values():
                m.to(device=device, dtype=self._dtype)
                for p in m.parameters():
                    p.requires_grad_(False)
            print(f"CLIP adapter: {adapter} ({len(self._adapters)} modules"
                  + (f", step {extra['step']}" if extra.get("step") else "") + ")")
        # STEERING (engine/steer.py): a fixed unit direction in the joint image/text space,
        # added to the normalized TARGET embedding before the semantic cosine. Asks for "the
        # target, but ascii-flavoured". Only the fc/semantic term lives in a space text and
        # image share, so this is the one term it can touch -- the geometric conv-map L2 and
        # dense_loss are untouched. The cosine re-normalizes, so the loss stays bounded in
        # [0,2] and only its DIRECTION moves; steer_weight is the whole knob.
        self.steer, self.steer_w = None, float(steer_weight)
        if steer is not None and self.steer_w != 0.0:
            if self.is_vit or self.is_convnext:
                raise ValueError("--clip-steer needs the joint-space fc embedding of an RN "
                                 "model (ViT/ConvNeXt geometric terms are token/map L2)")
            from unicasso.engine.steer import load_delta
            d, rec = load_delta(steer, device)
            if d.numel() != getattr(self.visual, "output_dim", d.numel()):
                raise ValueError(f"steer dim {d.numel()} != visual output dim "
                                 f"{self.visual.output_dim} ({steer})")
            if rec.get("model") not in (None, model_name):
                raise ValueError(f"steer vector was built for {rec['model']}, running "
                                 f"{model_name} -- the embedding spaces differ")
            self.steer = d.view(1, -1)
            print(f"CLIP steer: {steer} (kind={rec.get('kind','?')}, lambda={self.steer_w})")
        self.vit_layers = list(vit_layers)          # [(resblock_idx, weight), ...]
        self.vit_drop_cls = vit_drop_cls
        self.layer_weights = layer_weights
        self._feats = {}
        if self.is_vit:                             # hook the chosen resblocks; geometric = L2 on tokens
            blocks = self.visual.transformer.resblocks
            for idx, w in self.vit_layers:
                if w != 0:
                    blocks[idx].register_forward_hook(self._mk_hook(idx))
        elif self.is_convnext:                      # 4 conv stages ~ RN101 layer1..4; geometric = L2 on maps
            stages = self.visual.trunk.stages
            for i in range(1, 5):
                if layer_weights[i] != 0:
                    stages[i - 1].register_forward_hook(self._mk_hook(i))
        else:                                       # RN101: 0=stem, 1..4 = layer1..layer4 outputs
            layers = [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]
            for i, lyr in enumerate(layers, start=1):
                if layer_weights[i] != 0:
                    lyr.register_forward_hook(self._mk_hook(i))

    def _mk_hook(self, i):
        def hook(_m, _inp, out):
            self._feats[i] = out
        return hook

    def _sample_scale(self):
        """Log-uniform area scale in crop_scale -> density p(s) ∝ 1/s (small crops oversampled for
        coverage/variance; the s^alpha crop weight in forward un-biases it)."""
        lo, hi = self.crop_scale
        return lo * (hi / lo) ** random.random()

    def _sample_dims(self, H, W):
        """Log-uniform AREA scale + CLIPasso aspect-ratio jitter, base SQUARE. Crop area ~= s*H*W with
        aspect cw/ch = r in [lo,hi]; clamped to fit the image (effective area-fraction recomputed).
        This is torchvision RandomResizedCrop semantics: the resize-to-res stretch is then <= the
        ratio bound (in-distribution) instead of the full image aspect."""
        s = self._sample_scale()
        lo_r, hi_r = self.aspect_jitter
        r = lo_r * (hi_r / lo_r) ** random.random()
        area = s * H * W
        ch = max(8, min(H, int(round((area / r) ** 0.5))))
        cw = max(8, min(W, int(round((area * r) ** 0.5))))
        return ch, cw, (ch * cw) / float(H * W)            # effective area fraction after clamp

    def _rrc_params(self, H, W):
        """One random-resized-crop (square base + ratio jitter). Returns ((top,left,ch,cw), s_eff)."""
        ch, cw, s = self._sample_dims(H, W)
        top = random.randint(0, H - ch)
        left = random.randint(0, W - cw)
        return (top, left, ch, cw), s

    def _rrc_params_edge(self, H, W):
        """Edge-biased crop: same size sampling, but top-left from Beta(a,a) (U-shaped, a<1) -> the crop
        hugs a boundary/corner, over-covering edge cells to counteract the center bias of uniform placement."""
        ch, cw, s = self._sample_dims(H, W)
        a = self.edge_beta
        top = int(round(random.betavariate(a, a) * (H - ch)))
        left = int(round(random.betavariate(a, a) * (W - cw)))
        return (top, left, ch, cw), s

    def _solve_edge_frac(self, H, W, n=4000, G=24):
        """Monte-Carlo the per-(coarse-pixel) coverage of the uniform vs edge samplers, then solve the
        mixing fraction f* = -Cov(u,d)/Var(d) (d=e-u) that minimizes combined coverage variance (flattest)."""
        yc = (np.arange(G) + 0.5) / G * H
        xc = (np.arange(G) + 0.5) / G * W

        def cov(sampler):
            c = np.zeros((G, G))
            for _ in range(n):
                (top, left, ch, cw), _ = sampler(H, W)
                ym = (yc >= top) & (yc < top + ch); xm = (xc >= left) & (xc < left + cw)
                c[np.ix_(ym, xm)] += 1.0
            return (c / n).ravel()

        u, e = cov(self._rrc_params), cov(self._rrc_params_edge)
        d = e - u; up, dp = u - u.mean(), d - d.mean()
        vard = float((dp * dp).mean())
        f = 0.0 if vard < 1e-12 else -float((up * dp).mean()) / vard
        return float(np.clip(f, 0.0, 1.0))

    @torch.no_grad()
    def _reg_map(self, img3):
        """(1,3,H,W) white=1 -> half-res blurred INK map (1,1,H/2,W/2) for registration NCC."""
        m = F.avg_pool2d(1.0 - img3[:, :1], 2)
        m = F.conv2d(F.pad(m, (4, 4, 0, 0), mode="replicate"), self._reg_k)
        return F.conv2d(F.pad(m, (0, 0, 4, 4), mode="replicate"), self._reg_k.transpose(2, 3))

    @torch.no_grad()
    def _register(self, rm, tm_padded, top, left, ch, cw):
        """Best (dx, dy) full-res target shift (step 2px, within +/- reg_bx/by) aligning the target's
        ink field to the render crop's: normalized cross-correlation on the half-res maps, ALL
        candidate offsets scored in one unfold + one matmul (a python loop here cost 3.5x step
        time). tm_padded is the target map zero-padded by (bx2, by2) -- zero ink beyond the border
        is the correct semantics. Returns (0,0) when the render crop is near-blank."""
        h2, w2 = ch // 2, cw // 2
        bx2, by2 = self.reg_bx // 2, self.reg_by // 2
        a = rm[0, 0, top // 2:top // 2 + h2, left // 2:left // 2 + w2].reshape(-1)
        a = a - a.mean()
        na = a.norm()
        if float(na) < 1e-3:
            return 0, 0
        win = tm_padded[:, :, top // 2:top // 2 + h2 + 2 * by2,
                        left // 2:left // 2 + w2 + 2 * bx2]
        B = F.unfold(win, kernel_size=(h2, w2))[0].t()        # (n_offsets, h2*w2), row-major (dy, dx)
        B = B - B.mean(dim=1, keepdim=True)
        scores = (B @ a) / (B.norm(dim=1) * na + 1e-6)
        j = int(scores.argmax())
        n_x = 2 * bx2 + 1
        return 2 * (j % n_x - bx2), 2 * (j // n_x - by2)

    def _prep(self, img3, p):
        """Crop with params p then resize to res; img3: (1,3,H,W) in [0,1]."""
        top, left, ch, cw = p
        crop = img3[:, :, top:top + ch, left:left + cw]
        return F.interpolate(crop, size=(self.res, self.res), mode="bilinear", align_corners=False)

    def _aug_coords(self, H, W, device):
        """WARP path: source-pixel coords (res,res) folding a random crop + rotation + shear.
        Returns (x_src, y_src). When rotate=shear=0 this equals the pure crop, so the warp path
        only changes behavior once those are turned on."""
        res = self.res
        ch, cw, s = self._sample_dims(H, W)               # square base + ratio jitter (rotation/shear on top)
        top = random.randint(0, H - ch); left = random.randint(0, W - cw)
        ang = math.radians(random.uniform(-self.rotate_deg, self.rotate_deg))
        shx = math.tan(math.radians(random.uniform(-self.shear_deg, self.shear_deg)))
        shy = math.tan(math.radians(random.uniform(-self.shear_deg, self.shear_deg)))
        ca, sa = math.cos(ang), math.sin(ang)
        # forward M = R(ang) @ Shear; invert the 2x2 to map output -> source pixels
        m00, m01 = ca - sa * shy, ca * shx - sa
        m10, m11 = sa + ca * shy, sa * shx + ca
        det = m00 * m11 - m01 * m10
        det = det if abs(det) > 1e-6 else 1e-6
        i00, i01, i10, i11 = m11 / det, -m01 / det, -m10 / det, m00 / det
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
        gy, gx = torch.meshgrid(torch.linspace(0, 1, res, device=device),
                                torch.linspace(0, 1, res, device=device), indexing="ij")
        fx = left + gx * (cw - 1); fy = top + gy * (ch - 1)   # output -> crop region (full coords)
        dx, dy = fx - cx, fy - cy                              # inverse-affine about center
        return i00 * dx + i01 * dy + cx, i10 * dx + i11 * dy + cy, s, (top, left, ch, cw)

    def _bilinear_warp(self, img, x_src, y_src, fill=1.0):
        """Differentiable bilinear sample of img (B,C,H,W) at (x_src,y_src) (res,res). Backward is
        gather/scatter (MPS-supported), unlike grid_sample. Out-of-canvas -> fill (white=1)."""
        B, C, H, W = img.shape
        Ho, Wo = x_src.shape
        x0 = torch.floor(x_src); y0 = torch.floor(y_src)
        wx = (x_src - x0).view(1, 1, Ho, Wo); wy = (y_src - y0).view(1, 1, Ho, Wo)
        x0l, y0l = x0.long(), y0.long(); x1l, y1l = x0l + 1, y0l + 1
        flat = img.reshape(B, C, H * W)

        def g(xi, yi):
            idx = (yi.clamp(0, H - 1) * W + xi.clamp(0, W - 1)).reshape(1, 1, Ho * Wo).expand(B, C, Ho * Wo)
            return flat.gather(2, idx).reshape(B, C, Ho, Wo)

        I00, I01, I10, I11 = g(x0l, y0l), g(x1l, y0l), g(x0l, y1l), g(x1l, y1l)
        out = (1 - wy) * ((1 - wx) * I00 + wx * I01) + wy * ((1 - wx) * I10 + wx * I11)
        valid = ((x_src >= 0) & (x_src <= W - 1) & (y_src >= 0) & (y_src <= H - 1)).view(1, 1, Ho, Wo)
        return torch.where(valid, out, out.new_full((1,), fill))

    def _encode(self, batch):
        """batch: (B,3,res,res) in [0,1] -> (fc embedding, {layer: feat}). Hooks fill self._feats."""
        x = ((batch - self.mean) / self.std).to(self._dtype)
        self._feats = {}
        fc = self.visual(x)
        out = dict(self._feats)
        # DROP the instance's own references once the caller has its copy. The
        # hooked outputs carry grad_fn, so holding them holds that entire CLIP
        # backward graph -- and self._feats was only ever reassigned on the NEXT
        # encode, which meant the last photo's graph stayed live indefinitely.
        # Measured in the joint trainer at pool-refresh time: 4.58 GiB of live
        # cuda tensors, almost all batch-32 RN101 feature maps, surviving every
        # empty_cache() and squeezing the refinement workers. The returned dict
        # holds everything the caller needs, so this is free.
        self._feats = {}
        return fc, out

    def _encode_split(self, ra, ta, aa=None):
        """Encode crop set(s) -> (fc, feats) in canonical layout [R(n) | T(n) | A(n)]
        (render i at row i, target i at n+i, alt i at 2n+i; n = ra.shape[0]).
        No adapters: one batched forward (the original path). With adapters: render(+alt)
        through the ADAPTED tower, target through the frozen base (adapters toggled off,
        no_grad -- the target never carries gradient anyway), then re-assembled in the
        same layout so the loss code is path-agnostic."""
        if self._adapters is None:
            return self._encode(torch.cat([ra, ta] + ([aa] if aa is not None else []), dim=0))
        from unicasso.engine.clip_adapter import set_enabled
        n = ra.shape[0]
        fc_r, f_r = self._encode(ra if aa is None else torch.cat([ra, aa], dim=0))
        set_enabled(self._adapters, False)
        try:
            with torch.no_grad():
                fc_t, f_t = self._encode(ta)
        finally:
            set_enabled(self._adapters, True)
        fc = torch.cat([fc_r[:n], fc_t, fc_r[n:]], dim=0)
        feats = {i: torch.cat([f_r[i][:n], f_t[i], f_r[i][n:]], dim=0) for i in f_r}
        return fc, feats

    def _pair_loss(self, fc, feats, ir, it, ia, alt_w, cons_w, B_total):
        """Loss for ONE render/target(/alt) triple living at rows (ir, it, ia) of a
        canonical-layout encode. Shared by the sequential and batched aug paths."""
        fr, fa, ft = fc[ir:ir + 1], fc[ia:ia + 1] if ia is not None else None, fc[it:it + 1]
        if self.steer is not None:
            # ê_t* = normalize(ê_t + λ·Δ̂). cosine_similarity normalizes its arguments, so
            # only the pre-add normalize is explicit here: without it λ would mean something
            # different for every crop (embedding norms vary), and the knob would be useless.
            # The component of Δ̂ parallel to ê_t cancels under the renormalization -- the
            # ORTHOGONAL part is the entire steering effect (steer.py reports how big it is).
            # fp32 on both sides here: under --clip-fp16 the encoder returns fp16 and the
            # steered target is fp32, and cosine_similarity will not mix the two.
            ft = F.normalize(ft.float(), dim=-1) + self.steer_w * self.steer
            fr = fr.float()
            fa = fa.float() if fa is not None else None
        sem = (1.0 - torch.cosine_similarity(fr, ft, dim=1)).mean()
        l_main = self.fc_weight * sem
        l_alt = l_cons = fc.new_zeros(())
        if ia is not None:
            # the VIP/snap plane is scored against the SAME steered target: it is the shipping
            # image's copy of l_main, so a different target would make the two planes disagree
            l_alt = self.fc_weight * alt_w * (
                1.0 - torch.cosine_similarity(fa, ft, dim=1)).mean()
            l_cons = self.fc_weight * cons_w * (
                1.0 - torch.cosine_similarity(fr, fa, dim=1)).mean()
        if self.is_vit:                                  # geometric = L2 on transformer tokens
            vw = dict(self.vit_layers)
            for idx, tok in feats.items():
                t = tok if tok.shape[0] == B_total else tok.transpose(0, 1)   # -> (B, seq, dim)
                if self.vit_drop_cls:
                    t = t[:, 1:, :]
                l_main = l_main + vw[idx] * torch.square(t[ir] - t[it]).mean()
                if ia is not None:
                    l_alt = l_alt + vw[idx] * alt_w * torch.square(t[ia] - t[it]).mean()
                    l_cons = l_cons + vw[idx] * cons_w * torch.square(t[ir] - t[ia]).mean()
        else:                                            # geometric = L2 on conv feature maps
            for i, fmap in feats.items():
                l_main = l_main + self.layer_weights[i] * torch.square(
                    fmap[ir:ir + 1] - fmap[it:it + 1]).mean()
                if ia is not None:
                    l_alt = l_alt + self.layer_weights[i] * alt_w * torch.square(
                        fmap[ia:ia + 1] - fmap[it:it + 1]).mean()
                    l_cons = l_cons + self.layer_weights[i] * cons_w * torch.square(
                        fmap[ir:ir + 1] - fmap[ia:ia + 1]).mean()
        return l_main, l_alt, l_cons

    def _dense_run(self, img01, layers, max_side):
        """Fully-conv stem->layerN forward (grad-carrying); returns {layer: (C,h,w)}."""
        if img01.dim() == 3 and img01.shape[-1] == 3:
            img01 = img01.permute(2, 0, 1)[None]          # RGB probe/render
        elif img01.dim() == 2:
            img01 = img01[None, None]
        _, _, H, W = img01.shape
        scale = max_side / max(H, W)
        if scale < 1.0:
            img01 = F.interpolate(img01, size=(int(round(H * scale)), int(round(W * scale))),
                                  mode="bilinear", align_corners=False)
        x = (img01 if img01.shape[1] == 3 else img01.expand(1, 3, -1, -1)).to(self.device)
        x = ((x - self.mean) / self.std).to(self._dtype)
        v = self.visual
        x = v.act1(v.bn1(v.conv1(x)))      # ModifiedResNet stem
        x = v.act2(v.bn2(v.conv2(x)))
        x = v.act3(v.bn3(v.conv3(x)))
        x = v.avgpool(x)
        out = {}
        top = max(layers)
        for i, lyr in enumerate([v.layer1, v.layer2, v.layer3, v.layer4], start=1):
            x = lyr(x)
            if i in layers:
                out[i] = x[0]                # (C,h,w)
            if i >= top:
                break
        return out

    @torch.no_grad()
    def dense_features(self, img01, layers=(2, 3), max_side=448, base=False):
        """Whole-image conv feature maps for the requested layers (1..4). img01: (H,W) or (1,1,H,W)
        in [0,1] white=1. The RN101 conv trunk is fully convolutional (only the final attnpool needs
        a fixed size), so we run stem->layer{1..4} at the image's own aspect (long side = max_side) and
        read off the maps -- this IS the multi-scale densely-swept pyramid (layer2 fine ... layer4
        coarse). Returns {layer: (C,h,w)} float tensors. No crop, no resize-to-square distortion.
        base=True forces the frozen-base path (target/affinity features) when adapters are
        loaded; base=False = the adapted path (render-side features, the deploy default)."""
        if base and self._adapters is not None:
            from unicasso.engine.clip_adapter import set_enabled
            set_enabled(self._adapters, False)
            try:
                return self._dense_run(img01, layers, max_side)
            finally:
                set_enabled(self._adapters, True)
        return self._dense_run(img01, layers, max_side)

    def dense_loss(self, render, target, layers=(1, 2), max_side=544):
        """Dense geometric term: ONE fully-conv forward of the whole render, mean over feature
        locations of (1 - cos) vs the target's cached maps at `layers`. Perfectly even coverage
        at a fixed scale (no crop lottery, no center bias) with the conv stack's natural ~4-8px
        position tolerance: the mid-scale slot between pixel recon and global CLIP. Cheap:
        early-exits after max(layers)."""
        key = (tuple(layers), max_side)
        if getattr(self, "_dense_key", None) != key:
            with torch.no_grad():
                if self._adapters is not None:               # target maps = frozen base path
                    from unicasso.engine.clip_adapter import set_enabled
                    set_enabled(self._adapters, False)
                try:
                    tf = self._dense_run(target, layers, max_side)
                finally:
                    if self._adapters is not None:
                        set_enabled(self._adapters, True)
            self._dense_tgt = {l: F.normalize(f.float(), dim=0) for l, f in tf.items()}
            self._dense_key = key
        rf = self._dense_run(render, layers, max_side)
        loss = 0.0
        for l in layers:
            r = F.normalize(rf[l].float(), dim=0)
            loss = loss + (1.0 - (r * self._dense_tgt[l]).sum(0)).mean()
        return loss / len(layers)

    def forward(self, render, target, alt=None, alt_w=0.0, cons_w=0.0, return_parts=False):
        """render: (H,W) white=1, carries grad. target: (H,W) white=1, reference.
        Returns the scalar CLIPasso loss (geometric L2 + fc_weight * semantic cosine), aug-averaged.

        VIP/consolidation planes (alt + weights): `alt` is a second render view -- in swarm mode
        the STRAIGHT-THROUGH SNAP render (forward value = the hard shipping image, backward = the
        soft blend's Jacobian, the `ste`-mode trick at image level). Each crop then encodes THREE
        images in one batch (+50%% cost) and the loss adds
            alt_w  * d(alt, target)    -- the VIP plane: the SHIPPING image scored every step;
                                          breaks the tie equilibria the soft loss maintains
            cons_w * d(render, alt)    -- consolidation: charges the soft-hard gap directly,
                                          pulls mixtures toward REALIZABLE composites (ramp in
                                          LATE: it's incumbent-reinforcing during exploration)."""
        r3 = _as3(render)
        t3 = _as3(target.detach())
        use_alt = alt is not None and (alt_w > 0 or cons_w > 0)
        a3 = _as3(alt) if use_alt else None
        accum = self.microbatch > 0
        r3_src = a3_src = None
        if accum:                                # gradient-accumulation mode: crops hang off
            do_bwd = torch.is_grad_enabled() and render.requires_grad
            r3_src = r3                          # detached leaves; per-chunk backward frees
            r3 = r3.detach().requires_grad_(do_bwd)   # encoder activations, and the leaf grads
            if use_alt:                          # re-enter the render graph via a surrogate
                a3_src = a3
                a3 = a3.detach().requires_grad_(do_bwd)
        _, _, H, W = r3.shape
        if self._edge_f is None:                             # lazy auto-solve once (H,W known, post content-crop)
            self._edge_f = self._solve_edge_frac(H, W)
            print(f"  clip edge-frac (auto): f={self._edge_f:.2f} (beta {self.edge_beta}) -> edge crops/aug")
        n_edge = int(round(self._edge_f * self.n_aug))       # last n_edge crops use the edge-biased sampler
        rm = tm = None
        if self.reg_on:                                      # crop registration: half-res ink maps, once
            with torch.no_grad():
                rm = self._reg_map(r3)
                bx2, by2 = self.reg_bx // 2, self.reg_by // 2
                tm = F.pad(self._reg_map(t3), (bx2, bx2, by2, by2))   # zero ink beyond borders
        total = render.new_zeros(()); wsum = 0.0
        p_main = p_alt = p_cons = 0.0        # weighted component sums (return_parts diagnostics)
        ras, tas, aas, ss = [], [], [], []   # batched-aug collection (--clip-batch-aug)
        for i in range(self.n_aug):
            aa = None
            if self.warp_on:                                 # opt-in: crop + rotation/shear (edge-frac n/a)
                x_src, y_src, s, box = self._aug_coords(H, W, render.device)   # SAME warp for both
                ra = self._bilinear_warp(r3, x_src, y_src)
                if use_alt:
                    aa = self._bilinear_warp(a3, x_src, y_src)
                if self.reg_on and box[2] <= self.reg_max_h and box[3] <= self.reg_max_w:
                    dx, dy = self._register(rm, tm, *box)    # registered: offset the TARGET's source
                    ta = self._bilinear_warp(t3, x_src + dx, y_src + dy)   # coords (ignores the small
                else:                                        # rotation -- second-order)
                    ta = self._bilinear_warp(t3, x_src, y_src)
            else:                                            # known-good: crop only
                rrc = self._rrc_params_edge if i >= self.n_aug - n_edge else self._rrc_params
                p, s = rrc(H, W)                             # SAME crop for render + target
                ra = self._prep(r3, p)
                if use_alt:
                    aa = self._prep(a3, p)
                if self.reg_on and p[2] <= self.reg_max_h and p[3] <= self.reg_max_w:
                    dx, dy = self._register(rm, tm, *p)      # registered: shift the target's window
                    top2 = min(max(p[0] + dy, 0), H - p[2])
                    left2 = min(max(p[1] + dx, 0), W - p[3])
                    ta = self._prep(t3, (top2, left2, p[2], p[3]))
                else:
                    ta = self._prep(t3, p)
            if self.invert_frac > 0 and float(torch.rand(())) < self.invert_frac:
                ra = 1.0 - ra                                # inverted-polarity view: both sides flip,
                ta = 1.0 - ta                                # matching unchanged, judge = different features
                if aa is not None:
                    aa = 1.0 - aa
            if self.batch_aug or accum:                      # collect; encode after the loop
                ras.append(ra); tas.append(ta)
                if aa is not None:
                    aas.append(aa)
                ss.append(s)
                continue
            fc, feats = self._encode_split(ra, ta, aa)       # rows: 0=render, 1=target, [2=alt]
            l_main, l_alt, l_cons = self._pair_loss(fc, feats, 0, 1,
                                                    2 if aa is not None else None,
                                                    alt_w, cons_w, fc.shape[0])
            loss = l_main + l_alt + l_cons
            cw_ = s ** self.scale_alpha                      # crop weight (un-bias the 1/s sampling)
            total = total + cw_ * loss.float(); wsum += cw_   # fp16 loss -> float so it composes with fp32 terms
            p_main += cw_ * float(l_main); p_alt += cw_ * float(l_alt); p_cons += cw_ * float(l_cons)
        if accum:
            # Chunked encode + immediate backward (--clip-microbatch): identical math to the
            # batched path, but activations for at most `microbatch` crop triples exist at
            # once. Chunk losses are pre-normalized so per-chunk backward sums to the exact
            # full-batch gradient on the leaves.
            n = len(ras)
            wsum_mb = float(sum(float(sv) ** self.scale_alpha for sv in ss)) or 1e-9
            mb = self.microbatch
            for k0 in range(0, n, mb):
                k1 = min(k0 + mb, n)
                m = k1 - k0
                A3 = torch.cat(aas[k0:k1], dim=0) if aas else None
                fc, feats = self._encode_split(torch.cat(ras[k0:k1], dim=0),
                                               torch.cat(tas[k0:k1], dim=0), A3)
                B_total = m * (3 if A3 is not None else 2)
                chunk = fc.new_zeros(()).float()
                for j in range(m):
                    l_main, l_alt, l_cons = self._pair_loss(
                        fc, feats, j, m + j, 2 * m + j if A3 is not None else None,
                        alt_w, cons_w, B_total)
                    cw_ = float(ss[k0 + j]) ** self.scale_alpha
                    chunk = chunk + cw_ * (l_main + l_alt + l_cons).float()
                    p_main += cw_ * float(l_main); p_alt += cw_ * float(l_alt)
                    p_cons += cw_ * float(l_cons)
                chunk = chunk / wsum_mb
                total = total + chunk.detach()
                if do_bwd:
                    chunk.backward()
            out = total
            if do_bwd:
                sur = (r3_src * r3.grad.detach()).sum()
                if use_alt and a3.grad is not None:
                    sur = sur + (a3_src * a3.grad.detach()).sum()
                out = out + sur - sur.detach()   # value = total; grad = accumulated leaf grads
            if return_parts:
                return out, (p_main / wsum_mb, p_alt / wsum_mb, p_cons / wsum_mb)
            return out
        if self.batch_aug:
            # BATCHED aug path (--clip-batch-aug): one forward for all n_aug crop triples
            # instead of n_aug tiny forwards -- the dominant per-run win on CUDA. Layout
            # after _encode_split: [R(n) | T(n) | A(n)]. Memory scales with n_aug: meant
            # for CUDA; on MPS the sequential path is the safe default.
            n = len(ras)
            A3 = torch.cat(aas, dim=0) if aas else None
            fc, feats = self._encode_split(torch.cat(ras, dim=0), torch.cat(tas, dim=0), A3)
            B_total = n * (3 if A3 is not None else 2)
            for i in range(n):
                l_main, l_alt, l_cons = self._pair_loss(
                    fc, feats, i, n + i, 2 * n + i if A3 is not None else None,
                    alt_w, cons_w, B_total)
                cw_ = ss[i] ** self.scale_alpha
                total = total + cw_ * (l_main + l_alt + l_cons).float(); wsum += cw_
                p_main += cw_ * float(l_main); p_alt += cw_ * float(l_alt)
                p_cons += cw_ * float(l_cons)
        out = total / max(wsum, 1e-9)
        if return_parts:
            return out, (p_main / max(wsum, 1e-9), p_alt / max(wsum, 1e-9), p_cons / max(wsum, 1e-9))
        return out
