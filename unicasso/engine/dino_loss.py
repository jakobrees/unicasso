"""DINOv3 perceptual loss: match the render's DINO features to the target's.

Motivation (vs the CLIPasso/CLIP loss in clip_loss.py): we do IMAGE-to-IMAGE matching, so CLIP's
text-aligned embedding is dead weight -- it discards low-level structure. DINOv3 is built for dense,
spatially-coherent geometric features (correspondence / part structure), so it's a better backbone
for a structural perceptual loss. Loaded via timm; the weights are gated on Hugging Face
(a logged-in HF token is required).

Two terms, both averaged over the SAME random-resized crops applied to render + target (CLIPasso's
anti-adversarial trick -- still needed):
  - STRUCTURE (Splice / dino-vit-features style): MSE between the patch-token SELF-SIMILARITY
    matrices of render vs target. Self-similarity is appearance/domain INVARIANT -- it matches the
    pattern of "which regions resemble which", not absolute features -- which cancels the photo-vs-
    sketch domain gap (our render is a sparse line/blur image, out of DINO's photo distribution).
  - GLOBAL: 1 - cosine(CLS_render, CLS_target), for overall layout (small weight, like CLIPasso fc).

Same render polarity as CLIP loss: white bg = 1, ink = 0, expanded to 3 channels, ImageNet-normalized.
"""
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm


class DINOPerceptualLoss(nn.Module):
    def __init__(self, device, model_name="vit_base_patch16_dinov3.lvd1689m",
                 n_aug=4, crop_scale=(0.7, 1.0), res=224,
                 struct_weight=1.0, global_weight=0.1, cache_target=True, bank_size=32, scale_alpha=1.0):
        super().__init__()
        # ViT needs img_size to fit its pos-embed; ConvNeXt (the DINOv3-distilled CNN variants,
        # e.g. convnext_small.dinov3_lvd1689m) is fully convolutional and size-agnostic + much faster.
        kw = dict(pretrained=True, num_classes=0)
        if model_name.startswith("vit"):
            kw["img_size"] = res
        model = timm.create_model(model_name, **kw)
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model
        self.prefix = getattr(model, "num_prefix_tokens", 1)  # CLS (+ register tokens)
        self.device = device
        self.n_aug = n_aug
        self.crop_scale = crop_scale
        self.scale_alpha = scale_alpha   # crop weight = s^alpha (log-uniform sampling; alpha=1 unbiased)
        self.res = res
        self.struct_weight = struct_weight
        self.global_weight = global_weight
        dc = timm.data.resolve_model_data_config(model)
        self.mean = torch.tensor(dc["mean"], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(dc["std"], device=device).view(1, 3, 1, 1)
        # Target-feature cache: the target is fixed (unless --align warps it), so we precompute its
        # self-sim + CLS for a FIXED bank of crops once, then forward ONLY the render each step (~2x).
        self.cache_target = cache_target
        self.bank_size = max(bank_size, n_aug)
        self._bank = None        # list of fixed crop params
        self._tgt_ref = None     # identity of the cached target tensor (rebuild if it changes)
        self._tgt_sim = None     # (K, P, P) target self-similarity per bank crop
        self._tgt_cls = None     # (K, C) normalized target global token per bank crop

    def _rrc_params(self, H, W):
        lo, hi = self.crop_scale
        s = lo * (hi / lo) ** random.random()             # log-uniform: p(s) ∝ 1/s
        ch = max(8, int(round(H * (s ** 0.5))))
        cw = max(8, int(round(W * (s ** 0.5))))
        top = random.randint(0, H - ch)
        left = random.randint(0, W - cw)
        return (top, left, ch, cw), s

    def _prep(self, img3, p):
        top, left, ch, cw = p
        crop = img3[:, :, top:top + ch, left:left + cw]
        return F.interpolate(crop, size=(self.res, self.res), mode="bilinear", align_corners=False)

    def _tokens(self, batch):
        """batch: (B,3,res,res) in [0,1] -> (global (B,C), patches (B,P,C)). Backbone-agnostic:
        ViT returns tokens (CLS+register prefix, then patches); ConvNeXt returns a (B,C,H,W) map."""
        x = (batch - self.mean) / self.std
        feats = self.model.forward_features(x)
        if feats.dim() == 4:                          # conv backbone (B,C,H,W)
            return feats.mean(dim=(2, 3)), feats.flatten(2).transpose(1, 2)
        return feats[:, 0], feats[:, self.prefix:]    # ViT tokens

    @torch.no_grad()
    def dense_tokens(self, img01):
        """Whole-image patch tokens. img01: (H,W) or (1,1,H,W) in [0,1] white=1. ViT needs its fixed
        img_size, so the image is square-resized to res (sub-cell drift washes out after pooling to
        cells); ConvNeXt is size-agnostic. Returns (tokens (ph*pw, C), ph, pw)."""
        if img01.dim() == 2:
            img01 = img01[None, None]
        x = F.interpolate(img01.expand(1, 3, -1, -1).to(self.device), size=(self.res, self.res),
                          mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        feats = self.model.forward_features(x)
        if feats.dim() == 4:                                   # conv backbone (1,C,H,W)
            _, C, ph, pw = feats.shape
            return feats.flatten(2).transpose(1, 2)[0], ph, pw
        patches = feats[:, self.prefix:][0]                    # (P,C) ViT
        gs = getattr(self.model.patch_embed, "grid_size", None)
        if gs is None:
            n = int(round(patches.shape[0] ** 0.5)); gs = (n, n)
        return patches, gs[0], gs[1]

    @staticmethod
    def _self_sim(patches):
        pn = F.normalize(patches, dim=-1)
        return pn @ pn.transpose(1, 2)                 # (B, P, P)

    @torch.no_grad()
    def _build_target_cache(self, t3):
        sims, clss = [], []
        for p, _s in self._bank:
            cls, patches = self._tokens(self._prep(t3, p))   # (1,C), (1,P,C)
            sims.append(self._self_sim(patches)[0])           # (P,P)
            clss.append(F.normalize(cls, dim=-1)[0])          # (C,)
        self._tgt_sim = torch.stack(sims)                     # (K,P,P)
        self._tgt_cls = torch.stack(clss)                     # (K,C)

    def forward(self, render, target):
        """render: (H,W) white=1, carries grad. target: (H,W) white=1, reference. -> scalar."""
        r3 = render[None, None].expand(1, 3, -1, -1)
        _, _, H, W = r3.shape

        if self.cache_target:
            if self._bank is None:
                self._bank = [self._rrc_params(H, W) for _ in range(self.bank_size)]
            if self._tgt_ref is not target:              # (re)build only when the target changes
                self._build_target_cache(target.detach()[None, None].expand(1, 3, -1, -1))
                self._tgt_ref = target
            idx = random.sample(range(self.bank_size), self.n_aug)   # distinct crops from the bank
            crops = torch.cat([self._prep(r3, self._bank[i][0]) for i in idx], dim=0)  # (n_aug,3,res,res)
            cls, patches = self._tokens(crops)           # render only -- target is cached
            struct_pc = (self._self_sim(patches) - self._tgt_sim[idx]).pow(2).mean(dim=(1, 2))   # (n_aug,)
            glob_pc = 1.0 - (F.normalize(cls, dim=-1) * self._tgt_cls[idx]).sum(-1)              # (n_aug,)
            per = self.struct_weight * struct_pc + self.global_weight * glob_pc
            w = per.new_tensor([self._bank[i][1] ** self.scale_alpha for i in idx])              # crop weights
            return (w * per).sum() / w.sum().clamp_min(1e-9)

        # uncached path (e.g. --align warps the target every step): forward render + target together
        t3 = target.detach()[None, None].expand(1, 3, -1, -1)
        total = render.new_zeros(()); wsum = 0.0
        for _ in range(self.n_aug):
            p, s = self._rrc_params(H, W)                 # SAME crop for render + target
            cls, patches = self._tokens(torch.cat([self._prep(r3, p), self._prep(t3, p)], dim=0))
            sim = self._self_sim(patches)                 # (2, P, P)
            struct = (sim[0] - sim[1]).pow(2).mean()
            cn = F.normalize(cls, dim=-1)
            glob = 1.0 - (cn[0] * cn[1]).sum()
            w_ = s ** self.scale_alpha
            total = total + w_ * (self.struct_weight * struct + self.global_weight * glob); wsum += w_
        return total / max(wsum, 1e-9)
