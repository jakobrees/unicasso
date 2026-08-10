"""Non-local semantic affinity for the latent-affinity loss -- the "constrain the solution space"
alternative to candidate injection (inject.py). Per cell, find its top-k corresponding
cells by CNN visual similarity GATED by DINO semantics -- stretch(CNN_multiscale * DINO^beta) -- then
emit edges for a graph-Laplacian pull on the per-cell latents z (reuses asciify's consistency runtime:
Sum w_ij * ||z_i - z_j||^2). Injection introduces NEW glyph candidates (exploration, chaotic); this
just biases similar regions toward consistent latents (regularization, smooth, Adam-friendly).

The similarity helpers (cnn_sim/dino_sim/_stretch01/...) live HERE, with no training-side
imports, so asciify.py can import this cleanly.
"""
import numpy as np
import torch
import torch.nn.functional as F


def _cells_from_map(feat, GH, GW):
    """feat: (C,h,w) torch -> (M,C) L2-normalized per-cell features (bilinear resample to the grid).
    Decouples the backbone's native resolution from the glyph grid; recouples by pooling to cells."""
    f = F.interpolate(feat[None], size=(GH, GW), mode="bilinear", align_corners=False)[0]   # (C,GH,GW)
    return F.normalize(f.reshape(f.shape[0], -1).t(), dim=1)                                 # (M,C)


def _cosine01(fn):
    """(M,C) normalized -> (M,M) cosine mapped to [0,1]."""
    return (0.5 * (1.0 + fn @ fn.t())).cpu().numpy().astype(np.float32)


def _stretch01(S, lo=2.0, hi=98.0):
    """Percentile contrast-stretch a similarity matrix's off-diagonal to [0,1]. CNN/DINO cosines sit
    in a narrow high band (all natural patches positively correlate), so raw they read ~uniform;
    stretching the [lo,hi] percentiles to [0,1] makes them discriminative."""
    iu = ~np.eye(S.shape[0], dtype=bool)
    a, b = np.percentile(S[iu], [lo, hi])
    if b - a < 1e-6:
        return S
    out = np.clip((S - a) / (b - a), 0.0, 1.0).astype(np.float32)
    np.fill_diagonal(out, 1.0)
    return out


def cnn_sim(feat_t, layers, GH, GW, device, max_side=448, clip=None):
    """Multi-scale CLIP-conv similarity: one RN101 forward -> per-layer feature pyramid (layer1 fine
    .. layer4 coarse) resampled to cells, per-layer stretched cosine, averaged then re-stretched.
    Pass an existing CLIPPerceptualLoss as `clip` to avoid reloading the backbone."""
    from unicasso.engine.clip_loss import CLIPPerceptualLoss
    clip = clip if clip is not None else CLIPPerceptualLoss(device)
    maps = clip.dense_features(feat_t, layers=tuple(layers), max_side=max_side,
                               base=True)   # affinity edges describe the TARGET: base path
    mats = [_stretch01(_cosine01(_cells_from_map(maps[l], GH, GW))) for l in layers]
    return _stretch01(np.mean(mats, axis=0).astype(np.float32))


def dino_sim(feat_t, GH, GW, device, model_name="vit_base_patch16_dinov3.lvd1689m", dino=None):
    """DINOv3 patch-token similarity (semantic correspondence). Pass an existing DINOPerceptualLoss
    as `dino` to avoid reloading the backbone."""
    from unicasso.engine.dino_loss import DINOPerceptualLoss
    dino = dino if dino is not None else DINOPerceptualLoss(device, model_name=model_name)
    tok, ph, pw = dino.dense_tokens(feat_t)                # (P,C)
    feat = tok.t().reshape(tok.shape[1], ph, pw)           # (C,ph,pw)
    return _stretch01(_cosine01(_cells_from_map(feat, GH, GW)))


def gated_matrix(feat_t, GH, GW, device, layers=(1, 2, 3), beta=2.0,
                 dino_model="vit_base_patch16_dinov3.lvd1689m", max_side=448, clip=None):
    """CNN visual similarity GATED by DINO semantics: stretch(CNN * DINO^beta). Returns (M,M) [0,1]."""
    cnn = cnn_sim(feat_t, layers, GH, GW, device, max_side, clip)
    dn = dino_sim(feat_t, GH, GW, device, model_name=dino_model)
    return _stretch01(cnn * (dn ** beta))


def affinity_edges(A, topk, gamma=1.0, ink=None, min_ink=0.0, device="cpu"):
    """(M,M) gated similarity -> directed top-k edges (si, dj, w). Per row keep the k most-similar
    cells (self excluded); edge weight = A^gamma. Optionally drop edges touching near-empty cells
    (ink fraction < min_ink) so the pull budget isn't spent tying blank regions. w sums to 1.
    Mutual matches naturally appear as two edges -> pulled harder (desirable)."""
    M = A.shape[0]
    A = A.copy()
    np.fill_diagonal(A, -1.0)                              # exclude self
    if ink is not None and min_ink > 0:
        blank = ink < min_ink
        A[blank, :] = -1.0                                 # don't pull FROM a near-empty source cell
        A[:, blank] = -1.0                                 # ...or TO one
    k = min(max(1, topk), M - 1)
    idx = np.argpartition(-A, kth=k - 1, axis=1)[:, :k]    # (M,k) top-k cols per row
    si = np.repeat(np.arange(M), k)
    dj = idx.reshape(-1)
    w = A[si, dj]
    keep = w > 1e-6
    si, dj, w = si[keep], dj[keep], np.clip(w[keep], 0.0, 1.0) ** gamma
    w = w / (w.sum() + 1e-9)
    return (torch.from_numpy(si).long().to(device),
            torch.from_numpy(dj).long().to(device),
            torch.from_numpy(w).float().to(device))
