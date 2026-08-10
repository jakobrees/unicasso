"""Glyph-level retrieval diagnostics.

The latent's job in the ASCII pipeline is twofold: snapping (nearest codebook glyph) and a
coherent backward gradient. This module measures whether the *snapping metric* (latent L2)
is more perceptual than raw pixel L2, judged against a blurred-pixel "appearance at viewing
distance" oracle, and how augmentation-invariant the snapping is.
"""
import numpy as np
import torch
import torch.nn.functional as F


def _pairwise(x):
    """x: (N, D) -> (N, N) L2."""
    return np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)


def appearance_oracle(ink, kernel=4, stride=2):
    """Blurred-pixel distance matrix: the 'how similar do these look at distance' oracle.

    ink: (N, 1, H, W). Returns (N, N) L2 between avg-pooled glyphs.
    """
    pooled = F.avg_pool2d(ink, kernel, stride).reshape(ink.shape[0], -1).cpu().numpy()
    return _pairwise(pooled)


def pixel_distances(ink):
    """Raw full-resolution pixel L2 matrix (N, N)."""
    flat = ink.reshape(ink.shape[0], -1).cpu().numpy()
    return _pairwise(flat)


@torch.no_grad()
def latent_distances(model, ink):
    """Latent (mu) L2 matrix (N, N)."""
    mu, _ = model.encode(ink)
    return _pairwise(mu.cpu().numpy())


def _rank_agreement(metric_D, oracle_D):
    """Mean Spearman correlation between each glyph's metric ordering and the oracle's.

    High = the metric ranks other glyphs by appearance the way the oracle does.
    """
    from scipy.stats import spearmanr
    n = metric_D.shape[0]
    rhos = []
    for i in range(n):
        mask = np.arange(n) != i
        rho, _ = spearmanr(metric_D[i, mask], oracle_D[i, mask])
        if np.isfinite(rho):
            rhos.append(rho)
    return float(np.mean(rhos)) if rhos else float("nan")


def _nn1_oracle_cost(metric_D, oracle_D):
    """Mean oracle distance of each glyph's top-1 neighbor under `metric_D` (lower = the
    metric's nearest neighbor genuinely looks more similar)."""
    n = metric_D.shape[0]
    costs = []
    for i in range(n):
        d = metric_D[i].copy()
        d[i] = np.inf
        j = int(np.argmin(d))
        costs.append(oracle_D[i, j])
    return float(np.mean(costs))


@torch.no_grad()
def invariance_accuracy(model, ink, trials=8, **aug_kw):
    """Invariance metric: encode augmented views and check the nearest *clean* glyph
    in latent space is the glyph itself. 1.0 = perfectly augmentation-invariant snapping."""
    from unicasso.training import augment as gv_aug
    mu_clean = model.encode(ink)[0].cpu().numpy()  # (N, D) codebook
    N = mu_clean.shape[0]
    correct = total = 0
    for _ in range(trials):
        mu_v = model.encode(gv_aug.augment(ink, **aug_kw))[0].cpu().numpy()
        d = np.linalg.norm(mu_v[:, None, :] - mu_clean[None, :, :], axis=-1)  # (N, N)
        correct += int((d.argmin(axis=1) == np.arange(N)).sum())
        total += N
    return correct / total


def evaluate(model, ink, aug_kw=None):
    """Compare latent vs pixel snapping metrics against the appearance oracle, plus
    augmentation-invariance accuracy. aug_kw should match TRAINING augs (e.g. rot_deg=0
    when trained with --aug-rot 0) so invariance measures what the model was trained to be invariant
    to -- not rotation it was deliberately kept SENSITIVE to."""
    oracle = appearance_oracle(ink)
    pix = pixel_distances(ink)
    lat = latent_distances(model, ink)
    return {
        "rank_agreement_latent": _rank_agreement(lat, oracle),
        "rank_agreement_pixel": _rank_agreement(pix, oracle),
        "nn1_oracle_cost_latent": _nn1_oracle_cost(lat, oracle),
        "nn1_oracle_cost_pixel": _nn1_oracle_cost(pix, oracle),
        "invariance_acc": invariance_accuracy(model, ink, **(aug_kw or {})),
    }