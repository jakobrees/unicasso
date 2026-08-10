"""Latent-inspection harness, run after training so runs are comparable.

Produces diagnostic PNGs into an output dir:
  recon.png       -- original | reconstruction for a sample of glyphs
  pca.png         -- 2D PCA of the codebook (mu per glyph), colored by ink density
  interp.png      -- D(lerp(mu_a, mu_b)) for fixed glyph pairs (does the manifold morph
                     through glyph-like shapes, or ghost two glyphs together?)
  interp_snap.png -- the same interpolation plus the nearest real glyph at each step
  nn.png          -- each probe glyph + its nearest latent neighbors
  ink_axis.png    -- reserved-ink-axis alignment + tone sweep
  invariance.png  -- augmented views + the glyph each snaps to
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from unicasso.substrate import glyphs as G


# Default glyph pairs / probes for interpolation and NN panels (filtered to charset).
# Known-similar pairs (case / structural) test whether the manifold places "obviously
# close" glyphs near each other; contrast pairs span tone extremes.
SIMILAR_PAIRS = [("E", "F"), ("j", "J"), ("z", "Z"), ("o", "O"), ("c", "C"),
                 ("p", "P"), ("O", "0"), ("l", "1"), ("B", "8"), ("i", "l")]
CONTRAST_PAIRS = [("@", "."), ("M", " "), ("#", "-"), ("W", "i"), ("%", ".")]
INTERP_PAIRS = SIMILAR_PAIRS + CONTRAST_PAIRS
NN_PROBES = list("AaOo@#8MWil.-=+/?B\\│▓▒▐~╬╣ç╦^")


def _char_idx(chars, c):
    return chars.index(c) if c in chars else None


@torch.no_grad()
def codebook(model, ink):
    """mu for every glyph: (N, latent_dim) numpy."""
    mu, _ = model.encode(ink)
    return mu.detach().cpu().numpy()


@torch.no_grad()
def reconstruct(model, ink):
    """Reconstructed ink (N, 1, H, W) numpy from the deterministic (mu) path."""
    x_hat, _, _, _ = model(ink, sample=False)
    return x_hat.detach().cpu().numpy()


def _imshow_glyph(ax, ink_hw):
    # interpolation="nearest" so the hard low-res pixel grid renders crisp (the default
    # antialiased interp smears the pixels and makes glyphs look shifted/misaligned).
    ax.imshow(1.0 - np.squeeze(ink_hw), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])


def plot_recon(model, ink, chars, out, n=60, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(chars), size=min(n, len(chars)), replace=False)
    recon = reconstruct(model, ink)
    ink_np = ink.detach().cpu().numpy()

    cols = 10
    rows = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 2.0, rows * 1.1))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.axis("off")
    for k, i in enumerate(idx):
        r, c = divmod(k, cols)
        _imshow_glyph(axes[r, 2 * c], ink_np[i])
        axes[r, 2 * c].axis("on"); axes[r, 2 * c].set_xticks([]); axes[r, 2 * c].set_yticks([])
        _imshow_glyph(axes[r, 2 * c + 1], recon[i])
        axes[r, 2 * c + 1].axis("on"); axes[r, 2 * c + 1].set_xticks([]); axes[r, 2 * c + 1].set_yticks([])
    fig.suptitle("orig | recon", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_pca(model, ink, chars, out, annotate=44, seed=0):
    from sklearn.decomposition import PCA
    mu = codebook(model, ink)
    dens = G.ink_density(ink).detach().cpu().numpy()
    p2 = PCA(n_components=2).fit_transform(mu) if mu.shape[1] > 2 else mu

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(p2[:, 0], p2[:, 1], c=dens, cmap="viridis", s=28)
    fig.colorbar(sc, ax=ax, label="ink density")
    rng = np.random.default_rng(seed)
    for i in rng.choice(len(chars), size=min(annotate, len(chars)), replace=False):
        ch = chars[i]
        ax.annotate(ch if ch.strip() else "·", (p2[i, 0], p2[i, 1]), fontsize=8)
    ax.set_title("PCA of codebook (mu), colored by ink density")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


@torch.no_grad()
def plot_interp(model, ink, chars, out, steps=9, pairs=None):
    pairs = pairs or INTERP_PAIRS
    pairs = [(a, b) for a, b in pairs if _char_idx(chars, a) is not None and _char_idx(chars, b) is not None]
    if not pairs:
        return
    mu, _ = model.encode(ink)
    ts = torch.linspace(0, 1, steps, device=ink.device)

    fig, axes = plt.subplots(len(pairs), steps, figsize=(steps * 1.0, len(pairs) * 1.1))
    axes = np.atleast_2d(axes)
    for r, (a, b) in enumerate(pairs):
        za, zb = mu[_char_idx(chars, a)], mu[_char_idx(chars, b)]
        for c, t in enumerate(ts):
            z = (1 - t) * za + t * zb
            dec = model.decode(z.unsqueeze(0))[0]
            _imshow_glyph(axes[r, c], dec.cpu().numpy())
        axes[r, 0].set_ylabel(f"{a!r}->{b!r}", fontsize=8, rotation=0, labelpad=22)
    fig.suptitle("D(lerp(mu_a, mu_b))", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def _label(chars, i):
    c = chars[i]
    return c if c.strip() else "·"


@torch.no_grad()
def plot_interp_snap(model, ink, chars, out, steps=9, pairs=None):
    """Interpolate mu_a -> mu_b; at each step show the decoded glyph AND the nearest real
    glyph (latent L2). Reveals 'character mixing': when a point between two glyphs snaps to
    a *third* character (those steps are titled in red).

    Two rows per pair: [decoded D(z)] over [nearest real glyph bitmap].
    """
    pairs = pairs or INTERP_PAIRS
    pairs = [(a, b) for a, b in pairs if _char_idx(chars, a) is not None and _char_idx(chars, b) is not None]
    if not pairs:
        return
    mu, _ = model.encode(ink)                      # (N, L)
    mu_np = mu.cpu().numpy()
    ink_np = ink.detach().cpu().numpy()
    ts = torch.linspace(0, 1, steps, device=ink.device)

    nrows = 2 * len(pairs)
    fig, axes = plt.subplots(nrows, steps, figsize=(steps * 1.0, nrows * 0.95))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.axis("off")

    for r, (a, b) in enumerate(pairs):
        ia, ib = _char_idx(chars, a), _char_idx(chars, b)
        za, zb = mu[ia], mu[ib]
        zs = torch.stack([(1 - t) * za + t * zb for t in ts])      # (steps, L)
        dec = model.decode(zs).cpu().numpy()                       # (steps, 1, H, W)
        d = np.linalg.norm(mu_np[None, :, :] - zs.cpu().numpy()[:, None, :], axis=-1)
        nearest = d.argmin(axis=1)                                 # (steps,)

        dec_row, snap_row = 2 * r, 2 * r + 1
        for c in range(steps):
            _imshow_glyph(axes[dec_row, c], dec[c])
            j = int(nearest[c])
            # red title when the nearest glyph is neither endpoint -> a "third" character
            is_third = j not in (ia, ib)
            axes[dec_row, c].set_title(_label(chars, j), fontsize=8,
                                       color="red" if is_third else "black")
            _imshow_glyph(axes[snap_row, c], ink_np[j])
        axes[dec_row, 0].axis("on"); axes[dec_row, 0].set_xticks([]); axes[dec_row, 0].set_yticks([])
        axes[dec_row, 0].set_ylabel(f"{a!r}->{b!r}\ndecode", fontsize=7, rotation=0, labelpad=26)
        axes[snap_row, 0].axis("on"); axes[snap_row, 0].set_xticks([]); axes[snap_row, 0].set_yticks([])
        axes[snap_row, 0].set_ylabel("snap", fontsize=7, rotation=0, labelpad=26)

    fig.suptitle("interp decode (top) + nearest real glyph (bottom); red = third char", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_nn(model, ink, chars, out, k=6, probes=None):
    probes = probes or NN_PROBES
    probes = [c for c in probes if _char_idx(chars, c) is not None]
    if not probes:
        return
    mu = codebook(model, ink)
    ink_np = ink.detach().cpu().numpy()
    # Pairwise L2 in latent space.
    d = np.linalg.norm(mu[:, None, :] - mu[None, :, :], axis=-1)

    fig, axes = plt.subplots(len(probes), k + 1, figsize=((k + 1) * 1.0, len(probes) * 1.1))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.axis("off")
    for r, ch in enumerate(probes):
        i = _char_idx(chars, ch)
        order = np.argsort(d[i])  # self first
        _imshow_glyph(axes[r, 0], ink_np[i])
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        axes[r, 0].set_ylabel(ch if ch.strip() else "·", fontsize=8, rotation=0, labelpad=10)
        for c, j in enumerate(order[1:k + 1]):
            _imshow_glyph(axes[r, c + 1], ink_np[j])
    fig.suptitle("probe | nearest latent neighbors", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


@torch.no_grad()
def plot_ink_axis(model, ink, chars, out, axis=0, probes=None, lo=-2.5, hi=2.5, steps=9, seed=0):
    """Diagnostic for the reserved ink axis (--ink-weight). Two panels:

    (left) ALIGNMENT: mu[:,axis] vs true ink density -- does the encoder read ink onto the
           axis? (tight monotone line = yes).
    (right) TONE SWEEP (the decisive test): for a few glyphs, hold all dims fixed and sweep
           z[axis] from lo..hi, decode each. Darkness should change smoothly. If it doesn't,
           the encoder aligned the axis but the DECODER ignores it for tone -- the failure
           mode where ink is a permutation-invariant readout, not a generative lever.
    """
    probes = probes or NN_PROBES
    probes = [c for c in probes if _char_idx(chars, c) is not None][:8]
    mu, _ = model.encode(ink)
    mu_np = mu.cpu().numpy()
    dens = G.ink_density(ink).detach().cpu().numpy()
    decoded_ink = []  # mean ink of decode while sweeping, per glyph

    nrow = max(len(probes), 1)
    fig = plt.figure(figsize=(steps * 1.0 + 5, nrow * 1.0 + 0.5))
    gs = fig.add_gridspec(nrow, steps + 4)

    # Left: alignment scatter (spans all rows, first 4 cols).
    axs = fig.add_subplot(gs[:, :4])
    axs.scatter(mu_np[:, axis], dens, s=14, alpha=0.6)
    try:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(mu_np[:, axis], dens)
        axs.set_title(f"axis {axis} vs ink density (spearman {rho:.2f})", fontsize=9)
    except Exception:
        axs.set_title(f"axis {axis} vs ink density", fontsize=9)
    axs.set_xlabel(f"mu[:, {axis}]"); axs.set_ylabel("ink density")

    # Right: tone sweep, one glyph per row.
    sweep = torch.linspace(lo, hi, steps, device=ink.device)
    for r, ch in enumerate(probes):
        i = _char_idx(chars, ch)
        base = mu[i].clone()
        zs = base.unsqueeze(0).repeat(steps, 1)
        zs[:, axis] = sweep
        dec = model.decode(zs).cpu().numpy()  # (steps, 1, H, W)
        decoded_ink.append(dec.reshape(steps, -1).mean(axis=1))
        for c in range(steps):
            ax = fig.add_subplot(gs[r, 4 + c])
            _imshow_glyph(ax, dec[c])
            if c == 0:
                ax.set_ylabel(ch if ch.strip() else "·", fontsize=8, rotation=0, labelpad=8)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"reserved ink axis {axis}: alignment (left) + tone sweep z[{axis}]={lo}..{hi} (right)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


@torch.no_grad()
def plot_invariance(model, ink, chars, out, probes=None, n_aug=6, **aug_kw):
    """Invariance diagnostic: for each probe glyph show the clean glyph + several augmented
    views, titling each view with the glyph it SNAPS to (nearest clean latent). Red title =
    the augmentation broke the snap (it landed on a different character)."""
    from unicasso.training import augment as gv_aug
    probes = probes or NN_PROBES
    probes = [c for c in probes if _char_idx(chars, c) is not None][:10]
    if not probes:
        return
    mu_clean = model.encode(ink)[0]
    mu_np = mu_clean.cpu().numpy()
    ink_np = ink.detach().cpu().numpy()

    ncol = n_aug + 1
    fig, axes = plt.subplots(len(probes), ncol, figsize=(ncol * 1.0, len(probes) * 1.05))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.axis("off")
    for r, ch in enumerate(probes):
        i = _char_idx(chars, ch)
        _imshow_glyph(axes[r, 0], ink_np[i])
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        axes[r, 0].set_ylabel(ch if ch.strip() else "·", fontsize=8, rotation=0, labelpad=10)
        single = ink[i:i + 1]
        for c in range(n_aug):
            v = gv_aug.augment(single, **aug_kw)
            mv = model.encode(v)[0].cpu().numpy()[0]
            j = int(np.argmin(np.linalg.norm(mu_np - mv, axis=1)))
            _imshow_glyph(axes[r, c + 1], v[0].cpu().numpy())
            axes[r, c + 1].set_title(_label(chars, j), fontsize=8,
                                     color="black" if j == i else "red")
    fig.suptitle("clean | augmented views (title = snapped char; red = broke)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def run_all(model, ink, chars, outdir, ink_axis=0, aug_kw=None):
    """Write the diagnostic PNGs into outdir. aug_kw should match TRAINING augs so invariance.png
    shows what the model was trained to be invariant to (e.g. rot_deg=0 when --aug-rot 0)."""
    os.makedirs(outdir, exist_ok=True)
    plot_recon(model, ink, chars, os.path.join(outdir, "recon.png"))
    plot_pca(model, ink, chars, os.path.join(outdir, "pca.png"))
    plot_interp(model, ink, chars, os.path.join(outdir, "interp.png"))
    plot_interp_snap(model, ink, chars, os.path.join(outdir, "interp_snap.png"))
    plot_nn(model, ink, chars, os.path.join(outdir, "nn.png"))
    plot_ink_axis(model, ink, chars, os.path.join(outdir, "ink_axis.png"), axis=ink_axis)
    plot_invariance(model, ink, chars, os.path.join(outdir, "invariance.png"), **(aug_kw or {}))
    print(f"  wrote recon, pca, interp, interp_snap, nn, ink_axis, invariance -> {outdir}")
