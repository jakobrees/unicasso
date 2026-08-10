"""Small convolutional VAE over glyph bitmaps (ink space, 0=bg / 1=ink).

Latent dim default 12. The model always produces (mu, logvar); whether the latent is
sampled (reparameterized) and whether a KL term is applied is decided by the *training*
config, not here -- a deterministic AE and a VAE share this one model: the AE just uses
`mu` and sets the KL weight to 0.

Default cell geometry is 24x12. If you change CHAR_H/CHAR_W the conv shapes
below (two stride-2 steps -> H/4 x W/4) still hold as long as both are divisible by 4.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlyphVAE(nn.Module):
    def __init__(self, latent_dim=12, char_h=24, char_w=12):
        super().__init__()
        self.latent_dim = latent_dim
        self.char_h = char_h
        self.char_w = char_w
        self.fh = char_h // 4  # feature-map height after two stride-2 convs
        self.fw = char_w // 4
        self.flat = 128 * self.fh * self.fw

        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1), nn.GELU(),    # H   x W
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),   # H/2 x W/2
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(),  # H/4 x W/4
        )
        self.fc_mu = nn.Linear(self.flat, latent_dim)
        self.fc_logvar = nn.Linear(self.flat, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, self.flat)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.GELU(),  # H/2 x W/2
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.GELU(),   # H   x W
            nn.Conv2d(32, 1, 3, stride=1, padding=1),                        # H   x W
        )

    def encode(self, x):
        h = self.enc(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        h = self.fc_dec(z).view(-1, 128, self.fh, self.fw)
        return torch.sigmoid(self.dec(h))  # ink in [0, 1]

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x, sample=False, noise_std=0.0):
        """sample: VAE reparameterization (scale from logvar, regularized by KL).
        noise_std: additional FIXED-scale isotropic latent noise (denoising-AE
        regularizer); unlike reparam noise the encoder can't scale it away, so it works
        even at beta=0. Both default off -> deterministic z = mu."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if sample else mu
        if noise_std > 0:
            z = z + noise_std * torch.randn_like(z)
        return self.decode(z), mu, logvar, z


def multiscale_recon_loss(x_hat, x, kernel=4, stride=2):
    """Full-resolution MSE + one avg-pooled-scale MSE.

    Mirrors the rasterizer's multiscale loss (avg_pool2d, kernel 4 / stride 2) so the AE judges
    reconstructions the same way the ASCII renderer is judged: appearance at viewing
    distance, not exact edge placement.
    """
    full = F.mse_loss(x_hat, x)
    xp = F.avg_pool2d(x, kernel, stride)
    xhp = F.avg_pool2d(x_hat, kernel, stride)
    coarse = F.mse_loss(xhp, xp)
    return full + coarse, full.detach(), coarse.detach()


def kl_divergence(mu, logvar):
    """Mean-over-batch KL(q(z|x) || N(0, I))."""
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


def nt_xent(z1, z2, temp=0.2, neg_bias=None):
    """SimCLR NT-Xent / InfoNCE over two views. z1, z2: (N, D) embeddings of the SAME N
    glyphs under two augmentations. Pulls each glyph's two views together and pushes all
    other glyphs apart -- imposed directly on mu (the snapping space), no projection head.

    Cross-entropy where, for anchor i, the "correct class" is its other view; cosine
    similarities are the logits (scaled by temp). Lower temp = harsher on nearest negatives.

    neg_bias: optional (2N, 2N) additive bias on the logits for SOFT NEGATIVES -- e.g.
    log(1 - strength * S_teacher) so look-alike glyphs (per a frozen teacher) are down-
    weighted as negatives instead of being shoved apart. Must be 0 on the positive pairs.
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)              # (2N, 2N)
    sim = (z @ z.t()) / temp                    # (2N, 2N)
    sim.fill_diagonal_(float("-inf"))           # exclude self-similarity
    if neg_bias is not None:
        sim = sim + neg_bias
    targets = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)
    return F.cross_entropy(sim, targets)


def teacher_neg_bias(mu_teacher, strength=1.0, eps=1e-4):
    """Build the (2N, 2N) soft-negative bias from a frozen teacher's codebook.

    Entry = log(1 - strength * cos_sim_teacher(i, j)); tiled over the two views and zeroed
    on the positive pairs so they're never down-weighted. strength in [0,1]: 0 = plain
    NT-Xent, 1 = fully excise teacher-identical glyphs from the negative set.
    """
    z = F.normalize(mu_teacher, dim=1)
    S = (z @ z.t()).clamp(0.0, 1.0)                    # (N, N) cosine sim
    N = S.shape[0]
    bias = torch.log(1.0 - strength * S + eps)         # -> very negative as S -> 1
    big = bias.repeat(2, 2)                             # (2N, 2N), both views share glyph idx
    idx = torch.arange(N, device=S.device)
    big[idx, idx + N] = 0.0                             # protect positives
    big[idx + N, idx] = 0.0
    return big


def structure_affinity_loss(z, theta=None, qual=None, dens=None, cats=None,
                            orient_w=0.0, quality_w=0.0, ink_w=0.0, cat_w=0.0,
                            gate_pow=1.0, eps=1e-8):
    """Soft push/pull on a SAMPLED batch so latent similarity tracks structure relationally.

    z:     (B, L) latent (mu) of B randomly-sampled glyphs.
    theta: (B,) orientation (mod pi). qual: (B,) cleanliness [0,1]. dens: (B,) ink density [0,1].
    cats:  (B, C) category-membership matrix (0/1).

    Four independent, separately-weighted terms (all default off -> A/B-able), each an MSE(S, target)
    over off-diagonal pairs, S = cosine(z_i, z_j) in [-1, 1]:
      orient_w   target cos(2*dtheta), GATED by (q_i*q_j)^gate_pow so a messy glyph (M) isn't dragged
                 toward a clean line (|) just for sharing an axis.
      quality_w  target 1 - 2|q_i - q_j| -> builds the clean-vs-messy axis.
      ink_w      dens is a (B,K) descriptor [density, com_x, com_y] (each [0,1]); target
                 1 - 2*mean|dd| -> a relational TONE+PLACEMENT direction (how much ink AND where it
                 sits; density alone would conflate _ vs overline).
      cat_w      target 1 if the two glyphs share >=1 category else 0 -> clusters visually-similar
                 glyphs (navigability), without forcing different categories apart.

    A relational / soft-contrastive term: attracts where target affinity is high, relaxes where low.
    Keep the weights SMALL so it nudges, not dominates, recon (and so it doesn't fight uniformity).
    """
    zc = F.normalize(z, dim=1)
    S = zc @ zc.t()                                          # (B, B) in [-1, 1]
    B = z.shape[0]
    off = ~torch.eye(B, dtype=torch.bool, device=z.device)
    loss = z.new_zeros(())
    if orient_w > 0:
        tgt = torch.cos(2.0 * (theta[:, None] - theta[None, :]))
        gate = (qual[:, None] * qual[None, :] + eps).pow(gate_pow)
        loss = loss + orient_w * ((S - tgt) ** 2 * gate)[off].mean()
    if quality_w > 0:
        tgt = 1.0 - 2.0 * (qual[:, None] - qual[None, :]).abs()
        loss = loss + quality_w * ((S - tgt) ** 2)[off].mean()
    if ink_w > 0:
        d = dens if dens.dim() > 1 else dens[:, None]        # (B,K) descriptor: density + center-of-mass
        diff = (d[:, None, :] - d[None, :, :]).abs().mean(-1)  # mean L1 over components, in [0,1]
        tgt = 1.0 - 2.0 * diff
        loss = loss + ink_w * ((S - tgt) ** 2)[off].mean()
    if cat_w > 0:
        shared = ((cats.float() @ cats.float().t()) > 0).float()   # (B,B) share >=1 category
        loss = loss + cat_w * ((S - shared) ** 2)[off].mean()
    return loss


def uniformity_loss(z, t=2.0):
    """Wang & Isola uniformity: log E[exp(-t * ||z_i - z_j||^2)] on L2-normalized z. The decoupled
    'spread' force -- the uniformity half of InfoNCE, pairs with alignment_loss (positives-only).
    Minimizing it spreads embeddings toward uniform on the hypersphere (anti-collapse)."""
    z = F.normalize(z, dim=1)
    sq = (2.0 - 2.0 * (z @ z.t())).clamp_min(0.0)           # ||zi-zj||^2 for unit vectors (MPS-safe; pdist isn't)
    N = z.shape[0]
    off = ~torch.eye(N, dtype=torch.bool, device=z.device)
    return sq[off].mul(-t).exp().mean().log()


def alignment_loss(z1, z2):
    """Invariance WITHOUT negatives: pull the two views of each glyph together (1 - cosine),
    no repulsion. Recon keeps distinct glyphs apart, so we don't need the instance-
    discrimination push that also separates look-alikes (O/0/o) we WANT clustered for
    snapping. Cheaper alternative to nt_xent when hard negatives degrade the neighborhood."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    return (1.0 - (z1 * z2).sum(dim=1)).mean()
