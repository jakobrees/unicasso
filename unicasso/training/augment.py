"""Identity-preserving augmentations for the invariance-contrastive and denoising terms.

Each augmentation is a claim that "this is the same glyph", so the transforms are
conservative: sub-pixel shift, small scale, small rotation, light blur. Applied in INK space
(0 = white bg, 1 = ink) on the PADDED field so rotation/shift don't clip glyph content.

All warps run under no_grad and are detached -- the augmented image is a *constant* input to
the encoder (grad flows through the encoder, not the warp), so grid_sample's backward
(unsupported on MPS) is never invoked. Outside-the-glyph samples come back 0 (= white) via
padding_mode="zeros", which is exactly right in ink space.
"""
import math

import torch
import torch.nn.functional as F


def _gaussian_kernel(sigma, ksize, device):
    ax = torch.arange(ksize, device=device, dtype=torch.float32) - (ksize - 1) / 2.0
    k = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    return k / k.sum()


@torch.no_grad()
def _affine_sample(ink, angle, scale, tx, ty):
    """Pixel-space affine (rotation + uniform scale + translation), white (0) outside.

    ink: (B,1,H,W); angle (rad), scale, tx, ty (px): each (B,). Built in pixel coords then
    normalized -- affine_grid on a non-square cell would shear a rotation, so we do it by
    hand (mirrors raster.apply_transform, plus rotation).
    """
    B, _, H, W = ink.shape
    device = ink.device
    ys = torch.arange(H, device=device, dtype=torch.float32)
    xs = torch.arange(W, device=device, dtype=torch.float32)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")  # (H,W)
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    x = (gx - cx).view(1, H, W)
    y = (gy - cy).view(1, H, W)
    # inverse map (output -> source): rotate by -angle, divide by scale, undo translation
    cos = torch.cos(-angle).view(B, 1, 1)
    sin = torch.sin(-angle).view(B, 1, 1)
    s = scale.view(B, 1, 1)
    txb, tyb = tx.view(B, 1, 1), ty.view(B, 1, 1)
    x_src = (cos * (x - txb) - sin * (y - tyb)) / s + cx
    y_src = (sin * (x - txb) + cos * (y - tyb)) / s + cy
    gxn = 2.0 * x_src / (W - 1) - 1.0
    gyn = 2.0 * y_src / (H - 1) - 1.0
    grid = torch.stack([gxn, gyn], dim=-1)  # (B,H,W,2)
    return F.grid_sample(ink, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


@torch.no_grad()
def augment(ink, shift=2.0, scale=0.1, rot_deg=8.0,
            blur_sigma=0.8, blur_alpha=0.7, blur_p=0.5):
    """One augmented view of each glyph. ink: (B,1,H,W) ink space. Returns same shape,
    detached. Ranges are per-sample uniform; blur is applied stochastically and mixed in by a
    random alpha so it stays light (a 'same glyph at a slightly different rendering' claim)."""
    B = ink.shape[0]
    device = ink.device

    def rand(a, b):
        return torch.rand(B, device=device) * (b - a) + a

    angle = rand(-rot_deg, rot_deg) * (math.pi / 180.0)
    out = _affine_sample(ink, angle, rand(1 - scale, 1 + scale), rand(-shift, shift), rand(-shift, shift))

    if blur_sigma > 0 and blur_alpha > 0:
        k = _gaussian_kernel(blur_sigma, 5, device)
        b = F.pad(out, (2, 2, 2, 2), mode="reflect")
        b = F.conv2d(b, k.view(1, 1, 1, -1))   # horizontal
        b = F.conv2d(b, k.view(1, 1, -1, 1))   # vertical -> back to (B,1,H,W)
        alpha = rand(0.0, blur_alpha).view(B, 1, 1, 1)
        gate = (torch.rand(B, device=device) < blur_p).float().view(B, 1, 1, 1)
        out = out + gate * alpha * (b - out)

    return out.clamp(0, 1).detach()


def pixel_noise(ink, smooth_sigma=1.0, smooth_amp=0.08, sharp_frac=0.02, sharp_amp=0.5):
    """Corrupt glyph ink (N,1,H,W, [0,1], 1=ink) to mimic REAL image cells vs clean glyphs, so the
    encoder is robust to non-clean inputs (better warm-start). Two components:
      SMOOTH: a low-frequency field (reflect-padded Gaussian-blurred white noise, renormed to unit std,
              scaled by smooth_amp) -- gentle shading/anti-aliasing-like intensity variation.
      SPARSE SHARP: a fraction sharp_frac of pixels get a +/- sharp_amp spike -- specks/artifacts.
    Clamped to [0,1]. All under no_grad (it's a constant corruption of the input, like the geom augment)."""
    with torch.no_grad():
        out = ink
        if smooth_amp > 0:
            rad = max(1, int(round(3 * smooth_sigma)))
            ax = torch.arange(-rad, rad + 1, device=ink.device, dtype=ink.dtype)
            g = torch.exp(-(ax ** 2) / (2 * smooth_sigma * smooth_sigma))
            k = torch.outer(g, g); k = (k / k.sum()).view(1, 1, 2 * rad + 1, 2 * rad + 1)
            n = torch.randn_like(ink)
            n = F.conv2d(F.pad(n, (rad, rad, rad, rad), mode="reflect"), k)
            out = out + n / (n.std() + 1e-8) * smooth_amp
        if sharp_frac > 0 and sharp_amp > 0:
            mask = (torch.rand_like(ink) < sharp_frac).to(ink.dtype)
            spike = torch.sign(torch.randn_like(ink)) * sharp_amp
            out = out + mask * spike
        return out.clamp(0.0, 1.0)
