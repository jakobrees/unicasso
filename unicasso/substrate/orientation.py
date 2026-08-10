"""Structure-tensor orientation field: per-cell orientation measurement + a match loss.

A per-cell ORIENTATION match against the target's structure-tensor field addresses messy
glyph picks, shape priority, and cross-cell coherence at once: the target's orientation
field is smooth ALONG a contour (so neighbouring cells get pulled toward consistent
orientations for free) and it breaks the messy/clean tie toward the glyph whose
orientation matches the contour.

KEY over plain Sobel magnitude: magnitude is flip-blind (can't tell `/` from `\`). The
structure tensor gives orientation **mod pi** (so `/` at 45 deg and `\` at 135 deg are
distinct) plus a **coherence** in [0,1] that says how linear/oriented a patch is -- which
doubles as the gate for *which* cells should be directional at all (low coherence -> let the
density term pick the glyph).

Convention: image arrays are (x = column, right) and (y = row, DOWN). All thetas are returned
as the **contour** orientation (the direction the stroke RUNS), already rotated +90 deg off
the dominant gradient, so a horizontal bar `-` reads ~0 and a vertical bar `|` reads ~pi/2.
Drawn directly on an imshow (origin='upper') with (dx, dy) = (cos theta, sin theta).
"""
import math

import torch
import torch.nn.functional as F


def _sobel(img):
    """img: (N,1,H,W) -> (gx, gy), gradients in the (x-right, y-down) frame."""
    kx = torch.tensor([[-1.0, 0.0, 1.0],
                       [-2.0, 0.0, 2.0],
                       [-1.0, 0.0, 1.0]], dtype=img.dtype, device=img.device) / 8.0
    ky = kx.t().contiguous()
    gx = F.conv2d(img, kx.view(1, 1, 3, 3), padding=1)
    gy = F.conv2d(img, ky.view(1, 1, 3, 3), padding=1)
    return gx, gy


def _gaussian1d(sigma, device, dtype):
    radius = max(1, int(round(3.0 * sigma)))
    xs = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(xs * xs) / (2.0 * sigma * sigma))
    return k / k.sum()


def _gaussian_blur(img, sigma):
    """Separable Gaussian blur on (N,1,H,W). sigma<=0 is a no-op."""
    if sigma <= 0:
        return img
    k = _gaussian1d(sigma, img.device, img.dtype)
    r = (k.numel() - 1) // 2
    img = F.conv2d(img, k.view(1, 1, 1, -1), padding=(0, r))
    img = F.conv2d(img, k.view(1, 1, -1, 1), padding=(r, 0))
    return img


def structure_tensor(img, sigma=2.0):
    """Windowed structure tensor of (N,1,H,W). Returns (Jxx, Jyy, Jxy), each (N,1,H,W),
    Gaussian-pooled with `sigma` (the integration window -- larger = smoother orientation,
    coarser locality)."""
    gx, gy = _sobel(img)
    Jxx = _gaussian_blur(gx * gx, sigma)
    Jyy = _gaussian_blur(gy * gy, sigma)
    Jxy = _gaussian_blur(gx * gy, sigma)
    return Jxx, Jyy, Jxy


def orientation_coherence(Jxx, Jyy, Jxy, eps=1e-8):
    """From structure-tensor components -> (contour_theta, coherence, energy).

    contour_theta: orientation the stroke RUNS, mod pi, in [-pi/2, pi/2] (rotated +90 deg off
                   the dominant gradient).
    coherence:     (lambda1 - lambda2) / (lambda1 + lambda2) in [0,1]; 1 = perfectly linear
                   edge, 0 = isotropic (corner / texture / blank). The natural gate weight.
    energy:        lambda1 + lambda2 = trace = total gradient power (how much edge at all).
    """
    theta_grad = 0.5 * torch.atan2(2.0 * Jxy, Jxx - Jyy)   # dominant gradient orientation
    contour = theta_grad + math.pi / 2.0                    # rotate to along-edge
    # wrap back into [-pi/2, pi/2] (mod pi)
    contour = torch.remainder(contour + math.pi / 2.0, math.pi) - math.pi / 2.0
    # clamp (not +eps) inside the sqrt: a tiny in-sqrt floor would divide a near-zero trace on
    # BLANK cells and explode coherence -> ~0 spread on flat cells must give ~0 coherence.
    spread = torch.sqrt(torch.clamp((Jxx - Jyy) ** 2 + 4.0 * Jxy ** 2, min=0.0))
    trace = Jxx + Jyy
    coherence = spread / (trace + eps)
    return contour, coherence, trace


@torch.no_grad()
def glyph_orientations(ink, sigma=0.0):
    """One dominant orientation + coherence PER glyph by integrating the tensor over the whole
    glyph. ink: (N,1,H,W). Returns (theta, coherence, energy), each (N,) on cpu.

    sigma here pre-smooths the gradients (0 = raw); the integration is the full-glyph sum, so
    coherence answers "is this glyph globally linear (/, |, -) or a mess (@, M, %)?".
    """
    gx, gy = _sobel(ink)
    if sigma > 0:
        gx = _gaussian_blur(gx, sigma)
        gy = _gaussian_blur(gy, sigma)
    Jxx = (gx * gx).sum(dim=(1, 2, 3))
    Jyy = (gy * gy).sum(dim=(1, 2, 3))
    Jxy = (gx * gy).sum(dim=(1, 2, 3))
    theta, coh, energy = orientation_coherence(Jxx, Jyy, Jxy)
    return theta.cpu(), coh.cpu(), energy.cpu()


def _double_angle_field(img, char_h, char_w, sigma):
    """Per-cell double-angle orientation vector of (1,1,H,W) image.

    Returns (vx, vy, trace), each (1,1,GH,GW): vx = Jxx-Jyy, vy = 2*Jxy (the structure tensor in
    double-angle form -- direction encodes 2*theta, |(vx,vy)| is the anisotropy "spread"), and
    trace = Jxx+Jyy (total gradient power). avg-pooled over each char_h x char_w cell. Fully
    differentiable (sobel + gaussian + avg_pool are all convs), so a render's field carries grad.
    """
    Jxx, Jyy, Jxy = structure_tensor(img, sigma)
    Jxx = F.avg_pool2d(Jxx, (char_h, char_w))
    Jyy = F.avg_pool2d(Jyy, (char_h, char_w))
    Jxy = F.avg_pool2d(Jxy, (char_h, char_w))
    return Jxx - Jyy, 2.0 * Jxy, Jxx + Jyy


def orientation_match_loss(render, target, char_h, char_w, sigma=4.0, eps=1e-6):
    """Per-cell orientation alignment of a render to a target, gated by TARGET coherence.

    render, target: (H, W) images (white=1 or ink -- orientation is flip-invariant, so either
    works). `render` carries grad; `target` is detached (it's the reference). Both must be laid
    out on the GH*char_h x GW*char_w grid (row_gap=0).

    loss = sum_cell  coh_target * (1 - cos(2*dtheta))  /  sum_cell coh_target

    cos(2*dtheta) = <v_render_hat, v_target_hat> (double-angle dot, mod pi). coh_target gates so
    the term is ~0 on flat target cells (no contour to follow -> let density pick the glyph) and
    strong on contour cells. A blank render in a contour cell gets the full penalty (cos->0), so
    it's pushed to grow an edge of the right orientation. Returns (loss, per_cell_weight) where
    per_cell_weight = coh_target reshaped (GH, GW) for optional inspection.
    """
    r = render[None, None]
    t = target[None, None].detach()
    vxr, vyr, _ = _double_angle_field(r, char_h, char_w, sigma)
    vxt, vyt, trt = _double_angle_field(t, char_h, char_w, sigma)
    # Two separate magnitudes:
    #  - normalization (for cos2): eps INSIDE the sqrt so a blank render cell (v~0) doesn't give
    #    sqrt(0) with infinite derivative -> NaN grads.
    #  - coherence GATE: bare magnitude / trace, so a blank cell reads coherence ~0 (no contour
    #    to follow) instead of eps/eps = 1, which would gate empty regions at full weight.
    magr = torch.sqrt(vxr ** 2 + vyr ** 2 + eps ** 2)
    magt = torch.sqrt(vxt ** 2 + vyt ** 2 + eps ** 2)
    cos2 = (vxr * vxt + vyr * vyt) / (magr * magt)        # cos(2*dtheta) in [-1, 1]
    coh_t = torch.sqrt(vxt ** 2 + vyt ** 2) / (trt + eps)  # target coherence per cell (gate), in [0,1]
    loss = (coh_t * (1.0 - cos2)).sum() / (coh_t.sum() + eps)
    return loss, coh_t[0, 0]


@torch.no_grad()
def glyph_linearity(ink, eps=1e-8):
    """Spatial "is this glyph one thin line?" measure -- complements coherence.

    Coherence looks at the GRADIENT distribution, so a multi-stroke glyph dominated by one
    gradient axis (M, W -- both vertical-heavy) scores high even though it's messy. This looks
    at the INK's spatial distribution instead: PCA of the ink-pixel cloud (weighted by ink).

    Returns (elongation, theta, thinness), each (N,) on cpu:
      elongation:  (l1-l2)/(l1+l2) of the spatial covariance in [0,1]. 1 = a thin straight
                   stroke, 0 = a round/2-D blob. This is the line-likeness coherence misses.
      theta:       major-axis orientation (the direction the ink cloud runs), mod pi, directly
                   comparable to the structure-tensor contour orientation (agree on real lines).
      thinness:    sqrt(l2)/sqrt(l1) -- minor/major std ratio; small = thin (line), large = fat.
    """
    N, _, H, W = ink.shape
    ys, xs = torch.meshgrid(torch.arange(H, device=ink.device, dtype=ink.dtype),
                            torch.arange(W, device=ink.device, dtype=ink.dtype), indexing="ij")
    w = ink[:, 0]                                    # (N,H,W) ink weights
    m = w.sum(dim=(1, 2)) + eps
    cx = (w * xs).sum(dim=(1, 2)) / m
    cy = (w * ys).sum(dim=(1, 2)) / m
    dx = xs[None] - cx[:, None, None]
    dy = ys[None] - cy[:, None, None]
    sxx = (w * dx * dx).sum(dim=(1, 2)) / m
    syy = (w * dy * dy).sum(dim=(1, 2)) / m
    sxy = (w * dx * dy).sum(dim=(1, 2)) / m
    spread = torch.sqrt(torch.clamp((sxx - syy) ** 2 + 4.0 * sxy ** 2, min=0.0))
    tr = sxx + syy
    elong = spread / (tr + eps)
    theta = 0.5 * torch.atan2(2.0 * sxy, sxx - syy)
    theta = torch.remainder(theta + math.pi / 2.0, math.pi) - math.pi / 2.0
    l1 = 0.5 * (tr + spread)
    l2 = 0.5 * (tr - spread)
    thinness = torch.sqrt(torch.clamp(l2, min=0.0)) / (torch.sqrt(torch.clamp(l1, min=0.0)) + eps)
    return elong.cpu(), theta.cpu(), thinness.cpu()


@torch.no_grad()
def cell_orientations(img, grid_h, grid_w, char_h, char_w, sigma=2.0):
    """Per-cell orientation field of a single image laid out on the ASCII grid.

    img: (1,1,H,W) with H == grid_h*char_h, W == grid_w*char_w (row_gap=0 assumed -- the
         orientation field is a property of the source, computed before any row spacing).
    Returns (theta, coherence, energy), each (grid_h, grid_w). Each cell pools the windowed
    structure tensor over its char_h x char_w footprint -- exactly the granularity a per-cell
    loss term would see.
    """
    Jxx, Jyy, Jxy = structure_tensor(img, sigma)
    Jxx = F.avg_pool2d(Jxx, (char_h, char_w))
    Jyy = F.avg_pool2d(Jyy, (char_h, char_w))
    Jxy = F.avg_pool2d(Jxy, (char_h, char_w))
    theta, coh, energy = orientation_coherence(Jxx, Jyy, Jxy)
    return theta[0, 0], coh[0, 0], energy[0, 0]
