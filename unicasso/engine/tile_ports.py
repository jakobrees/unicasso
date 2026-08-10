"""Glyphs as tiles with edge PORTS: 1-D border profiles + a glyph-glyph compatibility matrix.

The proposal/coupling currency that the linear gradient score can't provide: what makes a glyph
"right" in line art is mostly what happens at its borders -- a stroke enters at height y with
angle theta and must continue coherently into the neighbor. Per glyph we reduce each border band
to a 1-D profile with 3 channels: [ink, ink*coh*cos2theta, ink*coh*sin2theta] (contour orientation
as a double angle, so a '/' and a '-' crossing at the same height are DIFFERENT ports), slightly
blurred along the edge so near-aligned crossings match softly.

Compatibility (precomputed once per font, image-independent):
    C_horiz[g, g'] = sim(right(g), left(g'))     "g' sits to the RIGHT of g"
    C_vert [g, g'] = sim(bottom(g), top(g'))     "g' sits BELOW g"
    sim(a, b) = <a, b> - gamma * ||a - b||^2
chosen so aligned crossings score positive, crossing-vs-blank scores NEGATIVE (this term IS the
broken-line penalty), and blank-vs-blank scores 0 (whitespace continuity is neutral, not rewarded).

Core functions are importable (for a nomination channel / a differentiable coupling w_i^T C w_j
later); the CLI is a PREVIEW: for chosen glyphs, print + render the top-compatible partners per
direction as joined two-cell images, so the matrix is judged by eye before it touches the optimizer.
"""
import argparse
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from unicasso.substrate import glyphs as G


DEFAULT_TILE_CHARS = ("─│┌┐└┘├┤┬┴┼═║╔╗╚╝╠╣╦╩╬╒╓╕╖╘╙╛╜╞╟╡╢╤╥╧╨╪╫"
                      "█░▒▓/\\_¬⌐-=≡·|")


def _gauss1d(sigma, device):
    rad = max(1, int(round(3 * sigma)))
    ax = torch.arange(-rad, rad + 1, device=device, dtype=torch.float32)
    k = torch.exp(-(ax ** 2) / (2 * sigma * sigma))
    return (k / k.sum()).view(1, 1, 1, -1), rad


def _blur2d(x, sigma):
    if sigma <= 0:
        return x
    k, r = _gauss1d(sigma, x.device)
    C = x.shape[1]
    kh = k.expand(C, 1, 1, k.shape[-1])
    kv = k.transpose(2, 3).expand(C, 1, k.shape[-1], 1)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), kh, groups=C)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), kv, groups=C)


def _contour_channels(ink, sigma=1.5, eps=1e-6):
    """ink (N,1,H,W) -> (N,2,H,W): ink- and coherence-weighted CONTOUR double-angle components.
    Structure tensor J = blur(grad grad^T); gradient double angle = (Jxx-Jyy, 2Jxy); the contour is
    the gradient rotated 90 deg, which in double-angle space is a sign flip. Weighted by coherence
    (|v|/trace) and by local ink so orientation only speaks where a stroke actually is."""
    kx = torch.tensor([[[[-0.5, 0.0, 0.5]]]], device=ink.device)
    gx = F.conv2d(F.pad(ink, (1, 1, 0, 0), mode="replicate"), kx)
    gy = F.conv2d(F.pad(ink, (0, 0, 1, 1), mode="replicate"), kx.transpose(2, 3))
    Jxx = _blur2d(gx * gx, sigma)
    Jyy = _blur2d(gy * gy, sigma)
    Jxy = _blur2d(gx * gy, sigma)
    vx, vy = Jxx - Jyy, 2.0 * Jxy
    mag = torch.sqrt(vx * vx + vy * vy) + eps
    coh = (mag / (Jxx + Jyy + eps)).clamp(0, 1)
    return torch.cat([-vx / mag, -vy / mag], dim=1) * coh * ink   # contour = gradient rotated (sign flip)


def edge_profiles(ink, band=3, blur_sigma=1.5, orient_weight=1.0, orient_sigma=1.5):
    """ink (N,1,H,W) in [0,1] -> {edge: (N,3,L)} 1-D border profiles, blurred along the edge.
    Channels: [ink, oriented-x, oriented-y]. L = H for left/right, W for top/bottom."""
    N, _, H, W = ink.shape
    ch = torch.cat([ink, orient_weight * _contour_channels(ink, orient_sigma)], dim=1)   # (N,3,H,W)
    prof = {
        "left": ch[:, :, :, :band].mean(dim=3),
        "right": ch[:, :, :, W - band:].mean(dim=3),
        "top": ch[:, :, :band, :].mean(dim=2),
        "bottom": ch[:, :, H - band:, :].mean(dim=2),
    }
    if blur_sigma > 0:
        k, r = _gauss1d(blur_sigma, ink.device)
        for e, p in prof.items():
            p = F.conv1d(F.pad(p, (r, r), mode="replicate"), k[0].expand(3, 1, -1), groups=3)
            prof[e] = p
    return prof


def _sim(A, B, gamma):
    """A, B: (N,3,L) -> (N,N) with sim = <a,b> - gamma*||a-b||^2 (aligned>0, crossing-vs-blank<0,
    blank-vs-blank=0)."""
    a, b = A.flatten(1), B.flatten(1)
    dot = a @ b.t()
    na, nb = (a * a).sum(1)[:, None], (b * b).sum(1)[None, :]
    return dot - gamma * (na + nb - 2.0 * dot)


def compat_matrices(prof, gamma=0.5):
    """-> C_horiz[g,g'] (g' RIGHT of g), C_vert[g,g'] (g' BELOW g), both (N,N)."""
    return _sim(prof["right"], prof["left"], gamma), _sim(prof["bottom"], prof["top"], gamma)


def _sim_cross(A, B, gamma):
    """A (M,3,L) cell profiles x B (N,3,L) glyph profiles -> (M,N) sim matrix."""
    a, b = A.flatten(1), B.flatten(1)
    dot = a @ b.t()
    na, nb = (a * a).sum(1)[:, None], (b * b).sum(1)[None, :]
    return dot - gamma * (na + nb - 2.0 * dot)


def target_port_scores(target, GH, GW, CH, CW, glyph_prof, gamma=0.5, band=3,
                       blur_sigma=1.5, orient_weight=1.0, orient_sigma=1.5):
    """Per-(cell, glyph) TARGET port match: how well does each glyph's edge wiring match what the
    target image does at each cell's four borders? Target boundary profiles use a band STRADDLING
    the boundary (+/-band px), so they capture the actual stroke crossing. Returns (M, N) =
    sum over the cell's 4 edges of sim(glyph edge profile, target boundary profile)."""
    ink = (1.0 - target)[None, None]                                  # (1,1,H,W) ink space
    ch = torch.cat([ink, orient_weight * _contour_channels(ink, orient_sigma)], dim=1)[0]  # (3,H,W)
    H, W = target.shape
    k, r = (None, 0)
    if blur_sigma > 0:
        k, r = _gauss1d(blur_sigma, target.device)

    def blur1d(p):                                                    # p (M,3,L)
        if k is None:
            return p
        return F.conv1d(F.pad(p, (r, r), mode="replicate"), k[0].expand(3, 1, -1), groups=3)

    M = GH * GW
    prof_t = {}
    ys = torch.arange(GH, device=target.device)
    xs = torch.arange(GW, device=target.device)
    # vertical boundaries (left/right edges): band around x = col*CW (left) and (col+1)*CW (right)
    for edge, xb in (("left", xs * CW), ("right", (xs + 1) * CW)):
        cols = []
        for x in xb.tolist():
            x0, x1 = max(0, x - band), min(W, x + band)
            cols.append(ch[:, :, x0:x1].mean(dim=2))                   # (3,H)
        colp = torch.stack(cols, dim=0)                                # (GW,3,H)
        cellp = colp.view(GW, 3, GH, CH).permute(2, 0, 1, 3).reshape(M, 3, CH)
        prof_t[edge] = blur1d(cellp)
    # horizontal boundaries (top/bottom): band around y = row*CH (top) and (row+1)*CH (bottom)
    for edge, yb in (("top", ys * CH), ("bottom", (ys + 1) * CH)):
        rows_ = []
        for y in yb.tolist():
            y0, y1 = max(0, y - band), min(H, y + band)
            rows_.append(ch[:, y0:y1, :].mean(dim=1))                  # (3,W)
        rowp = torch.stack(rows_, dim=0)                               # (GH,3,W)
        cellp = rowp.view(GH, 3, GW, CW).permute(0, 2, 1, 3).reshape(M, 3, CW)
        prof_t[edge] = blur1d(cellp)
    T = sum(_sim_cross(prof_t[e], glyph_prof[e], gamma) for e in ("left", "right", "top", "bottom"))
    return T                                                          # (M, N)


def load_overrides(path):
    """{'horiz': {'<g><g2>': 'ban', ...}, 'vert': {...}} -- g2 sits right-of/below g."""
    import json, os
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"horiz": {}, "vert": {}}


def apply_overrides(C_h, C_v, ov, chars, ban_val=-1e9):
    """Set banned pairs to ban_val (drops them from any ranking/proposal)."""
    idx = {c: i for i, c in enumerate(chars)}
    for key, C in (("horiz", C_h), ("vert", C_v)):
        for pair, v in ov.get(key, {}).items():
            if v == "ban" and len(pair) == 2 and pair[0] in idx and pair[1] in idx:
                C[idx[pair[0]], idx[pair[1]]] = ban_val
    return C_h, C_v


def interactive(args, ink, chars, C_h, C_v, allowed):
    """Click a glyph in the PALETTE -> click an EDGE on the big render -> top partners for that
    side appear as joined pairs -> click a pair to BAN/UNBAN it (saved to --overrides JSON on
    every toggle; consumed later by proposal machinery via load_overrides/apply_overrides)."""
    import json
    import matplotlib.pyplot as plt

    N = len(chars)
    H, W = ink.shape[2], ink.shape[3]
    bm = (1.0 - ink[:, 0]).numpy()                            # white=1 (N,H,W)
    if args.overrides is None:
        args.overrides = os.path.join(G.kit_dir(), "port_overrides.json")
    ov = load_overrides(args.overrides)
    # tile membership is SYMMETRIC: only whitelist glyphs give OR receive tile candidates,
    # so the palette shows the tile vocabulary only.
    vis = allowed.nonzero().flatten().tolist()
    pal_cols = 8 if len(vis) <= 96 else 16
    pal_rows = math.ceil(len(vis) / pal_cols)
    palette = np.ones((pal_rows * H, pal_cols * W))
    for j, i in enumerate(vis):
        r, c = divmod(j, pal_cols)
        palette[r * H:(r + 1) * H, c * W:(c + 1) * W] = bm[i]

    g0 = chars.index("─") if "─" in chars and allowed[chars.index("─")] else vis[0]
    state = {"g": g0, "edge": "right"}
    n_res = args.topn
    res_cols = min(6, n_res)
    res_rows = math.ceil(n_res / res_cols)

    fig = plt.figure(figsize=(16, 9))
    ax_pal = fig.add_axes([0.02, 0.04, 0.34, 0.90])
    ax_g = fig.add_axes([0.40, 0.55, 0.10, 0.36])
    res_axes = []
    for i in range(n_res):
        r, c = divmod(i, res_cols)
        res_axes.append(fig.add_axes([0.54 + c * 0.075, 0.62 - r * 0.30, 0.065, 0.26]))
    ax_pal.imshow(palette, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax_pal.set_title("palette (click a glyph)", fontsize=10)
    ax_pal.axis("off")
    current = {"results": []}                                  # [(g_left/top, g_right/bottom, key)]

    def results_for(g, edge):
        if edge == "right":
            scores, mk = C_h[g], lambda p: (chars[g] + chars[p], "horiz")
        elif edge == "left":
            scores, mk = C_h[:, g], lambda p: (chars[p] + chars[g], "horiz")
        elif edge == "bottom":
            scores, mk = C_v[g], lambda p: (chars[g] + chars[p], "vert")
        else:
            scores, mk = C_v[:, g], lambda p: (chars[p] + chars[g], "vert")
        v = scores.clone()
        v[~allowed] = -1e9
        order = torch.argsort(v, descending=True)[:n_res]
        return [(int(p), float(scores[p])) + mk(int(p)) for p in order]

    def draw():
        g, edge = state["g"], state["edge"]
        ax_g.clear()
        ax_g.imshow(bm[g], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        span = {"right": ([W - 0.5] * 2, [-0.5, H - 0.5]), "left": ([-0.5] * 2, [-0.5, H - 0.5]),
                "top": ([-0.5, W - 0.5], [-0.5] * 2), "bottom": ([-0.5, W - 0.5], [H - 0.5] * 2)}[edge]
        ax_g.plot(*span, color="#d62728", lw=4)
        ax_g.set_title(f"'{chars[g]}' -- click an edge\n(selected: {edge})", fontsize=10)
        ax_g.axis("off")
        current["results"] = results_for(g, edge)
        for ax, item in zip(res_axes, current["results"] + [None] * n_res):
            ax.clear()
            ax.axis("off")
            if item is None:
                continue
            p, s, pair, key = item
            if key == "horiz":
                a, b = (state["g"], p) if edge == "right" else (p, state["g"])
                tile = np.concatenate([bm[a], bm[b]], axis=1)
            else:
                a, b = (state["g"], p) if edge == "bottom" else (p, state["g"])
                tile = np.concatenate([bm[a], bm[b]], axis=0)
            ax.imshow(tile, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            banned = ov[key].get(pair) == "ban"
            ax.set_title(f"'{pair}' {s:+.1f}" + (" BANNED" if banned else ""),
                         fontsize=8, color="#d62728" if banned else "black")
            if banned:
                ax.plot([0, tile.shape[1] - 1], [0, tile.shape[0] - 1], color="#d62728", lw=2)
                ax.plot([0, tile.shape[1] - 1], [tile.shape[0] - 1, 0], color="#d62728", lw=2)
        n_ban = len(ov["horiz"]) + len(ov["vert"])
        fig.suptitle(f"tile ports -- click result pairs to ban/unban (auto-saved, {n_ban} bans -> "
                     f"{args.overrides})", fontsize=11)
        fig.canvas.draw_idle()

    def save():
        with open(args.overrides, "w", encoding="utf-8") as f:
            json.dump(ov, f, ensure_ascii=False, indent=1, sort_keys=True)

    def onclick(event):
        if event.inaxes == ax_pal and event.xdata is not None:
            j = int(event.ydata // H) * pal_cols + int(event.xdata // W)
            if 0 <= j < len(vis):
                state["g"] = vis[j]
                draw()
        elif event.inaxes == ax_g and event.xdata is not None:
            dx = event.xdata / (W - 1) - 0.5
            dy = event.ydata / (H - 1) - 0.5
            state["edge"] = (("right" if dx > 0 else "left") if abs(dx) * H > abs(dy) * W
                             else ("bottom" if dy > 0 else "top"))
            draw()
        elif event.inaxes in res_axes:
            i = res_axes.index(event.inaxes)
            if i < len(current["results"]):
                _, _, pair, key = current["results"][i]
                if ov[key].get(pair) == "ban":
                    del ov[key][pair]
                else:
                    ov[key][pair] = "ban"
                save()
                draw()

    fig.canvas.mpl_connect("button_press_event", onclick)
    draw()
    plt.show()


def main():
    p = argparse.ArgumentParser(description="Tile-port preview: which glyph edges align with which")
    p.add_argument("--glyphs", default="─═≈=~|║/\\_¬·╔┌L(▌ε",
                   help="query glyphs to preview (each gets a row of joined-pair renders)")
    p.add_argument("--topn", type=int, default=6, help="partners shown per direction")
    p.add_argument("--band", type=int, default=3, help="border band width (px)")
    p.add_argument("--gamma", type=float, default=0.5, help="mismatch penalty in sim = dot - gamma*||a-b||^2")
    p.add_argument("--orient-weight", type=float, default=1.0, help="orientation channels weight (0 = ink-only ports)")
    p.add_argument("--blur", type=float, default=1.5, help="profile blur along the edge (px)")
    p.add_argument("--ban-letters", action="store_true", help="drop Unicode letters from partner candidates")
    p.add_argument("--tile-chars", default="default",
                   help="restrict PARTNERS to this whitelist (the 'explicitly tiling' vocabulary; "
                        "queries stay unrestricted). 'default' = box+blocks+slashes+line punctuation "
                        "(ON by default); pass 'all' for no restriction")
    p.add_argument("--pairs-top", type=int, default=25, help="also print the globally most compatible pairs")
    p.add_argument("--interactive", action="store_true",
                   help="click glyph -> click edge -> see top partners -> click a pair to ban/unban "
                        "(saved to --overrides; static mode and future consumers honor the bans)")
    p.add_argument("--overrides", default=None,  # None -> <active kit>/port_overrides.json
                   help="hand-curation JSON: {'horiz': {'<gg2>': 'ban'}, 'vert': {...}}")
    p.add_argument("--out", default="./out/tile_ports",
                   help="output prefix -> _pairs.png + .npz")
    args = p.parse_args()
    if args.tile_chars == "default":
        args.tile_chars = DEFAULT_TILE_CHARS
    elif args.tile_chars == "all":
        args.tile_chars = None

    import unicodedata
    device = "cpu"
    ink, chars = G.load_glyphs(device=device, pad=0)             # (N,1,24,12) ink space
    prof = edge_profiles(ink, band=args.band, blur_sigma=args.blur, orient_weight=args.orient_weight)
    C_h, C_v = compat_matrices(prof, gamma=args.gamma)
    N = len(chars)
    allowed = torch.ones(N, dtype=torch.bool)
    if args.ban_letters:
        for i, c in enumerate(chars):
            if unicodedata.category(c).startswith("L"):
                allowed[i] = False
    if args.tile_chars:
        wl = set(args.tile_chars)
        for i, c in enumerate(chars):
            if c not in wl:
                allowed[i] = False
        print(f"tile whitelist: {int(allowed.sum())}/{N} glyphs eligible as partners")
    if args.interactive:
        interactive(args, ink, chars, C_h, C_v, allowed)         # bans NOT pre-applied: shown with X
        return
    if args.overrides is None:
        args.overrides = os.path.join(G.kit_dir(), "port_overrides.json")
    apply_overrides(C_h, C_v, load_overrides(args.overrides), chars)   # static: bans drop from rankings

    def top_partners(Crow, n):
        v = Crow.clone()
        v[~allowed] = -1e9
        idx = torch.argsort(v, descending=True)[:n]
        return [(int(i), float(v[i])) for i in idx]

    queries = [c for c in args.glyphs if c in chars and allowed[chars.index(c)]]
    skipped = [c for c in args.glyphs if c in chars and not allowed[chars.index(c)]]
    if skipped:
        print(f"(queries outside the tile set skipped -- membership is symmetric: {''.join(skipped)})")
    print(f"charset {N} glyphs | band {args.band}px gamma {args.gamma} orient-weight {args.orient_weight}")
    for q in queries:
        qi = chars.index(q)
        r = " ".join(f"'{chars[g]}'{s:+.1f}" for g, s in top_partners(C_h[qi], args.topn))
        l = " ".join(f"'{chars[g]}'{s:+.1f}" for g, s in top_partners(C_h[:, qi], args.topn))
        b = " ".join(f"'{chars[g]}'{s:+.1f}" for g, s in top_partners(C_v[qi], args.topn))
        t = " ".join(f"'{chars[g]}'{s:+.1f}" for g, s in top_partners(C_v[:, qi], args.topn))
        print(f"'{q}'  right-> {r}\n     <-left  {l}\n     below:  {b}\n     above:  {t}")
    iu = torch.triu_indices(N, N, offset=0)
    flat = C_h.clone(); flat[~allowed] = -1e9; flat[:, ~allowed] = -1e9
    vals, idx = flat.flatten().topk(args.pairs_top)
    print("\nglobally most compatible horizontal pairs:")
    print("  " + "  ".join(f"'{chars[i // N]}{chars[i % N]}'{v:+.1f}" for v, i in zip(vals.tolist(), idx.tolist())))

    # render: per query, joined two-cell images of its top partners (left->right, then top->bottom)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bm = 1.0 - ink[:, 0]                                          # white=1 for display
    n_col = 2 * args.topn
    fig, axes = plt.subplots(len(queries), n_col, figsize=(n_col * 1.15, len(queries) * 1.35))
    axes = np.atleast_2d(axes)
    for r_i, q in enumerate(queries):
        qi = chars.index(q)
        for j, (g, s) in enumerate(top_partners(C_h[qi], args.topn)):
            pair = torch.cat([bm[qi], bm[g]], dim=1)              # (24, 24) side by side
            axes[r_i, j].imshow(pair, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            axes[r_i, j].set_title(f"'{q}{chars[g]}' {s:+.1f}", fontsize=7)
        for j, (g, s) in enumerate(top_partners(C_v[qi], args.topn)):
            pair = torch.cat([bm[qi], bm[g]], dim=0)              # (48, 12) stacked
            ax = axes[r_i, args.topn + j]
            ax.imshow(pair, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(f"'{q}'/'{chars[g]}' {s:+.1f}", fontsize=7)
    for a in axes.flat:
        a.axis("off")
    fig.suptitle("left: horizontal partners (query|partner) -- right: vertical partners (query over partner)",
                 fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out + "_pairs.png", dpi=140, bbox_inches="tight")
    np.savez(args.out + ".npz", chars=np.array(list(chars)),
             C_horiz=C_h.numpy().astype(np.float32), C_vert=C_v.numpy().astype(np.float32),
             **{f"prof_{e}": v.numpy().astype(np.float32) for e, v in prof.items()})
    print(f"\nwrote {args.out}_pairs.png + {args.out}.npz (C_horiz, C_vert, profiles)")


if __name__ == "__main__":
    main()
