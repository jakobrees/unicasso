"""Click-through viewer for adaptation corruptions: original vs corrupted, zoomed.

    GLYPHVAE_FONT=sfmono python -m unicasso.adapter.corrupt_viewer path/to/output.txt
    GLYPHVAE_FONT=sfmono python -m unicasso.adapter.corrupt_viewer <folder-of-txts> --families fade,soft
        (--families takes any comma subset of the families listed in --help)

Panels: original render | corrupted render (changed cells framed red) | zoom on the
corrupted region (original above, corrupted below). 'soft' shows the POSITIVE
augmentation (green title, no mask) so blend quality can be judged in the same tool.

Keys:  →/space/click  next     ←  prev     r  regenerate current
       f  cycle family filter (all -> each active family in turn)     q  quit
"""
import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

from unicasso.adapter.corrupt import CorruptionSampler, FAMILIES, FIELD_FAMILIES, RANK_FAMILIES

from unicasso.substrate.glyphs import repo_path
VAE_DEFAULT = repo_path("weights/vae_sfmono/model.pt")


def collect_txts(paths):
    txts = []
    for p in paths:
        if os.path.isdir(p):
            for dp, _, fns in os.walk(p):
                txts += [os.path.join(dp, f) for f in sorted(fns) if f.endswith(".txt")]
        elif p.endswith(".txt"):
            txts.append(p)
    return txts


class Viewer:
    def __init__(self, sampler, txts, families, img_root=None):
        self.s = sampler
        self.txts = txts
        self.families = families            # active generation set (list)
        self.img_root = img_root
        self._data = None                   # lazy clip_adapt.Data (decorate w/ targets)
        self.filter_cycle = ["all"] + families
        self.filter_i = 0
        self.grids = {}                     # txt path -> index grid
        self.origs = {}                     # txt path -> rendered original (np)
        self.samples = []                   # generated history
        self.i = -1

        # our keys collide with mpl defaults (f=fullscreen, r=home, arrows=back/forward)
        for km in ("keymap.back", "keymap.forward", "keymap.home",
                   "keymap.fullscreen", "keymap.save"):
            if km in plt.rcParams:
                plt.rcParams[km] = []
        self.fig, self.axes = plt.subplots(1, 3, figsize=(16, 7))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.next()

    # ------------------------------------------------------------------ sample generation
    def _grid(self, path):
        if path not in self.grids:
            self.grids[path] = self.s.load_txt(path)
            self.origs[path] = self.s.render(self.grids[path]).cpu().numpy()
        return self.grids[path]

    def _active(self):
        f = self.filter_cycle[self.filter_i]
        return self.families if f == "all" else [f]

    def make(self):
        rng = self.s.rng
        path = self.txts[int(rng.integers(len(self.txts)))]
        grid = self._grid(path)
        fam = self._active()[int(rng.integers(len(self._active())))]
        if fam == "soft":
            img, desc = self.s.soften(grid)
            return dict(path=path, family="soft", desc=desc,
                        corr=img.cpu().numpy(), mask=None)
        if fam in RANK_FAMILIES:            # ranked dose pair: show BOTH members
            pr = self.s.noise_pair(grid, fam)
            if pr is None:                  # erode on a junction-poor grid
                return dict(path=path, family=fam, mask=None,
                            corr=self.s.render(grid).cpu().numpy(),
                            desc="too few break sites in this grid -- skipped")
            return dict(path=path, family=fam, desc=pr["desc"],
                        corr=self.s.render(pr["grid_hi"]).cpu().numpy(),
                        corr_lo=self.s.render(pr["grid_lo"]).cpu().numpy(),
                        mask=pr["mask_hi"].cpu().numpy())
        if fam == "decorate":               # trainer augmentation (clip_adapt.Data)
            return self._make_decorate(path, grid)
        out = self.s.corrupt(grid, family=fam)
        return dict(path=path, family=out["family"], desc=out["desc"],
                    corr=self.s.render(out["grid"]).cpu().numpy(),
                    mask=out["mask"].cpu().numpy())

    def _find_parent(self, path):
        if not self.img_root:
            return None
        from unicasso.adapter.clip_adapt import VARIANT_RE, EXTS
        base = VARIANT_RE.sub("", os.path.splitext(os.path.basename(path))[0])
        for suffix in ("", "_line"):
            for ext in EXTS:
                p = os.path.join(self.img_root, base + suffix + ext)
                if os.path.exists(p):
                    return p
        return None

    def _make_decorate(self, path, grid):
        """The trainer's --decorate-frac augmentation, made visible. With a matching
        parent image (--img-root): the REAL thing -- Data._decorate stamps target +
        render, zoom shows decorated target over decorated render. Without: patch is
        stamped into the grid's own all-space regions (morphology check only)."""
        img = self._find_parent(path)
        if img is not None:
            from unicasso.adapter.clip_adapt import Data
            if self._data is None:
                self._data = Data(self.s, "cpu")
            dims = (grid.shape[0] * self.s.CH, grid.shape[1] * self.s.CW)
            parent = self._data.parent(img, dims)
            g2, p2 = self._data._decorate(grid, parent, self.s.rng)
            if g2 is None:
                return dict(path=path, family="decorate", corr=self.s.render(grid).cpu().numpy(),
                            mask=None, desc="no blank window fit -- item trains undecorated")
            return dict(path=path, family="decorate",
                        desc=f"{int((g2 != grid).sum())} cells stamped (target + render)",
                        corr=self.s.render(g2).cpu().numpy(),
                        corr_lo=p2.cpu().numpy(),
                        zoom_labels="zoom: original / decorated TARGET / decorated render",
                        mask=(g2 != grid).cpu().numpy())
        from unicasso.adapter.synth_identity import make_patch          # no parent: grid-blank fallback
        import torch.nn.functional as tF
        GH, GW = grid.shape
        rng = self.s.rng
        gh = int(rng.integers(5, max(6, min(12, GH - 2))))
        gw = int(rng.integers(8, max(9, min(24, GW - 2))))
        blank = (grid == self.s.space).float()
        ok = tF.avg_pool2d(blank[None, None], (gh, gw), stride=1)[0, 0] >= 1.0 - 1e-6
        pos = torch.nonzero(ok)
        if pos.shape[0] == 0:
            return dict(path=path, family="decorate", corr=self.s.render(grid).cpu().numpy(),
                        mask=None, desc="no all-space window in grid (try --img-root)")
        y0, x0 = (int(v) for v in pos[int(rng.integers(pos.shape[0]))])
        patch = make_patch(rng, gh, gw, self.s)
        g2 = grid.clone()
        reg = g2[y0:y0 + gh, x0:x0 + gw]
        put = patch != self.s.space
        reg[put] = patch[put]
        return dict(path=path, family="decorate",
                    desc=f"{int(put.sum())} cells stamped (grid-blank fallback; "
                         "pass --img-root for the target side)",
                    corr=self.s.render(g2).cpu().numpy(),
                    mask=(g2 != grid).cpu().numpy())

    # ------------------------------------------------------------------ drawing
    def draw(self):
        smp = self.samples[self.i]
        orig = self.origs[smp["path"]]
        CH, CW = self.s.CH, self.s.CW
        for ax in self.axes:
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])

        self.axes[0].imshow(orig, cmap="gray", vmin=0, vmax=1)
        self.axes[0].set_title("original")
        self.axes[1].imshow(smp["corr"], cmap="gray", vmin=0, vmax=1)
        self.axes[1].set_title("corrupted" if smp["family"] != "soft" else "softened")

        if smp["mask"] is not None and smp["mask"].any():
            ys, xs = np.nonzero(smp["mask"])
            for y, x in zip(ys, xs):
                self.axes[1].add_patch(Rectangle((x * CW - 0.5, y * CH - 0.5), CW, CH,
                                                 fill=False, edgecolor="red", lw=0.8))
            # zoom: mask bbox + 2-cell margin, original stacked over corrupted
            GH, GW = smp["mask"].shape
            y0, y1 = max(ys.min() - 2, 0), min(ys.max() + 3, GH)
            x0, x1 = max(xs.min() - 2, 0), min(xs.max() + 3, GW)
        else:                               # soft: zoom on an inky center region
            GH = orig.shape[0] // CH
            GW = orig.shape[1] // CW
            y0, y1 = GH // 3, min(GH // 3 + 8, GH)
            x0, x1 = GW // 3, min(GW // 3 + 12, GW)
        oc = orig[y0 * CH:y1 * CH, x0 * CW:x1 * CW]
        cc = smp["corr"][y0 * CH:y1 * CH, x0 * CW:x1 * CW]
        div = np.full((3, oc.shape[1]), 0.5)
        if smp.get("corr_lo") is not None:  # dose pair: original / t_lo / t_hi stack
            lc = smp["corr_lo"][y0 * CH:y1 * CH, x0 * CW:x1 * CW]
            self.axes[2].imshow(np.concatenate([oc, div, lc, div, cc], axis=0),
                                cmap="gray", vmin=0, vmax=1)
            self.axes[2].set_title(smp.get("zoom_labels", "zoom: original / t_lo / t_hi"))
        else:
            self.axes[2].imshow(np.concatenate([oc, div, cc], axis=0),
                                cmap="gray", vmin=0, vmax=1)
            self.axes[2].set_title("zoom: original / corrupted")
        if smp["mask"] is not None and smp["mask"].any() and smp.get("corr_lo") is None:
            for y, x in zip(ys, xs):        # frame changed cells in the zoom's lower half too
                self.axes[2].add_patch(Rectangle(((x - x0) * CW - 0.5,
                                                  oc.shape[0] + 3 + (y - y0) * CH - 0.5),
                                                 CW, CH, fill=False, edgecolor="red", lw=0.8))

        kind = ("POSITIVE (augmentation)" if smp["family"] in ("soft", "decorate")
                else "NEGATIVE")
        self.fig.suptitle(
            f"[{self.i + 1}/{len(self.samples)}]  {smp['family']}  —  {smp['desc']}   "
            f"({kind})   filter: {self.filter_cycle[self.filter_i]}\n"
            f"{os.path.basename(smp['path'])}",
            color="green" if smp["family"] in ("soft", "decorate") else "black", fontsize=11)
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------ interaction
    def next(self):
        if self.i + 1 >= len(self.samples):
            self.samples.append(self.make())
        self.i += 1
        self.draw()

    def prev(self):
        if self.i > 0:
            self.i -= 1
            self.draw()

    def on_key(self, ev):
        if ev.key in ("right", " "):
            self.next()
        elif ev.key == "left":
            self.prev()
        elif ev.key == "r":
            self.samples[self.i] = self.make()
            self.draw()
        elif ev.key == "f":
            self.filter_i = (self.filter_i + 1) % len(self.filter_cycle)
            self.samples[self.i + 1:] = []   # regenerate queue under the new filter
            self.next() if self.i + 1 <= len(self.samples) else self.draw()
        elif ev.key == "q":
            plt.close(self.fig)

    def on_click(self, ev):
        if ev.button == 1 and ev.inaxes is not None:
            self.next()


def main():
    ap = argparse.ArgumentParser(description="click-through corruption viewer")
    ap.add_argument("paths", nargs="+", help=".txt files and/or folders (recursed)")
    ap.add_argument("--vae-ckpt", default=VAE_DEFAULT)
    ap.add_argument("--families",
                    default=",".join(FAMILIES + RANK_FAMILIES) + ",soft,decorate",
                    help=f"comma list from {FAMILIES + RANK_FAMILIES + ('soft', 'decorate')}")
    ap.add_argument("--img-root", default=None,
                    help="parent linework dir (find_pairs stem matching); gives the "
                         "'decorate' family its target side (without it: grid-blank "
                         "morphology check only)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    txts = collect_txts(args.paths)
    if not txts:
        raise SystemExit(f"no .txt files found under {args.paths}")
    fams = [f.strip() for f in args.families.split(",") if f.strip()]
    bad = [f for f in fams if f not in FAMILIES + RANK_FAMILIES + ("soft", "decorate")]
    if bad:
        raise SystemExit(f"unknown families: {bad}")
    print(f"{len(txts)} grid(s); families: {fams}")

    sampler = CorruptionSampler(args.vae_ckpt, device="cpu", seed=args.seed)
    viewer = Viewer(sampler, txts, fams, img_root=args.img_root)   # keep ref: weak mpl callbacks
    plt.show()
    del viewer


if __name__ == "__main__":
    main()
