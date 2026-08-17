"""CLIP steering vectors: a fixed direction added to the TARGET embedding so the render
is asked for "the line art, but ascii-flavoured" instead of "the line art".

The deployed perceptual loss (engine/clip_loss.py) scores each crop with

    sem = 1 - cos(e_render, e_target)                       [weight --clip-semantic-weight]
    geo = sum_l w_l * ||F_l(render) - F_l(target)||^2       [weight 1]

`sem` lives in RN101's JOINT image/text space (attnpool output, 512-d, the same space
`encode_text` writes into), so it is the one term a steering vector can act on -- the
geometric conv maps and the dense term have no shared space with text. Steering replaces
the target direction with

    ê_t* = normalize( normalize(e_target) + λ · Δ̂ )

leaving the loss a cosine in [0, 2]: the SCALE of the objective is untouched, only the
DIRECTION of the semantic pull rotates. λ (--clip-steer-weight) is the only knob, and
because Δ̂ is a unit vector, λ is interpretable -- λ=1 puts the steered target halfway
between the true target and the pure ascii direction.

Two ways to obtain Δ̂, both built here:

  text   Prompt ensembles through the text tower: mean("ASCII art ...") minus
         mean("a line drawing ..."), each side L2-normalized before averaging. Costs
         nothing to compute and needs no data, but it is CLIP's *linguistic* idea of
         the gap, which may not be the gap the renders actually exhibit.

  data   The gap as MEASURED on the adapter corpus: for each (ascii .txt render, line-art
         parent) pair, sample crops from the DEPLOY crop distribution (log-uniform area
         0.4-0.9 + ratio jitter, ink-gated), encode both sides, and average
         normalize(e_ascii) - normalize(e_line). This is the empirical mean domain shift
         in exactly the regime the loss operates in.

A third direction is IMPLIED by the LoRA/FiLM adapter (engine/clip_adapter.py) and is
recovered here for comparison: mean[ normalize(e_adapted(ascii)) - normalize(e_base(ascii)) ],
i.e. how far the adapter actually moves an ascii render's embedding. Note the SIGN
CONVENTION: `text`/`data` point line -> ascii and are added to the TARGET, while the
adapter moves the RENDER toward the line manifold. The two strategies close the same gap
from opposite ends, so an adapter that agrees with the measured gap scores a NEGATIVE
cosine against it. `compare` prints the matrix with this spelled out.

    python -m unicasso.engine.steer text --out weights/steer/text.pt
    GLYPHVAE_FONT=sfmono python -m unicasso.engine.steer data \
        --txt-root <corpus>/txts --img-root <corpus>/images \
        --out weights/steer/data.pt --adapter weights/clip_adapter/adapters_step500.pt
    python -m unicasso.engine.steer compare weights/steer/*.pt
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import open_clip
from open_clip import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

from unicasso.substrate.glyphs import repo_path

# Prompt ensembles. Averaged over the ensemble (each member normalized first) so the
# direction reflects the CONCEPT rather than one phrasing's idiosyncrasies -- the same
# reason CLIP zero-shot classification ensembles its templates.
POS_PROMPTS = [
    "ASCII art",
    "a picture drawn with ASCII characters",
    "an image made of text characters",
    "text art made of letters and symbols",
    "a monospaced character rendering of a picture",
]
NEG_PROMPTS = [
    "a line drawing",
    "a black and white line drawing",
    "a pen and ink line drawing",
    "a contour line illustration",
    "an outline sketch",
]


def _device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Encoder:
    """Frozen CLIP tower for both modalities. Adapters, when loaded, start DISABLED so
    `encode_image` is the bit-exact base path; `adapted()` toggles them on."""

    def __init__(self, device, model_name="RN101", pretrained="openai", adapter=None):
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model, self.visual, self.device = model, model.visual, device
        self.model_name, self.pretrained = model_name, pretrained
        self.mean = torch.tensor(OPENAI_DATASET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(OPENAI_DATASET_STD, device=device).view(1, 3, 1, 1)
        self.adapters = None
        if adapter:
            from unicasso.engine.clip_adapter import load_adapters, set_enabled
            self.adapters, extra = load_adapters(self.visual, adapter, device)
            for m in self.adapters.values():
                m.to(device)
                for p in m.parameters():
                    p.requires_grad_(False)
            set_enabled(self.adapters, False)            # base path by default
            print(f"adapter: {adapter} ({len(self.adapters)} modules"
                  + (f", step {extra['step']}" if extra.get("step") else "") + ")")

    def set_adapted(self, flag):
        from unicasso.engine.clip_adapter import set_enabled
        set_enabled(self.adapters, flag)

    @torch.no_grad()
    def encode_image(self, batch1):
        """(B,1,res,res) in [0,1], white=1 -> (B,D) joint-space embeddings."""
        x = (batch1.to(self.device).expand(-1, 3, -1, -1) - self.mean) / self.std
        return self.visual(x).float()

    @torch.no_grad()
    def encode_text(self, prompts):
        tok = open_clip.get_tokenizer(self.model_name)
        return self.model.encode_text(tok(prompts).to(self.device)).float()


def save_delta(path, delta, kind, enc, meta):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    torch.save(dict(delta=F.normalize(delta.detach().cpu().float(), dim=-1),
                    kind=kind, model=enc.model_name, pretrained=enc.pretrained,
                    meta=meta), path)
    print(f"wrote {path}  (kind={kind}, dim={delta.numel()})")


def load_delta(path, device="cpu"):
    """-> (unit delta (D,), record dict). Used by clip_loss at render time."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    d = F.normalize(ck["delta"].float(), dim=-1).to(device)
    return d, ck


# ------------------------------------------------------------------ text direction
def build_text(args):
    enc = Encoder(_device(), args.model, args.pretrained)
    pos, neg = args.pos or POS_PROMPTS, args.neg or NEG_PROMPTS
    ep = F.normalize(enc.encode_text(pos), dim=-1).mean(0)
    en = F.normalize(enc.encode_text(neg), dim=-1).mean(0)
    delta = F.normalize(ep - en, dim=-1)
    gap = float(F.cosine_similarity(F.normalize(ep, dim=-1), F.normalize(en, dim=-1), dim=0))
    print(f"text: {len(pos)} positive / {len(neg)} negative prompts")
    print(f"  cos(ascii-mean, line-mean) = {gap:+.4f}   (1 = no separation to steer along)")
    save_delta(args.out, delta, "text", enc,
               dict(pos=pos, neg=neg, prompt_cos=gap,
                    mean_pos=F.normalize(ep, dim=-1).cpu(),
                    mean_neg=F.normalize(en, dim=-1).cpu()))


# ------------------------------------------------------------------ data direction
def build_data(args):
    from unicasso.adapter.clip_adapt import find_pairs, inky_window, crop224
    from unicasso.adapter.corrupt import CorruptionSampler

    device = _device()
    enc = Encoder(device, args.model, args.pretrained, adapter=args.adapter)
    sampler = CorruptionSampler(repo_path(args.vae_ckpt), device="cpu")
    pairs = find_pairs(args.txt_root, args.img_root)
    if args.split:
        # Same train/holdout boundary the adapter was trained under (the two split.json files
        # are byte-identical), so a data-steered run and an adapter run are fit on exactly the
        # same parents and can be judged on exactly the same held-out images.
        import json
        sp = json.load(open(args.split, encoding="utf-8"))
        # EXCLUDE the holdout rather than intersect with train -- clip_adapt does the same
        # ("parents in the corpus but in neither list join train, so the split survives as the
        # corpus grows"). Intersecting instead would silently drop every parent added since the
        # split was written, and the vector would be fit on less data than the adapter was.
        hold = set(sp.get("holdout", []))
        before = len(pairs)
        pairs = [p for p in pairs if p["name"] not in hold]
        print(f"split {args.split}: {len(pairs)}/{before} parents kept "
              f"({before - len(pairs)} of {len(hold)} holdout names matched)")
    if args.limit:
        pairs = pairs[:args.limit]
    n_txt = sum(len(p["txts"]) for p in pairs)
    print(f"{len(pairs)} parents / {n_txt} ascii renders x {args.crops} crops "
          f"= {n_txt * args.crops} crop pairs | device {device}")

    rng = np.random.default_rng(args.seed)
    scale = tuple(args.scale)
    # running sums over crops (all in the joint space, each embedding unit-normalized
    # BEFORE differencing: the loss only ever sees direction, and unnormalized magnitude
    # differences between the two domains would otherwise dominate the mean)
    acc = dict(delta=torch.zeros(512, device=device), a=torch.zeros(512, device=device),
               l=torch.zeros(512, device=device), adapt=torch.zeros(512, device=device))
    cos_al, cos_adapt, n = [], [], 0
    len_d, len_adapt = [], []          # per-crop ||e_a - e_l||, for the coherence ratio below

    buf_a, buf_l = [], []

    def flush():
        nonlocal buf_a, buf_l, n
        if not buf_a:
            return
        A = torch.stack(buf_a)[:, None]                  # (B,1,224,224)
        L = torch.stack(buf_l)[:, None]
        ea = F.normalize(enc.encode_image(A), dim=-1)
        el = F.normalize(enc.encode_image(L), dim=-1)
        acc["delta"] += (ea - el).sum(0)
        acc["a"] += ea.sum(0)
        acc["l"] += el.sum(0)
        cos_al.append((ea * el).sum(-1).cpu())
        len_d.append((ea - el).norm(dim=-1).cpu())
        if enc.adapters is not None:
            enc.set_adapted(True)
            ea2 = F.normalize(enc.encode_image(A), dim=-1)
            enc.set_adapted(False)
            acc["adapt"] += (ea2 - ea).sum(0)
            cos_adapt.append((ea2 * ea).sum(-1).cpu())
            len_adapt.append((ea2 - ea).norm(dim=-1).cpu())
        n += A.shape[0]
        buf_a, buf_l = [], []

    for pair in tqdm(pairs, desc="pairs"):
        for txt in pair["txts"]:
            grid = sampler.load_txt(txt)
            ascii_img = sampler.render(grid)                       # (H,W) white=1
            H, W = ascii_img.shape
            parent = torch.from_numpy(
                np.asarray(Image.open(pair["img"]).convert("L").resize((W, H), Image.BILINEAR),
                           dtype=np.float32) / 255.0)
            for _ in range(args.crops):
                # ink-gated window: a blank-vs-blank crop pair carries no domain signal and
                # would drag the mean toward "whiteness" rather than toward "ascii-ness"
                p = inky_window(rng, parent, scale=scale)
                buf_a.append(crop224(ascii_img, p))
                buf_l.append(crop224(parent, p))
                if len(buf_a) >= args.batch:
                    flush()
    flush()

    cos_al = torch.cat(cos_al) if cos_al else torch.zeros(1)
    print(f"\n{n} crop pairs encoded")
    print(f"  cos(ascii, line) over crops = {cos_al.mean():+.4f} "
          f"+/- {cos_al.std():.4f}   <- the domain gap being closed")
    # COHERENCE: is the gap one direction, or scatter? ||mean of the per-crop differences||
    # divided by the mean of their lengths. 1.0 = every pair shifts exactly the same way (a
    # single vector captures the gap perfectly); ~1/sqrt(n) = the differences are random
    # directions and no fixed vector can represent them. This ratio is the entire premise of
    # linear steering, so it is worth printing before anyone trusts the vector.
    mean_len = float(torch.cat(len_d).mean())
    coh = float((acc["delta"] / max(n, 1)).norm()) / max(mean_len, 1e-9)
    print(f"  coherence = {coh:.4f}   (1 = one shared direction; "
          f"{1 / max(n, 1) ** 0.5:.4f} = random scatter at this n)")
    delta = F.normalize(acc["delta"] / max(n, 1), dim=-1)
    mean_a = F.normalize(acc["a"] / max(n, 1), dim=-1)
    mean_l = F.normalize(acc["l"] / max(n, 1), dim=-1)
    # How much of the direction SURVIVES the re-normalization in the loss: the component
    # of Δ̂ parallel to the target only rescales ê_t and cancels, so the orthogonal part is
    # the entire steering effect.
    par = float((delta * mean_l).sum())
    print(f"  cos(delta, mean line embedding) = {par:+.4f} "
          f"-> {100 * (1 - par ** 2) ** 0.5:.1f}% of the direction is orthogonal (= effective)")
    meta = dict(n_crops=n, n_pairs=len(pairs), n_txts=n_txt, scale=list(scale),
                crops_per_txt=args.crops, seed=args.seed, split=args.split,
                txt_root=args.txt_root, img_root=args.img_root,
                domain_cos=float(cos_al.mean()), domain_cos_std=float(cos_al.std()),
                coherence=coh, delta_dot_target=par,
                mean_ascii=mean_a.cpu(), mean_line=mean_l.cpu())
    save_delta(args.out, delta, "data", enc, meta)

    if enc.adapters is not None:
        ca = torch.cat(cos_adapt)
        d_ad = F.normalize(acc["adapt"] / max(n, 1), dim=-1)
        coh_a = float((acc["adapt"] / max(n, 1)).norm()) / max(float(torch.cat(len_adapt).mean()), 1e-9)
        print(f"\nadapter-implied direction:")
        print(f"  cos(adapted, base) over ascii crops = {ca.mean():+.4f} "
              f"+/- {ca.std():.4f}   <- how far the adapter moves a render")
        print(f"  coherence = {coh_a:.4f}   <- how much of that move is ONE fixed translation "
              f"(the rest is input-dependent)")
        stem = os.path.splitext(args.out)[0]
        save_delta(stem + "_adapter.pt", d_ad, "adapter", enc,
                   dict(n_crops=n, adapter=args.adapter, move_cos=float(ca.mean()),
                        coherence=coh_a,
                        note="points render -> line manifold; OPPOSITE sign to text/data "
                             "deltas, which point line -> ascii and are added to the target"))


# ------------------------------------------------------------------ compare
def compare(args):
    recs = []
    for p in args.paths:
        d, ck = load_delta(p)
        recs.append((os.path.basename(p), ck.get("kind", "?"), d, ck))
    print(f"\n{len(recs)} steering vector(s), dim {recs[0][2].numel()}\n")
    for name, kind, _, ck in recs:
        m = ck.get("meta", {})
        bits = [f"kind={kind}"]
        for k in ("n_crops", "domain_cos", "prompt_cos", "move_cos", "delta_dot_target"):
            if k in m:
                bits.append(f"{k}={m[k]:.4f}" if isinstance(m[k], float) else f"{k}={m[k]}")
        print(f"  {name:<28} {'  '.join(bits)}")

    w = max(len(n) for n, _, _, _ in recs)
    print("\ncosine matrix\n")
    print(" " * (w + 2) + "  ".join(f"{n[:10]:>10}" for n, _, _, _ in recs))
    for name, _, d, _ in recs:
        row = "  ".join(f"{float(d @ e):>10.4f}" for _, _, e, _ in recs)
        print(f"  {name:<{w}}" + row)

    kinds = {k: d for _, k, d, _ in recs}
    print("\nreading it:")
    if "text" in kinds and "data" in kinds:
        c = float(kinds["text"] @ kinds["data"])
        print(f"  text vs data      {c:+.4f}  -- both point line->ascii, so POSITIVE = CLIP's "
              f"linguistic idea of\n{' ' * 36}the gap matches the measured one. "
              f"{'agree' if c > 0.2 else 'weak agreement' if c > 0.05 else 'essentially unrelated'}.")
    for k in ("text", "data"):
        if k in kinds and "adapter" in kinds:
            c = float(kinds[k] @ kinds["adapter"])
            print(f"  {k} vs adapter   {c:+.4f}  -- opposite sign conventions, so NEGATIVE = same "
                  f"axis.\n{' ' * 36}"
                  f"{'the adapter learned this gap' if c < -0.2 else 'only loosely related' if c < -0.05 else 'the adapter learned something else'}.")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    common = dict(model="RN101", pretrained="openai")
    t = sub.add_parser("text", help="steering vector from text-prompt ensembles")
    t.add_argument("--out", required=True)
    t.add_argument("--pos", nargs="+", default=None, help=f"override; default: {POS_PROMPTS}")
    t.add_argument("--neg", nargs="+", default=None, help=f"override; default: {NEG_PROMPTS}")
    t.add_argument("--model", default=common["model"])
    t.add_argument("--pretrained", default=common["pretrained"])
    t.set_defaults(fn=build_text)

    d = sub.add_parser("data", help="steering vector measured on the ascii/line corpus")
    d.add_argument("--txt-root", required=True, help="corpus .txt renders")
    d.add_argument("--img-root", required=True, help="corpus line-art parents")
    d.add_argument("--out", required=True)
    d.add_argument("--vae-ckpt", default="weights/vae_sfmono/model.pt",
                   help="only used to build the glyph rasteriser for .txt -> image")
    d.add_argument("--adapter", default=None,
                   help="also recover this adapter's IMPLIED direction -> <out>_adapter.pt")
    d.add_argument("--split", default=None, metavar="SPLIT.JSON",
                   help="restrict to the split's TRAIN parents (e.g. data/corpus/split.json), so "
                        "the vector never sees the images it will be judged on")
    d.add_argument("--crops", type=int, default=8, help="crops sampled per ascii render")
    d.add_argument("--scale", type=float, nargs=2, default=[0.4, 0.9],
                   help="crop area fraction; match the deploy --clip-crop-scale")
    d.add_argument("--batch", type=int, default=32)
    d.add_argument("--limit", type=int, default=0, help="first N parents only (smoke test)")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--model", default=common["model"])
    d.add_argument("--pretrained", default=common["pretrained"])
    d.set_defaults(fn=build_data)

    c = sub.add_parser("compare", help="cosine matrix across saved steering vectors")
    c.add_argument("paths", nargs="+")
    c.set_defaults(fn=compare)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
