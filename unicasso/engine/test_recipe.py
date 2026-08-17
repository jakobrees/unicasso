"""Prove that `asciify image.jpg` with no flags reproduces the canonical recipe.

    python -m unicasso.engine.test_recipe

The recipe used to live in a 60-line shell invocation that had to be pasted
correctly every time. It is now the set of argparse defaults, which is far easier
to use and far easier to break: a one-character edit to a `default=` silently
changes every future render, and nothing else in the repo would notice.

This test reads `canonical_recipe.args`, parses a bare command line for each font
kit, and asserts every value matches -- the font-independent part from the argparse
defaults, and the per-kit part from that kit's profile.json "recipe" block.

Run it after touching any default in asciify.py or any kit profile.
"""
import os
import re
import shlex
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RECIPE = os.path.join(HERE, "canonical_recipe.args")
KITS = ("sfmono", "dejavu")


def parse_args_file(path):
    """{flag: [values]} from a recipe file; '#' comments and blank lines ignored."""
    text = "\n".join(l for l in open(path, encoding="utf-8")
                     if not l.lstrip().startswith("#"))
    toks = shlex.split(text)
    out, i = {}, 0
    while i < len(toks):
        t = toks[i]
        if not t.startswith("--"):
            i += 1
            continue
        vals, j = [], i + 1
        # a leading-minus token that parses as a number is a VALUE (e.g. -0.25)
        while j < len(toks) and (not toks[j].startswith("--") or re.match(r"^-\d", toks[j])):
            vals.append(toks[j])
            j += 1
        out[t] = vals
        i = j
    return out


def profile_recipe(kit):
    import json
    with open(os.path.join(os.path.dirname(os.path.dirname(HERE)),
                           "kits", kit, "profile.json")) as f:
        return json.load(f).get("recipe") or {}


def check(kit, want, prof):
    os.environ["GLYPHVAE_FONT"] = kit
    sys.argv = ["asciify", "image.jpg"]
    from unicasso.engine import asciify as A
    args = A.parse_args()

    bad = []
    for flag, vals in want.items():
        dest = flag[2:].replace("-", "_")
        got = getattr(args, dest, "<MISSING>")
        if not vals:                                   # boolean flag: recipe passes it
            exp, ok = True, got is True
        elif len(vals) > 1:                            # nargs list
            exp = [float(v) for v in vals]
            ok = isinstance(got, (list, tuple)) and [float(g) for g in got] == exp
        else:
            v = vals[0]
            try:
                exp = float(v)
                ok = got is not None and abs(float(got) - exp) < 1e-12
            except ValueError:
                exp, ok = v, str(got) == v
        if not ok:
            bad.append((flag, exp, got))

    # Path-valued recipe keys name repo-owned files, so asciify resolves them against
    # the repo root (the cwd is no longer chdir'd there); compare like for like.
    from unicasso.substrate.glyphs import repo_path
    PATH_KEYS = {"vae_ckpt", "clip_adapter", "init_vae_ckpt"}

    for dest, exp in prof.items():                     # per-kit block
        got = getattr(args, dest, "<MISSING>")
        if dest in PATH_KEYS and isinstance(exp, str):
            exp = repo_path(exp)
        ok = (abs(float(got) - float(exp)) < 1e-12
              if isinstance(exp, (int, float)) and not isinstance(exp, bool)
              else got == exp)
        if not ok:
            bad.append(("--" + dest.replace("_", "-") + " (profile)", exp, got))

    n = len(want) + len(prof)
    print(f"{kit:>8}: {n - len(bad)}/{n} values match the canonical recipe with no CLI flags")
    for f, e, g in bad:
        print(f"          MISMATCH {f:<34} want {e!r:<20} got {g!r}")
    return not bad


COLOUR_MODE = {
    # `--color` alone must give the canonical colour recipe.
    # --color-fit is the load-bearing one: probe_color = color and color_fit, so without
    # it live probing is silently disabled under --color and the run has no measured
    # tier at all (admissions on prediction, no evidence-based tabu, lottery births 0).
    # --color-contrast-learn without --color-contrast-tv gives a salt-and-pepper k field
    # (measured to drive whole regions flat), so those two move together.
    "color_fit": True,
    "color_contrast_learn": True,
    "color_contrast_tv": 0.05,
    "color_contrast_iters": 500,
    "color_div_weight": 1e-3,
    "recolor_min_contrast": 0.12,
}


def check_colour():
    from unicasso.engine import asciify as A
    sys.argv = ["asciify", "image.jpg", "--color"]
    args = A.parse_args()
    bad = [(k, v, getattr(args, k, "<MISSING>")) for k, v in COLOUR_MODE.items()
           if getattr(args, k, None) != v]
    n = len(COLOUR_MODE)
    print(f"  colour: {n - len(bad)}/{n} settings match the canonical colour recipe with just --color")
    for k, w, g in bad:
        print(f"          MISMATCH {k:<28} want {w!r:<10} got {g!r}")
    return not bad


def main():
    want = parse_args_file(RECIPE)
    ok = all(check(kit, want, profile_recipe(kit)) for kit in KITS)
    ok &= check_colour()
    # The adapter must stay opt-in -- mainly for speed, and because it changes what
    # CLIP rewards, so it should never switch on by accident. Not a quality verdict:
    # on sfmono it does not hurt output and it suppresses line over-extension.
    os.environ["GLYPHVAE_FONT"] = KITS[0]
    sys.argv = ["asciify", "image.jpg"]
    from unicasso.engine import asciify as A
    if A.parse_args().clip_adapter is not None:
        print("  MISMATCH --clip-adapter should default to None (adapters opt-in)")
        ok = False
    print("RECIPE OK" if ok else "RECIPE DRIFT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
