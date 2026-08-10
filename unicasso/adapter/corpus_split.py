"""Grouped train/holdout split for the adapter-training corpus.

    python -m unicasso.adapter.corpus_split <txt_root> --img-root <images> \
        --groups <corpus>/groups.json --out <corpus>/split.json [--holdout-frac 0.1]

Parents (find_pairs identity: txt stem minus variant suffix) that share the same
ORIGINAL photo — e.g. two different line-art conversions of one
photo — are listed in groups.json and always land on the SAME side of the split, so
the holdout never contains a linework twin of a training image.

Deterministic in (seed, group structure): parents are keyed by sorted group, shuffled
with the seed, and holdout groups are taken from the front until the target parent
count is reached. Re-running after the corpus grows keeps the same holdout groups
drawn first; brand-new parents mostly land in train. RE-RUNNABLE / no files touched
except --out.
"""
import argparse
import json
import os

import numpy as np

from unicasso.adapter.clip_adapt import find_pairs


def load_groups(path):
    """groups.json -> {parent_name: group_id}; keys starting with '_' are notes."""
    with open(path) as f:
        raw = json.load(f)
    owner = {}
    for gname, members in raw.items():
        if gname.startswith("_"):
            continue
        for m in members:
            if m in owner:
                raise SystemExit(f"parent {m!r} appears in groups {owner[m]!r} and {gname!r}")
            owner[m] = gname
    return owner


def main():
    ap = argparse.ArgumentParser(description="grouped corpus split by original photo")
    ap.add_argument("txt_root")
    ap.add_argument("--img-root", required=True)
    ap.add_argument("--groups", default=None, help="groups.json (same-photo parents)")
    ap.add_argument("--holdout-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="split.json path")
    args = ap.parse_args()

    pairs = find_pairs(args.txt_root, args.img_root)
    owner = load_groups(args.groups) if args.groups else {}
    unknown = sorted(set(owner) - {p["name"] for p in pairs})
    if unknown:
        print(f"note: {len(unknown)} grouped parent(s) not (yet) in the corpus: {unknown}")

    groups = {}
    for p in pairs:
        groups.setdefault(owner.get(p["name"], p["name"]), []).append(p)

    gids = sorted(groups)
    order = np.random.default_rng(args.seed).permutation(len(gids))
    target = max(1, round(args.holdout_frac * len(pairs)))
    hold_g, n_hold = [], 0
    for k in order:
        if n_hold >= target:
            break
        hold_g.append(gids[k])
        n_hold += len(groups[gids[k]])

    hold = sorted(p["name"] for g in hold_g for p in groups[g])
    train = sorted(p["name"] for p in pairs if p["name"] not in set(hold))
    n_txt = {s: sum(len(p["txts"]) for p in pairs if p["name"] in set(names))
             for s, names in (("train", train), ("holdout", hold))}
    multi = [g for g in hold_g if len(groups[g]) > 1]
    print(f"{len(pairs)} parents ({len(gids)} photo-groups) -> "
          f"{len(train)} train ({n_txt['train']} txts) / {len(hold)} holdout ({n_txt['holdout']} txts)")
    print(f"holdout groups: {hold_g}" + (f"  [multi-parent: {multi}]" if multi else ""))
    with open(args.out, "w") as f:
        json.dump(dict(train=train, holdout=hold, holdout_groups=hold_g,
                       seed=args.seed, holdout_frac=args.holdout_frac), f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
