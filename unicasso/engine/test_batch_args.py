"""Prove the batch renders the CANONICAL RECIPE plus a named, deliberate short list.

    python -m unicasso.engine.test_batch_args

The corpus is a public artifact: it should document the configuration users actually get,
so `batch_asciify.BASE_ARGS` must equal the asciify defaults (the canonical recipe, see
canonical_recipe.args) and the active kit profile, departing only where a corpus-scale
concern has no meaning for a single render.

This gate parses `defaults + BASE_ARGS + variant` and compares every resulting value with
`defaults + ADDITIONS + variant`. Anything that differs is a departure nobody declared --
which is exactly how the batch drifted before: it once carried a second, hand-maintained
copy of the whole recipe, and silently kept `--pool-probe-rate 0` / `--pool-probe-per-chan 1`
after the canonical recipe raised them to 1 / 2. That turned continuous, measured proposals
off for an entire corpus without anything reporting it.

To add a batch-only flag, add it to BOTH lists in the same commit. If it does not belong in
ADDITIONS, it does not belong in the batch.
"""
import sys

# The batch's deliberate additions -- and nothing else.
ADDITIONS = [
    # coarser slot-trajectory sampling: ~26 MB/run at the default 25 is not worth it
    # across a whole corpus
    "--pool-record-interval", "100",
]
# --vae-ckpt is resolved per environment; compared separately.
IGNORE = {"vae_ckpt", "input_image", "output", "output_text", "pool_record",
          "anim", "loss_curve", "clip_diagnostic"}

# Settings the corpus depends on being exactly these. Checked by NAME as well as by the
# diff above, so a change to the canonical recipe itself cannot silently alter the corpus.
PINNED = {
    "iters": 3500,
    "mode": "swarm",
    "pool_probe": True,          # measured verification of nominations
    "pool_probe_rate": 1,        # CONTINUOUS proposals: probe every iteration (memoised)
    "pool_probe_memo_ttl": 200,
    "pool_probe_per_chan": 2,
    "pool_probe_batches": 4,
    "pool_probe_spacing": 9,
    "swarm_w_temp_cycles": 2,
    "pool_nominate_end_frac": 0.85,
    "clip_adapter": None,        # no domain adapter
    "clip_steer": None,          # no steering
    "early_stop_patience": 6,    # early stop ON (now a canonical default, inherited)
    "early_stop_frac": 0.86,     # arms just past nomination-end (0.85 x iters)
}


def parse(extra):
    from unicasso.engine import asciify as A
    sys.argv = ["asciify", "image.jpg", "--vae-ckpt", "x.pt"] + [str(a) for a in extra]
    return A.parse_args()


def main():
    from unicasso.engine import batch_asciify as B
    ok = True
    for label, variant in B.VARIANTS:
        got = parse([*B.BASE_ARGS, *variant])
        want = parse([*ADDITIONS, *variant])
        bad = [(k, getattr(want, k), getattr(got, k))
               for k in vars(want) if k not in IGNORE and getattr(want, k) != getattr(got, k)]
        n = len([k for k in vars(want) if k not in IGNORE])
        print(f"  variant {label}: {n - len(bad)}/{n} settings = canonical recipe + additions")
        for k, w, g in bad:
            print(f"      UNDECLARED DEPARTURE {k:<28} canonical {w!r} -> batch {g!r}")
        for k, v in sorted(PINNED.items()):
            if getattr(got, k) != v:
                print(f"      PIN BROKEN {k:<34} expected {v!r}, got {getattr(got, k)!r}")
                bad.append(k)
        ok &= not bad
    print("BATCH ARGS OK" if ok else "BATCH ARGS DRIFT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
