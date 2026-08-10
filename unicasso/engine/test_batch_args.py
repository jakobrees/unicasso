"""Prove the trimmed BASE_ARGS still produces exactly the old batch configuration.

    python -m unicasso.engine.test_batch_args

`batch_asciify.BASE_ARGS` used to spell out ~95 flags, which was a second copy of
the recipe: it drifted from the single-image one, and once the canonical recipe
became the argparse defaults it also started silently inheriting changes for every
flag it did *not* set (`--pool-probe-rate` 0 -> 1 and `--pool-probe-per-chan`
1 -> 2 both moved this way).

It now lists only the deliberate departures. That is a better shape, but it means
correctness depends on the defaults underneath it, so this test pins the result:
it parses `defaults + BASE_ARGS + variant` and compares every value against the
namespace the old explicit list produced. Any drift in a default that the batch
silently relies on shows up here as a named mismatch.

The reference below is the fully-explicit list as it stood before the trim.
"""
import sys

# The pre-trim BASE_ARGS, verbatim. Do not "tidy" this -- it is the reference.
REFERENCE = [
    "--iters", "5000",
    "--mode", "swarm", "--swarm-k", "3", "--swarm-knn-k", "4", "--swarm-w-cap", "0.5",
    "--swarm-w-decay", "0.02",
    "--swarm-w-temp", "1.0", "--swarm-w-temp-end", "0.03", "--swarm-w-temp-end-frac", "0.9",
    "--swarm-w-temp-cycles", "3", "--swarm-w-temp-cycle-decay", "0.75",
    "--swarm-purge-cycles", "--swarm-purge-keep", "2", "--swarm-purge-reseed", "1",
    "--swarm-w-noise", "0.1", "--swarm-phase-gates", "--swarm-phase-frac", "0.75",
    "--swarm-epilogue-frac", "0.15", "--swarm-epilogue-cycles",
    "--swarm-epilogue-temp", "0.5", "--swarm-epilogue-noise", "0.4",
    "--swarm-blank-close", "10", "--swarm-blank-close-ink", "0.04",
    "--swarm-boost-cooldown", "50", "--swarm-boost-ink", "0.04",
    "--z-noise", "0.7", "--z-noise-end", "0.2", "--z-noise-schedule", "cosine",
    "--z-noise-adapt", "nn1", "--z-noise-commit", "0.05", "--z-noise-quench-frac", "0.75",
    "--commit", "0.1",
    "--coord-weight", "0.1", "--coord-weight-end", "0.02", "--coord-weight-end-frac", "0.25",
    "--orient-weight", "1e-2",
    "--pixel-nudge-weight", "0.2", "--pixel-nudge-weight-end", "0.01",
    "--pixel-nudge-weight-schedule", "cosine", "--pixel-nudge-weight-end-frac", "0.6",
    "--clip-snap-weight", "0.2", "--clip-snap-mode", "winner", "--clip-snap-gate", "entropy",
    "--clip-snap-start-frac", "0.2",
    "--pool-nominate-rate", "25", "--pool-nominate-start-frac", "0.05",
    "--pool-nominate-end-frac", "0.85",
    "--pool-channels", "latent,blend,affinity,join,pixel,ports",
    "--pool-join-coord-sigma", "0.2", "--pool-join-every", "2",
    "--pool-probe", "--pool-probe-batches", "3",
    "--pool-probe-spacing", "11", "--pool-probe-window", "5",
    "--pool-ports-floor", "1.0", "--pool-ports-gamma", "1.0",
    "--pool-diversity-weight", "1e-3", "--pool-diversity-weight-end", "1e-3",
    "--pool-diversity-weight-start-frac", "0.2", "--pool-grace", "50",
    "--knn-temp", "0.8", "--knn-temp-end", "0.08", "--knn-temp-schedule", "cosine",
    "--knn-temp-warmup-frac", "0.0", "--knn-temp-end-frac", "1.0",
    "--pool-affinity-margin", "-0.25", "--pool-latent-margin", "-0.25",
    "--recon-weight", "0.25", "--recon-weight-end", "0.07",
    "--recon-weight-schedule", "cosine", "--recon-weight-end-frac", "0.4",
    "--clip-weight", "1.2", "--clip-scale-alpha", "0.8", "--clip-aug", "16",
    "--clip-rotate", "6", "--clip-shear", "5", "--clip-aspect-jitter", "0.75", "1.3333",
    "--affinity-weight", "0.0", "--affinity-weight-end", "5e-3",
    "--affinity-weight-schedule", "cosine", "--affinity-weight-start-frac", "0.4",
    "--clip-dense-weight", "0.3",
    "--clip-consist-weight", "0.15", "--clip-consist-start-frac", "0.45",
    "--target-white", "0.93",
    "--blank-weight", "0.2",
    "--swarm-epilogue-skip-final", "--swarm-epilogue-cycles-max", "2",
    "--anim-interval", "10",
    "--empty-weight", "0.18", "--empty-temp-scale", "0.3", "--empty-thresh", "0.95",
    "--empty-window", "5", "--empty-gamma", "2.0", "--empty-anneal", "1",
    "--empty-noise-scale", "0.3", "--empty-clean-target", "--empty-clean-sigma", "4.0",
    "--clip-content-crop", "--clip-content-pad", "2",
]
# The batch ran with pool-probe-rate/per-chan at their OLD defaults, which the
# canonical recipe has since raised. The reference must carry them explicitly.
REFERENCE += ["--pool-probe-rate", "0", "--pool-probe-per-chan", "1"]
# --vae-ckpt is resolved per environment; compared separately.
IGNORE = {"vae_ckpt", "input_image", "output", "output_text", "pool_record",
          "anim", "loss_curve", "clip_diagnostic"}


def parse(extra):
    from unicasso.engine import asciify as A
    sys.argv = ["asciify", "image.jpg", "--vae-ckpt", "x.pt"] + list(extra)
    return A.parse_args()


def main():
    from unicasso.engine import batch_asciify as B
    ok = True
    for label, variant in B.VARIANTS:
        got = parse([str(a) for a in B.BASE_ARGS] + [str(a) for a in variant])
        want = parse(REFERENCE + [str(a) for a in variant])
        bad = [(k, getattr(want, k), getattr(got, k))
               for k in vars(want) if k not in IGNORE
               and getattr(want, k) != getattr(got, k)]
        n = len([k for k in vars(want) if k not in IGNORE])
        print(f"  variant {label}: {n - len(bad)}/{n} settings identical to the pre-trim BASE_ARGS")
        for k, w, g in bad:
            print(f"      DRIFT {k:<34} was {w!r:<18} now {g!r}")
        ok &= not bad
    print("BATCH ARGS OK" if ok else "BATCH ARGS DRIFT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
