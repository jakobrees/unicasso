"""Engine worker for the refinement pools: one asciify job per process.

Reads one JSON-encoded argv list from stdin, runs
`unicasso.engine.asciify.main()` in-process and prints `DONE <rc>` on its own
stdout line. The ASCIIFY_PERSIST model cache is armed so a worker COULD run
many jobs, but pool_manager.RollingPool launches a fresh process per job (the
cache does not survive --color-lite; see _run_fleet) and runs them in waves of
--pool-workers -- on a CUDA box, arm the MPS daemon first
(`nvidia-cuda-mps-control -d`; NOT reboot-persistent) so the concurrent
workers share the GPU efficiently.

Standalone use:
    echo '["photo.png","--color","--iters","200",...]' \
        | python -m unicasso.training.em_worker
"""

import contextlib
import json
import os
import sys
import traceback

os.environ["ASCIIFY_PERSIST"] = "1"


def main():
    from unicasso.engine import asciify
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        argv = json.loads(line)
        sys.argv = ["asciify"] + [str(a) for a in argv]
        rc = 0
        try:
            # job output -> stderr: the worker's REAL stdout carries only the
            # DONE protocol lines the orchestrator reads
            with contextlib.redirect_stdout(sys.stderr):
                asciify.main()
        except SystemExit as e:
            rc = int(e.code or 0)
        except Exception:
            traceback.print_exc()
            rc = 1
        print(f"DONE {rc}", flush=True)


if __name__ == "__main__":
    main()
