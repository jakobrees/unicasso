"""Cross-language parity gate: the browser decoder must match the Python encoder.

    python -m unicasso.artcode.test_parity            # corpus round-trip + node parity
    python -m unicasso.artcode.test_parity --quick    # a 12-piece sample

This is the test that matters. A range coder carries no checksum: if the JS
decoder and the Python encoder disagree anywhere -- one signed-vs-unsigned shift,
one float where an integer was meant -- the stream desynchronises and the rest
decodes into plausible-looking garbage instead of raising. Nothing else in the
suite would catch it, and it would ship as art that renders wrong on the web and
right on the desk.

Run this before deploying the site, and after ANY change to the range coder, the
table format, or the model.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

from . import decode, encode, load_charset, load_model, to_base32
from .fit_tables import load_corpus

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVER = os.path.join(HERE, "parity_driver.js")
DEFAULT_JS = os.path.normpath(os.path.join(
    HERE, "..", "..", "..", "jakobrees.com", "public", "a", "artcode.js"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="sample 12 pieces instead of all")
    ap.add_argument("--js", default=None, help="path to artcode.js (the shipped decoder)")
    ap.add_argument("--table", default=None, help="model table (defaults to the shipped one)")
    args = ap.parse_args(argv)

    js = args.js or DEFAULT_JS
    if not os.path.exists(js):
        print(f"FAIL: browser decoder not found at {js}", file=sys.stderr)
        return 2

    model = load_model(args.table) if args.table else load_model()
    chars = model.chars
    table_path = args.table or os.path.join(HERE, "tables", "uac1_sfmono.bin")

    corpus = load_corpus()
    if args.quick:
        corpus = corpus[:: max(1, len(corpus) // 12)][:12]

    # Stage 1 -- Python must round-trip its own output before parity means anything.
    cases, expected, py_fail = [], {}, []
    for name, _group, text in corpus:
        payload = encode(text, chars, model)
        if decode(payload, chars, model) != text:
            py_fail.append(name)
        cases.append({"name": name, "b32": to_base32(payload)})
        expected[name] = text
    print(f"python round-trip : {len(corpus) - len(py_fail)}/{len(corpus)} exact")
    if py_fail:
        print(f"  FAILED: {py_fail[:5]}", file=sys.stderr)

    # Stage 2 -- the shipped JS must produce byte-identical text.
    with tempfile.TemporaryDirectory() as td:
        cpath = os.path.join(td, "cases.json")
        with open(cpath, "w") as f:
            json.dump(cases, f)
        env = dict(os.environ, ARTCODE_JS=os.path.abspath(js))
        try:
            proc = subprocess.run(["node", DRIVER, table_path, cpath],
                                  capture_output=True, text=True, env=env, timeout=600)
        except FileNotFoundError:
            print("FAIL: node not found -- cannot verify the browser decoder", file=sys.stderr)
            return 2
        if proc.returncode != 0:
            print(f"FAIL: node driver exited {proc.returncode}\n{proc.stderr}", file=sys.stderr)
            return 2
        results = json.loads(proc.stdout)

    js_fail, errs = [], []
    for r in results:
        if "error" in r:
            errs.append((r["name"], r["error"]))
        elif r["text"] != expected[r["name"]]:
            js_fail.append(r["name"])
    ok = len(results) - len(js_fail) - len(errs)
    print(f"js parity        : {ok}/{len(results)} byte-identical to Python")
    for name, err in errs[:5]:
        print(f"  ERROR {name}: {err}", file=sys.stderr)
    for name in js_fail[:5]:
        print(f"  MISMATCH {name}", file=sys.stderr)

    bad = bool(py_fail or js_fail or errs)
    print("PARITY FAILED" if bad else "PARITY OK")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
