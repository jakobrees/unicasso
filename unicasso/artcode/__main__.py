"""Turn an ASCII-art text file into a QR code (or a QR code back into the art).

    python -m unicasso.artcode art.txt                    # QR in the terminal
    python -m unicasso.artcode art.txt --png code.png     # ...and/or a PNG
    python -m unicasso.artcode art.txt --url-only         # just print the URL
    python -m unicasso.artcode --decode 'HTTPS://...#ABC' # QR payload -> the art

The whole artwork travels inside the URL fragment, so the code needs no server
and no stored record: scanning it hands the decoder page everything it needs.
"""
import argparse
import sys

from . import (BASE_URL, decode, encode, from_base32, load_charset,
               load_model, to_base32)
from .codec import canonicalize


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m unicasso.artcode",
                                 description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", help="path to an ASCII-art .txt ('-' for stdin)")
    ap.add_argument("--decode", metavar="URL_OR_BASE32",
                    help="decode a URL/fragment back to the artwork and print it")
    ap.add_argument("--url", default=BASE_URL,
                    help=f"URL prefix the fragment hangs off (default {BASE_URL}). Keep the "
                         "HOST uppercase for QR alphanumeric mode; the PATH is case-sensitive "
                         "and must match the deployed page exactly")
    ap.add_argument("--png", metavar="PATH", help="also write the QR as a PNG")
    ap.add_argument("--scale", type=int, default=8, help="PNG pixels per module")
    ap.add_argument("--error", default="m", choices=list("lmqh"),
                    help="QR error-correction level; higher survives more damage "
                         "but holds less (default m)")
    ap.add_argument("--url-only", action="store_true", help="print the URL, no QR")
    ap.add_argument("--no-ansi", action="store_true",
                    help="draw the QR without black-on-white ANSI (for light terminals)")
    ap.add_argument("--table", help="model table path (defaults to the shipped one)")
    ap.add_argument("--charset", help="charset file (defaults to the shipped one)")
    args = ap.parse_args(argv)

    chars = load_charset(args.charset) if args.charset else load_charset()
    model = load_model(args.table) if args.table else load_model()

    if args.decode:
        frag = args.decode.split("#", 1)[1] if "#" in args.decode else args.decode
        sys.stdout.write(decode(from_base32(frag), chars, model))
        return 0

    if not args.input:
        ap.error("give an input .txt, or --decode a URL")

    text = sys.stdin.read() if args.input == "-" else open(args.input, encoding="utf-8").read()
    canon = canonicalize(text)
    lines = canon.split("\n")[:-1] if canon else []
    if not lines:
        ap.error("input is empty")

    payload = encode(canon, chars, model)
    url = args.url + to_base32(payload)

    # Encoding is worthless if it does not come back identical -- check every time.
    if decode(payload, chars, model) != canon:
        print("ERROR: round-trip mismatch, refusing to emit a broken code", file=sys.stderr)
        return 1

    cells = len(lines) * len(lines[0])
    print(f"{len(lines[0])}x{len(lines)} = {cells} cells -> {len(payload)} B "
          f"({8*len(payload)/cells:.2f} bits/cell), {len(url)} char URL", file=sys.stderr)

    if args.url_only:
        print(url)
        return 0

    from . import qr as qrmod
    code = qrmod.make(url, error=args.error)
    print(qrmod.describe(code, url), file=sys.stderr)
    print(qrmod.to_terminal(code, ansi=not args.no_ansi))
    print(url, file=sys.stderr)
    if args.png:
        qrmod.to_png(code, args.png, scale=args.scale)
        print(f"wrote {args.png}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
