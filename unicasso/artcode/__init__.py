"""UAC1 artcode: an ASCII piece packed small enough to travel inside a QR code.

The grid is range-coded against an order-2 (left, above) context model fitted on
the corpus, base32'd, and hung off a URL fragment. A static page decodes it
client-side, so a piece needs no server, no database and no stored record --
the artwork *is* the link.

    from unicasso import artcode
    payload = artcode.encode(text)                 # bytes
    url     = artcode.make_url(text)               # https://.../#BASE32
    text    = artcode.decode(payload)

This package is stdlib-only (torch is not required); `qr.py` additionally wants
`segno`, which is a pure-Python optional extra.
"""
import os

from .codec import (canonicalize, decode_bytes, encode_text,
                    from_base32, to_base32)
from .model import ArtcodeModel

DEFAULT_TABLE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "tables", "uac1_sfmono.bin")

# The live decoder.
#
# The HOST is upper-case on purpose, and it is not cosmetic: QR's alphanumeric mode
# packs 5.5 bits/char against byte mode's 8, and hostnames are case-insensitive so
# shouting costs nothing there.
#
# The PATH is lower-case because it must be. URL paths ARE case-sensitive, and the
# page is deployed at /a/. Requesting /A/ does not 404 on Cloudflare Pages -- it
# serves the site's index page instead -- so the decoder receives HTML where it
# expects its model table and reports "could not decode". qr.segments() keeps the
# few lower-case characters in a short byte-mode segment so the payload itself
# stays alphanumeric; the trailing slash matters too, or the page's relative
# fetches resolve against the site root.
BASE_URL = "HTTPS://JAKOBREES.COM/a/#"

_model = None


def load_model(table_path=None):
    """Load (and cache) the shipped model table."""
    global _model
    if _model is None or table_path is not None:
        path = table_path or DEFAULT_TABLE
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"model table missing: {path}\n"
                f"build it with: python -m unicasso.artcode.fit_tables")
        with open(path, "rb") as f:
            m = ArtcodeModel.from_bytes(f.read())
        if table_path is not None:
            return m
        _model = m
    return _model


def load_charset(path=None):
    """The charset the table was fitted against.

    Read from the table itself, not from kits/: a table and a charset that
    disagree decode cleanly into the *wrong* glyphs, and that is not a failure
    anyone would notice until the art looked subtly scrambled. An explicit
    `path` still overrides, for building tables against a different kit.
    """
    if path is not None:
        from .fit_tables import read_charset
        return read_charset(path)
    return load_model().chars


def encode(text, chars=None, model=None):
    return encode_text(text, chars or load_charset(), model or load_model())


def decode(payload, chars=None, model=None):
    return decode_bytes(payload, chars or load_charset(), model or load_model())


def make_url(text, base=BASE_URL, chars=None, model=None):
    return base + to_base32(encode(text, chars, model))


def decode_url(url, chars=None, model=None):
    frag = url.split("#", 1)[1] if "#" in url else url
    return decode(from_base32(frag), chars, model)
