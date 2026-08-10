"""UAC1 wire format: glyph grid <-> bytes <-> base32 URL fragment.

    byte 0      0x01                version
    byte 1      flags: bit0 colour (reserved, always 0)
                       bits1-3 alphabet id
                       bits4-7 reserved
    varint      width
    varint      height
    bitstream   range-coded, W*H symbols, raster order

Then RFC-4648 base32, uppercase, unpadded. Base32 rather than base64 so the whole
URL can be uppercase and the QR encoder can use ALPHANUMERIC mode (5.5 bits per
character) instead of byte mode (8 bits per character). Base32 wastes 5/5.5 = 9%
of the alphanumeric capacity but byte-mode base64 wastes 6/8 = 25%, so base32
nets ~21% more art per QR.

ROUND-TRIP CONTRACT: the codec preserves the *canonical* grid -- every line padded
with spaces to the maximum line width, terminated by a newline. `canonicalize()`
is the identity for 297 of the 298 corpus files (the corpus is already
full-width padded); the one ragged file gains trailing spaces, which is visually
identical but not byte-identical. Compare against `canonicalize(text)`, not the
raw file, when asserting round-trip equality.
"""
from .rangecoder import RangeEncoder, RangeDecoder

VERSION = 1


def canonicalize(text):
    """Pad every line to the max width with spaces; guarantee one trailing newline."""
    lines = text.split("\n")
    while lines and lines[-1] == "":
        lines.pop()
    if not lines:
        return ""
    w = max(len(l) for l in lines)
    return "".join(l.ljust(w) + "\n" for l in lines)


def _put_varint(out, v):
    while True:
        b = v & 0x7F
        v >>= 7
        out.append(b | (0x80 if v else 0))
        if not v:
            return


def _get_varint(buf, pos):
    v = shift = 0
    while True:
        b = buf[pos]
        pos += 1
        v |= (b & 0x7F) << shift
        if not (b & 0x80):
            return v, pos
        shift += 7


def _encode_symbol(enc, model, left, above, sym):
    """PPM-C escape chain: (left,above) -> (left) -> dense order-0."""
    tb = model.total_bits
    t2 = model.l2.get(model.ctx2(left, above))
    if t2 is not None:
        hit = t2.find(sym)
        if hit is not None:
            enc.encode(hit[0], hit[1], tb)
            return
        enc.encode(t2.esc_cum, t2.esc_freq, tb)
    t1 = model.l1.get(left)
    if t1 is not None:
        hit = t1.find(sym)
        if hit is not None:
            enc.encode(hit[0], hit[1], tb)
            return
        enc.encode(t1.esc_cum, t1.esc_freq, tb)
    enc.encode(model.l0_cums[sym], model.l0[sym], tb)


def _decode_symbol(dec, model, left, above):
    tb = model.total_bits
    t2 = model.l2.get(model.ctx2(left, above))
    if t2 is not None:
        v = dec.get_freq(tb)
        hit = t2.symbol_at(v)
        if hit is not None:
            dec.update(hit[1], hit[2])
            return hit[0]
        dec.update(t2.esc_cum, t2.esc_freq)
    t1 = model.l1.get(left)
    if t1 is not None:
        v = dec.get_freq(tb)
        hit = t1.symbol_at(v)
        if hit is not None:
            dec.update(hit[1], hit[2])
            return hit[0]
        dec.update(t1.esc_cum, t1.esc_freq)
    v = dec.get_freq(tb)
    lo, hi = 0, len(model.l0) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if model.l0_cums[mid] + model.l0[mid] <= v:
            lo = mid + 1
        else:
            hi = mid
    dec.update(model.l0_cums[lo], model.l0[lo])
    return lo


def encode_text(text, chars, model):
    """Canonical text -> UAC1 payload bytes."""
    canon = canonicalize(text)
    lines = canon.split("\n")[:-1] if canon else []
    h = len(lines)
    w = len(lines[0]) if h else 0
    index = {c: i for i, c in enumerate(chars)}

    out = bytearray()
    out.append(VERSION)
    out.append((model.alphabet_id & 0x07) << 1)
    _put_varint(out, w)
    _put_varint(out, h)

    enc = RangeEncoder()
    prev = [model.border] * w              # context row above; border before row 0
    for row in lines:
        left = model.border
        cur = [0] * w
        for j, ch in enumerate(row):
            sym = index.get(ch)
            if sym is None:
                # Out-of-charset (hand-made .txt): escape, then the raw codepoint.
                _encode_symbol(enc, model, left, prev[j], model.esc_lit)
                cp = ord(ch)
                tmp = bytearray()
                _put_varint(tmp, cp)
                for b in tmp:
                    enc.encode_bits(b, 8)
                cur[j] = model.oov
            else:
                _encode_symbol(enc, model, left, prev[j], sym)
                cur[j] = sym
            left = cur[j]
        prev = cur
    return bytes(out) + enc.finish()


def decode_bytes(blob, chars, model):
    """UAC1 payload bytes -> canonical text."""
    if not blob or blob[0] != VERSION:
        raise ValueError(f"unsupported UAC1 version {blob[0] if blob else 'empty'}")
    flags = blob[1]
    if flags & 0x01:
        raise ValueError("colour payloads are not supported by this decoder")
    alphabet_id = (flags >> 1) & 0x07
    if alphabet_id != model.alphabet_id:
        raise ValueError(f"payload wants alphabet {alphabet_id}, "
                         f"decoder has {model.alphabet_id}")
    pos = 2
    w, pos = _get_varint(blob, pos)
    h, pos = _get_varint(blob, pos)

    dec = RangeDecoder(blob[pos:])
    prev = [model.border] * w
    lines = []
    for _ in range(h):
        left = model.border
        cur = [0] * w
        row = []
        for j in range(w):
            sym = _decode_symbol(dec, model, left, prev[j])
            if sym == model.esc_lit:
                cp = shift = 0
                while True:
                    b = dec.decode_bits(8)
                    cp |= (b & 0x7F) << shift
                    if not (b & 0x80):
                        break
                    shift += 7
                row.append(chr(cp))
                cur[j] = model.oov
            else:
                row.append(chars[sym])
                cur[j] = sym
            left = cur[j]
        lines.append("".join(row))
        prev = cur
    return "".join(l + "\n" for l in lines)


# ---- base32 for the URL fragment ---------------------------------------

import base64


def to_base32(payload):
    return base64.b32encode(payload).decode("ascii").rstrip("=")


def from_base32(s):
    s = s.strip().upper()
    pad = (-len(s)) % 8
    return base64.b32decode(s + "=" * pad)
