"""Order-2 context model for UAC1, and the binary table both languages load.

The model is exactly what it sounds like: n-gram style conditional frequency
tables. For each cell we condition on the glyph to the LEFT and the glyph
directly ABOVE, look up that pair's frequency table, and hand it to the range
coder. Measured held-out on the 298-file corpus (5-fold, group-disjoint):

    order-0 (plain corpus frequencies)     2.68 bits/cell
    order-1 (left only)                    1.98
    order-2 (left, above)                  1.84   <- this
    order-2 (left, two-to-the-left)        1.97   (the second-left cell adds ~nothing)

The cell above carries the information; a second horizontal neighbour does not.

Three levels with PPM-C escape coding. If the (left, above) table has never seen
this glyph, emit its ESCAPE symbol and retry at (left); escape again and fall
through to a dense order-0 table where every glyph has non-zero frequency, so
coding always terminates.

All smoothing and backoff is baked in at table-build time and the shipped
frequencies are integers summing to 2**TOTAL_BITS. Nothing is computed from
floats at runtime -- that is what makes the JS decoder reproducible.

Symbol spaces (deliberately distinct):
  coding alphabet   0..G-1 glyphs, G = ESC_LIT (literal codepoint follows)  -> A = G+1
  context values    0..G-1 glyphs, G = OOV (a literal), G+1 = BORDER        -> C = G+2
"""
import struct

# v2 embeds the charset in the table. A table and a charset that disagree produce
# a clean-looking decode of the wrong glyphs -- silent corruption -- so they ship
# as one artifact and the JS decoder reads both from the same file.
# v3 adds the kit's cell geometry for the same reason: the art was optimised
# against a cell of a specific aspect ratio (sfmono 18x36 = 2.00, dejavu 16x34 =
# 2.125), and a renderer that picks its own line spacing shows the piece
# distorted. CSS `line-height: 1` gives a cell of 1/0.6 = 1.67 and squashes an
# sfmono piece ~17% vertically, which is subtle enough to look like bad art
# rather than a bug.
MAGIC = b"UAC1TBL\x03"
TOTAL_BITS = 12
TOTAL = 1 << TOTAL_BITS


def quantize(counts, total=TOTAL):
    """Integer frequencies summing to exactly `total`, every entry >= 1.

    `counts` is an ordered list of (key, count). Order is preserved and matters:
    cumulative frequencies are a prefix sum in this order, so the encoder and the
    JS decoder must build the list identically (we sort by symbol id, escape last).
    """
    keys = [k for k, _ in counts]
    raw = [c for _, c in counts]
    s = sum(raw)
    if s <= 0:
        raw = [1] * len(raw)
        s = len(raw)
    if len(keys) > total:
        raise ValueError(f"cannot quantize {len(keys)} symbols into {total}")
    freqs = [max(1, (c * total) // s) for c in raw]
    diff = total - sum(freqs)
    if diff > 0:
        # give the surplus to the largest bucket -- deterministic, ties by index
        i = max(range(len(freqs)), key=lambda k: (freqs[k], -k))
        freqs[i] += diff
    while diff < 0:
        # reclaim from the largest bucket that can spare it
        i = max(range(len(freqs)), key=lambda k: (freqs[k], -k))
        take = min(-diff, freqs[i] - 1)
        if take <= 0:
            raise ValueError("cannot fit frequencies into total")
        freqs[i] -= take
        diff += take
    assert sum(freqs) == total
    return list(zip(keys, freqs))


class ContextTable:
    """One context's frequency table. Escape occupies the final slot."""

    __slots__ = ("syms", "freqs", "cums", "esc_cum", "esc_freq")

    def __init__(self, pairs, esc_freq):
        self.syms = [s for s, _ in pairs]
        self.freqs = [f for _, f in pairs]
        self.cums = []
        c = 0
        for f in self.freqs:
            self.cums.append(c)
            c += f
        self.esc_cum = c
        self.esc_freq = esc_freq

    def find(self, sym):
        """(cum, freq) for sym, or None if this context has never seen it."""
        for i, s in enumerate(self.syms):
            if s == sym:
                return self.cums[i], self.freqs[i]
            if s > sym:
                return None
        return None

    def symbol_at(self, v):
        """Inverse of find(): which symbol owns cumulative slot v, or None for escape."""
        for i, c in enumerate(self.cums):
            if c <= v < c + self.freqs[i]:
                return self.syms[i], c, self.freqs[i]
        return None


class ArtcodeModel:
    def __init__(self, chars, l0, l1, l2, alphabet_id=0, total_bits=TOTAL_BITS,
                 cell_w=18, cell_h=36):
        self.chars = chars
        # Kit cell geometry in pixels. Only the RATIO matters downstream: a
        # renderer sets line-height = (cell_h/cell_w) * measured_advance so each
        # text cell has the same shape as the cell the optimiser drew into.
        self.cell_w = cell_w
        self.cell_h = cell_h
        n_glyphs = len(chars)
        self.n_glyphs = n_glyphs
        self.alphabet = n_glyphs + 1          # coding alphabet A (glyphs + ESC_LIT)
        self.ctx_card = n_glyphs + 2          # context cardinality C
        self.esc_lit = n_glyphs
        self.oov = n_glyphs
        self.border = n_glyphs + 1
        self.alphabet_id = alphabet_id
        self.total_bits = total_bits
        self.l0 = l0                          # dense list of A freqs
        self.l1 = l1                          # {left: ContextTable}
        self.l2 = l2                          # {left*C+above: ContextTable}
        self.l0_cums = []
        c = 0
        for f in l0:
            self.l0_cums.append(c)
            c += f
        assert c == (1 << total_bits), f"L0 sums to {c}, expected {1 << total_bits}"

    def ctx2(self, left, above):
        return left * self.ctx_card + above

    # ---- serialisation -------------------------------------------------

    def to_bytes(self):
        out = bytearray(MAGIC)
        out += struct.pack("<BBHHI", self.alphabet_id, self.total_bits,
                           self.alphabet, len(self.l1), len(self.l2))
        out += struct.pack("<HH", self.cell_w, self.cell_h)
        # Charset as raw code points: unambiguous in both languages, and immune to
        # the UTF-8 / surrogate-pair splitting mistakes a string would invite.
        out += struct.pack(f"<{self.n_glyphs}I", *[ord(c) for c in self.chars])
        out += struct.pack(f"<{self.alphabet}H", *self.l0)
        for ctx in sorted(self.l1):
            t = self.l1[ctx]
            out += struct.pack("<HH", ctx, len(t.syms))
            for s, f in zip(t.syms, t.freqs):
                out += struct.pack("<HH", s, f)
            out += struct.pack("<H", t.esc_freq)
        for ctx in sorted(self.l2):
            t = self.l2[ctx]
            out += struct.pack("<IH", ctx, len(t.syms))
            for s, f in zip(t.syms, t.freqs):
                out += struct.pack("<HH", s, f)
            out += struct.pack("<H", t.esc_freq)
        return bytes(out)

    @classmethod
    def from_bytes(cls, blob):
        if blob[:8] != MAGIC:
            raise ValueError("not a UAC1 table")
        off = 8
        alphabet_id, total_bits, alphabet, n_l1, n_l2 = struct.unpack_from("<BBHHI", blob, off)
        off += 10
        cell_w, cell_h = struct.unpack_from("<HH", blob, off)
        off += 4
        n_glyphs = alphabet - 1
        chars = "".join(chr(cp) for cp in struct.unpack_from(f"<{n_glyphs}I", blob, off))
        off += 4 * n_glyphs
        l0 = list(struct.unpack_from(f"<{alphabet}H", blob, off))
        off += 2 * alphabet
        l1, l2 = {}, {}
        for _ in range(n_l1):
            ctx, cnt = struct.unpack_from("<HH", blob, off)
            off += 4
            pairs = []
            for _ in range(cnt):
                s, f = struct.unpack_from("<HH", blob, off)
                off += 4
                pairs.append((s, f))
            (esc,) = struct.unpack_from("<H", blob, off)
            off += 2
            l1[ctx] = ContextTable(pairs, esc)
        for _ in range(n_l2):
            ctx, cnt = struct.unpack_from("<IH", blob, off)
            off += 6
            pairs = []
            for _ in range(cnt):
                s, f = struct.unpack_from("<HH", blob, off)
                off += 4
                pairs.append((s, f))
            (esc,) = struct.unpack_from("<H", blob, off)
            off += 2
            l2[ctx] = ContextTable(pairs, esc)
        return cls(chars, l0, l1, l2, alphabet_id, total_bits, cell_w, cell_h)


def build_from_counts(chars, c0, c1, c2, min_context_count=80, total_bits=TOTAL_BITS,
                      esc_alpha=1.0, esc_alpha_l1=None, cell_w=18, cell_h=36):
    """Fit tables from raw counts.

    chars: the charset the counts were taken over (stored in the table)
    c0: Counter over coding symbols
    c1: {left: Counter}
    c2: {left*C+above: Counter}

    Escape frequency generalises PPM method C: the escape gets
    `esc_alpha * (number of distinct symbols seen)`, which prices "something new
    happens here" by how varied the context has been so far. alpha=1 is method C,
    alpha=0.5 is method D. Lower alpha bets harder on the context being complete,
    which pays off when the tables are well fitted. Tuned by measurement --
    see `fit_tables --sweep`.

    `min_context_count` drops thin contexts entirely, and pruning hard turns out
    to help *both* axes: a thinly-observed context predicts badly and charges a
    large escape, so falling straight through to a well-estimated parent codes
    better. Measured held-out, raising it from 2 to 80 improved 1.974 -> 1.936
    bits/cell while shrinking the table 282 -> 83 KiB. The optimum is a flat
    plateau over roughly 32-80; the high end is chosen because the ratio
    difference there is under a byte per piece and the table ships to every
    visitor. Beyond ~128 the model degrades back toward order-1.
    """
    total = 1 << total_bits
    n_glyphs = len(chars)
    alphabet = n_glyphs + 1
    if esc_alpha_l1 is None:
        esc_alpha_l1 = esc_alpha

    dense = [c0.get(s, 0) for s in range(alphabet)]
    l0_pairs = quantize([(s, dense[s] + 1) for s in range(alphabet)], total)
    l0 = [f for _, f in l0_pairs]

    def make(counter, alpha):
        items = sorted(counter.items())
        total_c = sum(c for _, c in items)
        esc = max(1, int(round(alpha * len(items))))
        # An escape can never be cheaper than the context is uncertain, but it
        # also must not swamp a context that is genuinely deterministic.
        esc = min(esc, max(1, total_c))
        q = quantize(items + [(-1, esc)], total)
        return ContextTable(q[:-1], q[-1][1])

    l1 = {ctx: make(cn, esc_alpha_l1)
          for ctx, cn in c1.items() if sum(cn.values()) >= min_context_count}
    l2 = {ctx: make(cn, esc_alpha)
          for ctx, cn in c2.items() if sum(cn.values()) >= min_context_count}
    return ArtcodeModel(chars, l0, l1, l2, total_bits=total_bits,
                        cell_w=cell_w, cell_h=cell_h)
