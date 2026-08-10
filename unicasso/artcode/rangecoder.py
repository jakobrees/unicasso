"""Integer range coder (LZMA-style) for the UAC1 artcode format.

Why a range coder and not Huffman: Huffman must spend a whole number of bits per
symbol, so a glyph the model predicts at p=0.96 still costs 1 bit instead of the
0.06 bits it is worth. Half of every piece is margin whitespace predicted at
~0.96, and measuring it on the corpus put per-context Huffman at 2.587 bits/cell
against 1.835 for this coder -- 41% worse, which is most of the payload.

A range coder instead narrows one interval by the symbol's probability at every
step and emits a single number inside the final interval, so the total cost is
sum(-log2 p) with the rounding paid ONCE for the whole message (<2 bytes), not
once per symbol.

PARITY IS THE WHOLE GAME. This file is mirrored byte-for-byte in JavaScript
(`public/a/artcode.js` on the website). A single differing bit desynchronises the
decoder for the rest of the stream and produces garbage, not a clean error. Two
rules keep the two implementations identical:

  1. Every `total` is a power of two, passed as `total_bits`, so `range // total`
     is an exact shift with no division rounding to disagree about.
  2. All state is masked to explicit widths here, because JavaScript has no
     uint32 and its bitwise operators silently coerce to *signed* 32-bit. The JS
     side therefore uses `* 256` / `% 2**32` arithmetic rather than `<<`, and
     this side masks so that the two agree.

`low` is the only value allowed above 32 bits (it reaches 2^40 between
normalisations, which is exactly how the carry is detected).
"""

MASK32 = 0xFFFFFFFF
TOP = 1 << 24


class RangeEncoder:
    """Encodes symbols given as (cumulative frequency, frequency) in a power-of-two total."""

    def __init__(self):
        self.low = 0
        self.range = MASK32
        self.cache = 0
        self.cache_size = 1
        self.out = bytearray()

    def _shift_low(self):
        # Carry propagation: if low overflowed 32 bits, the +1 has to ripple back
        # through the run of pending 0xFF bytes we deliberately withheld.
        if (self.low & MASK32) < 0xFF000000 or (self.low >> 32) != 0:
            carry = (self.low >> 32) & 0xFF
            temp = self.cache
            while True:
                self.out.append((temp + carry) & 0xFF)
                temp = 0xFF
                self.cache_size -= 1
                if self.cache_size == 0:
                    break
            self.cache = (self.low >> 24) & 0xFF
        self.cache_size += 1
        self.low = (self.low << 8) & MASK32

    def encode(self, start, size, total_bits):
        """Narrow the interval to [start, start+size) out of 2**total_bits."""
        self.range >>= total_bits
        self.low += start * self.range
        self.range *= size
        while self.range < TOP:
            self.range = (self.range << 8) & MASK32
            self._shift_low()

    def encode_bits(self, value, nbits):
        """Encode a raw value under a uniform model (used by the literal escape)."""
        self.encode(value, 1, nbits)

    def finish(self):
        for _ in range(5):
            self._shift_low()
        return bytes(self.out)


class RangeDecoder:
    """Mirror of RangeEncoder. Two-phase: get_freq() then update()."""

    def __init__(self, data):
        self.data = data
        self.pos = 0
        self.range = MASK32
        self.code = 0
        # Five bytes, the first of which is the encoder's initial cache byte (always
        # 0) and shifts harmlessly out of the 32-bit window.
        for _ in range(5):
            self.code = ((self.code << 8) | self._byte()) & MASK32

    def _byte(self):
        if self.pos < len(self.data):
            b = self.data[self.pos]
            self.pos += 1
            return b
        # Reading past the end is normal: the encoder's flush bytes are implicit.
        self.pos += 1
        return 0

    def get_freq(self, total_bits):
        """Which slot of 2**total_bits the next symbol lies in."""
        self.range >>= total_bits
        v = self.code // self.range
        top = (1 << total_bits) - 1
        return top if v > top else v

    def update(self, start, size):
        self.code -= start * self.range
        self.range *= size
        while self.range < TOP:
            self.code = ((self.code << 8) | self._byte()) & MASK32
            self.range = (self.range << 8) & MASK32

    def decode_bits(self, nbits):
        v = self.get_freq(nbits)
        self.update(v, 1)
        return v
