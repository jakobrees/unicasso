# artcode — a QR code that *is* the artwork

An ASCII grid compressed small enough to fit inside a QR code. The whole grid
travels in the URL fragment, so it needs no server, no database and no stored
record: scanning the code hands the decoder page everything it needs, and the
link keeps working for as long as the page exists.

```bash
pip install segno                                   # or: pip install -e '.[qr]'

python -m unicasso.artcode art.txt                  # QR in the terminal
python -m unicasso.artcode art.txt --png code.png   # ...and a PNG
python -m unicasso.artcode art.txt --url-only       # just the URL
python -m unicasso.artcode --decode 'HTTPS://…#ABC' # back to the grid
```

From the optimizer, encoding the same grid it just wrote to disk:

```bash
python -m unicasso.engine.asciify photo.jpg --qr
python -m unicasso.engine.asciify photo.jpg --qr-image code.png
```

## How big is it

Held-out, 5-fold group-disjoint over the 298-file corpus, measuring real encoder
output (not an entropy estimate):

| | bits/cell | median | p90 | worst |
|---|---|---|---|---|
| order-0 Huffman (corpus frequencies) | 2.68 | 464 B | 840 B | 1623 B |
| deflate / gzip | 2.54 | 394 B | 757 B | 1523 B |
| **UAC1 (order-2 + range coder)** | **1.94** | **299 B** | **627 B** | **1403 B** |

Every corpus grid fits a single QR at error level M, clustering around V13
(69×69 modules) and topping out at V31. Verified by decoding the rendered PNGs
with zxing: 298/298 recover the exact source text.

## Format

```
byte 0      0x01                version
byte 1      flags: bit0 colour (reserved, 0), bits1-3 alphabet id, bits4-7 reserved
varint      width
varint      height
bitstream   range-coded, W*H symbols, raster order
```

then RFC-4648 base32, uppercase, unpadded.

**Why base32 and a mostly-uppercase URL.** QR's alphanumeric mode packs 5.5 bits
per character against byte mode's 8. Base32 is uppercase and hostnames are
case-insensitive, so the payload and host stay in alphanumeric mode. The *path*
cannot: URL paths are case-sensitive, the decoder lives at `/a/`, and requesting
`/A/` does not 404 on Cloudflare Pages -- it serves the site index, so the decoder
gets HTML where it expects its table and reports "could not decode". The trailing
slash is load-bearing too, or the page's relative fetches resolve against the site
root. `qr.segments()` therefore keeps `/a/#` in a short byte-mode segment. Base32
wastes 5/5.5 = 9% of that capacity where byte-mode base64 wastes 6/8 = 25%, so
this nets ~21% more art per symbol.

**Why the '#' still works.** '#' is not in QR's alphanumeric set, so a fragment
URL would normally drag the whole string into byte mode — measured, that costs 4
QR versions at the median (V17/85×85 instead of V13/69×69). `qr.segments()`
emits mixed-mode segments instead: alphanumeric prefix, a one-character byte
segment for the '#', alphanumeric payload. Overhead is ~5 bytes and the result
matches a pure-alphanumeric URL exactly — so we keep the fragment, and with it
the property that the grid is never sent to the server.

**Round-trip contract.** The codec preserves the *canonical* grid: every line
space-padded to the maximum width, one trailing newline. `canonicalize()` is the
identity for 297 of 298 corpus files; the one ragged file gains trailing spaces,
which is visually identical but not byte-identical. Compare against
`canonicalize(text)`, not the raw file.

## The model

Order-2 context modelling: for each cell, condition on the glyph to the **left**
and the glyph directly **above**, then range-code against that pair's frequency
table. Measured alternatives, held-out:

| context | bits/cell | |
|---|---|---|
| order-0 | 2.68 | plain corpus frequencies |
| (left) | 1.98 | |
| (above) | 2.40 | horizontal runs dominate vertical |
| (left, left-2) | 1.97 | the second-left cell adds ~nothing |
| **(left, above)** | **1.84** | shipped |
| (left, above, above-left) | 1.86 | worse — the diagonal is implied |
| (left, above, above-right) | 1.78 | 3% better, 3.5× the table; available if wanted |

(Those are blended-backoff estimates; the shipped PPM-C escape coder measures
1.94, and the gap is the price of exact integer reproducibility.)

**Things that sound like they should help and don't.** Margin whitespace is
51.8% of all cells but only 2.4% of the payload — the model already prices a
leading margin space at 0.061 bits. A start-column + end-of-row grammar can
therefore save at most 2.4%, while an 8-bit start column per row *costs* ~8%.
An oracle per-row selector flag (choose the cheaper encoding per row, 1 bit
each) loses 0.97% and helps 2 of 298 images. All dropped.

**Why a range coder and not Huffman per context.** Huffman spends a whole bit
minimum per symbol, but a glyph predicted at p=0.96 is worth 0.06 bits. Over the
same order-2 tables, per-context Huffman measures 2.587 bits/cell against 1.835
— 41% worse, and barely better than order-0 Huffman (2.809). Whole-bit rounding
destroys ~90% of the gain from context modelling.

## Rebuilding the tables

```bash
python -m unicasso.artcode.fit_tables            # write tables/uac1_sfmono.bin
python -m unicasso.artcode.fit_tables --report   # held-out measurement
```

`min_context_count` (default 80) prunes thin contexts, and pruning hard helps
*both* axes: a thinly-observed context predicts badly and charges a large
escape, so falling through to a well-estimated parent codes better. Raising it
from 2 to 80 improved 1.974 → 1.936 bits/cell while shrinking the table
282 → 85 KiB. The optimum is a flat plateau over ~32–80; past ~128 the model
degrades toward order-1.

The table embeds the charset it was fitted against. A table and a charset that
disagree decode cleanly into the *wrong* glyphs — silent corruption nobody would
notice until the art looked subtly scrambled — so they ship as one artifact.

## Parity is the gate

```bash
python -m unicasso.artcode.test_parity
```

Runs the **shipped browser decoder** (`jakobrees.com/public/a/artcode.js`) under
node against every corpus payload and diffs the output. A range coder carries no
checksum: if the JS and Python sides disagree by one bit, the stream
desynchronises and the rest decodes to plausible-looking garbage rather than
raising. Nothing else would catch it.

Two rules keep them identical, and both are load-bearing:

1. Every `total` is a power of two, so `range / total` is an exact shift with no
   division rounding to disagree about.
2. **No bitwise operators on values that reach 2^31 in the JS decoder.**
   JavaScript's `<<`, `|` and `&` coerce to *signed* 32-bit, which silently
   mangles `range` and `code`. The JS side uses `*`, `/`, `%` and `Math.floor`.

All smoothing and backoff happens at table-build time; runtime is pure integer
table lookup, with no floating point anywhere in the coding path.

Run it before deploying the site, and after any change to the range coder, the
table format, or the model.
