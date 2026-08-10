"""Render a UAC1 URL as a QR code -- in the terminal, or as a PNG.

Requires `segno` (pure Python, MIT, no compiled deps):

    pip install segno            # or: pip install -e '.[qr]'

Why the URL is uppercase: QR has a dedicated ALPHANUMERIC mode covering
0-9 A-Z and a few punctuation marks at 5.5 bits per character, against 8 bits per
character in byte mode. Base32 is uppercase and the hostname is case-insensitive,
so an all-uppercase URL stays in alphanumeric mode and fits ~21% more art per
symbol. Lowercase anywhere in the string silently forces byte mode and costs you
that margin.

'#' is NOT in the alphanumeric set, so a fragment URL would ordinarily drag the
whole string into byte mode -- measured, that costs 4 QR versions at the median
(V17/85x85 instead of V13/69x69) and 7 at the worst. The fix is mixed-mode
segments: alphanumeric for the prefix, a one-character byte segment for the '#',
alphanumeric again for the payload. The mode-switch overhead is ~5 bytes and the
result matches a pure-alphanumeric URL exactly, so we keep the fragment -- and
with it the property that the artwork is never sent to the server.
"""

BLOCKS = {
    (0, 0): " ",     # both halves light
    (1, 0): "▀",  # upper half dark
    (0, 1): "▄",  # lower half dark
    (1, 1): "█",  # both halves dark
}

# QR alphanumeric mode's 45-character set.
ALNUM = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:")


def _segno():
    try:
        import segno
    except ImportError as e:
        raise ImportError(
            "QR output needs the 'segno' package (pure Python, no compiled deps).\n"
            "  pip install segno        # or: pip install -e '.[qr]'") from e
    return segno


def is_alphanumeric(text):
    return all(c in ALNUM for c in text)


def assert_alphanumeric(text, what="URL"):
    if not is_alphanumeric(text):
        bad = sorted({c for c in text if c not in ALNUM})
        raise ValueError(
            f"{what} leaves QR alphanumeric mode (costs ~21% capacity); "
            f"offending characters: {''.join(bad)!r}. Keep the whole URL uppercase.")


def segments(url):
    """Split a URL into QR segments: maximal alphanumeric runs stay in alphanumeric
    mode, everything else drops to byte mode for as few characters as possible.

    Two parts of a canonical artcode URL are unavoidably outside QR's alphanumeric
    set, and both matter:

      '#'   the fragment delimiter.
      '/a/' the decoder path. URL PATHS ARE CASE-SENSITIVE -- only the hostname is
            not -- so this cannot be upper-cased to fit alphanumeric mode. Doing so
            requests a path that does not exist, and a static host that serves an
            index page for unmatched paths (Cloudflare Pages does) returns HTML
            where the decoder expects its table, which surfaces as "could not
            decode" rather than as a 404.

    Isolating those runs costs ~5 bytes of mode-switch overhead and keeps the long
    payload -- the part that actually determines the QR version -- at 5.5 bits per
    character instead of 8.

    Returns a list of (text, segno_mode_constant) for segno.make().
    """
    from segno import consts
    alnum, byte = consts.MODE_ALPHANUMERIC, consts.MODE_BYTE
    runs, run, run_alnum = [], "", None
    for ch in url:
        a = ch in ALNUM
        if run and a != run_alnum:
            runs.append([run, run_alnum])
            run = ""
        run, run_alnum = run + ch, a
    if run:
        runs.append([run, run_alnum])

    # Coalesce tiny alphanumeric runs trapped between byte runs ('/' inside "/a/#").
    # A mode switch costs ~4 mode bits + a 9-13 bit count field, while carrying one
    # character as byte instead of alphanumeric costs 2.5 -- so for short runs,
    # staying in byte mode is cheaper than switching out and back.
    SWITCH_COST_CHARS = 5
    merged = []
    for i, (text, is_alnum) in enumerate(runs):
        prev_byte = merged and not merged[-1][1]
        next_byte = i + 1 < len(runs) and not runs[i + 1][1]
        if is_alnum and prev_byte and next_byte and len(text) < SWITCH_COST_CHARS:
            merged[-1][0] += text                      # absorb into the byte run
            continue
        if merged and merged[-1][1] == is_alnum:
            merged[-1][0] += text
        else:
            merged.append([text, is_alnum])
    return [(t, alnum if a else byte) for t, a in merged]


def make(url, error="m"):
    """Build the QR symbol. `error` is one of l/m/q/h (rising redundancy)."""
    segno = _segno()
    try:
        return segno.make(segments(url), error=error)
    except segno.DataOverflowError as e:
        raise ValueError(
            f"{len(url)} characters will not fit in a single QR code at error level "
            f"'{error}'. Use a lower error level, or render the piece at a smaller "
            f"grid width.") from e


def matrix(qr, border=4):
    """Module matrix including the quiet zone (4 modules is the spec minimum)."""
    return [list(row) for row in qr.matrix_iter(border=border)]


def to_terminal(qr, border=4, ansi=True):
    """Half-block render: two module rows per text row, so the code stays square.

    With `ansi`, each line is wrapped in black-on-white so the code scans on a
    dark terminal (a QR reader needs dark modules on a light field, which is the
    opposite of a normal dark-background terminal). Without it, the blocks are
    drawn in the terminal's own foreground colour, which only scans on a
    light-background terminal or after inverting.
    """
    rows = matrix(qr, border=border)
    if len(rows) % 2:
        rows.append([0] * len(rows[0]))
    out = []
    for i in range(0, len(rows), 2):
        top, bot = rows[i], rows[i + 1]
        line = "".join(BLOCKS[(1 if t else 0, 1 if b else 0)] for t, b in zip(top, bot))
        out.append(f"\x1b[30;47m{line}\x1b[0m" if ansi else line)
    return "\n".join(out)


def to_png(qr, path, scale=8, border=4):
    qr.save(path, scale=scale, border=border)
    return path


def describe(qr, url):
    v = qr.version
    vs = f"V{v}" if isinstance(v, int) else str(v)
    n = qr.symbol_size(border=0)[0]
    mode = "alphanumeric" + (" (mixed, '#' in byte mode)" if "#" in url else "")
    return (f"QR {vs} ({n}x{n} modules), error level {qr.error.upper()}, "
            f"{mode}, {len(url)} chars")
