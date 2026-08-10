"""Rasterization substrate: the glyph-bitmap LUT and target-image loading.

Configuration is module-global by design: `substrate.glyphs` writes the active
font kit's geometry and font selection into this module, and every consumer
(engine, adapter, curation tools) reads the same view. The globals below hold
placeholder values until a kit is loaded.

This module is a from-scratch reimplementation of the project's original
rasterizer interface, written against a behavioral specification and verified
by parity tests (byte-identical bitmap LUTs and image tensors for the shipped
kits). It contains no code from prior implementations.
"""
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

# ---------------------------------------------------------------- geometry
# All kit-dependent; substrate.glyphs.load_glyphs() overwrites these.
CHAR_WIDTH = 16
CHAR_HEIGHT = 34
GRID_WIDTH = 60
GRID_HEIGHT = 30
ROW_GAP = 0
IMAGE_WIDTH = CHAR_WIDTH * GRID_WIDTH
IMAGE_HEIGHT = CHAR_HEIGHT * GRID_HEIGHT + ROW_GAP * (GRID_HEIGHT - 1)

# ---------------------------------------------------------------- charset + fonts
CHARS = ""
NUM_CHARS = 0
BANNED_CHARS = []
# 7-bit ASCII renders with the "printer" font; everything else with the first
# loadable fallback. Kit profiles point both at the same file.
PRINTER_FONT = None
PRINTER_FONT_SIZE = 24
PRINTER_Y_OFFSET = 0
FALLBACK_FONTS = []
FALLBACK_FONT_SIZE = 24
FALLBACK_Y_OFFSET = 0
DRAW_ANCHOR = None          # e.g. "ls" = left/baseline (kit profiles use this)

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")


def _load_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return None          # unreadable/missing file: caller decides on fallback
    except TypeError as e:
        # older Pillow rejects fractional sizes; rounding would break the kit's
        # exact-advance calibration, so refuse instead of degrading silently
        raise RuntimeError(
            f"Pillow too old for fractional font sizes ({e}); pip install -U 'pillow>=10'")


def create_char_bitmaps():
    """Render CHARS into the glyph LUT: (N, CHAR_HEIGHT, CHAR_WIDTH) float32 on
    DEVICE, antialiased, 1.0 = white background, 0.0 = full ink.

    Codepoints < 127 use PRINTER_FONT at PRINTER_Y_OFFSET; the rest use the
    first loadable FALLBACK_FONTS entry at FALLBACK_Y_OFFSET. The y offset is
    interpreted through DRAW_ANCHOR (kit profiles anchor "ls", so it is the
    baseline row within the cell). Banned or unrenderable chars come out blank.
    """
    printer = _load_font(PRINTER_FONT, PRINTER_FONT_SIZE) if PRINTER_FONT else None
    if printer:
        print(f"[raster] printer font {PRINTER_FONT} @ {PRINTER_FONT_SIZE}pt")
    fallback = None
    for path in FALLBACK_FONTS:
        fallback = _load_font(path, FALLBACK_FONT_SIZE)
        if fallback:
            print(f"[raster] fallback font {path} @ {FALLBACK_FONT_SIZE}pt")
            break
    if printer is None and fallback is None:
        raise RuntimeError("no usable font: neither PRINTER_FONT nor any FALLBACK_FONTS loaded")

    cells = np.empty((len(CHARS), CHAR_HEIGHT, CHAR_WIDTH), dtype=np.uint8)
    n_printer = 0
    for i, ch in enumerate(CHARS):
        use_printer = printer is not None and ord(ch) < 127
        font = printer if use_printer else (fallback or printer)
        y = PRINTER_Y_OFFSET if use_printer else FALLBACK_Y_OFFSET
        n_printer += use_printer
        cell = Image.new("L", (CHAR_WIDTH, CHAR_HEIGHT), 255)
        if ch not in BANNED_CHARS:
            draw = ImageDraw.Draw(cell)
            if DRAW_ANCHOR:
                draw.text((0, y), ch, font=font, fill=0, anchor=DRAW_ANCHOR)
            else:
                draw.text((0, y), ch, font=font, fill=0)
        cells[i] = np.asarray(cell, dtype=np.uint8)
    print(f"Character bitmaps shape: torch.Size([{len(CHARS)}, {CHAR_HEIGHT}, {CHAR_WIDTH}])")
    print(f"Using printer font: {n_printer} chars, fallback font: {len(CHARS) - n_printer} chars")
    return (torch.from_numpy(cells).to(DEVICE).float() / 255.0)


def load_target_image(image_path, keep_rgb=False):
    """Load an image as the optimization target at the current grid geometry.

    Returns (IMAGE_HEIGHT, IMAGE_WIDTH) float32 on DEVICE in [0, 1] with
    white = 1.0; with keep_rgb=True, (IMAGE_HEIGHT, IMAGE_WIDTH, 3) RGB.
    EXIF orientation is applied so photos come in upright.
    """
    with Image.open(image_path) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB" if keep_rgb else "L")
        im = im.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.LANCZOS)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).to(DEVICE)


def native_size(image_path):
    """Return (width, height) of an image after applying EXIF orientation (upright),
    matching how load_target_image() ingests it."""
    with Image.open(image_path) as im:
        im = ImageOps.exif_transpose(im)
        return im.size  # (w, h)


def grid_height_for_aspect(image_w, image_h, grid_w, char_w, char_h, row_gap):
    """Choose the number of character rows so the rendered output preserves the source
    image's aspect ratio, given the (non-square) character-cell geometry.

    A cell occupies char_w px wide and (char_h + row_gap) px tall on the physical output,
    so a square image needs grid_h < grid_w (e.g. 12x30 cells -> grid_w ~2.5x grid_h).

    rendered_w / rendered_h == image_w / image_h
      where rendered_w = grid_w * char_w, rendered_h = grid_h * (char_h + row_gap)
    """
    cell_h = char_h + row_gap
    gh = round(grid_w * char_w * image_h / (cell_h * image_w))
    return max(1, int(gh))
