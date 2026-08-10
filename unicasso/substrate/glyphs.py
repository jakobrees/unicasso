"""Glyph bitmaps for the glyph-VAE.

Uses the rasterizer's glyph rendering (`raster.create_char_bitmaps`, same fonts / sizes /
cell geometry) so the latent space is built over the very bitmaps the ASCII optimizer
snaps to. We work internally in **ink space** (0 = white background, 1 = ink) -- the
inverse of the rasterizer's bitmap convention (1 = white) -- because zeros are a natural
"nothing here" padding value for the conv stack. Convert back with `1 - ink` for display.
"""
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def repo_path(path):
    """Resolve a REPO-OWNED relative path (kits, fonts, bundled weights) against the
    repo root; absolute paths and None pass through.

    This replaces an import-time `os.chdir(REPO_ROOT)`. The chdir made every
    relative path in the codebase resolve from the repo, which was convenient and
    wrong: it broke `pip install` (the package chdir'd into site-packages), it
    silently redirected the CALLER's relative paths -- an output filename, an input
    image -- to somewhere they did not mean, and it made the process cwd a hidden
    global that any library in the same interpreter also felt.

    The rule now: paths the REPO owns resolve through here; paths the USER supplies
    stay relative to their own working directory, which is what they expect.
    """
    if not path or os.path.isabs(path):
        return path
    return os.path.join(REPO_ROOT, path)

from unicasso.substrate import raster as train


PROFILE_ALIASES = {
    "sfmono": "kits/sfmono/profile.json",
    "dejavu": "kits/dejavu/profile.json",
}


def active_profile_path(profile=None):
    """Resolve the active font profile to an absolute profile.json path (or None)."""
    profile = profile if profile is not None else (os.environ.get("GLYPHVAE_FONT") or "dejavu")
    return repo_path(PROFILE_ALIASES.get(profile, profile))


def kit_dir(profile=None, default="kits/sfmono"):
    """Directory of the active font kit; tools default their kit-scoped files here."""
    ppath = active_profile_path(profile)
    return os.path.dirname(ppath) if ppath else repo_path(default)


def build_charset():
    """Default charset: cp437 32..255 minus banned chars."""
    chars = "".join(bytes([i]).decode("cp437") for i in range(32, 256))
    chars = "".join(c for c in chars if c not in train.BANNED_CHARS)
    return chars


def read_charset_file(path):
    """One glyph per line, char in the first tab-separated field (font_kit format)."""
    chars = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.rstrip("\n"):
                chars.append(line.split("\t")[0])
    return "".join(chars)


def load_glyphs(device=None, pad=0, profile=None):
    """Render the charset to an ink tensor.

    Args:
        pad: white margin (px) added on every side -- int, or (pad_h, pad_w) for
             per-axis margins. Rotate/shift/scale augmentations need headroom or they
             clip the glyph; pad to a div-by-4 field (e.g. pad=4 -> 24x12 becomes
             32x20; sfmono 36x18 needs (4, 3) -> 44x24) so the conv stack still
             tiles. The margin is constant white for every glyph, so it leaves
             nearest-neighbor ranking unchanged -- free for snapping.
        profile: font profile -- an alias ("sfmono") or a profile.json path (font_kit
             output). Defaults to the GLYPHVAE_FONT env var, and to the bundled
             "dejavu" kit when that is unset.

    Returns:
        ink:   (N, 1, CHAR_H+2pad, CHAR_W+2pad) float tensor, 0 = white bg, 1 = ink
        chars: the N-character string (ink[k] is the bitmap of chars[k])
    """
    device = torch.device(device) if device is not None else train.DEVICE
    train.DEVICE = device

    profile = profile if profile is not None else (os.environ.get("GLYPHVAE_FONT") or "dejavu")
    if profile:
        import json
        ppath = repo_path(PROFILE_ALIASES.get(profile, profile))
        if not os.path.exists(ppath):
            raise RuntimeError(
                f"font kit not found at {ppath}. unicasso's kits/, fonts/ and weights/ "
                "live in the repository, not inside the installed package -- install "
                "editable from a checkout (pip install -e .) or run from the repo root.")
        with open(ppath) as f:
            prof = json.load(f)
        # profiles may store the font repo-relative (bundled libre fonts) or absolute
        # (macOS system fonts); both must work with no dependence on the cwd
        font = repo_path(prof["font"])
        if not os.path.exists(font):
            # portability: profiles store absolute font paths (macOS system fonts) that
            # don't exist off-machine; fall back to a copy bundled next to the profile
            kit = os.path.dirname(ppath)
            for cand in (os.path.join(kit, "fonts", os.path.basename(font)),
                         os.path.join(kit, os.path.basename(font))):
                if os.path.exists(cand):
                    print(f"[glyphs] profile font missing ({font}); using bundled {cand}")
                    font = cand
                    break
            else:
                raise FileNotFoundError(
                    f"profile font not found: {font} -- copy the .otf into {kit}/fonts/")
        # Fail fast if the font file is unreadable: the rasterizer's own fallback chain
        # would silently substitute PIL's default font, poisoning every downstream fit.
        try:
            from PIL import ImageFont
            ImageFont.truetype(font, 16)
        except Exception as e:
            raise RuntimeError(f"kit font failed to load: {font} ({e})")
        # One font for everything, baseline-anchored so the line box fills the cell the
        # way the terminal lays it out (overdrawing box glyphs clip to the cell = the
        # cross-cell seam behavior; blocks keep their real line-box stripe).
        train.PRINTER_FONT = font
        train.PRINTER_FONT_SIZE = prof["size"]
        train.PRINTER_Y_OFFSET = prof["baseline_y"]
        train.FALLBACK_FONTS = [font]
        train.FALLBACK_FONT_SIZE = prof["size"]
        train.FALLBACK_Y_OFFSET = prof["baseline_y"]
        train.DRAW_ANCHOR = "ls"
        train.CHAR_HEIGHT = prof["cell_h"]
        train.CHAR_WIDTH = prof["cell_w"]
        train.IMAGE_WIDTH = train.CHAR_WIDTH * train.GRID_WIDTH
        train.IMAGE_HEIGHT = (train.CHAR_HEIGHT * train.GRID_HEIGHT
                              + train.ROW_GAP * (train.GRID_HEIGHT - 1))
        train.ROW_GAP = 0
        train.BANNED_CHARS = []
        cf = prof["charset_file"]
        chars = read_charset_file(repo_path(cf))
        print(f"[glyphs] font profile '{profile}': {font} @ {prof['size']:.2f}px, "
              f"cell {train.CHAR_WIDTH}x{train.CHAR_HEIGHT}, {len(chars)} glyphs")

    train.CHARS = chars
    train.NUM_CHARS = len(chars)

    bitmaps = train.create_char_bitmaps().to(device)  # (N, H, W), 1 = white, 0 = ink
    ink = (1.0 - bitmaps).unsqueeze(1).contiguous()    # (N, 1, H, W), 1 = ink
    ph, pw = (pad, pad) if isinstance(pad, int) else pad
    if ph > 0 or pw > 0:
        import torch.nn.functional as F
        ink = F.pad(ink, (pw, pw, ph, ph), mode="constant", value=0.0)  # white margin
    return ink, chars


def ink_density(ink):
    """Mean ink per glyph in [0, 1] (1 = fully inked). ink: (N, 1, H, W) or (N, ...)."""
    flat = ink.reshape(ink.shape[0], -1)
    return flat.mean(dim=1)


def to_bitmap(ink):
    """ink (0=bg,1=ink) -> displayable bitmap (1=white,0=ink), numpy (H, W)."""
    arr = ink.detach().cpu().numpy()
    arr = np.squeeze(arr)
    return 1.0 - arr
