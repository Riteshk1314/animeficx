"""Text masks, style estimation (fill / outline colour) and English layout."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\comicbd.ttf",            # cartoon-friendly, ships with Windows
    r"C:\Windows\Fonts\arialbd.ttf",
    "/System/Library/Fonts/Supplemental/Comic Sans MS Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def default_font() -> str:
    for f in FONT_CANDIDATES:
        if os.path.exists(f):
            return f
    raise FileNotFoundError("no default font found; pass --font path/to/font.ttf")


def _odd(n: int) -> int:
    n = max(3, int(n))
    return n if n % 2 else n + 1


def _color_dist(img: np.ndarray, bgr) -> np.ndarray:
    return np.abs(img.astype(np.int16) - np.asarray(bgr, np.int16)).max(axis=2)


def stroke_mask(crop: np.ndarray, box: tuple[int, int, int, int], fill_bgr=None,
                diff_thresh: int = 38, color_tol: int = 70, grow: int = 0) -> np.ndarray:
    """Pixel mask of the text strokes (plus outline) inside `box` (crop coordinates).

    The local background is estimated with a large median filter, which wipes out
    thin strokes but keeps flat anime colours and gradients. Pixels that differ from
    it, and (when known) are close to the text's fill colour, are text. Restricting
    to the fill colour keeps overlapping logos/characters out of the mask. The result
    is dilated to swallow outlines and anti-aliasing (`grow` px more when residue was found).
    """
    x0, y0, x1, y1 = box
    bh = max(1, y1 - y0)
    k = _odd(min(255, bh * 0.9))
    bg = cv2.medianBlur(crop, k)
    diff = cv2.absdiff(crop, bg).max(axis=2)
    m = np.zeros(diff.shape, np.uint8)
    core = diff[y0:y1, x0:x1] > diff_thresh
    if fill_bgr is not None:
        by_color = core & (_color_dist(crop[y0:y1, x0:x1], fill_bgr) < color_tol)
        if by_color.sum() > 0.25 * core.sum():
            core = by_color
    m[y0:y1, x0:x1] = core.astype(np.uint8) * 255

    area = (x1 - x0) * (y1 - y0)
    cover = (m[y0:y1, x0:x1] > 0).sum() / max(1, area)
    if cover < 0.02 or cover > 0.85:  # estimate failed: fall back to the whole box
        m[:] = 0
        m[y0:y1, x0:x1] = 255
    d = max(3, int(round(bh * 0.10))) + max(0, grow)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1)))
    return m


@dataclass
class Style:
    fill: tuple[int, int, int]           # RGB
    outline: tuple[int, int, int] | None  # RGB, or None for no outline

    @property
    def fill_bgr(self) -> tuple[int, int, int]:
        return self.fill[::-1]


def _lum(c) -> float:
    r, g, b = c
    return 0.299 * r + 0.587 * g + 0.114 * b


def estimate_style(crop: np.ndarray, box: tuple[int, int, int, int]) -> Style:
    x0, y0, x1, y1 = box
    region = crop[y0:y1, x0:x1]
    bh = max(1, y1 - y0)
    bg = cv2.medianBlur(crop, _odd(min(255, bh * 0.9)))[y0:y1, x0:x1]
    diff = cv2.absdiff(region, bg).max(axis=2).astype(np.float32)
    strong = diff > 40
    if strong.sum() < 30:
        return Style(fill=(255, 255, 255), outline=(0, 0, 0))

    # Cluster the high-contrast pixels; the text fill is the cluster with the most
    # contrast mass (count x contrast). Other clusters are outline, anti-aliasing,
    # or whatever overlaps the box (a logo, a character's hair...).
    px = region[strong].astype(np.float32)
    k = min(4, len(px))
    _, labels, centers = cv2.kmeans(px, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
                                    3, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    dvals = diff[strong]
    score = [(labels == i).sum() * dvals[labels == i].mean() if (labels == i).any() else 0 for i in range(k)]
    fill_bgr = centers[int(np.argmax(score))]
    fill = tuple(int(v) for v in fill_bgr[::-1])
    bg_rgb = np.median(bg.reshape(-1, 3), axis=0)[::-1]

    # Look at a thin ring just outside the strokes: a clearly different colour there
    # (from both the fill and the background) is an outline.
    core = (_color_dist(region, fill_bgr) < 50) & strong
    core_u8 = core.astype(np.uint8) * 255
    r = max(3, int(bh * 0.02))  # outlines are thin: look only a few px outside the strokes
    ring = cv2.dilate(core_u8, np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0
    ring &= ~(cv2.dilate(core_u8, np.ones((3, 3), np.uint8)) > 0)
    candidate = (255, 255, 255) if _lum(fill) < 128 else (0, 0, 0)
    outline = None
    if ring.sum() > 50:
        ring_rgb = np.median(region[ring], axis=0)[::-1]
        if np.abs(ring_rgb - np.array(fill)).max() > 60 and np.abs(ring_rgb - bg_rgb).max() > 20:
            outline = tuple(int(v) for v in ring_rgb)
    if outline is None and abs(_lum(fill) - _lum(bg_rgb)) < 70:
        outline = candidate  # low contrast against background: add one so it stays legible
    return Style(fill=fill, outline=outline)


@lru_cache(maxsize=64)
def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def _wrap(text: str, font, max_w: float) -> list[str] | None:
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if font.getlength(trial) <= max_w:
            cur = trial
        else:
            if not cur:
                return None  # single word too wide at this size
            lines.append(cur)
            cur = w
            if font.getlength(cur) > max_w:
                return None
    if cur:
        lines.append(cur)
    return lines


def _layout(text: str, box_w: int, box_h: int, style: Style, font_path: str,
            width_slack: float, max_lines: int, max_size: int | None):
    max_w, max_h = box_w * width_slack, box_h * 1.05
    top = max(8, int(box_h * 1.1))
    if max_size:
        top = min(top, max_size)
    for size in range(top, 7, -1):
        font = _font(font_path, size)
        ow = max(1, round(size * 0.06)) if style.outline else 0
        lines = _wrap(text, font, max_w - 2 * ow)
        if not lines or len(lines) > max_lines:
            continue
        ascent, descent = font.getmetrics()
        line_h = int((ascent + descent) * 0.92)
        if line_h * len(lines) + 2 * ow <= max_h:
            return font, lines, line_h, ow
    font = _font(font_path, 8)
    return font, [text], sum(font.getmetrics()), 0


def fit_size(text: str, box_w: int, box_h: int, style: Style, font_path: str,
             width_slack: float = 1.0, max_lines: int = 3) -> int:
    """Largest font size at which `text` fits the box."""
    return _layout(text, box_w, box_h, style, font_path, width_slack, max_lines, None)[0].size


def render_text(text: str, box_w: int, box_h: int, style: Style, font_path: str,
                width_slack: float = 1.0, max_lines: int = 3, max_size: int | None = None) -> Image.Image:
    """Render `text` as large as possible (up to `max_size`) inside (box_w x box_h). Returns RGBA."""
    font, lines, line_h, ow = _layout(text, box_w, box_h, style, font_path, width_slack, max_lines, max_size)
    return _draw(font, lines, line_h, ow, box_w, box_h, style)


def _draw(font, lines, line_h, ow, box_w, box_h, style: Style) -> Image.Image:
    W = int(max(box_w, max(font.getlength(l) for l in lines) + 2 * ow + 4))
    H = int(max(box_h, line_h * len(lines) + 2 * ow + 4))
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    top = (H - line_h * len(lines)) / 2
    for i, line in enumerate(lines):
        draw.text((W / 2, top + line_h * (i + 0.5)), line, font=font, anchor="mm",
                  fill=style.fill + (255,),
                  stroke_width=ow, stroke_fill=(style.outline + (255,)) if style.outline else None)
    return img


def paste_rgba(frame: np.ndarray, rgba: Image.Image, cx: float, cy: float) -> None:
    """Alpha-composite an RGBA PIL image onto a BGR frame, centred at (cx, cy). In place."""
    arr = np.asarray(rgba)
    h, w = arr.shape[:2]
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    fx0, fy0 = max(0, x0), max(0, y0)
    fx1, fy1 = min(frame.shape[1], x0 + w), min(frame.shape[0], y0 + h)
    if fx1 <= fx0 or fy1 <= fy0:
        return
    src = arr[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]
    a = src[..., 3:4].astype(np.float32) / 255
    rgb_as_bgr = src[..., 2::-1].astype(np.float32)
    dst = frame[fy0:fy1, fx0:fx1].astype(np.float32)
    frame[fy0:fy1, fx0:fx1] = (dst * (1 - a) + rgb_as_bgr * a).round().astype(np.uint8)
