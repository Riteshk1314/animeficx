"""Japanese text detection and reading (EasyOCR / CRAFT), plus optional manga-ocr."""

from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np

JP_CHAR = re.compile(r"[぀-ヿㇰ-ㇿ一-鿿ｦ-ﾟ]")
LATIN_OR_DIGIT = re.compile(r"[A-Za-z0-9]")


def japanese_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if JP_CHAR.match(c)) / len(chars)


def looks_japanese(text: str) -> bool:
    """EasyOCR's ja+en model reads logos/UI as Latin; real Japanese text comes back mostly kana/kanji."""
    if not JP_CHAR.search(text):
        return False
    return japanese_ratio(text) >= 0.5 and len(LATIN_OR_DIGIT.findall(text)) <= 1


@dataclass
class Detection:
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 (exclusive)
    text: str  # EasyOCR's reading, only used for filtering / track splitting
    conf: float

    @property
    def area(self) -> int:
        return (self.box[2] - self.box[0]) * (self.box[3] - self.box[1])


class TextDetector:
    def __init__(self, gpu: bool = True, min_height: int = 12,
                 ignore: list[tuple[int, int, int, int]] | None = None):
        import easyocr

        self.reader = easyocr.Reader(["ja", "en"], gpu=gpu, verbose=False)
        self.min_height = min_height
        self.ignore = ignore or []

    def _ignored(self, box) -> bool:
        x0, y0, x1, y1 = box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        return any(ix0 <= cx <= ix1 and iy0 <= cy <= iy1 for ix0, iy0, ix1, iy1 in self.ignore)

    def __call__(self, frame_bgr: np.ndarray) -> list[Detection]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.reader.readtext(
            rgb,
            paragraph=False,
            width_ths=0.8,       # merge spaced-out katakana like "ク マ" into one box
            text_threshold=0.6,
            low_text=0.35,
            link_threshold=0.4,
        )
        out: list[Detection] = []
        for pts, text, conf in results:
            pts = np.asarray(pts, dtype=np.float32)
            x0, y0 = np.floor(pts.min(axis=0)).astype(int)
            x1, y1 = np.ceil(pts.max(axis=0)).astype(int)
            box = (max(0, x0), max(0, y0), min(w, x1), min(h, y1))
            if box[3] - box[1] < self.min_height or box[2] - box[0] < self.min_height:
                continue
            if not looks_japanese(text) or self._ignored(box):
                continue
            out.append(Detection(box, text, float(conf)))
        return out


class TextReader:
    """manga-ocr: a transformer OCR trained on manga, robust to stylised Japanese.
    Optional (`uv sync --extra manga`); raises ImportError when not installed."""

    def __init__(self):
        from manga_ocr import MangaOcr

        self.mocr = MangaOcr()

    def __call__(self, crop_bgr: np.ndarray) -> str:
        from PIL import Image

        # manga-ocr resizes to a square input; a long single line gets squashed and loses
        # dakuten (デ -> テ). Pad to a square with the background colour instead.
        h, w = crop_bgr.shape[:2]
        s = max(h, w)
        bg = np.median(np.concatenate([crop_bgr[0], crop_bgr[-1]]), axis=0).astype(np.uint8)
        canvas = np.full((s, s, 3), bg, np.uint8)
        canvas[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = crop_bgr
        img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        return self.mocr(img).strip()
