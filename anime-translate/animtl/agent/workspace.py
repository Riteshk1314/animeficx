"""Shared state for the agents: the video, regions.json, and image helpers.

regions.json works as a blackboard: perception writes tracks into it, each agent reads it,
adds its decision (with the reason) and saves it, and the renderer draws whatever is there.
Agents never touch pixels themselves; they can only change these structured fields:
`ja` (reading), `en` (English), `style` (mask growth / colours) and `agent` (notes).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from ..editing.render import _full_box, read_frames
from ..editing.typeset import Style, default_font, fit_size
from ..video import probe


class Workspace:
    def __init__(self, video: str | Path, work: Path):
        self.video, self.work = Path(video), Path(work)
        self.info = probe(video)
        self.W, self.H = self.info.width, self.info.height
        self.font = default_font()
        self.load()

    # ---- regions.json -------------------------------------------------------------
    def load(self) -> None:
        self.data = json.loads((self.work / "regions.json").read_text(encoding="utf-8"))
        self.tracks = {t["id"]: t for t in self.data["tracks"]}

    def save(self) -> None:
        (self.work / "regions.json").write_text(json.dumps(self.data, ensure_ascii=False, indent=1),
                                                encoding="utf-8")

    def note(self, tid: int, key: str, value) -> None:
        self.tracks[tid].setdefault("agent", {})[key] = value

    # ---- geometry -----------------------------------------------------------------
    def mid(self, tid: int) -> int:
        t = self.tracks[tid]
        return (t["first"] + t["last"]) // 2

    def box(self, tid: int, frame: int | None = None) -> tuple:
        t = self.tracks[tid]
        f = self.mid(tid) if frame is None else frame
        return tuple(t["boxes"][str(f)])

    def text_box(self, tid: int) -> tuple:
        """Box of the whole text (for cut-off text, as it would be if fully on screen)."""
        return _full_box(self.box(tid), self.tracks[tid].get("full_box"), self.W, self.H)

    def shots_with_text(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for tid, t in sorted(self.tracks.items(), key=lambda kv: (kv[1]["shot"], kv[1]["first"])):
            out.setdefault(t["shot"], []).append(tid)
        return out

    def shot_mid(self, shot: int) -> int:
        tids = self.shots_with_text()[shot]
        # the frame where the most of this shot's text is on screen
        frames = [f for tid in tids for f in map(int, self.tracks[tid]["boxes"])]
        return max(set(frames), key=lambda f: (sum(str(f) in self.tracks[t]["boxes"] for t in tids), -abs(
            f - (min(frames) + max(frames)) // 2)))

    # ---- images -------------------------------------------------------------------
    def frames(self, wanted: set[int]) -> dict[int, np.ndarray]:
        return read_frames(self.video, set(wanted))

    @staticmethod
    def crop(frame: np.ndarray, box: tuple, pad: float = 0.35, min_h: int = 96) -> np.ndarray:
        """Crop around a box with some context, upscaled so small text has enough pixels."""
        x0, y0, x1, y1 = box
        h = y1 - y0
        p = int(max(6, pad * h))
        H, W = frame.shape[:2]
        c = frame[max(0, y0 - p):min(H, y1 + p), max(0, x0 - p):min(W, x1 + p)]
        if c.shape[0] < min_h:
            s = min_h / c.shape[0]
            c = cv2.resize(c, (int(c.shape[1] * s), min_h), interpolation=cv2.INTER_CUBIC)
        return c

    @staticmethod
    def stack(a: np.ndarray, b: np.ndarray, gap: int = 8) -> np.ndarray:
        """Before (top) / after (bottom) with a white gap."""
        w = max(a.shape[1], b.shape[1])
        pad = lambda im: np.pad(im, ((0, 0), (0, w - im.shape[1]), (0, 0)), constant_values=255)
        return np.vstack([pad(a), np.full((gap, w, 3), 255, np.uint8), pad(b)])

    # ---- layout check used by the translator ----------------------------------------
    def fit(self, tid: int, english: str) -> dict:
        """How large `english` would be drawn in this track's box, vs the Japanese it replaces."""
        x0, y0, x1, y1 = self.text_box(tid)
        bw, bh = x1 - x0, y1 - y0
        size = fit_size(english, bw, bh, Style((0, 0, 0), (255, 255, 255)), self.font)
        ratio = size / max(1, bh)
        ok = ratio >= 0.5 and size >= 14
        return {"fits": ok, "font_px": int(size), "japanese_px": int(bh), "box_width_px": int(bw),
                "advice": "ok" if ok else
                f"too long: it would be drawn at {size}px, under half the {bh}px original. Use fewer letters."}
