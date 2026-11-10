"""Link per-frame text detections into tracks, fill gaps, and stabilise boxes."""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher

import numpy as np

from .detect import Detection

Box = tuple[int, int, int, int]


def iou(a: Box, b: Box) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if inter == 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua


def text_sim(a: str, b: str) -> float:
    if not a or not b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


@dataclass
class Track:
    id: int
    shot: int
    boxes: dict[int, Box] = field(default_factory=dict)       # frames the text was seen in
    texts: list[str] = field(default_factory=list)            # EasyOCR readings
    confs: list[float] = field(default_factory=list)          # EasyOCR recognition confidence
    samples: list[tuple[int, np.ndarray]] = field(default_factory=list)  # (area, crop) best crops for OCR

    @property
    def last_frame(self) -> int:
        return max(self.boxes)

    @property
    def last_box(self) -> Box:
        return self.boxes[self.last_frame]

    @property
    def last_text(self) -> str:
        return self.texts[-1] if self.texts else ""

    def add(self, frame_idx: int, det: Detection, frame: np.ndarray, keep: int = 5) -> None:
        self.boxes[frame_idx] = det.box
        self.texts.append(det.text)
        self.confs.append(det.conf)
        x0, y0, x1, y1 = det.box
        pad = max(2, int(0.15 * (y1 - y0)))  # manga-ocr reads better with a little margin
        H, W = frame.shape[:2]
        x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
        # keep the `keep` largest crops (most fully visible text) for OCR voting
        if len(self.samples) < keep or det.area > self.samples[-1][0]:
            self.samples.append((det.area, frame[y0:y1, x0:x1].copy()))
            self.samples.sort(key=lambda s: -s[0])
            del self.samples[keep:]


class ShotTracker:
    """Greedy IoU + text-similarity tracker, reset at every shot boundary."""

    def __init__(self, shot: int, next_id: int, max_gap: int = 4,
                 min_iou: float = 0.3, min_text_sim: float = 0.35):
        self.shot = shot
        self.next_id = next_id
        self.max_gap = max_gap
        self.min_iou = min_iou
        self.min_text_sim = min_text_sim
        self.active: list[Track] = []
        self.done: list[Track] = []

    def update(self, frame_idx: int, dets: list[Detection], frame: np.ndarray) -> None:
        # retire stale tracks
        still = []
        for t in self.active:
            (still if frame_idx - t.last_frame <= self.max_gap else self.done).append(t)
        self.active = still

        pairs = []
        for di, d in enumerate(dets):
            for ti, t in enumerate(self.active):
                ov = iou(d.box, t.last_box)
                if ov >= self.min_iou and text_sim(d.text, t.last_text) >= self.min_text_sim:
                    pairs.append((ov, di, ti))
        pairs.sort(reverse=True)
        used_d, used_t = set(), set()
        for _, di, ti in pairs:
            if di in used_d or ti in used_t:
                continue
            used_d.add(di)
            used_t.add(ti)
            self.active[ti].add(frame_idx, dets[di], frame)
        for di, d in enumerate(dets):
            if di not in used_d:
                t = Track(self.next_id, self.shot)
                self.next_id += 1
                t.add(frame_idx, d, frame)
                self.active.append(t)

    def hold(self, frame_idx: int, prev_idx: int) -> None:
        """`frame_idx` is a pixel copy of `prev_idx`: whatever was seen there is still there,
        in the same place. Extends those tracks without running the detector."""
        for t in self.active:
            if t.last_frame == prev_idx:
                t.boxes[frame_idx] = t.boxes[prev_idx]

    def finish(self) -> list[Track]:
        self.done.extend(self.active)
        self.active = []
        return self.done


def densify(boxes: dict[int, Box]) -> dict[int, Box]:
    """Linearly interpolate boxes for frames between detections (missed detections / stride)."""
    frames = sorted(boxes)
    out: dict[int, Box] = {}
    for a, b in zip(frames, frames[1:]):
        ba, bb = np.array(boxes[a], float), np.array(boxes[b], float)
        for f in range(a, b):
            t = (f - a) / (b - a)
            out[f] = tuple(int(round(v)) for v in ba * (1 - t) + bb * t)
    out[frames[-1]] = boxes[frames[-1]]
    return out


def stabilise(boxes: dict[int, Box]) -> dict[int, Box]:
    """Group frames where the box barely moves into runs and give each run one union box.

    Static text then gets one rock-steady box (no jittering English, and the
    renderer can reuse a single clean plate). Moving text keeps per-frame boxes.
    """
    frames = sorted(boxes)
    out: dict[int, Box] = {}
    run: list[int] = []
    anchor: Box | None = None

    def flush():
        if not run:
            return
        arr = np.array([boxes[f] for f in run])
        ub = (int(arr[:, 0].min()), int(arr[:, 1].min()), int(arr[:, 2].max()), int(arr[:, 3].max()))
        for f in run:
            out[f] = ub

    for f in frames:
        b = boxes[f]
        if anchor is not None:
            tol = max(3, int(0.08 * (anchor[3] - anchor[1])))
            if all(abs(p - q) <= tol for p, q in zip(b, anchor)):
                run.append(f)
                continue
        flush()
        run, anchor = [f], b
    flush()
    return out
