"""Find persistent overlays (channel logos, watermarks, 'NEW EPISODES' bugs).

They sit at exactly the same pixels in every shot while the picture under them
changes, so comparing one frame per shot isolates them. Those pixels are then
protected: never inpainted, and composited back on top of the English text.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..video import iter_frames


def compute_overlay(video: str | Path, shots: list[tuple[int, int]], tol: int = 32,
                    min_shots: int = 3) -> np.ndarray | None:
    picks = {(s + e) // 2 for s, e in shots if e - s >= 3}
    if len(picks) < min_shots:
        return None
    frames = []
    last = max(picks)
    for idx, frame in iter_frames(video, lambda i: i in picks):
        if frame is not None:
            frames.append(frame.astype(np.int16))
        if idx >= last:
            break
    stack = np.stack(frames)                        # (n, H, W, 3)
    spread = (stack.max(axis=0) - stack.min(axis=0)).max(axis=2)
    mask = (spread <= tol).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # drop speckles that matched by coincidence; logos are solid blobs
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 150:
            keep[lab == i] = 255
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return cv2.dilate(keep, np.ones((3, 3), np.uint8))
