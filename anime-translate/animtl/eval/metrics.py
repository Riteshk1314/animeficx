"""Objective metrics for a translated video. No reference translation needed.

  text_removal     Japanese detections (OCR, conf >= 0.3) in sampled frames: input vs output.
                   Did the Japanese actually go away?
  fidelity         PSNR between input and output *outside* the regions the renderer may
                   edit. Measures collateral damage; only the lossy re-encode should show up.
  edited_area      Share of the frame that was changed, on frames that have text.
  stability        Frame-to-frame change inside edited boxes while the source is static.
                   Flicker from per-frame inpainting shows up here; 0 means rock steady.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from ..editing.render import _clip, _crop_rect, _full_box
from ..perception.detect import looks_japanese
from ..video import frames_differ


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0:
        return float("inf")
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def evaluate(video: str | Path, output: str | Path, work: Path, sample_every: int = 6,
             detector=None) -> dict:
    data = json.loads((Path(work) / "regions.json").read_text(encoding="utf-8"))
    W, H = data["width"], data["height"]
    tracks = [t for t in data["tracks"] if t.get("en")]
    boxes: dict[int, list] = {}
    for t in tracks:
        for f, b in t["boxes"].items():
            boxes.setdefault(int(f), []).append(_clip(_full_box(tuple(b), t.get("full_box"), W, H), W, H))
    text_frames = sorted(boxes)
    sampled = set(text_frames[::sample_every])

    if detector is None:
        from ..perception.detect import TextDetector
        detector = TextDetector()
    # Detections on the protected logo overlay are not text we could or should remove (OCR
    # sometimes reads a stylised logo as kana), so they don't count.
    ov = cv2.imread(str(Path(work) / "overlay.png"), cv2.IMREAD_GRAYSCALE)
    ov = (ov > 0) if ov is not None and ov.shape == (H, W) else np.zeros((H, W), bool)

    def on_logo(box) -> bool:
        x0, y0, x1, y1 = box
        return ov[y0:y1, x0:x1].mean() > 0.2 if (x1 > x0 and y1 > y0) else False

    jp = lambda fr: sum(1 for d in detector(fr)
                        if d.conf >= 0.3 and looks_japanese(d.text) and not on_logo(d.box))

    cin, cout = cv2.VideoCapture(str(video)), cv2.VideoCapture(str(output))
    n_in = n_out = 0
    psnr_out, edited, flicker_out, flicker_in = [], [], [], []
    prev = None
    idx = 0
    while True:
        ok1, a = cin.read()
        ok2, b = cout.read()
        if not (ok1 and ok2):
            break
        bx = boxes.get(idx)
        if bx:
            region = np.zeros((H, W), bool)
            for box in bx:
                x0, y0, x1, y1 = _crop_rect(box, W, H)
                region[y0:y1, x0:x1] = True
            psnr_out.append(_psnr(a[~region], b[~region]))
            changed = cv2.absdiff(a, b).max(axis=2) > 8
            edited.append(float((changed & region).mean()))
            if idx in sampled:
                n_in += jp(a)
                n_out += jp(b)
            if prev is not None and prev[0] == idx - 1 and prev[3] == bx:
                for box in bx:
                    x0, y0, x1, y1 = box
                    if not frames_differ(a[y0:y1, x0:x1], prev[1][y0:y1, x0:x1], min_px=2):
                        flicker_out.append(float(cv2.absdiff(b[y0:y1, x0:x1], prev[2][y0:y1, x0:x1]).mean()))
                        flicker_in.append(float(cv2.absdiff(a[y0:y1, x0:x1], prev[1][y0:y1, x0:x1]).mean()))
            prev = (idx, a, b, bx)
        idx += 1

    return {
        "text_removal": {"sampled_frames": len(sampled), "japanese_in_input": n_in,
                         "japanese_in_output": n_out,
                         "removed_pct": round(100 * (1 - n_out / n_in), 1) if n_in else None},
        "fidelity": {"psnr_outside_edits_db": round(float(np.mean(psnr_out)), 2) if psnr_out else None,
                     "note": "re-encode only; higher is better, > 40 dB is visually identical"},
        "edited_area_pct": round(100 * float(np.mean(edited)), 2) if edited else 0.0,
        "stability": {"static_pairs": len(flicker_out),
                      "mean_abs_change_output": round(float(np.mean(flicker_out)), 3) if flicker_out else None,
                      "mean_abs_change_input": round(float(np.mean(flicker_in)), 3) if flicker_in else None},
    }
