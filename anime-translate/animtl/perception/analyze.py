"""Stage 1: find every piece of Japanese text, track it over time, and read it."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from math import ceil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .detect import TextDetector, TextReader
from .overlay import compute_overlay
from .track import ShotTracker, Track, densify, iou, stabilise, text_sim
from ..video import detect_shots, frames_differ, iter_frames, probe


def _vote(readings: list[str]) -> str:
    counts = Counter(r for r in readings if r)
    if not counts:
        return ""
    best = max(counts.values())
    return max((r for r, c in counts.items() if c == best), key=len)


def _weighted_readings(texts: list[str], confs: list[float]) -> list[str]:
    """EasyOCR's per-frame readings of one track, best first (summed confidence)."""
    score: dict[str, float] = defaultdict(float)
    for t, c in zip(texts, confs):
        if t:
            score[t] += c
    return sorted(score, key=lambda t: -score[t])


def _manga_reader():
    try:
        return TextReader()
    except ImportError:
        return None


def _drop_watermarks(tracks: list[Track], min_shots: int = 3) -> list[Track]:
    """Text sitting at the same spot across many different shots is a channel logo, not story text."""
    keep = []
    for t in tracks:
        shots = {o.shot for o in tracks
                 if iou(o.last_box, t.last_box) > 0.6 and text_sim(o.last_text, t.last_text) > 0.5}
        if len(shots) < min_shots:
            keep.append(t)
    return keep


def _touches_edge(box, W, H, m: int = 2) -> bool:
    return box[0] <= m or box[1] <= m or box[2] >= W - m or box[3] >= H - m


def _resolve(tracks: list[dict], W: int, H: int, min_conf: float) -> list[dict]:
    """Drop low-confidence noise; map text cut off by the frame edge to its full reading.

    EasyOCR's Japanese model happily 'reads' foliage and logos, but with near-zero
    confidence. Real text that is only partly in frame (a scrolling list) also scores
    low, so it is kept when it is part of a confident reading in the same shot.
    """
    confident = [t for t in tracks if t["conf"] >= min_conf]
    out = []
    for t in tracks:
        edge = any(_touches_edge(b, W, H) for b in t["boxes"].values())
        parent = None
        if edge:
            reads = {r for r in (t["ja"], t["easyocr"], *t["ja_readings"]) if r}
            # a partly visible word is usually a piece of the full one, give or take a misread
            # character at the cut (ペンダ for the top of パンダ): longest common run of characters
            overlap = lambda r, p: SequenceMatcher(None, r, p).find_longest_match().size
            cands = []
            for p in confident:
                if p is t or p["shot"] != t["shot"] or not p["ja"]:
                    continue
                hit = max((n for r in reads if (n := overlap(r, p["ja"])) >= max(1, ceil(0.6 * len(r)))),
                          default=0)
                if hit:
                    cands.append((hit, -abs(p["first"] - t["first"]), p))
            if cands:
                parent = max(cands, key=lambda c: c[:2])[2]
        if parent is not None:
            pboxes = np.array(list(parent["boxes"].values()))
            t["ja"] = parent["ja"]
            t["full_box"] = [int(v) for v in np.median(pboxes, axis=0)]  # where the whole text sits when visible
            out.append(t)
        elif t["conf"] >= min_conf:
            out.append(t)
    return out


def analyze(video: str | Path, work: Path, probe_stride: int = 10, det_stride: int = 1,
            min_dets: int = 3, min_conf: float = 0.3, ignore=None, gpu: bool = True,
            manga: bool = True, max_hold: int = 30) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    info = probe(video)
    print(f"[analyze] {info.width}x{info.height} @ {info.fps:.2f} fps, {info.n_frames} frames")

    shots = detect_shots(video, info.n_frames)
    shot_of = np.zeros(max(info.n_frames, shots[-1][1]), dtype=np.int32)
    for i, (s, e) in enumerate(shots):
        shot_of[s:e] = i
    print(f"[analyze] {len(shots)} shots")

    overlay = compute_overlay(video, shots)
    if overlay is not None:
        cv2.imwrite(str(work / "overlay.png"), overlay)
        print(f"[analyze] persistent overlay (logos) covers {100 * (overlay > 0).mean():.1f}% of frame "
              f"-> {work / 'overlay.png'} (protected from editing)")

    detector = TextDetector(gpu=gpu, ignore=ignore)

    # Pass A: sparse probe, to skip shots that contain no Japanese text at all.
    text_shots: set[int] = set()
    probe_wanted = lambda i: i < len(shot_of) and (i - shots[shot_of[i]][0]) % probe_stride == 0
    for idx, frame in tqdm(iter_frames(video, probe_wanted), total=info.n_frames, desc="probe", unit="f"):
        if frame is not None and shot_of[idx] not in text_shots and detector(frame):
            text_shots.add(int(shot_of[idx]))
    print(f"[analyze] shots with text: {sorted(text_shots)}")

    # Pass B: dense detection + tracking inside text shots only.
    dense_wanted = lambda i: (i < len(shot_of) and int(shot_of[i]) in text_shots
                              and (i - shots[shot_of[i]][0]) % det_stride == 0)
    # Most frames are copies of the previous one (held drawings, still title cards and UI),
    # so the detector only runs when the picture changed since it last ran; on copies the
    # tracks are simply extended. `max_hold` forces a fresh look now and then regardless.
    tracks: list[Track] = []
    tracker: ShotTracker | None = None
    next_id = 0
    last_seen = last_idx = None
    n_run = n_held = 0
    for idx, frame in tqdm(iter_frames(video, dense_wanted), total=info.n_frames, desc="detect", unit="f"):
        if frame is None:
            continue
        shot = int(shot_of[idx])
        if tracker is None or tracker.shot != shot:
            if tracker is not None:
                tracks += tracker.finish()
                next_id = tracker.next_id
            tracker = ShotTracker(shot, next_id, max_gap=4 * det_stride)
            last_seen = None
        if last_seen is not None and idx - last_seen[0] < max_hold and not frames_differ(frame, last_seen[1]):
            tracker.hold(idx, last_idx)
            n_held += 1
        else:
            tracker.update(idx, detector(frame), frame)
            last_seen = (idx, frame)
            n_run += 1
        last_idx = idx
    if tracker is not None:
        tracks += tracker.finish()

    del detector
    print(f"[analyze] detector ran on {n_run} frames, {n_held} unchanged frames reused its result")
    tracks = [t for t in tracks if len(t.boxes) >= min_dets]
    tracks = _drop_watermarks(tracks)

    # Reading: EasyOCR's own per-frame readings, voted. When EasyOCR is unsure (< 0.5) and
    # manga-ocr is installed, manga-ocr reads the 5 largest crops instead (it is better on
    # stylised lettering). It is only loaded if some track actually needs it.
    kept = []
    for t in tracks:
        conf = float(np.median(t.confs))
        if conf < min_conf and not any(_touches_edge(b, info.width, info.height) for b in t.boxes.values()):
            continue  # noise; only edge-clipped text gets a second chance in _resolve
        kept.append((t, conf))
    reader = _manga_reader() if manga and any(conf < 0.5 for _, conf in kept) else None
    if manga and reader is None and any(conf < 0.5 for _, conf in kept):
        print("[analyze] manga-ocr not installed; using EasyOCR readings for low-confidence text")

    out_tracks = []
    for t, conf in tqdm(kept, desc="read", unit="track"):
        easy = _weighted_readings(t.texts, t.confs)
        readings = [reader(crop) for _, crop in t.samples] if reader and conf < 0.5 else []
        boxes = stabilise(densify(t.boxes))
        out_tracks.append({
            "id": t.id,
            "shot": t.shot,
            "ja": easy[0] if conf >= 0.5 or not readings else _vote(readings),
            "ja_readings": (readings or easy[1:])[:5],
            "easyocr": easy[0],
            "conf": round(conf, 3),
            "n_dets": len(t.confs),
            "en": None,
            "first": min(boxes),
            "last": max(boxes),
            "boxes": {str(f): list(b) for f, b in sorted(boxes.items())},
        })
    del reader
    out_tracks = _resolve(out_tracks, info.width, info.height, min_conf)
    print(f"[analyze] kept {len(out_tracks)}/{len(tracks)} tracks after confidence filtering")
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    data = {
        "video": str(Path(video).resolve()),
        "width": info.width, "height": info.height, "fps": info.fps, "n_frames": info.n_frames,
        "shots": shots,
        "tracks": out_tracks,
    }
    (work / "regions.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[analyze] {len(out_tracks)} text tracks -> {work / 'regions.json'}")
    for t in out_tracks:
        print(f"  #{t['id']:<3} shot {t['shot']:<2} frames {t['first']}-{t['last']}  {t['ja']}")
    return data
