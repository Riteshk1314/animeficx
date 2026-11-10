"""Video I/O: probing, sequential frame reading, shot detection, ffmpeg encoding."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    n_frames: int


def probe(path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    info = VideoInfo(
        path=str(path),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=float(cap.get(cv2.CAP_PROP_FPS)) or 30.0,
        n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    cap.release()
    return info


def iter_frames(path: str | Path, wanted=None) -> Iterator[tuple[int, np.ndarray | None]]:
    """Yield (index, BGR frame) for every frame in order.

    If `wanted(idx) -> bool` is given, frames it rejects are skipped without
    pixel conversion and yielded as None (decoding still happens, but grab()
    is much cheaper than read()).
    """
    cap = cv2.VideoCapture(str(path))
    idx = 0
    try:
        while True:
            if wanted is not None and not wanted(idx):
                if not cap.grab():
                    break
                yield idx, None
            else:
                ok, frame = cap.read()
                if not ok:
                    break
                yield idx, frame
            idx += 1
    finally:
        cap.release()


def frames_differ(a: np.ndarray, b: np.ndarray | None, tol: int = 12, min_px: int = 16,
                  scale: int = 4) -> bool:
    """True if the picture really changed between `a` and `b` (frames or same-size crops).

    Anime holds drawings for several frames and title cards / UI screens sit still for
    seconds, so most frames are copies of the previous one. Work done on a frame can be
    reused on every following frame until this returns True.

    The comparison runs at 1/`scale` resolution: averaging over scale x scale blocks wipes
    out encoder noise and grain (scattered, weak), while a letter appearing or moving still
    changes a cluster of blocks. "Changed" = more than `min_px` blocks differ by > `tol`.
    """
    if b is None or a.shape != b.shape:
        return True
    if scale > 1:
        h, w = a.shape[:2]
        size = (max(1, w // scale), max(1, h // scale))
        a = cv2.resize(a, size, interpolation=cv2.INTER_AREA)
        b = cv2.resize(b, size, interpolation=cv2.INTER_AREA)
    d = cv2.absdiff(a, b)
    if d.ndim == 3:
        d = d.max(axis=2)
    return int(np.count_nonzero(d > tol)) > min_px


def detect_shots(path: str | Path, n_frames: int, threshold: float = 27.0) -> list[tuple[int, int]]:
    """Return [(start, end_exclusive), ...] covering the whole video."""
    from scenedetect import ContentDetector, detect

    scenes = detect(str(path), ContentDetector(threshold=threshold), show_progress=False)
    shots = [(s.get_frames(), e.get_frames()) for s, e in scenes]
    if not shots:
        return [(0, n_frames)]
    # make sure the list covers [0, n_frames) without gaps
    shots[0] = (0, shots[0][1])
    shots[-1] = (shots[-1][0], max(shots[-1][1], n_frames))
    return shots


class FfmpegWriter:
    """Pipe raw BGR frames into ffmpeg and mux the untouched audio from `audio_src`."""

    def __init__(self, out_path: str | Path, info: VideoInfo, audio_src: str | Path,
                 encoder: str = "x264", crf: int = 16):
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg not found on PATH")
        if encoder == "nvenc":
            vcodec = ["-c:v", "h264_nvenc", "-preset", "p6", "-rc", "vbr", "-cq", str(crf), "-b:v", "0"]
        else:
            vcodec = ["-c:v", "libx264", "-preset", "slow", "-crf", str(crf)]
        cmd = [
            ffmpeg, "-v", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{info.width}x{info.height}", "-r", f"{info.fps}",
            "-i", "-",
            "-i", str(audio_src),
            "-map", "0:v:0", "-map", "1:a?",
            *vcodec, "-pix_fmt", "yuv420p",
            "-c:a", "copy", "-shortest",
            str(out_path),
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg failed while encoding")
