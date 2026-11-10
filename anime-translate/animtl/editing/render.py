"""Stage 3: erase the Japanese, typeset the English, re-encode with the original audio.

Only pixels inside each text mask are replaced; everything else in the frame is
passed through untouched (apart from the final lossy encode, which is near-transparent
at the default CRF 16).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .inpaint import Lama
from .typeset import Style, default_font, estimate_style, fit_size, paste_rgba, render_text, stroke_mask
from ..video import FfmpegWriter, frames_differ, iter_frames, probe


@dataclass
class CleanPlate:
    box: tuple
    rect: tuple            # crop rect in frame coords
    orig: np.ndarray       # original crop when the patch was made
    mask: np.ndarray
    patch: np.ndarray      # inpainted crop


def _crop_rect(box, W, H):
    x0, y0, x1, y1 = box
    m = max(24, int((y1 - y0) * 0.8))
    return max(0, x0 - m), max(0, y0 - m), min(W, x1 + m), min(H, y1 + m)


def _full_box(box, full_box, W, H, m: int = 2):
    """For text cut off by the frame edge, rebuild the box of the whole text (it may
    extend off-screen) from where the same text sits when fully visible. Erasing uses
    it clipped to the frame; the English is drawn into it and clipped like the original."""
    if not full_box:
        return box
    x0, y0, x1, y1 = box
    fx0, fy0, fx1, fy1 = full_box
    fw, fh = fx1 - fx0, fy1 - fy0
    if y1 >= H - m or y0 <= m:          # scrolled through top/bottom: same column as when visible
        x0, x1 = min(x0, fx0), max(x1, fx1)
        if y1 >= H - m:
            y1 = max(y1, y0 + fh)
        else:
            y0 = min(y0, y1 - fh)
    if x1 >= W - m or x0 <= m:          # slid through left/right: same row
        y0, y1 = min(y0, fy0), max(y1, fy1)
        if x1 >= W - m:
            x1 = max(x1, x0 + fw)
        else:
            x0 = min(x0, x1 - fw)
    return x0, y0, x1, y1


def _clip(box, W, H):
    return max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])


def _columns(tracks: dict, W: int, H: int, tol: float = 0.25) -> list[list[tuple]]:
    """Group labels that form one UI column/list: same shot, centres aligned, similar height."""
    groups: list[list[tuple]] = []
    for tid, t in sorted(tracks.items()):
        mid = str((t["first"] + t["last"]) // 2)
        b = _full_box(tuple(t["boxes"][mid]), t.get("full_box"), W, H)
        w, h, cx = b[2] - b[0], b[3] - b[1], (b[0] + b[2]) / 2
        for g in groups:
            shot, gw, gh, gcx = g[0][1], g[0][2], g[0][3], g[0][4]
            if (t["shot"] == shot and abs(cx - gcx) < 0.3 * max(w, gw)
                    and max(h, gh) <= min(h, gh) * (1 + tol)):
                g.append((tid, t["shot"], w, h, cx))
                break
        else:
            groups.append([(tid, t["shot"], w, h, cx)])
    return groups


def _shared_layout(tracks: dict, styles: dict, font: str, W: int, H: int):
    """Labels in one column share a width (the widest Japanese label in it) and one
    font size (the largest at which every English label fits). English is usually
    longer than the katakana it replaces, and a list should look uniform."""
    caps: dict[int, int] = {}
    widths: dict[int, int] = {}
    for g in _columns(tracks, W, H):
        gw = max(item[2] for item in g)
        cap = min(fit_size(tracks[tid]["en"], gw, h, styles[tid], font) for tid, _, _, h, _ in g)
        for tid, *_ in g:
            caps[tid], widths[tid] = cap, gw
    return caps, widths


def _can_reuse(ref: CleanPlate, box, rect, ocrop, mask) -> bool:
    if ref.box != box or ref.rect != rect:
        return False
    # every text pixel now must already be covered by the plate's mask
    new_text = (mask > 0) & (ref.mask == 0)
    if new_text.sum() > 0.01 * max(1, (mask > 0).sum()):
        return False
    # and the surroundings must not have changed (camera still, nothing moving through)
    d = cv2.absdiff(ocrop, ref.orig).max(axis=2)[ref.mask == 0]
    return d.size == 0 or (d.mean() < 2.5 and np.percentile(d, 99.5) < 24)


def _load_tracks(work: Path) -> tuple[dict, dict]:
    data = json.loads((work / "regions.json").read_text(encoding="utf-8"))
    return data, {t["id"]: t for t in data["tracks"] if t.get("en")}


class Renderer:
    """Edits single frames: erase the Japanese, draw the English, keep logos on top.

    Built once per video (samples styles, plans layout, loads LaMa), then `edit()` can be
    called on any frame: in order (full render, with clean-plate reuse) or out of order
    (the agent's QA re-rendering a few stills after changing a track).

    Per-track overrides, read from the track's optional "style" dict in regions.json
    (the agent's repair step writes these):
      grow     extra mask dilation in px (residue left around the strokes)
      fill     [r, g, b] text colour        outline  [r, g, b] or null
    """

    def __init__(self, video: str | Path, work: Path, font: str | None = None,
                 only: set[int] | None = None):
        self.video, self.work = video, work
        self.info = probe(video)
        self.W, self.H = self.info.width, self.info.height
        self.font = font or default_font()
        _, tracks = _load_tracks(work)
        self.tracks = {k: v for k, v in tracks.items() if only is None or k in only}
        W, H = self.W, self.H

        # per frame: (track id, box to erase, box to draw the English into)
        self.per_frame: dict[int, list[tuple[int, tuple, tuple]]] = {}
        for tid, t in self.tracks.items():
            for f, b in t["boxes"].items():
                full = _full_box(tuple(b), t.get("full_box"), W, H)
                self.per_frame.setdefault(int(f), []).append((tid, _clip(full, W, H), full))

        # Text style (fill/outline colour) is sampled once per track on its middle frame,
        # which is more representative than the first (fade-ins, text still writing on).
        mids: dict[int, list[int]] = {}
        for tid in self.tracks:
            mids.setdefault(self.mid(tid), []).append(tid)
        self.styles: dict[int, Style] = {}
        for idx, frame in read_frames(video, set(mids)).items():
            for tid in mids[idx]:
                box = tuple(self.tracks[tid]["boxes"][str(idx)])
                rx0, ry0, rx1, ry1 = _crop_rect(box, W, H)
                self.styles[tid] = estimate_style(frame[ry0:ry1, rx0:rx1],
                                                  (box[0] - rx0, box[1] - ry0, box[2] - rx0, box[3] - ry0))
        for tid, t in self.tracks.items():
            o = t.get("style") or {}
            s = self.styles[tid]
            fill = tuple(o["fill"]) if o.get("fill") else s.fill
            outline = (tuple(o["outline"]) if o["outline"] else None) if "outline" in o else s.outline
            self.styles[tid] = Style(fill=fill, outline=outline)

        self.size_cap, self.col_width = _shared_layout(self.tracks, self.styles, self.font, W, H)

        self.overlay = None
        if (work / "overlay.png").exists():
            ov = cv2.imread(str(work / "overlay.png"), cv2.IMREAD_GRAYSCALE)
            if ov is not None and ov.shape == (H, W) and ov.any():
                self.overlay, self.overlay_px = ov, ov > 0

        self.lama = Lama() if self.per_frame else None
        self.plates: dict[int, CleanPlate] = {}
        self.text_cache: dict[tuple, object] = {}
        self.n_inpaint = self.n_reuse = self.n_copy = 0
        self._last = None  # previous edited frame: (items, original, changed-pixel mask, output)

    def mid(self, tid: int) -> int:
        t = self.tracks[tid]
        return (t["first"] + t["last"]) // 2

    def edit(self, idx: int, frame: np.ndarray) -> np.ndarray:
        """Return the edited frame (the input array is modified in place)."""
        items = self.per_frame.get(idx)
        if not items:
            return frame
        W, H = self.W, self.H
        last = self._last
        # Same text in the same places, and nothing changed around it: the edit is identical,
        # so copy the edited pixels from the last output instead of erasing and drawing again.
        # (Pixels outside the edit always come fresh from this frame, so motion elsewhere is kept.)
        if last is not None and last[0] == items and not any(
                frames_differ(frame[r[1]:r[3], r[0]:r[2]], last[1][r[1]:r[3], r[0]:r[2]], min_px=2)
                for r in (_crop_rect(box, W, H) for _, box, _ in items)):
            frame[last[2]] = last[3][last[2]]
            self.n_copy += 1
            return frame
        orig = frame.copy()

        # 1) erase every text region (masks are computed on the untouched original)
        for tid, box, _ in items:
            rect = _crop_rect(box, W, H)
            rx0, ry0, rx1, ry1 = rect
            ocrop = orig[ry0:ry1, rx0:rx1]
            ref = self.plates.get(tid)
            # cheap check first: same pixels as when this track's clean plate was made
            reuse = (ref is not None and ref.box == box and ref.rect == rect
                     and not frames_differ(ocrop, ref.orig, min_px=2))
            if not reuse:
                lbox = (box[0] - rx0, box[1] - ry0, box[2] - rx0, box[3] - ry0)
                grow = int((self.tracks[tid].get("style") or {}).get("grow", 0))
                mask = stroke_mask(ocrop, lbox, self.styles[tid].fill_bgr, grow=grow)
                if self.overlay is not None:
                    mask[self.overlay[ry0:ry1, rx0:rx1] > 0] = 0  # never paint over logos
                reuse = ref is not None and _can_reuse(ref, box, rect, ocrop, mask)
            if reuse:
                patch, pmask = ref.patch, ref.mask
                self.n_reuse += 1
            else:
                patch = self.lama(frame[ry0:ry1, rx0:rx1], mask)
                pmask = mask
                self.plates[tid] = CleanPlate(box, rect, ocrop.copy(), mask, patch)
                self.n_inpaint += 1

            a = cv2.GaussianBlur(pmask.astype(np.float32) / 255, (0, 0), 1.5)
            a = np.maximum(a, (pmask > 0).astype(np.float32))[..., None]
            region = frame[ry0:ry1, rx0:rx1].astype(np.float32)
            frame[ry0:ry1, rx0:rx1] = (region * (1 - a) + patch.astype(np.float32) * a).round().astype(np.uint8)

        # 2) typeset the English on top
        for tid, _, tb in items:
            bw, bh = max(tb[2] - tb[0], self.col_width.get(tid, 0)), tb[3] - tb[1]
            key = (tid, bw, bh)
            if key not in self.text_cache:
                self.text_cache[key] = render_text(self.tracks[tid]["en"], bw, bh, self.styles[tid], self.font,
                                                   max_size=self.size_cap.get(tid))
            paste_rgba(frame, self.text_cache[key], (tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2)

        # 3) logos / watermarks stay on top, exactly as broadcast
        if self.overlay is not None:
            frame[self.overlay_px] = orig[self.overlay_px]

        self._last = (items, orig, np.any(frame != orig, axis=2), frame.copy())
        return frame

    def stills(self, frames: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
        """Edit a few frames out of order (QA previews). Returns edited copies."""
        out = {}
        for idx in sorted(frames):
            self._last = None
            out[idx] = self.edit(idx, frames[idx].copy())
        return out


def read_frames(video: str | Path, wanted: set[int]) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    if not wanted:
        return out
    last = max(wanted)
    for idx, frame in iter_frames(video, lambda i: i in wanted):
        if frame is not None:
            out[idx] = frame
        if idx >= last:
            break
    return out


def render(video: str | Path, work: Path, out: Path, font: str | None = None,
           encoder: str = "x264", crf: int = 16, preview: bool = True) -> dict:
    data, _ = _load_tracks(work)
    skipped = [t["id"] for t in data["tracks"] if not t.get("en")]
    if skipped:
        print(f"[render] tracks without English are left as-is: {skipped}")
    r = Renderer(video, work, font)
    preview_at = {r.mid(tid): tid for tid in r.tracks}
    prev_dir = work / "preview"
    if preview:
        prev_dir.mkdir(exist_ok=True)

    writer = FfmpegWriter(out, r.info, video, encoder=encoder, crf=crf)
    for idx, frame in tqdm(iter_frames(video), total=r.info.n_frames, desc="render", unit="f"):
        src = frame.copy() if preview and idx in preview_at else None
        frame = r.edit(idx, frame)
        if src is not None:
            cv2.imwrite(str(prev_dir / f"track{preview_at[idx]:03d}_f{idx}.png"), np.vstack([src, frame]))
        writer.write(frame)
    writer.close()
    print(f"[render] inpainted {r.n_inpaint} regions, reused clean plates {r.n_reuse} times, "
          f"copied {r.n_copy} unchanged frames -> {out}")
    if preview:
        print(f"[render] before/after stills in {prev_dir}")
    return {"inpainted": r.n_inpaint, "plate_reuse": r.n_reuse, "frames_copied": r.n_copy}
