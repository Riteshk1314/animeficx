"""Command line: animtl run|analyze|translate|render."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ignore(values: list[str] | None):
    out = []
    for v in values or []:
        x, y, w, h = (int(p) for p in v.split(","))
        out.append((x, y, x + w, y + h))
    return out


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Japanese in the Windows console

    p = argparse.ArgumentParser(prog="animtl", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, needs_video=True):
        if needs_video:
            sp.add_argument("video", type=Path)
        sp.add_argument("--work", type=Path, help="working dir (default: <video>_work)")

    def analyze_opts(sp):
        sp.add_argument("--probe-stride", type=int, default=10, help="frames between probes when looking for text shots")
        sp.add_argument("--det-stride", type=int, default=1, help="frames between detections inside text shots")
        sp.add_argument("--ignore", action="append", metavar="X,Y,W,H", help="screen area to never touch (repeatable)")
        sp.add_argument("--cpu", action="store_true", help="run OCR on CPU")
        sp.add_argument("--no-manga-ocr", action="store_true",
                        help="don't use manga-ocr even if installed (EasyOCR readings only)")

    def translate_opts(sp):
        sp.add_argument("--translator", default="google", choices=["google", "none"],
                        help="google (free endpoint, or Cloud API if GOOGLE_TRANSLATE_API_KEY is set) "
                             "or none (fill regions.json by hand)")
        sp.add_argument("--force", action="store_true", help="re-translate tracks that already have English")

    def render_opts(sp):
        sp.add_argument("-o", "--out", type=Path)
        sp.add_argument("--font", help="TTF/OTF font for the English text")
        sp.add_argument("--encoder", default="x264", choices=["x264", "nvenc"])
        sp.add_argument("--crf", type=int, default=16)
        sp.add_argument("--no-preview", action="store_true")

    sp = sub.add_parser("run", help="analyze + translate + render")
    common(sp); analyze_opts(sp); translate_opts(sp); render_opts(sp)
    sp = sub.add_parser("analyze", help="detect, track and OCR the Japanese text")
    common(sp); analyze_opts(sp)
    sp = sub.add_parser("translate", help="fill English into regions.json")
    common(sp, needs_video=False); translate_opts(sp)
    sp = sub.add_parser("render", help="erase + typeset + encode")
    common(sp); render_opts(sp)
    sp = sub.add_parser("agent", help="agentic pipeline: a local vision-language model reads, "
                        "translates, checks and repairs (needs Ollama)")
    common(sp); render_opts(sp)
    sp.add_argument("--model", default=None, help="Ollama model (default: qwen3-vl:4b or $ANIMTL_MODEL)")
    sp.add_argument("--rounds", type=int, default=2, help="max critique/repair rounds")
    sp.add_argument("--reanalyze", action="store_true", help="redo perception even if regions.json exists")
    sp.add_argument("--resume", action="store_true", help="continue an interrupted run from its last finished phase")
    sp.add_argument("--cpu", action="store_true", help="run OCR on CPU")
    sp = sub.add_parser("eval", help="objective metrics: input video vs translated output")
    common(sp)
    sp.add_argument("output", type=Path)

    a = p.parse_args(argv)
    if a.cmd == "translate" and a.work is None:
        p.error("translate needs --work")
    work = a.work or a.video.with_name(a.video.stem + "_work")
    out = getattr(a, "out", None) or (a.video.with_name(a.video.stem + "_en.mp4") if hasattr(a, "video") else None)

    if a.cmd == "agent":
        from .agent.llm import DEFAULT_MODEL
        from .agent.orchestrator import AgentPipeline
        AgentPipeline(a.video, work, out, model=a.model or DEFAULT_MODEL, max_rounds=a.rounds,
                      encoder=a.encoder, crf=a.crf, font=a.font).run(reanalyze=a.reanalyze, gpu=not a.cpu, resume=a.resume)
        return
    if a.cmd == "eval":
        import json
        from .eval.metrics import evaluate
        print(json.dumps(evaluate(a.video, a.output, work), indent=1))
        return

    if a.cmd in ("run", "analyze"):
        from .perception.analyze import analyze
        analyze(a.video, work, probe_stride=a.probe_stride, det_stride=a.det_stride,
                ignore=_ignore(a.ignore), gpu=not a.cpu, manga=not a.no_manga_ocr)
    if a.cmd in ("run", "translate"):
        from .translate import translate
        translate(work, backend=a.translator, force=a.force)
    if a.cmd in ("run", "render"):
        from .editing.render import render
        render(a.video, work, out, font=a.font, encoder=a.encoder, crf=a.crf, preview=not a.no_preview)


if __name__ == "__main__":
    main()
