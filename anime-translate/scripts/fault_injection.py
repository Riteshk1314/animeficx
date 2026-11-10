"""Fault injection: does the critique -> repair loop actually catch and fix problems?

On a clean run every track passes QA in round 1, which says nothing about whether the
critic would notice a bad edit or whether the repair agent picks a sensible fix. This script
breaks a finished agent run on purpose and runs only the QA loop on it:

  fault: the English is drawn in the colour of the background behind it, with no outline
         (invisible text). The deterministic checks can't see this; only the read-back test can.
  expected: the critic fails these tracks (it can't read the English back), the repair agent
         gives them an outline, and they pass in round 2.

Usage:  uv run python scripts/fault_injection.py ..\\Demo.mp4 ..\\Demo_agent_work [track ids...]
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from animtl.agent.llm import DEFAULT_MODEL  # noqa: E402
from animtl.agent.orchestrator import AgentPipeline  # noqa: E402
from animtl.agent.workspace import Workspace  # noqa: E402


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    video, src = Path(sys.argv[1]), Path(sys.argv[2])
    work = src.with_name(src.name + "_fault")
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(src, work, ignore=shutil.ignore_patterns("preview", "*.jsonl"))

    ws = Workspace(video, work)
    targets = [int(x) for x in sys.argv[3:]] or [
        min(ws.tracks, key=lambda t: ws.tracks[t]["shot"]),          # the first track (title)
        max((t for t in ws.tracks if ws.tracks[t].get("en")),         # and the widest menu label
            key=lambda t: (ws.tracks[t]["shot"], ws.box(t)[2] - ws.box(t)[0]))]
    frames = ws.frames({ws.mid(t) for t in targets})
    for tid in targets:
        x0, y0, x1, y1 = ws.box(tid)
        crop = frames[ws.mid(tid)][y0:y1, x0:x1]
        bg = np.median(crop.reshape(-1, 3), axis=0)[::-1]  # BGR -> RGB: the dominant colour
        ws.tracks[tid]["style"] = {"fill": [int(v) for v in bg], "outline": None}
        print(f"injected fault: track #{tid} '{ws.tracks[tid]['en']}' drawn in {bg.astype(int).tolist()}, no outline")
    ws.save()

    p = AgentPipeline(video, work, work.parent / (work.name + ".mp4"), model=DEFAULT_MODEL)
    p.open()
    p.ws, p.shots = ws, ws.shots_with_text()
    p.frames = ws.frames({ws.mid(x) for x in ws.tracks})
    # only the sabotaged tracks and one healthy control track are checked
    control = next(t for t in ws.tracks if t not in targets and ws.tracks[t].get("en"))
    ws.tracks = {k: v for k, v in ws.tracks.items() if k in targets + [control]}
    try:
        p.qa_loop()
    finally:
        ws.tracks = {t["id"]: t for t in ws.data["tracks"]}
    stats = p.trace.close()

    print("\nQA history:")
    print(json.dumps(p.report["qa"], ensure_ascii=False, indent=1))
    rounds = p.report["qa"]
    caught = set(rounds[0]["failed"]) == {str(t) for t in targets}
    fixed = len(rounds) > 1 and not rounds[-1]["failed"]
    print(f"\ncontrol track #{control} {'passed' if str(control) not in rounds[0]['failed'] else 'FAILED'}")
    print(f"faults caught in round 1: {caught}   all fixed by the last round: {fixed}")
    print(f"model calls: {stats['llm_calls']}, tool calls: {stats['tool_calls']}, "
          f"{stats['llm_seconds']} s in the model; trace: {work / 'agent_trace.jsonl'}")
    sys.exit(0 if caught and fixed else 1)


if __name__ == "__main__":
    main()
