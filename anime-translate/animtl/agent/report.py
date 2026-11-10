"""Human-readable report of an agent run (what it decided, why, and how good the result is)."""

from __future__ import annotations

from pathlib import Path


def write_markdown(rep: dict, ws, path: Path) -> Path:
    L = [f"# Agent run report: {ws.video.name}", ""]
    s = rep.get("agent_stats", {})
    L += [f"Model: `{rep['model']}` (local, Ollama) · {s.get('llm_calls', 0)} model calls · "
          f"{s.get('tool_calls', 0)} tool calls · {s.get('prompt_tokens', 0):,} prompt + "
          f"{s.get('output_tokens', 0):,} output tokens · {s.get('llm_seconds', 0)} s in the model · "
          f"{s.get('wall_seconds', 0)} s total", ""]

    L += ["## Screens", "", "| Shot | Kind | What the vision model saw | Tone |", "|---|---|---|---|"]
    for k, v in rep.get("scenes", {}).items():
        L.append(f"| {k} | {v['kind']} | {v['description']} | {v['tone']} |")

    L += ["", "## Readings corrected", ""]
    ch = rep.get("reading_changes", [])
    if ch:
        L += ["| OCR read | Agent reading | Model's reason (first sentence) |", "|---|---|---|"]
        L += [f"| {c['ocr']} | {c['agent']} | {c['reason'].split('. ')[0].rstrip('.')}. |" for c in ch]
    else:
        L.append("None: every OCR reading was confirmed.")

    L += ["", "## Translations", "",
          "| Japanese (final) | Dictionary gloss given to the model | English | Tracks |", "|---|---|---|---|"]
    seen: dict[tuple, list] = {}
    for tid, t in sorted(ws.tracks.items()):
        lit = (t.get("agent", {}).get("translation") or {}).get("dictionary", "")
        seen.setdefault((t["ja"], lit, t.get("en") or ""), []).append(tid)
    for (ja, lit, en), tids in seen.items():
        L.append(f"| {ja} | {lit} | {en or '*(left as is)*'} | {', '.join(f'#{x}' for x in tids)} |")

    L += ["", "## Quality checks and repairs", ""]
    for r in rep.get("qa", []):
        L.append(f"- Round {r['round']}: {r['checked'] - len(r['failed'])}/{r['checked']} tracks passed.")
        for tid, reasons in r["failed"].items():
            fix = r.get("repairs", {}).get(tid, "no repair (round budget used)")
            L.append(f"  - #{tid}: {'; '.join(reasons)} → **{fix}**")

    L += ["", "## Flagged for a human", ""]
    flags = rep.get("review_flags", [])
    if flags:
        L += [f"- Shot {f['shot']} (frame {f['frame']}): `{f['text']}` near x={f['area'][0]}-{f['area'][2]}, "
              f"y={f['area'][1]}-{f['area'][3]}" for f in flags]
    else:
        L.append("Nothing: the reviewer found no Japanese left on the final frames.")

    m = rep.get("metrics", {})
    if m:
        tr, fi, st = m["text_removal"], m["fidelity"], m["stability"]
        L += ["", "## Metrics", "", "| Metric | Value |", "|---|---|",
              f"| Japanese detections, input → output ({tr['sampled_frames']} sampled frames) | "
              f"{tr['japanese_in_input']} → {tr['japanese_in_output']} ({tr['removed_pct']}% removed) |",
              f"| PSNR outside edited regions | {fi['psnr_outside_edits_db']} dB |",
              f"| Share of frame edited (frames with text) | {m['edited_area_pct']}% |",
              f"| Change in edited boxes while source is static (output / input) | "
              f"{st['mean_abs_change_output']} / {st['mean_abs_change_input']} |"]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path
