"""The agentic pipeline: perceive -> understand -> read -> translate -> render -> critique -> repair.

Design: a deterministic orchestrator runs the phases in a fixed order, and LLM agents make the
judgement calls inside each phase. The model is a 4B vision-language model running locally,
so its autonomy is scoped to decisions it is good at (reading, wording, judging an image), and
every decision passes a guardrail before it is applied:

  - the reader may only *correct* an OCR reading (it must stay similar), never invent text;
  - the translator's lines are checked against the real layout engine before they're accepted;
  - QA combines a deterministic check (re-running OCR on the result) with the vision critic;
  - repairs go through typed tools with clamped parameters, and the loop has a round budget;
  - agents only edit regions.json; pixels are changed only by the renderer, inside text masks.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..editing.render import Renderer, _clip, render
from ..perception.detect import looks_japanese
from ..perception.track import iou
from .agents import CriticAgent, ReaderAgent, RepairAgent, SceneAgent, TranslatorAgent, similar
from .dictionary import Dictionary
from .llm import LLM, encode_image
from .trace import Trace
from .workspace import Workspace


def _luminance(rgb) -> float:
    """WCAG relative luminance of an sRGB colour."""
    def lin(c):
        c = c / 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(v) for v in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b) -> float:
    la, lb = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def legibility(fill, outline, bg) -> float:
    """Contrast ratio that decides legibility: text against the background, or, with an
    outline, text against its outline (the outline then separates it from any background)."""
    return max(contrast(fill, bg), contrast(fill, outline) if outline else 0.0)


def _reading_evidence(d: Dictionary, ja: str) -> str:
    """What the dictionary says about an OCR reading, phrased as evidence for the reader."""
    hits = d.search(ja, 1)
    if hits and ja in (hits[0]["reading"], hits[0]["word"]):
        return f"{ja} is a dictionary word ({hits[0]['meaning']})."
    if hits and len(hits[0]["reading"]) == len(ja) and similar(hits[0]["reading"], ja) >= 0.6:
        h = hits[0]
        return (f"no entry for {ja}; the closest entry, {h['reading']} ({h['meaning']}), "
                f"differs by a look-alike character.")
    parts = d.segment(ja)
    if parts and "".join(p["text"] for p in parts) == ja:
        return "splits into dictionary words: " + d.gloss(ja)
    return f"no dictionary entry found for {ja}."


def _japanese_in(detector, frame, box) -> list[str]:
    """Japanese that OCR still finds over `box` in `frame` (deterministic residue check)."""
    x0, y0, x1, y1 = box
    m = max(8, (y1 - y0) // 2)
    H, W = frame.shape[:2]
    rx0, ry0, rx1, ry1 = max(0, x0 - m), max(0, y0 - m), min(W, x1 + m), min(H, y1 + m)
    found = []
    for d in detector(frame[ry0:ry1, rx0:rx1]):
        if d.conf >= 0.3 and looks_japanese(d.text):
            found.append(d.text)
    return found


class AgentPipeline:
    def __init__(self, video: Path, work: Path, out: Path, model: str, max_rounds: int = 2,
                 encoder: str = "x264", crf: int = 16, font: str | None = None):
        self.video, self.work, self.out = Path(video), Path(work), Path(out)
        self.max_rounds, self.encoder, self.crf, self.font = max_rounds, encoder, crf, font
        self.model = model
        self.detector = None
        self.report: dict = {"model": model, "done": []}

    def open(self, append: bool = False) -> None:
        """Start the trace and the model client (separate from run() so tests can drive single phases)."""
        self.trace = Trace(self.work / "agent_trace.jsonl", append=append)
        self.llm = LLM(self.model, trace=self.trace)
        self.dict = Dictionary(self.work / "dictionary_cache.json")

    PHASES = ("understand", "read", "translate", "qa_loop", "final_render", "review", "evaluate")

    def _checkpoint(self) -> None:
        """Persist progress after every phase: all decisions live in regions.json, everything
        else in agent_report.json, so an interrupted run can continue with --resume."""
        self.ws.save()
        (self.work / "agent_report.json").write_text(json.dumps(self.report, ensure_ascii=False, indent=1),
                                                     encoding="utf-8")

    # ------------------------------------------------------------------------------
    def run(self, reanalyze: bool = False, gpu: bool = True, resume: bool = False) -> dict:
        saved = self.work / "agent_report.json"
        if resume and saved.exists() and not reanalyze:
            self.report = json.loads(saved.read_text(encoding="utf-8"))
            if self.report.get("model") != self.model:
                raise SystemExit(f"--resume: the saved run used {self.report.get('model')}, not {self.model}")
        else:
            resume = False
        self.report.setdefault("done", [])
        self.open(append=resume)
        if not self.llm.available():
            raise SystemExit(f"Ollama model '{self.llm.model}' is not available. Is Ollama running "
                             f"(`ollama serve`)? Is the model installed (`ollama pull {self.llm.model}`)?")
        t = self.trace
        if resume:
            t.phase("resume", f"phases already done: {', '.join(self.report['done']) or 'none'}")
            ws = self.ws = Workspace(self.video, self.work)
            self.scenes = {int(k): v for k, v in self.report.get("scenes", {}).items()}
        else:
            t.phase("1 perceive", "shot detection, text detection, tracking, OCR (deterministic)")
            if reanalyze or not (self.work / "regions.json").exists():
                from ..perception.analyze import analyze
                analyze(self.video, self.work, gpu=gpu)
            else:
                t.note("orchestrator", "reusing existing regions.json (use --reanalyze to redo)")
            ws = self.ws = Workspace(self.video, self.work)
            for tr in ws.tracks.values():
                tr.setdefault("ja_ocr", tr["ja"])
                tr["ja"] = tr["ja_ocr"]
                tr["en"], tr["style"], tr["agent"] = None, {}, {}
        self.shots = ws.shots_with_text()
        self.frames = ws.frames({ws.shot_mid(s) for s in self.shots} | {ws.mid(x) for x in ws.tracks})

        for name in self.PHASES:
            if name in self.report["done"]:
                continue
            getattr(self, name)()
            self.report["done"].append(name)
            self._checkpoint()

        stats = t.close()
        prev = self.report.get("agent_stats", {}) if resume else {}
        self.report["agent_stats"] = {k: round(prev.get(k, 0) + v, 1) if isinstance(v, float) else prev.get(k, 0) + v
                                      for k, v in stats.items()}
        self._checkpoint()
        from .report import write_markdown
        md = write_markdown(self.report, self.ws, self.work / "agent_report.md")
        print(f"\nreport: {md}\ntrace:  {self.work / 'agent_trace.jsonl'}\nvideo:  {self.out}")
        return self.report

    # ------------------------------------------------------------------------------
    def understand(self) -> None:
        ws, t = self.ws, self.trace
        t.phase("2 understand", "vision model describes each screen that has text")
        agent = SceneAgent(self.llm)
        self.scenes = {}
        for s, tids in self.shots.items():
            texts = list(dict.fromkeys(ws.tracks[x]["ja"] for x in tids))
            self.scenes[s] = agent.run(self.frames[ws.shot_mid(s)], texts)
        ws.data["scenes"] = {str(k): v for k, v in self.scenes.items()}
        ws.save()
        self.report["scenes"] = ws.data["scenes"]

    # ------------------------------------------------------------------------------
    def read(self) -> None:
        ws, t = self.ws, self.trace
        t.phase("3 read", "vision model checks every OCR reading against the image")
        agent = ReaderAgent(self.llm)
        groups: dict[tuple, list[int]] = {}
        for tid, tr in ws.tracks.items():
            groups.setdefault((tr["shot"], tr["ja"]), []).append(tid)
        changes = []
        for (shot, ja), tids in groups.items():
            full = [x for x in tids if not ws.tracks[x].get("full_box")] or tids  # prefer uncut text
            rep = max(full, key=lambda x: (lambda b: (b[2] - b[0]) * (b[3] - b[1]))(ws.box(x)))
            tr = ws.tracks[rep]
            cands = list(dict.fromkeys(c for c in (ja, tr.get("easyocr"), *tr.get("ja_readings", [])) if c))
            evidence = _reading_evidence(self.dict, ja)
            r = agent.run(ws.crop(self.frames[ws.mid(rep)], ws.box(rep)), cands,
                          self.scenes[shot]["description"], evidence)
            new = (r.get("reading") or "").strip()
            conf = max(ws.tracks[x]["conf"] for x in tids)
            # guardrails: corrections only (still Japanese, similar to what OCR saw, about the
            # same length); text can only be dropped if OCR itself was unsure
            if not r.get("is_text", True) and conf < 0.5:
                decision = "dropped as non-text"
                for x in tids:
                    ws.tracks[x]["en"] = ""
            elif (new and new != ja and looks_japanese(new) and similar(new, ja) >= 0.5
                  and abs(len(new) - len(ja)) <= 1):
                decision = f"corrected {ja} -> {new}"
                for x in tids:
                    ws.tracks[x]["ja"] = new
                changes.append({"shot": shot, "ocr": ja, "agent": new, "reason": r.get("reason", "")})
            elif new and new != ja:
                decision = f"rejected suggestion {new} (too different from OCR {ja})"
            else:
                decision = "confirmed"
            for x in tids:
                ws.note(x, "reading", {"decision": decision, "reason": r.get("reason", ""),
                                       "evidence": evidence})
            t.note("reader", f"{ja}: {decision}")
        ws.save()
        self.report["reading_changes"] = changes

    # ------------------------------------------------------------------------------
    def translate(self) -> None:
        ws, t = self.ws, self.trace
        t.phase("4 translate", "per screen: draft, check layout fit with a tool, submit")
        agent = TranslatorAgent(self.llm, ws, self.trace, self.dict)
        for s, tids in self.shots.items():
            live = [x for x in tids if ws.tracks[x]["en"] != ""]
            uniq = list(dict.fromkeys(ws.tracks[x]["ja"] for x in live))
            lines = {i + 1: {"ja": ja, "tids": [x for x in live if ws.tracks[x]["ja"] == ja]}
                     for i, ja in enumerate(uniq)}
            if not lines:
                continue
            res = agent.run(lines, self.scenes[s])
            for lid, line in lines.items():
                en = res.get(lid)
                for x in line["tids"]:
                    ws.tracks[x]["en"] = en or None
                    ws.note(x, "translation", {"by": "agent" if en else "missing",
                                               "dictionary": agent.gloss.get(lid, "")})
        missing = [x for x, tr in ws.tracks.items() if tr["en"] is None]
        if missing:  # fall back to plain MT rather than leave Japanese on screen
            from ..translate import google
            try:
                table = google(list({ws.tracks[x]["ja"] for x in missing}))
                for x in missing:
                    ws.tracks[x]["en"] = table.get(ws.tracks[x]["ja"])
                    ws.note(x, "translation", {"by": "google fallback"})
                t.note("orchestrator", f"translator skipped {len(missing)} tracks; filled by Google MT")
            except Exception as e:
                t.note("orchestrator", f"translator skipped {len(missing)} tracks; no fallback ({e})")
        ws.save()

    # ------------------------------------------------------------------------------
    def check(self, tids: list[int]) -> dict[int, dict]:
        """Render the tracks' middle frames with the current regions.json and judge them."""
        ws = self.ws
        if self.detector is None:
            from ..perception.detect import TextDetector
            self.detector = TextDetector()
        ws.save()
        r = Renderer(self.video, self.work, self.font)
        want = {ws.mid(x) for x in tids}
        edited = r.stills({f: self.frames[f] for f in want})
        critic = CriticAgent(self.llm)
        failures = {}
        for tid in tids:
            tr, f = ws.tracks[tid], ws.mid(tid)
            box = _clip(ws.text_box(tid), ws.W, ws.H)
            residue = _japanese_in(self.detector, edited[f], box)
            pair = ws.stack(ws.crop(self.frames[f], box), ws.crop(edited[f], box))
            c = critic.run(pair, tr["en"])
            cut = bool(tr.get("full_box"))  # English cut by the frame edge can't be read back in full
            readback = similar(c["english_text_in_bottom"], tr["en"])
            # Legibility is measured, not asked: a VLM reads even faint text back, so read-back
            # proves the text is *there*, not that a viewer can read it.
            x0, y0, x1, y1 = box
            bg = tuple(int(v) for v in np.median(self.frames[f][y0:y1, x0:x1].reshape(-1, 3), axis=0)[::-1])
            style = r.styles[tid]
            cr = legibility(style.fill, style.outline, bg)
            reasons = []
            if residue:
                reasons.append(f"OCR still finds Japanese {residue}")
            if c["japanese_visible_in_bottom"]:
                reasons.append("vision check sees Japanese left")
            if not cut and readback < 0.6:
                reasons.append(f"English reads back as '{c['english_text_in_bottom']}', expected '{tr['en']}'")
            if cr < 3.0:
                tone = "light" if _luminance(bg) > 0.4 else "dark"
                reasons.append(f"low contrast: text {list(style.fill)} with outline "
                               f"{list(style.outline) if style.outline else 'none'} on a {tone} background "
                               f"{list(bg)} is {cr:.1f}:1, needs 3:1")
            elif c.get("english_contrast") == "weak":
                reasons.append("vision check says the English is faint against the background")
            if not c["looks_natural"] and c["problem"]:
                reasons.append(c["problem"])
            ws.note(tid, "qa", {"residue_ocr": residue, "critic": c, "readback": round(readback, 2),
                                "contrast": round(cr, 2), "passed": not reasons})
            if reasons:
                failures[tid] = {"reasons": reasons, "image": encode_image(pair, 768)}
        ws.save()
        return failures

    def qa_loop(self) -> None:
        ws, t = self.ws, self.trace
        t.phase("5 critique + repair", f"render stills, check them, fix failures (max {self.max_rounds} rounds)")
        todo, history = list(ws.tracks), []
        todo = [x for x in todo if ws.tracks[x]["en"]]
        for rnd in range(1, self.max_rounds + 2):
            failures = self.check(todo)
            t.note("critic", f"round {rnd}: {len(todo) - len(failures)}/{len(todo)} passed",
                   failed={k: v["reasons"] for k, v in failures.items()})
            history.append({"round": rnd, "checked": len(todo),
                            "failed": {str(k): v["reasons"] for k, v in failures.items()}})
            if not failures or rnd > self.max_rounds:
                break
            actions = RepairAgent(self.llm, ws, self.trace).run(
                failures, {k: v["image"] for k, v in failures.items()})
            history[-1]["repairs"] = {str(k): v for k, v in actions.items()}
            for tid, a in actions.items():
                ws.note(tid, "repair", ws.tracks[tid]["agent"].get("repair", []) + [a])
            ws.save()
            todo = [x for x in failures if not actions.get(x, "").startswith("accept")]
            if not todo:
                break
        self.report["qa"] = history

    # ------------------------------------------------------------------------------
    def final_render(self) -> None:
        self.trace.phase("6 render", "full video with the agent's decisions")
        self.ws.save()
        self.llm.unload()  # give LaMa the GPU memory
        self.report["render"] = render(self.video, self.work, self.out, font=self.font,
                                       encoder=self.encoder, crf=self.crf)

    def _review_regions(self, shot: int, frame: int) -> list[tuple]:
        """Where missed text is likely: around the text that was found. Untranslated rows of a
        list or menu sit next to the translated ones. Each found box is widened by about two
        text heights; overlapping areas are merged."""
        ws = self.ws
        rects = []
        for tid in self.shots[shot]:
            b = ws.tracks[tid]["boxes"].get(str(frame))
            if not b or not ws.tracks[tid].get("en"):
                continue
            x0, y0, x1, y1 = b
            h = y1 - y0
            rects.append([max(0, x0 - h), max(0, y0 - 2 * h), min(ws.W, x1 + h), min(ws.H, y1 + 2 * h)])
        merged = True
        while merged:
            merged = False
            for i in range(len(rects)):
                for j in range(i + 1, len(rects)):
                    a, b = rects[i], rects[j]
                    if a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]:
                        rects[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                        del rects[j]
                        merged = True
                        break
                if merged:
                    break
        return [tuple(r) for r in rects]

    def review(self) -> None:
        """Look at the finished frames for Japanese the pipeline never detected (recall check).

        A whole 1546x856 frame shrunk to the model's input size makes half-visible small text
        unreadable, so the reviewer looks at full-resolution crops around the text that was
        found, plus the whole frame for anything elsewhere."""
        ws, t = self.ws, self.trace
        t.phase("7 review", "vision model looks for Japanese that perception missed")
        from ..editing.render import read_frames
        mids = {s: ws.shot_mid(s) for s in self.shots}
        outs = read_frames(self.out, set(mids.values()))
        schema = {"type": "object", "properties": {"remaining": {"type": "array", "items": {
            "type": "object", "properties": {"text": {"type": "string", "maxLength": 40}},
            "required": ["text"]}}}, "required": ["remaining"]}
        prompt = ("This image is from an anime frame that was translated from Japanese to English. "
                  "List any Japanese characters (kana or kanji) still visible, including text partly "
                  "cut off at an edge. Ignore English, logos and channel watermarks. If there is none, "
                  "return an empty list.")
        flags, seen = [], set()
        for s, f in mids.items():
            img = outs[f]
            areas = [(0, 0, ws.W, ws.H)] + self._review_regions(s, f)
            for k, (x0, y0, x1, y1) in enumerate(areas):
                try:
                    r = self.llm.chat_json([{"role": "user", "content": prompt,
                                             "images": [encode_image(img[y0:y1, x0:x1], 896)]}],
                                           schema, agent="reviewer", step=f"shot {s} area {k}")
                except ValueError:
                    continue
                for x in r.get("remaining", []):
                    text = x.get("text", "").strip()
                    if looks_japanese(text) and (s, text) not in seen:
                        seen.add((s, text))
                        flags.append({"shot": s, "frame": f, "text": text, "area": [x0, y0, x1, y1]})
            n = sum(1 for x in flags if x["shot"] == s)
            t.note("reviewer", f"shot {s}: {n} untranslated text(s) flagged for a human")
        self.report["review_flags"] = flags

    def evaluate(self) -> None:
        self.trace.phase("8 evaluate", "objective metrics on the output video")
        self.llm.unload()
        from ..eval.metrics import evaluate
        self.report["metrics"] = evaluate(self.video, self.out, self.work, detector=self.detector)
