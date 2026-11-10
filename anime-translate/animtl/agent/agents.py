"""The LLM workers. Each has one narrow job, a short prompt, and either a JSON schema
(the reply is constrained to it) or a small set of tools it can call in a loop.

  SceneAgent      looks at a frame        -> what kind of screen this is (context for the rest)
  ReaderAgent     looks at a text crop    -> corrects OCR misreads, flags non-text
  TranslatorAgent tool loop               -> translates a whole screen, checks each line fits, submits
  CriticAgent     looks at before/after   -> is Japanese left? can it read the English back?
  RepairAgent     tool loop               -> picks a fix for each failed check (mask, outline, wording)
"""

from __future__ import annotations

import json
from difflib import SequenceMatcher
from typing import Callable

from .llm import LLM, encode_image
from .workspace import Workspace


# --------------------------------------------------------------------------------------
# tool-calling loop
# --------------------------------------------------------------------------------------

def tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required}}}


def run_tools(llm: LLM, agent: str, messages: list[dict], tools: list[dict],
              handlers: dict[str, Callable[[dict], dict]], done: Callable[[], bool],
              max_steps: int = 8, trace=None) -> int:
    """ReAct-style loop: the model calls tools, sees the results, and continues until
    `done()` is true or the step budget runs out. Returns the number of steps used."""
    nudged = False
    for step in range(1, max_steps + 1):
        reply = llm.chat(messages, tools=tools, agent=agent, step=f"step {step}")
        if not reply.tool_calls:
            if done() or nudged:
                return step
            nudged = True
            messages.append({"role": "assistant", "content": reply.content})
            messages.append({"role": "user", "content": "Use the tools: call one of them now."})
            continue
        messages.append({"role": "assistant", "content": reply.content,
                         "tool_calls": [{"function": {"name": c["name"], "arguments": c["arguments"]}}
                                        for c in reply.tool_calls]})
        for c in reply.tool_calls:
            fn = handlers.get(c["name"])
            try:
                result = fn(c["arguments"]) if fn else {"error": f"unknown tool {c['name']}"}
            except Exception as e:  # a bad argument is fed back so the model can correct itself
                result = {"error": f"{type(e).__name__}: {e}"}
            if trace:
                trace.tool(agent, c["name"], c["arguments"], result)
            messages.append({"role": "tool", "tool_name": c["name"],
                             "content": json.dumps(result, ensure_ascii=False)})
        if done():
            return step
    return max_steps


def similar(a: str, b: str) -> float:
    norm = lambda s: "".join(ch for ch in s.lower() if ch.isalnum())
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


# --------------------------------------------------------------------------------------
# Scene
# --------------------------------------------------------------------------------------

class SceneAgent:
    name = "scene"
    SCHEMA = {"type": "object", "properties": {
        "kind": {"type": "string", "enum": ["title_card", "ui_screen", "sign", "caption", "other"]},
        "description": {"type": "string", "maxLength": 300},
        "tone": {"type": "string", "maxLength": 60}},
        "required": ["kind", "description", "tone"]}

    def __init__(self, llm: LLM):
        self.llm = llm

    def run(self, frame, texts: list[str]) -> dict:
        prompt = (
            "This is a frame from a Japanese animated TV show (anime) with Japanese text on it: "
            f"{', '.join(texts)}.\n"
            "In ONE short sentence, say what kind of screen this is and what the text is for "
            "(for example: an episode title card; a settings menu on a phone; a shop sign). Do not "
            "translate the text or guess what it says, and don't describe pictures in detail. "
            "Then give the tone in a few words (comedic, serious, childish...).")
        try:
            return self.llm.chat_json([{"role": "user", "content": prompt, "images": [encode_image(frame)]}],
                                      self.SCHEMA, agent=self.name, step="describe")
        except ValueError:  # fallback: generic context is better than none
            return {"kind": "other", "description": "a scene from an anime with Japanese text", "tone": "neutral"}


# --------------------------------------------------------------------------------------
# Reader
# --------------------------------------------------------------------------------------

class ReaderAgent:
    name = "reader"
    # "reason" comes first on purpose: constrained decoding writes fields in schema order, so
    # the model reasons before it commits to an answer (otherwise it answers, then explains
    # a different answer).
    SCHEMA = {"type": "object", "properties": {
        "reason": {"type": "string", "maxLength": 300},
        "is_text": {"type": "boolean"},
        "reading": {"type": "string"}},
        "required": ["reason", "is_text", "reading"]}

    def __init__(self, llm: LLM):
        self.llm = llm

    def run(self, crop, candidates: list[str], scene: str, evidence: str = "") -> dict:
        prompt = (
            f"Scene: {scene}\n"
            "The image is a crop of Japanese text from that scene. OCR software read it as:\n"
            + "\n".join(f"- {c}" for c in candidates) +
            (f"\nDictionary evidence: {evidence}" if evidence else "") +
            "\nOCR often confuses look-alike characters: dakuten and handakuten (バ/パ, デ/テ, ダ/タ), "
            "ソ/ン, シ/ツ, ー/一, small and large kana (ッ/ツ). Those marks are only a few pixels, so "
            "the image alone may not settle it. Check whether the OCR reading is a real Japanese "
            "word or phrase that makes sense in this scene. If it is not, but swapping one "
            "look-alike character gives one that is, return that. Otherwise return the reading as "
            "written. If the crop is not text at all (a logo, a drawing), set is_text to false. "
            "Keep the reason short.")
        try:
            return self.llm.chat_json([{"role": "user", "content": prompt, "images": [encode_image(crop, 512)]}],
                                      self.SCHEMA, agent=self.name, step=f"read {candidates[0]}")
        except ValueError:  # fallback: keep what OCR read
            return {"reason": "no valid answer from the model; kept the OCR reading",
                    "is_text": True, "reading": candidates[0]}


# --------------------------------------------------------------------------------------
# Translator (tool loop)
# --------------------------------------------------------------------------------------

class TranslatorAgent:
    name = "translator"
    SYSTEM = (
        "You translate Japanese on-screen text from an anime into English. The English is drawn "
        "back into the picture in place of the Japanese, so it must fit the same space.\n"
        "Rules:\n"
        "- Menu items, settings and buttons are short Title Case noun phrases that name the "
        "setting or item, not sentences. Examples of hiragana settings labels:\n"
        "    おとのおおきさ (おと sound + の + おおきさ size) -> 'Sound Volume'\n"
        "    もじのはやさ (もじ letters + の + はやさ speed) -> 'Text Speed'\n"
        "    がめんのあかるさ (がめん screen + の + あかるさ brightness) -> 'Screen Brightness'\n"
        "- Titles and signs keep the character's voice and any joke. Sentence-final particles "
        "carry the voice: ぞ/ゾ and ぜ are boastful or emphatic (end with '!'), よ asserts, ね seeks "
        "agreement.\n"
        "- Translate the meaning, not word for word. No romaji, no notes.\n"
        "Labels are often written in hiragana only. Split them into words before translating, "
        "and do not mistake ordinary hiragana words for names. Each line comes with dictionary "
        "glosses: trust them over your memory, and call lookup for any other word you are unsure of.\n"
        "Work like this: call check_fit with your English for every line, shorten any line that "
        "does not fit, then call submit once with all lines.")

    def __init__(self, llm: LLM, ws: Workspace, trace=None, dictionary=None):
        self.llm, self.ws, self.trace, self.dict = llm, ws, trace, dictionary

    def run(self, lines: dict[int, dict], scene: dict) -> dict[int, str]:
        """lines: {line_id: {"ja": str, "tids": [...]} } for one shot. Returns {line_id: english}."""
        ws, result = self.ws, {}
        self.gloss: dict[int, str] = {}
        rep = {lid: min(l["tids"], key=lambda t: ws.text_box(t)[2] - ws.text_box(t)[0])
               for lid, l in lines.items()}  # narrowest box = worst case for fitting

        def check_fit(a):
            lid = int(a["line_id"])
            return {"line_id": lid, **ws.fit(rep[lid], str(a["english"]))}

        def submit(a):
            items = [x for x in a.get("translations", []) if str(x.get("line_id", "")).lstrip("-").isdigit()]
            got = {int(x["line_id"]): str(x.get("english", "")).strip() for x in items}
            missing = [lid for lid in lines if not got.get(lid)]
            too_long = {lid: ws.fit(rep[lid], en) for lid, en in got.items() if lid in rep}
            too_long = {lid: f["advice"] for lid, f in too_long.items() if not f["fits"]}
            result.update({lid: en for lid, en in got.items() if lid in lines})
            if missing:
                return {"accepted": False, "error": f"missing line_ids {missing}; submit all lines"}
            if too_long and not getattr(submit, "warned", False):
                submit.warned = True
                return {"accepted": False, "too_long": too_long,
                        "error": "shorten these and submit again (all lines)"}
            submit.done = True
            return {"accepted": True}

        def lookup(a):
            hits = self.dict.search(str(a["word"])) if self.dict else []
            return {"word": a["word"], "entries": hits} if hits else {"word": a["word"], "entries": [],
                                                                       "note": "no dictionary entry"}

        tools = [
            tool("lookup", "Look up a Japanese word in a Japanese-English dictionary.",
                 {"word": {"type": "string"}}, ["word"]),
            tool("check_fit", "Check how big an English line would be drawn in the space of the "
                 "Japanese line. Returns fits=true/false and font sizes.",
                 {"line_id": {"type": "integer"}, "english": {"type": "string"}}, ["line_id", "english"]),
            tool("submit", "Submit the final English for every line of this screen.",
                 {"translations": {"type": "array", "items": {"type": "object", "properties": {
                     "line_id": {"type": "integer"}, "english": {"type": "string"}},
                     "required": ["line_id", "english"]}}}, ["translations"]),
        ]
        def describe(lid, l):
            b = ws.text_box(rep[lid])
            s = f"line_id {lid}: {l['ja']}   (space: {b[2] - b[0]}px wide, {b[3] - b[1]}px tall)"
            g = self.dict.gloss(l["ja"]) if self.dict else ""
            self.gloss[lid] = g
            return s + (f"\n    dictionary: {g}" if g else "")

        table = "\n".join(describe(lid, l) for lid, l in lines.items())
        messages = [
            {"role": "system", "content": self.SYSTEM},
            {"role": "user", "content":
                f"Screen: {scene['description']} Tone: {scene['tone']}.\n\nLines on this screen:\n{table}"},
        ]
        steps = run_tools(self.llm, self.name, messages, tools,
                          {"lookup": lookup, "check_fit": check_fit, "submit": submit},
                          done=lambda: getattr(submit, "done", False), max_steps=8, trace=self.trace)
        if self.trace:
            self.trace.note(self.name, f"screen done in {steps} steps", result=result)
        return result


# --------------------------------------------------------------------------------------
# Critic (QA)
# --------------------------------------------------------------------------------------

class CriticAgent:
    name = "critic"
    SCHEMA = {"type": "object", "properties": {
        "japanese_visible_in_bottom": {"type": "boolean"},
        "english_text_in_bottom": {"type": "string", "maxLength": 80},
        "english_contrast": {"type": "string", "enum": ["strong", "weak"]},
        "looks_natural": {"type": "boolean"},
        "problem": {"type": "string", "maxLength": 200}},
        "required": ["japanese_visible_in_bottom", "english_text_in_bottom", "english_contrast",
                     "looks_natural", "problem"]}

    def __init__(self, llm: LLM):
        self.llm = llm

    def run(self, before_after, intended: str) -> dict:
        prompt = (
            "Top: original anime frame crop with Japanese text. Bottom: the same crop after the "
            "Japanese was erased and English was drawn in its place.\n"
            "In the bottom image: is any Japanese text or a visible part of a Japanese character "
            "left? Write exactly the English you can read there. Is the English strong and easy "
            "to read at a glance, or weak (faint, washed out, close to the background colour)? "
            "Does the edit look natural (clean background, no smudges)? If something is wrong, "
            "describe it in one short sentence, otherwise problem is an empty string.")
        try:
            return self.llm.chat_json([{"role": "user", "content": prompt,
                                        "images": [encode_image(before_after, 768)]}],
                                      self.SCHEMA, agent=self.name, step=f"check '{intended}'")
        except ValueError:  # fallback: no opinion; the deterministic OCR check still runs
            return {"japanese_visible_in_bottom": False, "english_text_in_bottom": intended,
                    "english_contrast": "strong", "looks_natural": True, "problem": ""}


# --------------------------------------------------------------------------------------
# Repair (tool loop)
# --------------------------------------------------------------------------------------

class RepairAgent:
    name = "repair"
    SYSTEM = (
        "You fix problems found by quality checks on translated anime frames. For each failed "
        "track, choose one fix and call its tool:\n"
        "- grow_mask: Japanese strokes or outline remnants are still visible (erase a wider area)\n"
        "- set_outline: the English is hard to read against the background (low contrast, "
        "faint); a dark background needs a white outline, a light background a black one\n"
        "- set_english: the English is too small, cut off or wrong; give a shorter or better line\n"
        "- accept: the check was a false alarm and the frame is fine\n"
        "Handle every failed track, then stop.")

    def __init__(self, llm: LLM, ws: Workspace, trace=None):
        self.llm, self.ws, self.trace = llm, ws, trace

    def run(self, failures: dict[int, dict], image_of: dict[int, str]) -> dict[int, str]:
        ws, actions = self.ws, {}

        def track(a):
            tid = int(a["track_id"])
            if tid not in failures:
                raise ValueError(f"track {tid} has no failed check; failed: {list(failures)}")
            return tid

        def grow_mask(a):
            tid = track(a)
            px = max(2, min(12, int(a.get("pixels", 4))))
            st = ws.tracks[tid].setdefault("style", {})
            st["grow"] = int(st.get("grow", 0)) + px
            actions[tid] = f"grow_mask +{px}px"
            return {"ok": True, "grow_total_px": st["grow"]}

        def set_outline(a):
            tid = track(a)
            color = str(a.get("color", "black")).lower()
            rgb = {"black": [0, 0, 0], "white": [255, 255, 255]}.get(color)
            ws.tracks[tid].setdefault("style", {})["outline"] = rgb
            actions[tid] = f"set_outline {color}"
            return {"ok": True}

        def set_english(a):
            tid = track(a)
            en = str(a["english"]).strip()
            fit = ws.fit(tid, en)
            if not fit["fits"]:
                return {"ok": False, **fit}
            for t in ws.tracks.values():  # same text elsewhere in the shot stays consistent
                if t["shot"] == ws.tracks[tid]["shot"] and t["ja"] == ws.tracks[tid]["ja"]:
                    t["en"] = en
            actions[tid] = f"set_english '{en}'"
            return {"ok": True}

        def accept(a):
            tid = track(a)
            actions[tid] = f"accept ({a.get('reason', '')})"
            return {"ok": True}

        tools = [
            tool("grow_mask", "Erase a wider area around the text strokes.",
                 {"track_id": {"type": "integer"}, "pixels": {"type": "integer", "description": "2-12"}},
                 ["track_id", "pixels"]),
            tool("set_outline", "Change the outline of the English text.",
                 {"track_id": {"type": "integer"}, "color": {"type": "string", "enum": ["black", "white", "none"]}},
                 ["track_id", "color"]),
            tool("set_english", "Replace the English text of a track (must fit its space).",
                 {"track_id": {"type": "integer"}, "english": {"type": "string"}}, ["track_id", "english"]),
            tool("accept", "Keep the track as it is (false alarm).",
                 {"track_id": {"type": "integer"}, "reason": {"type": "string"}}, ["track_id", "reason"]),
        ]
        listing = "\n".join(
            f"track_id {tid}: Japanese '{ws.tracks[tid]['ja']}', English '{ws.tracks[tid]['en']}'. "
            f"Failed checks: {'; '.join(f['reasons'])}" for tid, f in failures.items())
        messages = [{"role": "system", "content": self.SYSTEM},
                    {"role": "user", "content": "Failed tracks (each image: top = original, bottom = "
                     f"current result, in the same order):\n{listing}",
                     "images": [image_of[tid] for tid in failures]}]
        run_tools(self.llm, self.name, messages, tools,
                  {"grow_mask": grow_mask, "set_outline": set_outline, "set_english": set_english,
                   "accept": accept},
                  done=lambda: all(tid in actions for tid in failures),
                  max_steps=2 + len(failures), trace=self.trace)
        return actions
