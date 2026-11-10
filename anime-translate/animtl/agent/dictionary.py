"""Japanese-English dictionary lookups (JMdict, via the free jisho.org API), cached on disk.

A 4B model doesn't know every word. It confuses はやさ ("speed") with やすい ("easy"),
and will happily invent a character called "Kapipara". Retrieval grounds it in two ways:
  - before the model is asked anything, each line is split into dictionary words and glossed
    (retrieval-augmented prompting);
  - the translator can also call `lookup` itself for anything else (tool use).
jisho's search is fuzzy, so looking up a misread word (カピパラ) returns the real one (カピバラ),
which is exactly the evidence the reader agent needs. Offline, lookups return nothing and the
agents carry on without hints.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

URL = "https://jisho.org/api/v1/search/words?keyword="
PARTICLES = set("のをはがにでともへや")


def kana(s: str) -> str:
    """Katakana -> hiragana, so 'イヌ' on screen matches the dictionary reading 'いぬ' (犬)."""
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)


class Dictionary:
    def __init__(self, cache: Path):
        self.cache_path = cache
        try:
            self.cache = json.loads(cache.read_text(encoding="utf-8"))
        except Exception:
            self.cache = {}
        self.online = True

    def _save(self) -> None:
        try:
            self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def search(self, word: str, n: int = 3) -> list[dict]:
        """[{word, reading, meaning}] best first; [] if not found or offline."""
        word = word.strip()
        if not word:
            return []
        if word in self.cache:
            return self.cache[word][:n]
        if not self.online:
            return []
        try:
            req = urllib.request.Request(URL + urllib.parse.quote(word), headers={"User-Agent": "animtl"})
            with urllib.request.urlopen(req, timeout=15) as r:
                data = json.loads(r.read().decode("utf-8"))["data"]
        except Exception:
            self.online = False  # don't keep paying timeouts
            return []
        out = []
        for e in data[:5]:
            j = e["japanese"][0]
            out.append({"word": j.get("word") or j.get("reading", ""), "reading": j.get("reading", ""),
                        "meaning": "; ".join(e["senses"][0]["english_definitions"][:3])})
        self.cache[word] = out
        self._save()
        time.sleep(0.2)
        return out[:n]

    def segment(self, text: str, max_words: int = 6) -> list[dict]:
        """Greedy longest-match split into dictionary words, each with its gloss.

        Matching is kana-insensitive (anime UIs write animal names in katakana, dictionaries
        list them under hiragana readings). Ties go to jisho's ranking, so サル is 猿 (monkey),
        not ＳＡＬ. If the rest can't be matched, segmentation stops: a wrong gloss is worse
        than none."""
        rest, out = text, []
        while rest and len(out) < max_words:
            best = None
            key = kana(rest)
            for e in self.search(rest, 5):
                for form in (e["reading"], e["word"]):
                    f = kana(form) if form else ""
                    if f and key.startswith(f) and (best is None or len(f) > best[0]):
                        best = (len(f), e)
            if best:
                n, e = best
                out.append({"text": rest[:n], "word": e["word"], "meaning": e["meaning"]})
                rest = rest[n:]
            elif rest[0] in PARTICLES:
                out.append({"text": rest[0], "word": rest[0], "meaning": "(particle)"})
                rest = rest[1:]
            else:
                break
        return out

    def gloss(self, text: str) -> str:
        """One line for a prompt: 'はなす = 話す (to talk) + はやさ = 速さ (speed)'."""
        parts = self.segment(text)
        return " + ".join(f"{p['text']} = {p['word']} ({p['meaning']})" if p["word"] != p["text"]
                          else f"{p['text']} ({p['meaning']})" for p in parts)
