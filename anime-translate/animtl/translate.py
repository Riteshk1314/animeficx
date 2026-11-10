"""Stage 2: translate the recognised Japanese into short on-screen English.

Backends:
  google  - Google Translate. With GOOGLE_TRANSLATE_API_KEY (or GOOGLE_API_KEY) set it uses
            the official Cloud Translation API; without a key it uses the free endpoint that
            translate.google.com itself uses (no account needed, fine for a few dozen strings).
  none    - leave `en` empty; fill regions.json by hand.

Edit regions.json afterwards to fix anything; `render` uses whatever is in `en`.
"""

from __future__ import annotations

import json
import os
import string
import time
import urllib.parse
import urllib.request
from pathlib import Path

FREE_URL = "https://translate.googleapis.com/translate_a/single"
CLOUD_URL = "https://translation.googleapis.com/language/translate/v2"
USER_AGENT = "Mozilla/5.0 (animtl)"


def _unique_strings(tracks: list[dict], force: bool) -> list[str]:
    seen: dict[str, None] = {}
    for t in tracks:
        if t["ja"] and (force or not t.get("en")):
            seen.setdefault(t["ja"], None)
    return list(seen)


def _get_json(req: urllib.request.Request, tries: int = 3):
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))


def _google_free(strings: list[str], target: str) -> list[str]:
    out = []
    for s in strings:
        q = urllib.parse.urlencode({"client": "gtx", "sl": "ja", "tl": target, "dt": "t", "q": s})
        data = _get_json(urllib.request.Request(f"{FREE_URL}?{q}", headers={"User-Agent": USER_AGENT}))
        out.append("".join(part[0] for part in data[0] if part and part[0]))
        time.sleep(0.2)  # be polite to an endpoint without a quota
    return out


def _google_cloud(strings: list[str], key: str, target: str) -> list[str]:
    out = []
    for i in range(0, len(strings), 100):  # API limit: 128 strings per request
        body = json.dumps({"q": strings[i:i + 100], "source": "ja", "target": target,
                           "format": "text"}).encode("utf-8")
        req = urllib.request.Request(f"{CLOUD_URL}?key={urllib.parse.quote(key)}", data=body,
                                     headers={"Content-Type": "application/json"})
        out += [t["translatedText"] for t in _get_json(req)["data"]["translations"]]
    return out


def _tidy(ja: str, en: str) -> str:
    """On-screen text conventions: labels in Title Case, no stray sentence period."""
    en = " ".join(en.split())
    if not en:
        return en
    sentence = ja.endswith(("。", "！", "？", "!", "?", "…")) or len(en.split()) > 4
    if sentence:
        return en[0].upper() + en[1:]
    if en.endswith("."):
        en = en[:-1]
    return string.capwords(en)


def google(strings: list[str], target: str = "en") -> dict[str, str]:
    key = os.environ.get("GOOGLE_TRANSLATE_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    print(f"[translate] Google Translate ({'Cloud API' if key else 'free endpoint'})")
    raw = _google_cloud(strings, key, target) if key else _google_free(strings, target)
    return {s: _tidy(s, r) for s, r in zip(strings, raw)}


def translate(work: Path, backend: str = "google", force: bool = False) -> dict:
    path = work / "regions.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    tracks = data["tracks"]
    strings = _unique_strings(tracks, force)
    if not strings:
        print("[translate] nothing to translate (use --force to redo existing translations)")
        return data

    print(f"[translate] {len(strings)} unique strings via {backend}")
    if backend == "google":
        table = google(strings)
    elif backend == "none":
        table = {}
    else:
        raise ValueError(f"unknown translator: {backend}")

    for t in tracks:
        if t["ja"] in table and (force or not t.get("en")):
            t["en"] = table[t["ja"]]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    for ja in strings:
        print(f"  {ja}  ->  {table.get(ja, '(untranslated)')}")
    return data
