"""Structured trace of everything the agents do: every model call, tool call and decision.

Written as JSON lines to <work>/agent_trace.jsonl (one event per line) and echoed to the
console in a readable form. The trace is how you audit an agent: what it saw, what it
decided, which tool it ran, and what came back.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


def _short(v, n: int = 160) -> str:
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


class Trace:
    def __init__(self, path: Path, echo: bool = True, append: bool = False):
        self.path, self.echo = path, echo
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a" if append else "w", encoding="utf-8")
        self.t0 = time.time()
        self.stats = {"llm_calls": 0, "tool_calls": 0, "prompt_tokens": 0, "output_tokens": 0,
                      "llm_seconds": 0.0}

    def _write(self, ev: dict) -> None:
        ev["t"] = round(time.time() - self.t0, 2)
        self.f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        self.f.flush()

    def phase(self, name: str, detail: str = "") -> None:
        self._write({"type": "phase", "name": name, "detail": detail})
        if self.echo:
            print(f"\n== {name} {('- ' + detail) if detail else ''}")

    def llm(self, agent: str, step: str, messages: list[dict], reply) -> None:
        s = self.stats
        s["llm_calls"] += 1
        s["prompt_tokens"] += reply.prompt_tokens
        s["output_tokens"] += reply.output_tokens
        s["llm_seconds"] += reply.seconds
        n_img = sum(len(m.get("images") or []) for m in messages)
        self._write({"type": "llm", "agent": agent, "step": step, "images": n_img,
                     "prompt": messages[-1].get("content", "")[:2000],
                     "reply": reply.content[:4000], "tool_calls": reply.tool_calls,
                     "prompt_tokens": reply.prompt_tokens, "output_tokens": reply.output_tokens,
                     "seconds": round(reply.seconds, 2)})
        if self.echo:
            what = ", ".join(f"{c['name']}({_short(c['arguments'], 90)})" for c in reply.tool_calls) \
                or _short(reply.content)
            print(f"  [{agent}] {step} ({n_img} img, {reply.seconds:.1f}s) -> {what}")

    def tool(self, agent: str, name: str, args: dict, result) -> None:
        self.stats["tool_calls"] += 1
        self._write({"type": "tool", "agent": agent, "name": name, "args": args, "result": result})
        if self.echo:
            print(f"  [{agent}]   tool {name} -> {_short(result, 120)}")

    def note(self, agent: str, text: str, **data) -> None:
        self._write({"type": "note", "agent": agent, "text": text, **data})
        if self.echo:
            print(f"  [{agent}] {text}")

    def close(self) -> dict:
        self.stats["llm_seconds"] = round(self.stats["llm_seconds"], 1)
        self.stats["wall_seconds"] = round(time.time() - self.t0, 1)
        self._write({"type": "summary", **self.stats})
        self.f.close()
        return self.stats
