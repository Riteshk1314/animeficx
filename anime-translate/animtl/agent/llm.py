"""Minimal client for a local vision-language model served by Ollama.

Only the standard library is used (no SDK). Supports what the agents need:
  - images in messages (the model *looks* at frames and crops),
  - tool calling (the model decides which tool to run and with what arguments),
  - structured output (a JSON schema the reply must match; Ollama enforces it with
    grammar-constrained decoding, so a 4B model can't return malformed JSON).
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.request
from dataclasses import dataclass, field

import cv2
import numpy as np

# The instruct variant answers directly; the default "qwen3-vl:4b" tag is the thinking variant,
# which spends thousands of hidden reasoning tokens per call (minutes on a laptop GPU).
DEFAULT_MODEL = os.environ.get("ANIMTL_MODEL", "qwen3-vl:4b-instruct")
HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
if not HOST.startswith("http"):
    HOST = "http://" + HOST


def encode_image(img_bgr: np.ndarray, max_side: int = 768) -> str:
    """BGR array -> base64 PNG, downscaled so the image costs a bounded number of tokens."""
    h, w = img_bgr.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img_bgr = cv2.resize(img_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buf.tobytes()).decode()


@dataclass
class Reply:
    content: str
    tool_calls: list[dict] = field(default_factory=list)  # [{"name": ..., "arguments": {...}}]
    prompt_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0

    def json(self):
        return json.loads(self.content)


class LLM:
    def __init__(self, model: str = DEFAULT_MODEL, num_ctx: int = 8192, temperature: float = 0.2,
                 trace=None):
        self.model, self.num_ctx, self.temperature, self.trace = model, num_ctx, temperature, trace

    def _post(self, path: str, body: dict, timeout: float = 600) -> dict:
        req = urllib.request.Request(HOST + path, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def available(self) -> bool:
        try:
            tags = self._post("/api/show", {"model": self.model}, timeout=10)
            return bool(tags)
        except Exception:
            return False

    def chat(self, messages: list[dict], tools: list[dict] | None = None, schema: dict | None = None,
             agent: str = "", step: str = "", max_tokens: int = 1024,
             temperature: float | None = None) -> Reply:
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,  # answers directly; reasoning goes in the visible tool calls / JSON
            "keep_alive": "10m",
            # num_predict bounds a degenerate repetition loop (small models occasionally get stuck)
            "options": {"num_ctx": self.num_ctx, "num_predict": max_tokens,
                        "temperature": self.temperature if temperature is None else temperature},
        }
        if tools:
            body["tools"] = tools
        if schema:
            body["format"] = schema
        t0 = time.time()
        r = self._post("/api/chat", body)
        msg = r.get("message", {})
        calls = []
        for c in msg.get("tool_calls") or []:
            fn = c.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            calls.append({"name": fn.get("name", ""), "arguments": args})
        reply = Reply(content=msg.get("content", "") or "", tool_calls=calls,
                      prompt_tokens=r.get("prompt_eval_count", 0) or 0,
                      output_tokens=r.get("eval_count", 0) or 0,
                      seconds=time.time() - t0)
        if self.trace is not None:
            self.trace.llm(agent, step, messages, reply)
        return reply

    def chat_json(self, messages: list[dict], schema: dict, agent: str = "", step: str = "",
                  max_tokens: int = 400, retries: int = 2) -> dict:
        """Structured call that always returns parsed JSON or raises ValueError.

        Grammar-constrained decoding guarantees the *shape* of the JSON, not that the model
        finishes it: a model stuck repeating a phrase inside a string hits the token cap and
        the JSON ends unterminated. Then retry, asking for brevity, with a little more
        randomness to break the loop.
        """
        msgs = messages
        for attempt in range(retries + 1):
            r = self.chat(msgs, schema=schema, agent=agent, max_tokens=max_tokens,
                          step=step + (f" (retry {attempt})" if attempt else ""),
                          temperature=None if attempt == 0 else 0.7)
            try:
                return r.json()
            except json.JSONDecodeError:
                last = dict(messages[-1])
                last["content"] = last.get("content", "") + (
                    "\n\nBe brief: every text field is one short sentence at most.")
                msgs = messages[:-1] + [last]
        raise ValueError(f"{agent}: no valid JSON after {retries + 1} attempts")

    def unload(self) -> None:
        """Free the GPU memory before LaMa / EasyOCR need it."""
        try:
            self._post("/api/generate", {"model": self.model, "keep_alive": 0}, timeout=30)
        except Exception:
            pass
