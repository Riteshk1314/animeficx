# animefix: pitch notes (not part of the repo)

## One line
An agentic pipeline that replaces Japanese text burned into anime with English. Computer-vision
models do the pixel work, and a local vision-language-model agent reads, translates, checks and
repairs its own output, offline on a laptop GPU.

## 60-second version
Burned-in text in anime (title cards, phone screens, signs) is part of the image, so
translating it means detecting it, reading it, translating it, erasing it without damaging the
art, and redrawing English in the same style, consistently across frames.

I split the system in two:
- **Deterministic CV** for everything that touches pixels: CRAFT text detection, tracking, LaMa
  inpainting, typesetting.
- **An agent layer** for judgement: a 4B vision-language model (Qwen3-VL) running locally through
  Ollama. It corrects OCR misreads using dictionary evidence, translates a whole screen at once
  using tools (a dictionary lookup, and a fit check that runs the real layout engine), then
  critiques the rendered result and repairs failures through typed tools.

The agents only edit structured state, so a bad model decision can't damage a frame. On my test
clip the agent fixed every error the Google Translate baseline made, removed 100% of detectable
Japanese, and left everything outside the edits untouched (40.6 dB PSNR, re-encoding loss only).

The most interesting lesson came from fault injection. I made the English nearly invisible and
the vision model still "read" it back perfectly. So the critic now *measures* contrast instead
of asking the model. Use the model for judgement; measure what can be measured.

## Résumé bullets (pick 2–3)
- Built an agentic video-translation pipeline: a local 4B vision-language model (Qwen3-VL via
  Ollama) orchestrates CV tools (CRAFT OCR, LaMa inpainting) to replace burned-in Japanese text in
  anime; 100% of detectable text removed with 40.6 dB PSNR outside edits, on an 8 GB laptop GPU.
- Designed tool-using agents (ReAct loops, JSON-schema-constrained outputs, dictionary retrieval,
  layout-grounded fit checks) with guardrails, an evaluator-optimizer critique/repair loop,
  checkpoint/resume, and a full JSONL trace of every model and tool call.
- Corrected all four translation errors of an OCR → Google Translate baseline, including OCR
  misreads, by grounding a small model in dictionary evidence; validated the repair loop with fault
  injection (2/2 faults caught and fixed in one round).
- Cut redundant compute by about 45% for detection and 50% for rendering by exploiting held
  frames in animation (change detection at 1/4 resolution, inpainted-patch reuse).

## 3-minute demo
1. Before/after video side by side (title card, then the menu).
2. `agent_report.md`: the corrected reading with the model's reason, the translation table with
   dictionary glosses, the flag for the cut-off row.
3. `agent_trace.jsonl`: one translator turn, where the model calls `lookup`, then `check_fit`,
   then `submit`.
4. `scripts/fault_injection.py`: the critic catches invisible text, and the repair agent picks a
   black outline.

## Questions you'll likely get

**Why not let the agent plan everything itself?**
A 4B model is unreliable at long-horizon planning, but good at narrow judgements. So a
deterministic orchestrator runs the phases, and LLM workers make the decisions inside them
(orchestrator-workers plus evaluator-optimizer). Autonomy goes where the model is reliable.

**Why a local 4B model instead of GPT or Claude?**
No API cost, it runs offline, and the constraint was an 8 GB GPU shared with EasyOCR and LaMa.
The design compensates for the small model: retrieval, measured checks, constrained decoding,
guardrails. Any Ollama model can be swapped in with `--model`.

**How do you know the agent's output is good?**
Several signals:
- an OCR re-check for leftover Japanese;
- a WCAG contrast ratio;
- a read-back test;
- the VLM's own judgement;
- objective video metrics (text removed, PSNR outside edits, flicker);
- fault injection to prove the critic can catch failures.

**What went wrong while building it?**
Use the "What testing changed" table in `docs/DESIGN.md`:
- the thinking-model latency (124 s per step);
- the answer-before-reason ordering in the JSON schema;
- runaway generations;
- the scene description biasing the translator;
- the read-back blind spot.

**How do you stop the model from making things worse?**
- Agents never touch pixels; they only edit `regions.json`.
- Corrections must stay similar to what OCR saw.
- Tool parameters are clamped.
- Steps and rounds are budgeted.
- Every agent has a fallback.
- Logos are protected at the pixel level.

**What would you do next?**
- Turn reviewer flags into new tracks using the VLM's bounding-box output (grounding).
- Video inpainting (ProPainter) for moving text.
- Build a small labelled set to measure translation quality (chrF/COMET) and calibrate the critic.
