"""animtl: replace burned-in Japanese text in anime with English.

perception/  find, track and read the text (deterministic CV)
agent/       local vision-language model agents: read, translate, critique, repair
editing/     erase (LaMa) and typeset (deterministic, the only code that changes pixels)
eval/        objective metrics for an output video
"""

__version__ = "0.2.0"
