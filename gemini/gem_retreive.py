#!/usr/bin/env python
"""
retreive.py (Gemini)

This project originally used OpenAI's Batch API (submit -> retrieve).
For Gemini, we run generation locally (no separate "retrieve" step).

This script is kept for backwards compatibility with your old workflow:
  1) prompt.py   -> writes batchinput_*.jsonl
  2) generate.py -> calls Gemini and writes batchoutput_*.jsonl
  3) retreive.py -> (this file) simply calls generate.py again (safe/no-op if outputs exist)

So you can still run retreive.py out of habit, and it won't break anything.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

try:
    # If you renamed generate.py -> gem_generate.py
    from gem_generate import run  # type: ignore
except Exception:
    from generate import run



@hydra.main(version_base=None, config_path=".", config_name="setting")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
