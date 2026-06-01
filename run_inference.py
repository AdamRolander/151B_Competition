#!/usr/bin/env python
"""Single entry point for grading.

Usage:
    python run_inference.py

Loads the base model (Qwen/Qwen3-4B-Thinking-2507) for MCQ and the fine-tuned
FRQ LoRA adapter (Aerolandaz/cap8-frq) from the HuggingFace Hub, runs the full
routed pipeline on data/private.jsonl, and writes results/submission.csv.

All hyperparameters are the final ones used for our submission and live inside
infer.run_inference().
"""
import sys
from pathlib import Path
import os
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")

# infer.py lives one level below the repo root (e.g. src/infer.py).
# Locate it automatically and put its directory on the import path so its
# internal `import prompts` / `from postprocess import ...` resolve.
_ROOT = Path(__file__).resolve().parent
_candidates = sorted(_ROOT.glob("*/infer.py"))
if not _candidates:
    raise FileNotFoundError(
        f"Could not find */infer.py under {_ROOT}. "
        "Place run_inference.py at the repo root."
    )
sys.path.insert(0, str(_candidates[0].parent))

from infer import run_inference  # noqa: E402

if __name__ == "__main__":
    run_inference()