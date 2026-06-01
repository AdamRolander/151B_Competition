# CSE 151B Math Reasoning Competition — Final Submission

Routed inference over **Qwen3-4B-Thinking-2507**: the base model answers
multiple-choice questions, and a fine-tuned LoRA adapter
([`Aerolandaz/cap8-frq`](https://huggingface.co/Aerolandaz/cap8-frq)) answers
free-response questions, with adaptive multi-sampling and per-position symbolic
voting.

## Hardware & runtime

- **GPU:** NVIDIA RTX 5090
- **Approximate total inference time on the private set (502 questions):** 7 hours
- Single GPU, bf16, no quantization. vLLM serves the base model and applies the
  FRQ adapter per-request, so only one model is resident in VRAM.

## Setup

Tested with **torch 2.8.0+cu128 (CUDA 12.8)**. `requirements.txt` intentionally
omits `torch` and the CUDA wheels — install a torch build matching your CUDA
**first**, then the rest:

```bash
# 1) install torch for your CUDA (example: CUDA 12.8). Skip if your image
#    already ships a CUDA-enabled torch.
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# 2) install the inference dependencies
pip install -r requirements.txt
```

## Model weights

Both models are fetched automatically on the first run:

- **Base (MCQ):** `Qwen/Qwen3-4B-Thinking-2507` — downloaded by vLLM from the Hub.
- **FRQ adapter:** `Aerolandaz/cap8-frq` (public LoRA adapter) — downloaded by
  `run_inference()` via `huggingface_hub.snapshot_download` and applied as a
  vLLM `LoRARequest`. No login or token required.

The only data file that must already be present is the private test set at
**`data/private.jsonl`** (included in this repo).

## Reproducing the submission

A single function runs the entire pipeline end-to-end — model loading, routed
generation, post-processing (per-position sympy voting, format cleaning), and
CSV output. No manual or intermediate steps.

```bash
VLLM_USE_DEEP_GEMM=0 python run_inference.py
```

The `VLLM_USE_DEEP_GEMM=0` flag is **required**: the model is bf16, so it does
not use FP8 kernels, and this skips vLLM's optional FP8 (DeepGEMM) warmup that
otherwise errors when the `deep_gemm` package is absent. Please run with exactly
this command.

Equivalently, from Python:

```python
import os
os.environ["VLLM_USE_DEEP_GEMM"] = "0"
from infer import run_inference   # infer.py is in src/
run_inference()                   # -> writes results/submission.csv
```

Output is written to **`results/submission.csv`** (columns `id,response`, full
reasoning traces, CSV-quoted).

## Final configuration (baked into `run_inference()`)

All hyperparameters below are the exact values used for our submission
(71.64% validation accuracy as of 2026-05-25).

| Tier             | Model           | Samples (N) | Temperature | Aggregation               |
| ---------------- | --------------- | ----------- | ----------- | ------------------------- |
| MCQ              | base            | 5           | 0.4         | letter majority           |
| FRQ, 1 answer    | `cap8-frq` LoRA | 7           | 0.8         | per-position sympy voting |
| FRQ, 2–3 answers | `cap8-frq` LoRA | 11          | 0.8         | per-position sympy voting |
| FRQ, 4+ answers  | `cap8-frq` LoRA | 17          | 0.8         | per-position sympy voting |

Shared sampling: `top_p=0.95`, `top_k=20`, `max_tokens=30000`, `seed=42`.
Question tier is determined from the `options` field (MCQ) and the number of
`[ANS]` placeholders in the question (FRQ).

Validation breakdown: MCQ 80.60% (54/67), FRQ 67.16% (90/134), Overall 71.64%.

## Repository layout

```
run_inference.py        # single entry point (calls infer.run_inference)
requirements.txt        # inference deps; torch + CUDA wheels intentionally omitted
src/
  infer.py              # routed inference pipeline + run_inference()
  prompts.py            # v3 prompts (build_prompt)
  postprocess.py        # clean_response post-processing
starter/
  judger.py             # competition sympy judger
  utils.py
data/
  private.jsonl         # private test set
results/                # outputs (submission.csv written here)
```

## Notes on reproducibility

Outputs are not byte-identical run to run (sampling temperature > 0), but the
configuration is fixed, so overall performance is stable. The validation run
used `val.jsonl` (tier inferred from gold-answer count); the private run infers
the same tiers from `[ANS]` counts, which match by dataset construction.
