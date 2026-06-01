"""Inference pipeline for the CSE 151B Math Reasoning Competition.

Examples
--------
# Smoke test (5 questions, ~30s):
python src/infer.py --data data/public.jsonl --out results/smoke.jsonl --limit 5

# Full baseline on public set (with local eval):
python src/infer.py --data data/public.jsonl --out results/baseline_public.jsonl

# Submission run on private set:
python src/infer.py --data data/private.jsonl --out results/baseline_private.jsonl \\
    --no-eval --csv results/submission_baseline.csv

# Self-consistency (8 samples per question, majority vote):
python src/infer.py --data data/public.jsonl --out results/sc8_public.jsonl \\
    --n-samples 8 --vote majority

# With a LoRA adapter:
python src/infer.py --data data/public.jsonl --out results/lora_public.jsonl \\
    --lora /path/to/adapter

# ROUTED + ADAPTIVE MULTI-SAMPLE (new):
#   - MCQ: base model + v3 prompt, N=5, simple letter-majority vote
#   - FRQ: LoRA adapter, adaptive N by tier, per-position judger voting
python src/infer.py --data data/private.jsonl --out results/routed_private.jsonl \\
    --csv results/submission_routed.csv --no-eval \\
    --lora-frq checkpoints/qwen3-4b-rft-v3-frq-cap8/checkpoint-1680 \\
    --n-mcq 5 --n-frq1 7 --n-frq-multi 11 --n-frq-hard 17

Notes
-----
- Uses vLLM with bf16 (no quantization). Fits on a 5090 (32 GB) easily.
- Sampling params follow Qwen3-Thinking-2507 official recommendations.
- Routed mode: MCQ runs without LoRA, FRQ runs with the --lora-frq adapter.
  Both share one vLLM instance — adapter applied via LoRARequest per call.
- FRQ voting uses sympy via judger.Judger — per-answer-position voting,
  not whole-string voting. Same logic as scripts/aggregate_multisample.py.
- Record order is preserved in the CSV via original-index tracking.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

# Make sure we can import judger.py from the starter repo and prompts.py from this folder
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))                # for prompts.py
sys.path.insert(0, str(ROOT / "starter"))    # for judger.py + utils.py (clone the starter repo here)

from prompts import build_prompt              # noqa: E402

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"


# ─── IO helpers ────────────────────────────────────────────────────────────────
def load_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path, "r", encoding="utf-8")]


def write_jsonl(path: str, records: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(path: str, records: list[dict], picked_response: list[str]) -> None:
    """Write a Kaggle-format submission CSV with proper escaping."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["id", "response"])
        for r, resp in zip(records, picked_response):
            w.writerow([r["id"], resp])


# ─── Answer extraction & MCQ scoring (matches what the judger does) ────────────
def strip_thinking(text: str) -> str:
    """Return the post-thinking portion of the response."""
    end = text.rfind("</think>")
    return text[end + len("</think>"):] if end >= 0 else text

def _rebuild_with_boxes(response: str, answers: list[str]) -> str:
    """Keep the response's thinking trace but replace the post-</think> boxed
    group with `answers` (the per-position voted winners). The judger reads only
    the final boxed group, so this submits the voted tuple verbatim."""
    group = " ".join(f"\\boxed{{{a}}}" for a in answers)
    end = response.rfind("</think>")
    if end >= 0:
        return response[:end + len("</think>")] + "\n" + group
    return group

def extract_mcq_letter(text: str) -> str:
    """Extract the chosen letter from a model response."""
    search = strip_thinking(text)
    matches = list(re.finditer(r"\\boxed\{\s*([A-Za-z])\s*\}", search))
    if matches:
        return matches[-1].group(1).upper()
    matches2 = re.findall(r"\b([A-Z])\b", search.upper())
    return matches2[-1] if matches2 else ""


def extract_freeform_answers(text: str) -> list[str]:
    """Extract the LAST contiguous group of \\boxed{...} from post-thinking text.

    Mirrors judger.extract_all_boxed so we can do majority voting locally.
    """
    text = strip_thinking(text)
    entries: list[tuple[int, int, str]] = []
    i = 0
    while True:
        idx = text.find("\\boxed{", i)
        if idx < 0:
            break
        bs = idx + len("\\boxed{")
        depth = 1
        j = bs
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            content = text[bs:j - 1]
            if content:
                entries.append((idx, j, content))
            i = j
        else:
            break
    if not entries:
        return []
    last = [entries[-1]]
    for k in range(len(entries) - 2, -1, -1):
        gap = text[entries[k][1]:entries[k + 1][0]]
        if re.match(r"^[\s,\$\.\;\:\-\&\\]*$", gap):
            last.insert(0, entries[k])
        else:
            break
    return [e[2].strip() for e in last]


# ─── Voting strategies ────────────────────────────────────────────────────────
def vote_majority(responses: list[str], is_mcq: bool) -> tuple[str, str]:
    """Pick the response whose extracted answer is the modal answer.

    String-based voting. Used for MCQ (letter). Kept for backward compat.
    Returns (chosen_response_text, key_used_for_voting).
    """
    if len(responses) == 1:
        return responses[0], ""

    if is_mcq:
        keys = [extract_mcq_letter(r) for r in responses]
    else:
        keys = [" ||| ".join(extract_freeform_answers(r)) for r in responses]

    # Drop empty keys from the count; if all empty, fall back to first response
    counts = Counter(k for k in keys if k)
    if not counts:
        return responses[0], ""
    winner_key, _ = counts.most_common(1)[0]
    # Pick the SHORTEST response whose key matches the winner (cleaner CoT)
    candidates = [(len(r), r) for r, k in zip(responses, keys) if k == winner_key]
    candidates.sort()
    return candidates[0][1], winner_key


def vote_frq_per_position(
    responses: list[str],
    expected_count: int,
    judger,
) -> tuple[str, list[str]]:
    """Per-answer-position majority voting using sympy equivalence.

    For a question with `expected_count` answers:
      1. Extract per-position answers from every response.
      2. For each position, cluster equivalent extracted answers via sympy
         (judger.auto_judge in pairwise mode).
      3. Pick the modal cluster's representative answer per position.
      4. Pick the response that matches the most per-position winners.
         Tiebreak: shortest response.

    Returns (chosen_response_text, list_of_per_position_winning_answers).
    """
    if len(responses) == 1:
        return responses[0], extract_freeform_answers(responses[0])

    # Extract per-response answers; drop responses whose box count doesn't match
    extracted: list[list[str]] = []
    valid_responses: list[str] = []
    for r in responses:
        ans = extract_freeform_answers(r)
        if len(ans) == expected_count:
            extracted.append(ans)
            valid_responses.append(r)

    # Fallback: if no response has the right box count, use string-based voting
    # on whatever was extracted.
    if not extracted:
        chosen, _ = vote_majority(responses, is_mcq=False)
        return chosen, extract_freeform_answers(chosen)

    # For each position, cluster equivalent answers via sympy.
    # We use judger.auto_judge pairwise (treats one as gold, one as pred) to
    # decide equivalence. To avoid O(N^2) per position, we cluster greedily:
    # for each new answer, check if it's equivalent to any existing cluster
    # representative; if so, add to that cluster, else start new cluster.
    n_positions = expected_count
    winning_answers: list[str] = []

    for pos in range(n_positions):
        pos_answers = [e[pos] for e in extracted]
        clusters: list[list[str]] = []  # each cluster is list of strings
        for ans in pos_answers:
            assigned = False
            for cluster in clusters:
                # Check if `ans` is equivalent to any answer in this cluster.
                # Use the cluster's first member as the representative.
                try:
                    is_equiv = judger.auto_judge(
                        pred=f"\\boxed{{{ans}}}",
                        gold=[cluster[0]],
                        options=[[]],
                    )
                except Exception:
                    is_equiv = (ans.strip() == cluster[0].strip())
                if is_equiv:
                    cluster.append(ans)
                    assigned = True
                    break
            if not assigned:
                clusters.append([ans])
        # Modal cluster wins this position. Representative = first in cluster.
        clusters.sort(key=len, reverse=True)
        winning_answers.append(clusters[0][0])

    # Pick the response that matches the most position-winners.
    # Match = sympy-equivalent at every position.
    best_score = -1
    best_response = valid_responses[0]
    best_len = len(best_response)
    for r, ans in zip(valid_responses, extracted):
        score = 0
        for pos in range(n_positions):
            try:
                if judger.auto_judge(
                    pred=f"\\boxed{{{ans[pos]}}}",
                    gold=[winning_answers[pos]],
                    options=[[]],
                ):
                    score += 1
            except Exception:
                if ans[pos].strip() == winning_answers[pos].strip():
                    score += 1
        if score > best_score or (score == best_score and len(r) < best_len):
            best_score = score
            best_response = r
            best_len = len(r)

    return best_response, winning_answers


# ─── Tier classification ──────────────────────────────────────────────────────
def classify_tier(item: dict) -> str:
    """Return one of: 'mcq', 'frq1', 'frq_multi', 'frq_hard'.

    For private data without 'answer', tier is inferred from question text by
    counting [ANS] placeholders. Fallback: assume frq1.
    """
    if item.get("options"):
        return "mcq"
    # Try to use gold/answer length if available (val/public sets)
    if "answer" in item:
        ans = item["answer"]
        if isinstance(ans, list):
            n = len(ans)
        else:
            n = 1
    else:
        # Count [ANS] placeholders as a proxy
        question = item.get("question", "")
        n = question.count("[ANS]")
        if n == 0:
            n = 1
    if n == 1:
        return "frq1"
    elif n <= 3:
        return "frq_multi"
    else:
        return "frq_hard"


# ─── Local eval ────────────────────────────────────────────────────────────────
def eval_records(records: list[dict], picked_responses: list[str]) -> dict:
    """Run the official judger over picked responses and report accuracy."""
    from judger import Judger
    judger = Judger(strict_extract=False)

    n_mcq_c = n_mcq_t = 0
    n_free_c = n_free_t = 0
    failures: list[dict] = []

    for r, response in zip(records, picked_responses):
        gold = r["gold"]
        if r["is_mcq"]:
            correct = extract_mcq_letter(response) == str(gold).strip().upper()
            n_mcq_t += 1
            n_mcq_c += int(correct)
        else:
            gold_list = gold if isinstance(gold, list) else [gold]
            try:
                correct = judger.auto_judge(
                    pred=response,
                    gold=gold_list,
                    options=[[]] * len(gold_list),
                )
            except Exception:
                correct = False
            n_free_t += 1
            n_free_c += int(correct)
        if not correct:
            failures.append({"id": r["id"], "is_mcq": r["is_mcq"], "gold": gold})

    n_t = n_mcq_t + n_free_t
    n_c = n_mcq_c + n_free_c
    summary = {
        "overall_acc": n_c / max(n_t, 1),
        "overall": (n_c, n_t),
        "mcq_acc": n_mcq_c / max(n_mcq_t, 1),
        "mcq": (n_mcq_c, n_mcq_t),
        "free_acc": n_free_c / max(n_free_t, 1),
        "free": (n_free_c, n_free_t),
        "failures": failures,
    }
    print("=" * 60)
    print("LOCAL EVAL")
    print("=" * 60)
    print(f"  MCQ:       {n_mcq_c}/{n_mcq_t}  ({100 * summary['mcq_acc']:.2f}%)")
    print(f"  Free-form: {n_free_c}/{n_free_t}  ({100 * summary['free_acc']:.2f}%)")
    print(f"  Overall:   {n_c}/{n_t}  ({100 * summary['overall_acc']:.2f}%)")
    print("=" * 60)
    return summary


# ─── Generation helpers ───────────────────────────────────────────────────────
def build_prompts_for_items(items: list[dict], tokenizer) -> list[str]:
    """Build chat-formatted prompts for a list of input items."""
    prompts = []
    for item in items:
        sys_p, usr_p = build_prompt(item["question"], item.get("options"))
        msgs = [{"role": "system", "content": sys_p},
                {"role": "user", "content": usr_p}]
        prompts.append(tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True))
    return prompts


def generate_batch(llm, prompts, n_samples, args, lora_request=None,
                   temperature_override=None):
    """Generate completions for a batch of prompts. Returns list-of-lists
    (one inner list of N completions per prompt)."""
    from vllm import SamplingParams
    temperature = temperature_override if temperature_override is not None \
        else args.temperature
    sampling = SamplingParams(
        n=n_samples,
        max_tokens=args.max_tokens,
        temperature=temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=0.0,
        presence_penalty=0.0,
    )
    if lora_request is not None:
        outputs = llm.generate(prompts, sampling_params=sampling,
                               lora_request=lora_request)
    else:
        outputs = llm.generate(prompts, sampling_params=sampling)
    return [[c.text for c in o.outputs] for o in outputs]


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora-mcq", default=None,
                    help="Adapter for MCQ (None = base).")
    ap.add_argument("--lora-frq1", default=None,
                    help="Adapter for single-answer FRQ (None = base).")
    ap.add_argument("--lora-frq-multi", default=None,
                    help="Adapter for 2-3 answer FRQ (None = base; "
                         "falls back to --lora-frq if unset).")
    ap.add_argument("--lora-frq-hard", default=None,
                    help="Adapter for 4+ answer FRQ (None = base; "
                         "falls back to --lora-frq if unset).")
    ap.add_argument("--model", default=MODEL_ID,
                help="Base MODEL_ID (default) or a full fine-tuned "
                     "checkpoint dir, for full-SFT eval.")
    ap.add_argument("--data", required=True, help="Input JSONL (public or private)")
    ap.add_argument("--out", required=True, help="Output raw JSONL with all samples per question")
    ap.add_argument("--csv", default=None, help="Also write Kaggle CSV submission to this path")
    ap.add_argument("--no-eval", action="store_true",
                    help="Set this for the private set (no ground-truth available)")
    ap.add_argument("--n-samples", type=int, default=1,
                    help="Self-consistency: number of completions per question. "
                         "Ignored when routed (--lora-frq) mode is used.")
    ap.add_argument("--vote", choices=["majority", "first"], default="first",
                    help="When n-samples > 1, how to pick the final response. "
                         "Ignored in routed mode (routed always uses judger voting).")
    ap.add_argument("--limit", type=int, default=None, help="Smoke test: only first N items")
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--max-tokens", type=int, default=30000,
                    help="Max generation tokens (leave headroom for prompt)")
    ap.add_argument("--temperature", type=float, default=0.6,
                    help="Default temperature. In routed mode, can be "
                         "overridden per-tier by --temp-mcq and --temp-frq.")
    ap.add_argument("--temp-mcq", type=float, default=None,
                    help="MCQ temperature in routed mode (overrides "
                         "--temperature). Previous screen: 0.4 worked best "
                         "on base+v3 MCQ.")
    ap.add_argument("--temp-frq", type=float, default=None,
                    help="FRQ temperature in routed mode (overrides "
                         "--temperature). Previous screen: 0.8 worked best "
                         "on sft-rft-v2 FRQ.")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    # Single-model LoRA (unchanged behavior)
    ap.add_argument("--lora", default=None, help="Path to single LoRA adapter directory")
    # Routed mode
    ap.add_argument("--lora-frq", default=None,
                    help="LoRA adapter for FRQ generation. When set, enables "
                         "routed mode: MCQ runs on base, FRQ runs with this "
                         "adapter. Per-tier N from --n-mcq/--n-frq1/etc.")
    ap.add_argument("--n-mcq", type=int, default=5,
                    help="Samples per MCQ in routed mode")
    ap.add_argument("--n-frq1", type=int, default=7,
                    help="Samples per single-answer FRQ in routed mode")
    ap.add_argument("--n-frq-multi", type=int, default=11,
                    help="Samples per 2-3 answer FRQ in routed mode")
    ap.add_argument("--n-frq-hard", type=int, default=17,
                    help="Samples per 4+ answer FRQ in routed mode")
    args = ap.parse_args()

    # Late imports so --help is fast without GPU drivers
    from vllm import LLM
    from transformers import AutoTokenizer

    data = load_jsonl(args.data)
    if args.limit:
        data = data[:args.limit]
    print(f"Loaded {len(data)} questions from {args.data}")
    has_eval = (not args.no_eval) and all("answer" in d for d in data)

    # Tag every record with its original index so the final CSV preserves order
    # regardless of partitioning, vLLM scheduling, or any merge step.
    for i, item in enumerate(data):
        item["_orig_idx"] = i

    routed_mode = any([args.lora_frq, args.lora_mcq, args.lora_frq1,
                       args.lora_frq_multi, args.lora_frq_hard])

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    llm_kwargs = dict(
        model=args.model,
        tokenizer=MODEL_ID,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=True,
        seed=args.seed,
    )
    _distinct_adapters = {p for p in [
        args.lora, args.lora_mcq, args.lora_frq1,
        args.lora_frq_multi or args.lora_frq,
        args.lora_frq_hard or args.lora_frq,
    ] if p}
    if _distinct_adapters or args.lora_frq:
        llm_kwargs.update(enable_lora=True,
                          max_loras=max(1, len(_distinct_adapters)),
                          max_lora_rank=128)

    print("Loading model...")
    llm = LLM(**llm_kwargs)

    # Container for per-record responses, keyed by _orig_idx
    responses_by_idx: dict[int, list[str]] = {}
    tier_by_idx: dict[int, str] = {}

    t0 = time.time()

    if routed_mode:
        # ─── Partition input by tier ───────────────────────────────────────
        tier_groups: dict[str, list[dict]] = defaultdict(list)
        for item in data:
            tier = classify_tier(item)
            tier_by_idx[item["_orig_idx"]] = tier
            tier_groups[tier].append(item)

        print("Routed mode:")
        for tier, items in tier_groups.items():
            print(f"  {tier}: {len(items)} questions")

        from vllm.lora.request import LoRARequest
        from safetensors import safe_open
        import os

        mcq_temp = args.temp_mcq if args.temp_mcq is not None else args.temperature
        frq_temp = args.temp_frq if args.temp_frq is not None else args.temperature

        # Per-tier plan: (adapter_path_or_None, n_samples, temperature).
        # None => base model. Multi/hard fall back to --lora-frq.
        tier_plan = {
            "mcq":       (args.lora_mcq,                        args.n_mcq,       mcq_temp),
            "frq1":      (args.lora_frq1,                       args.n_frq1,      frq_temp),
            "frq_multi": (args.lora_frq_multi or args.lora_frq, args.n_frq_multi, frq_temp),
            "frq_hard":  (args.lora_frq_hard  or args.lora_frq, args.n_frq_hard,  frq_temp),
        }

        # One LoRARequest per DISTINCT adapter path (unique id), verified up front.
        adapter_reqs: dict[str, LoRARequest] = {}
        for path in sorted({p for (p, _, _) in tier_plan.values() if p}):
            sf_path = os.path.join(path, "adapter_model.safetensors")
            with safe_open(sf_path, framework="pt") as f:
                n_tensors = len(list(f.keys()))
            adapter_id = len(adapter_reqs) + 1
            adapter_reqs[path] = LoRARequest(f"adapter_{adapter_id}", adapter_id, path)
            print(f"[LoRA verification] {n_tensors} tensors in {sf_path} (id={adapter_id})")

        for tier_name in ["mcq", "frq1", "frq_multi", "frq_hard"]:
            items = tier_groups.get(tier_name, [])
            if not items:
                continue
            adapter_path, n_samples, temp = tier_plan[tier_name]
            lora_req = adapter_reqs.get(adapter_path)   # None => base
            label = os.path.basename(adapter_path.rstrip("/")) if adapter_path else "base"
            print(f"\nGenerating {tier_name} ({len(items)} questions, "
                  f"N={n_samples}, temp={temp}, model={label})...")
            prompts = build_prompts_for_items(items, tok)
            responses_list = generate_batch(
                llm, prompts, n_samples, args, lora_request=lora_req,
                temperature_override=temp)
            for item, responses in zip(items, responses_list):
                responses_by_idx[item["_orig_idx"]] = responses

    else:
        # ─── Single-model legacy path (unchanged) ──────────────────────────
        prompts = build_prompts_for_items(data, tok)
        if args.lora:
            from safetensors import safe_open
            import os
            sf_path = os.path.join(args.lora, "adapter_model.safetensors")
            with safe_open(sf_path, framework="pt") as f:
                n_tensors = len(list(f.keys()))
            print(f"[LoRA verification] {n_tensors} tensors in {sf_path}")

            from vllm.lora.request import LoRARequest
            lora_req = LoRARequest("adapter", 1, args.lora)
            responses_list = generate_batch(
                llm, prompts, args.n_samples, args, lora_request=lora_req)
        else:
            responses_list = generate_batch(
                llm, prompts, args.n_samples, args, lora_request=None)
        for item, responses in zip(data, responses_list):
            responses_by_idx[item["_orig_idx"]] = responses

    dt = time.time() - t0
    n_tok_estimate = sum(len(r.split()) for resps in responses_by_idx.values()
                         for r in resps) * 1.3  # rough word→token estimate
    print(f"\nGeneration took {dt:.1f}s "
          f"(~{int(n_tok_estimate)} tokens, "
          f"~{int(n_tok_estimate / dt)} tok/s)")

    # ─── Build records in ORIGINAL ORDER ──────────────────────────────────
    records: list[dict] = []
    for item in data:  # iterate in original order
        idx = item["_orig_idx"]
        responses = responses_by_idx[idx]
        rec = {
            "id": item.get("id"),
            "is_mcq": bool(item.get("options")),
            "responses": responses,
        }
        if routed_mode:
            rec["tier"] = tier_by_idx[idx]
        if "answer" in item:
            rec["gold"] = item["answer"]
        records.append(rec)
    write_jsonl(args.out, records)
    print(f"Saved raw outputs ({len(records)} records) to {args.out}")

    # ─── Pick one response per question via voting ────────────────────────
    if routed_mode:
        from judger import Judger
        judger = Judger(strict_extract=False)
        from postprocess import clean_response
        picked = []
        for r, item in zip(records, data):
            if r["is_mcq"]:
                chosen, _ = vote_majority(r["responses"], is_mcq=True)
                picked.append(chosen)
            else:
                if "answer" in item:
                    ans = item["answer"]
                    expected_count = len(ans) if isinstance(ans, list) else 1
                else:
                    expected_count = item.get("question", "").count("[ANS]") or 1
                chosen, winning = vote_frq_per_position(
                    r["responses"], expected_count, judger)
                # Self-consistency fix: submit the per-position VOTED tuple, not
                # the single best-matching sample (which can disagree on a
                # position and zero out a multiplicatively-scored question).
                if winning:
                    chosen = _rebuild_with_boxes(chosen, winning)
                # Post-processing: tuple/count reconcile + round + decimalize.
                chosen = clean_response(chosen, expected_count)
                picked.append(chosen)
    elif args.n_samples == 1 or args.vote == "first":
        picked = [r["responses"][0] for r in records]
    else:
        picked = []
        for r in records:
            chosen, _ = vote_majority(r["responses"], r["is_mcq"])
            picked.append(chosen)

    # ─── Write CSV (in original order, guaranteed by records ordering) ────
    if args.csv:
        write_csv(args.csv, records, picked)
        print(f"Wrote Kaggle CSV submission to {args.csv}")

    # ─── Local eval ───────────────────────────────────────────────────────
    if has_eval:
        summary = eval_records(records, picked)
        sm_path = Path(args.out).with_suffix(".summary.json")
        with open(sm_path, "w") as f:
            json.dump({k: v for k, v in summary.items() if k != "failures"}, f, indent=2)
        with open(Path(args.out).with_suffix(".failures.jsonl"), "w") as f:
            for fr in summary["failures"]:
                f.write(json.dumps(fr) + "\n")
        print(f"Saved summary to {sm_path}")


def run_inference(
    data: str = "data/private.jsonl",
    out: str = "results/routed_private.jsonl",
    csv: str = "results/submission.csv",
    frq_repo: str = "Aerolandaz/cap8-frq",
) -> str:
    """Single entry point. Reproduces the 71.64%-val routed config end-to-end:
      MCQ  -> base model,  N=5,  temp 0.4, letter-majority vote
      FRQ  -> cap8-frq LoRA (Hub), N=7/11/17 by tier, temp 0.8,
              per-position sympy voting + clean_response post-processing.
    Downloads the FRQ adapter from the Hub, then runs the exact same pipeline
    as the validated CLI command. Produces the Kaggle CSV at `csv`."""
    import sys as _sys
    from huggingface_hub import snapshot_download

    adapter_dir = snapshot_download(repo_id=frq_repo)  # local dir w/ adapter_model.safetensors

    saved_argv = _sys.argv
    _sys.argv = [
        "infer.py",
        "--data", data,
        "--out", out,
        "--csv", csv,
        "--no-eval",                       # private set has no gold
        "--lora-frq", adapter_dir,         # routed mode: base=MCQ, this=FRQ
        "--n-mcq", "5",
        "--n-frq1", "7",
        "--n-frq-multi", "11",
        "--n-frq-hard", "17",
        "--temp-mcq", "0.4",
        "--temp-frq", "0.8",
    ]
    try:
        main()
    finally:
        _sys.argv = saved_argv
    return csv


if __name__ == "__main__":
    main()