"""Prompts v3 for the CSE 151B math reasoning competition.

Changes from v2 (driven by failure-diff between v2 and SFT runs):

1. Programmatic [ANS] count injection.
   v2 told the model "Count the [ANS] placeholders CAREFULLY". LLMs are bad at
   counting; Python isn't. We substitute the literal count, addressing the
   multiplicative penalty observed at n>=4 sub-answers.

2. Dynamic rule routing.
   The free-form prompt only injects rules for answer types that the question
   text hints at. Always-on: format rules, anti-patterns, basic answer styles.
   Conditionally-on: tuple, interval, multi-letter, boolean, word-fill.

3. New anti-patterns from SFT regression spot-checks:
   - No \\text{} / \\mathrm{} / \\textbf{} wrappers (saw \\boxed{\\text{A}})
   - No Markdown after </think> (saw "**Option A**: ...", bullet lists, headers)
   - No inline boxes in prose (saw "Therefore the mean is \\boxed{250}, and...")
   - No display math (\\[...\\]) after </think>

4. MCQ prompt tightened: explicit \\text{} ban + soft instruction to weigh
   each option in <think> before committing.

The judger extracts the LAST CONTIGUOUS GROUP of \\boxed{} after </think>, where
contiguity allows whitespace, commas, $, and basic punctuation between boxes
but NOT words. Every anti-pattern below is constructed to keep the model in
this extraction-friendly regime.
"""
from __future__ import annotations

import re
from typing import Optional


# ─── Always-on free-form blocks ────────────────────────────────────────────────
_FREEFORM_HEADER = """You are an expert mathematician. Reason inside <think>...</think>, then output ONLY the final answers.

OUTPUT FORMAT (the grader extracts the LAST CONTIGUOUS GROUP of \\boxed{...} after </think>):
- Boxes must be CONTIGUOUS — separated only by whitespace, commas, or punctuation.
  No words, no equations, no LaTeX commands between or around boxes.
- After the LAST box, write NOTHING. No "so the answer is", no period, no follow-up. Stop."""


_ANTI_PATTERNS = """
ANTI-PATTERNS (these break extraction — the grader will mark you WRONG):
- DO NOT wrap your answer in \\text{}, \\mathrm{}, or \\textbf{}.
  Write \\boxed{A} not \\boxed{\\text{A}}.   Write \\boxed{42} not \\boxed{\\mathrm{42}}.
- DO NOT use Markdown after </think>. No **bold**, no bullet lists, no headers,
  no "Option A: ..." analyses. Just the boxes.
- DO NOT inline a box inside a sentence.
  WRONG: "Therefore the mean is \\boxed{250}, and the standard error is..."
  RIGHT: \\boxed{250} \\boxed{6}
- DO NOT emit display math (\\[ ... \\]) or other equations after </think>.
- DO NOT use \\left( \\right) — use plain parentheses.
- DO NOT use \\, or \\; (LaTeX spacing) inside the box.
- DO NOT add units inside the box (NOT \\boxed{42 meters}).
- DO NOT add \\approx or \\pm decoration."""


_CORE_STYLE = """
ANSWER STYLES (always relevant):
- INTEGER: \\boxed{42}
- EXACT FRACTION: \\boxed{\\frac{1}{2}} or \\boxed{1/2}
- EXACT SURD/CONSTANT: \\boxed{\\sqrt{3}}, \\boxed{2\\pi/3}
- ALGEBRAIC EXPRESSION: \\boxed{2x^2 + 3x - 1}, \\boxed{6e^{16x}}
- DECIMAL (NOT exact): give AT LEAST 10 SIGNIFICANT DIGITS. Carry full precision through
  intermediate steps. The grader uses tight relative tolerance — 143.2 is wrong
  when the true answer is 143.2242292337.
- EQUATION (whole equation in one box): \\boxed{x + y = 43}, \\boxed{y = 470x - 390}."""


# ─── Conditional rule blocks ──────────────────────────────────────────────────
_RULE_TUPLE = """
TUPLE / ORDERED PAIR / POINT (this question hints at one):
- A tuple answer goes in ONE box: \\boxed{(2, -2)}    NOT \\boxed{2} \\boxed{-2}
- A triple: \\boxed{(80, -160, 320)}.
- A tuple counts as ONE [ANS], not multiple."""


_RULE_INTERVAL = """
INTERVAL (this question hints at one):
- Emit in ONE box. Use the WORD 'infinity', NOT \\infty.
- Examples: \\boxed{(-8, infinity)}, \\boxed{(-infinity, 60]}, \\boxed{[0, 60]},
  \\boxed{(-infinity, -2.5) U (2.5, infinity)}"""


_RULE_MULTI_LETTER = """
SELECT-ALL-THAT-APPLY (this question hints at one):
- Emit ONE box with the concatenated letters in alphabetical order.
- Examples: \\boxed{BCEG}, \\boxed{AC}, \\boxed{DG}.
- NEVER use a separate box per letter."""


_RULE_BOOLEAN = """
YES/NO / TRUE/FALSE (this question hints at one):
- Match the case used in the question. (Yes/No) -> \\boxed{Yes} or \\boxed{No}.
  (YES/NO) -> uppercase. Default to lowercase if unclear.
- TRUE/FALSE: capital T/F -> \\boxed{True} or \\boxed{False}."""


_RULE_WORD_FILL = """
WORD-FILL (this question hints at one):
- Patterns like (LARGER/SMALLER), (INCREASING/DECREASING), (REJECT/FAIL TO REJECT):
  emit the chosen word in its OWN box, in the case shown by the question.
- A word slot followed by a number slot = TWO boxes.
- Common forms: \\boxed{LARGER}, \\boxed{INCREASING}, \\boxed{REJECT}."""


# ─── MCQ prompt ───────────────────────────────────────────────────────────────
SYSTEM_MCQ = """You are an expert mathematician. Reason inside <think>...</think>, then select the single best option.

Inside <think>, weigh each option systematically before committing. Eliminate distractors with explicit reasoning, then commit to one letter.

OUTPUT FORMAT (after </think>):
- Output ONLY the chosen letter wrapped in \\boxed{}. Examples: \\boxed{C}, \\boxed{F}.
- The letter must match one of the labels in the Options list (A, B, C, ...).
- DO NOT wrap the letter: write \\boxed{C} not \\boxed{\\text{C}} or \\boxed{(C)}.
- After the box, write nothing. No period, no explanation. Just stop."""


# ─── Routing helpers ──────────────────────────────────────────────────────────
def _count_ans(question: str) -> int:
    """Programmatic [ANS] placeholder count.

    Defaults to 1 if no [ANS] is present (some questions phrase the prompt
    without a marker, e.g. 'Find x.')."""
    return max(question.count("[ANS]"), 1)


def _detect_features(question: str) -> set[str]:
    """Return tags suggested by the question text.

    Errs on the liberal side — false positives cost a few lines of context
    budget but never break extraction. False negatives would leave the model
    without rule guidance for that answer type, so we prefer over-inclusion.
    """
    q = question
    ql = q.lower()
    feats: set[str] = set()

    # Boolean / yes-no
    if re.search(r"\(?\s*(?:yes\s*/\s*no|y\s*/\s*n|true\s*/\s*false)\s*\)?", ql):
        feats.add("boolean")

    # Word-fill: slash-separated all-caps options, with or without parens.
    # Catches (LARGER/SMALLER), INCREASING/DECREASING, (REJECT/FAIL TO REJECT), etc.
    if re.search(r"[A-Z]{3,}(?:\s+[A-Z]+)*(?:\s*/\s*[A-Z]{3,}(?:\s+[A-Z]+)*)+", q):
        feats.add("word_fill")

    # Interval — keyword or notation hints
    if any(k in ql for k in (
        "interval", "domain", "range of values", "set of all",
        "values of x for which", "values of x such that"
    )):
        feats.add("interval")
    if "\\cup" in q or "infinity" in ql or re.search(r"\\?infty", q):
        feats.add("interval")

    # Tuple / ordered pair / point
    if re.search(r"\bordered\s+(pair|triple)\b|\bcoordinates?\b", ql):
        feats.add("tuple")
    if re.search(r"\bthe point\b|\bfind\s+the\s+point\b", ql):
        feats.add("tuple")
    if re.search(r"\(\s*x\s*,\s*y\b|\(\s*x\s*,\s*y\s*,\s*z\b", ql):
        feats.add("tuple")

    # Multi-letter / select-all-that-apply
    if any(k in ql for k in (
        "select all", "all that apply", "which of the following are",
        "choose all", "mark all", "check all"
    )):
        feats.add("multi_letter")

    return feats


def _build_count_block(n: int) -> str:
    plural = "s" if n != 1 else ""
    entries = "entries" if n != 1 else "entry"
    return (
        f"\nQUESTION STRUCTURE:\n"
        f"- This question contains exactly {n} [ANS] placeholder{plural}.\n"
        f"- Output exactly {n} \\boxed{{...}} {entries} after </think>, in question order.\n"
        f"- A {n}-answer question with the wrong number of boxes is automatically WRONG."
    )


# ─── Public API ───────────────────────────────────────────────────────────────
def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for one question.

    Signature is unchanged from v2, so infer.py needs no edits.
    """
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return SYSTEM_MCQ, f"{question}\n\nOptions:\n{opts_text}"

    n = _count_ans(question)
    feats = _detect_features(question)

    parts = [
        _FREEFORM_HEADER,
        _build_count_block(n),
        _ANTI_PATTERNS,
        _CORE_STYLE,
    ]
    # Stable rule order, only the ones we detect
    if "tuple" in feats:
        parts.append(_RULE_TUPLE)
    if "interval" in feats:
        parts.append(_RULE_INTERVAL)
    if "multi_letter" in feats:
        parts.append(_RULE_MULTI_LETTER)
    if "boolean" in feats:
        parts.append(_RULE_BOOLEAN)
    if "word_fill" in feats:
        parts.append(_RULE_WORD_FILL)

    system = "\n".join(parts)
    return system, question


# ─── Self-test (run directly: python prompts.py) ──────────────────────────────
if __name__ == "__main__":
    examples = [
        ("Find the sum of the first $325$ positive even whole numbers. Sum: [ANS]", None),
        ("A roasted turkey... (a) ... is [ANS] Fahrenheit. (b) ... is [ANS] hours.", None),
        ("Find the domain of f(x) = log(x-2). Domain: [ANS]", None),
        ("Find the ordered pair (x, y) such that x + y = 5 and x - y = 1. [ANS]", None),
        ("Which of the following are true? Select all that apply. [ANS]", None),
        ("Is f increasing? (Yes/No) [ANS]", None),
        ("State whether the value (LARGER/SMALLER) and by how much. [ANS] by [ANS]", None),
        ("Pick the best option.", ["A. 1", "B. 2", "C. 3"]),
    ]
    for q, opts in examples:
        sys_p, _ = build_prompt(q, opts)
        n = _count_ans(q) if not opts else "MCQ"
        feats = sorted(_detect_features(q)) if not opts else ["MCQ"]
        print(f"\n{'=' * 70}")
        print(f"Q: {q[:80]}{'...' if len(q) > 80 else ''}")
        print(f"  n={n}  features={feats}  prompt_lines={sys_p.count(chr(10)) + 1}")