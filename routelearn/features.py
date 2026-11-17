"""Cheap, dependency-free prompt features (numpy only). ~microseconds per prompt."""
import math
import re

import numpy as np

REASON_KW = ["prove", "analy", "design", "architecture", "trade-off", "tradeoff", "step by step",
             "compare", "optimi", "evaluate", "derive", "refactor", "debug", "critique",
             "strateg", "why", "formal"]
CODE_KW = ["def ", "class ", "function", "stack trace", "sql", "regex", "api", "algorithm",
           "python", "implement", "query"]
EXPLAIN_KW = ["explain", "summar", "write", "draft", "describe", "difference between", "rewrite"]
CONSTRAINT_KW = ["must", "exactly", "at least", "at most", "do not", "only", "format"]
SIMPLE_PAT = r"^(what is|who is|define|translate|capital of|give one fact|who wrote)\b"
MULTI_PAT = r"\b(first|then|finally|and also|step \d)\b"
MATH_PAT = r"[∑∫√≤≥]|\d+\s*[\^*/]\s*\d+|\bequation\b|\btheorem\b|\bproof\b"

FEATURE_NAMES = ["bias", "log_words", "n_sentences", "n_reason", "n_code", "code_block",
                 "math", "multi_part", "simple_pattern", "constraints", "explain_task",
                 "avg_word_len", "digit_ratio", "question_marks"]
DIM = len(FEATURE_NAMES)
IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}


def _count(low: str, kws) -> int:
    return sum(bool(re.search(r"\b" + re.escape(k), low)) for k in kws)


def extract(prompt: str, history: list[dict[str, str]] | None = None) -> np.ndarray:
    prior_user_turns = [
        message["content"] for message in (history or [])
        if message.get("role") == "user"
    ][-4:]
    prompt = "\n".join(prior_user_turns + [prompt])
    p = prompt.strip()
    low = p.lower()
    words = p.split()
    nw = max(len(words), 1)
    f = {
        "bias": 1.0,
        "log_words": math.log1p(nw) / math.log(200),
        "n_sentences": min(len(re.findall(r"[.!?]+(?:\s|$)", p)), 10) / 5,
        "n_reason": min(_count(low, REASON_KW), 4) / 2,
        "n_code": min(_count(low, CODE_KW), 4) / 2,
        "code_block": float("```" in p),
        "math": float(bool(re.search(MATH_PAT, low))),
        "multi_part": float(bool(re.search(MULTI_PAT, low))),
        "simple_pattern": float(bool(re.match(SIMPLE_PAT, low)) and nw < 15),
        "constraints": min(_count(low, CONSTRAINT_KW), 3) / 3,
        "explain_task": float(_count(low, EXPLAIN_KW) > 0),
        "avg_word_len": (sum(len(w) for w in words) / nw) / 6,
        "digit_ratio": min(sum(c.isdigit() for c in p) / max(len(p), 1) * 5, 1.0),
        "question_marks": min(p.count("?"), 3) / 3,
    }
    return np.array([f[n] for n in FEATURE_NAMES], dtype=float)
