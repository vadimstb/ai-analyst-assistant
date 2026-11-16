"""Hallucination guard: every number the LLM writes must exist in the facts computed by pandas."""
from __future__ import annotations

import json
import re

import numpy as np

NUMBER = re.compile(r"(?<![\w.])[-−]?\d{1,3}(?:[  ,]\d{3})+(?:[.,]\d+)?|(?<![\w.])[-−]?\d+(?:[.,]\d+)?")


def _numbers_in(obj) -> list[float]:
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += _numbers_in(k) if isinstance(k, str) else []  # value_counts keys carry numbers too
            out += _numbers_in(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out += _numbers_in(v)
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float, np.integer, np.floating)):
        if not (isinstance(obj, float) and np.isnan(obj)):
            out.append(float(obj))
    elif isinstance(obj, str):
        out += [n for n in (_parse(m) for m in NUMBER.findall(obj)) if n is not None]
    return out


def _parse(token: str) -> float | None:
    t = token.replace("−", "-").replace(" ", " ")
    if re.fullmatch(r"-?\d{1,3}(?:[ ,]\d{3})+(?:[.,]\d+)?", t):
        t = re.sub(r"[ ,](?=\d{3}\b)", "", t)
    t = t.replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return None


def verify_numbers(llm_output, facts, rel_tol: float = 0.02) -> dict:
    """Compares numbers in the LLM answer with numbers in the facts.

    Small integers (≤ 12, e.g. "top 5", "3 hypotheses") and years are skipped. A number counts as verified if it
    matches a fact value within 2 %, also after ×100 / ÷100 (share vs percent) or rounding to K/M.
    """
    known = np.array(sorted(set(_numbers_in(facts))) or [np.nan])
    candidates = np.concatenate([known, known * 100, known / 100, known / 1000, known / 1e6, known * 7])
    text = llm_output if isinstance(llm_output, str) else json.dumps(llm_output, ensure_ascii=False)
    checked, unverified = 0, []
    for m in NUMBER.finditer(text):
        val = _parse(m.group())
        if val is None or (abs(val) <= 12 and float(val).is_integer()) or (1990 <= val <= 2100 and float(val).is_integer()):
            continue
        checked += 1
        ok = np.any(np.isclose(candidates, val, rtol=rel_tol, atol=0.051)) or \
            np.any(np.isclose(np.abs(candidates), abs(val), rtol=rel_tol, atol=0.051))
        if not ok:
            ctx = text[max(0, m.start() - 50): m.end() + 30].replace("\n", " ")
            unverified.append({"value": m.group(), "context": ctx})
    return {"numbers_checked": checked, "verified": checked - len(unverified), "unverified": unverified}
