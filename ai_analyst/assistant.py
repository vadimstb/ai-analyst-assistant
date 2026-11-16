"""AI analyst: pandas computes the numbers, the LLM interprets them, verify.py checks the LLM's numbers."""
from __future__ import annotations

import json

import pandas as pd

from . import profiling, prompts
from .llm import LLMError, get_backend
from .safe_exec import UnsafeCodeError, run_code
from .verify import verify_numbers


def _facts_block(facts) -> str:
    return "<facts>\n" + json.dumps(facts, ensure_ascii=False, indent=1, default=str) + "\n</facts>"


class AIAnalyst:
    def __init__(self, backend=None):
        self.backend = backend or get_backend()

    def _ask(self, system, user, schema, facts):
        answer = self.backend.structured(system, user, schema)
        return answer, verify_numbers(answer.model_dump(), facts)

    # 1. Automatic EDA
    def auto_eda(self, df: pd.DataFrame, raw: pd.DataFrame | None = None) -> dict:
        profile = profiling.profile_dataframe(df, raw)
        answer, check = self._ask(prompts.EDA_SYSTEM,
                                  "Інтерпретуй профіль датасету.\n" + _facts_block(profile),
                                  prompts.EdaInterpretation, profile)
        return {"profile": profile, "interpretation": answer.model_dump(), "verification": check}

    # 2. Hypothesis generation
    def suggest_hypotheses(self, df: pd.DataFrame) -> dict:
        facts = profiling.hypothesis_facts(df)
        if not facts:
            raise ValueError("У даних немає категоріальних розрізів або дат – гіпотези не з чого будувати")
        context = {"rows": len(df), "columns": {c: str(t) for c, t in df.dtypes.items()}}
        user = ("Опис датасету:\n" + json.dumps(context, ensure_ascii=False) +
                "\nКандидатні спостереження:\n" + _facts_block(facts))
        answer, check = self._ask(prompts.HYPOTHESES_SYSTEM, user, prompts.HypothesisSet, facts)
        known = {f["id"] for f in facts}
        for h in answer.hypotheses:
            h.fact_ids = [i for i in h.fact_ids if i in known]  # drop ids the model invented
        return {"facts": facts, "hypotheses": answer.model_dump()["hypotheses"], "verification": check}

    # 3. Natural language -> pandas
    def ask(self, df: pd.DataFrame, question: str, max_fixes: int = 1) -> dict:
        schema = profiling.schema_for_llm(df)
        user = "<dataset>\n" + json.dumps(schema, ensure_ascii=False, default=str) + "\n</dataset>\n" \
               f"<question>{question}</question>"
        answer = self.backend.structured(prompts.NL2PANDAS_SYSTEM, user, prompts.PandasAnswer)
        attempts = []
        for attempt in range(max_fixes + 1):
            if not answer.can_answer:
                return {"question": question, "can_answer": False, "reason": answer.reason_if_not,
                        "attempts": attempts}
            try:
                result = run_code(answer.code, df)
                attempts.append({"code": answer.code, "error": None})
                return {"question": question, "can_answer": True, "code": answer.code,
                        "explanation": answer.explanation, "result": result, "attempts": attempts}
            except (UnsafeCodeError, Exception) as e:  # noqa: BLE001 – any runtime error is fed back to the model
                attempts.append({"code": answer.code, "error": f"{type(e).__name__}: {e}"})
                if attempt == max_fixes:
                    break
                fix = prompts.NL2PANDAS_FIX.format(code=answer.code, error=f"{type(e).__name__}: {e}")
                answer = self.backend.structured(prompts.NL2PANDAS_SYSTEM, user + "\n" + fix, prompts.PandasAnswer)
        return {"question": question, "can_answer": False,
                "reason": "Не вдалося отримати робочий код: " + attempts[-1]["error"], "attempts": attempts}

    # 4. Anomaly explanation
    def explain_change(self, df, date_col, metric, period_a, period_b, dims=None) -> dict:
        change = profiling.compute_change(df, date_col, metric, period_a, period_b, dims)
        answer, check = self._ask(prompts.ANOMALY_SYSTEM, "Поясни зміну метрики.\n" + _facts_block(change),
                                  prompts.AnomalyExplanation, change)
        return {"change": change, "explanation": answer.model_dump(), "verification": check}

    # 5. Executive summary
    def executive_summary(self, metrics: dict, context: str = "") -> dict:
        if not metrics:
            raise ValueError("Немає метрик для summary")
        user = (f"Контекст: {context}\n" if context else "") + _facts_block(metrics)
        answer, check = self._ask(prompts.SUMMARY_SYSTEM, user, prompts.ExecutiveSummary, metrics)
        return {"metrics": metrics, "summary": answer.model_dump(), "verification": check}


__all__ = ["AIAnalyst", "LLMError"]
