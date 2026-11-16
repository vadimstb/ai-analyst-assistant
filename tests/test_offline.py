"""Offline tests (no LLM calls): pandas computations, sandbox, verifier and the assistant with a fake backend.

Run: python -m pytest -q
"""
import pandas as pd
import pytest

from ai_analyst import profiling
from ai_analyst.assistant import AIAnalyst
from ai_analyst.prompts import ExecutiveSummary, PandasAnswer
from ai_analyst.safe_exec import UnsafeCodeError, run_code
from ai_analyst.verify import verify_numbers


@pytest.fixture
def df():
    return pd.DataFrame({
        "user_id": range(1, 201),
        "date": pd.date_range("2024-01-01", periods=200, freq="D"),
        "channel": ["ads", "organic"] * 100,
        "amount": [10.0, 20.0] * 100,
    })


class FakeBackend:
    """Returns prepared answers in the order they are requested."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def structured(self, system, user, schema):
        self.calls.append(user)
        return self.answers.pop(0)


def test_change_decomposition_sums_to_total(df):
    res = profiling.compute_change(df, "date", "sum:amount", ("2024-01-01", "2024-01-31"),
                                   ("2024-02-01", "2024-02-29"), ["channel"])
    assert res["period_a"]["value"] == 460.0 and res["period_b"]["value"] == 440.0  # 16×10+15×20, 15×20+14×10
    assert sum(r["change"] for r in res["breakdown"]["channel"]) == pytest.approx(res["change"])


def test_empty_period_raises(df):
    with pytest.raises(ValueError):
        profiling.compute_change(df, "date", "count", ("2020-01-01", "2020-01-02"), ("2024-01-01", "2024-01-05"))


@pytest.mark.parametrize("code", [
    "import os\nresult = 1",
    "result = open('x').read()",
    "result = df.__class__",
    "df.to_csv('x.csv'); result = 1",
    "x = 1",
])
def test_sandbox_blocks_unsafe_code(df, code):
    with pytest.raises(UnsafeCodeError):
        run_code(code, df)


def test_sandbox_runs_pandas(df):
    assert run_code("result = df.groupby('channel')['amount'].sum()", df)["organic"] == 2000.0


def test_verifier_flags_invented_number():
    facts = {"revenue": 28636.64, "share": 0.1596}
    check = verify_numbers("Виручка 28 636.64, падіння 15.96 %, а у вівторок було 99 999", facts)
    assert check["numbers_checked"] == 3 and [u["value"] for u in check["unverified"]] == ["99 999"]


def test_ask_repairs_failing_code(df):
    broken = PandasAnswer(can_answer=True, reason_if_not="", code="result = df['missing'].sum()", explanation="")
    fixed = PandasAnswer(can_answer=True, reason_if_not="", code="result = df['amount'].sum()", explanation="")
    backend = FakeBackend(broken, fixed)
    res = AIAnalyst(backend).ask(df, "яка виручка?")
    assert res["result"] == 3000.0 and len(res["attempts"]) == 2 and "KeyError" in backend.calls[1]


def test_ask_refuses_unrelated_question(df):
    backend = FakeBackend(PandasAnswer(can_answer=False, reason_if_not="Не про дані", code="", explanation=""))
    assert AIAnalyst(backend).ask(df, "яка погода?")["can_answer"] is False


def test_summary_verification_catches_hallucination():
    answer = ExecutiveSummary(title="Виручка 900", what_happened=["Виручка зросла до 1 234 567"],
                              why_it_matters=[], next_steps=[])
    res = AIAnalyst(FakeBackend(answer)).executive_summary({"revenue": 900})
    assert [u["value"] for u in res["verification"]["unverified"]] == ["1 234 567"]
