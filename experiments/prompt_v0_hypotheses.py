"""Prompt iteration experiment: the naive v0 hypothesis prompt (raw describe() + head(), no facts, no rules).

Run: python -m experiments.prompt_v0_hypotheses data/dating_payments.csv
The answer is checked with the same verifier against the facts used by the current prompt (v2).
"""
import json
import sys

from ai_analyst import profiling
from ai_analyst.llm import get_backend
from ai_analyst.prompts import HypothesisSet
from ai_analyst.verify import verify_numbers

V0_SYSTEM = "You are a data analyst. Look at the data and suggest 5-10 hypotheses worth checking. Answer in Ukrainian."


def main(path: str) -> None:
    df = profiling.load_csv(path)
    user = ("describe():\n" + df.describe(include="all").to_string() + "\n\nhead():\n" + df.head(10).to_string())
    answer = get_backend().structured(V0_SYSTEM, user, HypothesisSet)
    facts = profiling.hypothesis_facts(df)
    check = verify_numbers(answer.model_dump(), {"facts": facts, "describe": json.loads(df.describe().to_json())})
    print(json.dumps({"hypotheses": [h.model_dump() for h in answer.hypotheses], "verification": check},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
