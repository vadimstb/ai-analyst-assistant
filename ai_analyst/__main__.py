"""CLI: python -m ai_analyst <command> data.csv [...]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import profiling, report
from .assistant import AIAnalyst
from .llm import LLMError, get_backend


def _period(text: str) -> tuple[str, str]:
    start, _, end = text.partition(":")
    if not end:
        raise argparse.ArgumentTypeError("Період задається як YYYY-MM-DD:YYYY-MM-DD")
    return start, end


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ai_analyst", description="AI-асистент продуктового аналітика")
    ap.add_argument("--backend", choices=["auto", "anthropic", "claude-cli"], default=None,
                    help="LLM-бекенд (за замовчуванням auto)")
    ap.add_argument("--out", help="зберегти звіт у Markdown-файл")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("eda", help="автоматичний EDA")
    p.add_argument("csv")
    p = sub.add_parser("hypotheses", help="генерація гіпотез")
    p.add_argument("csv")
    p = sub.add_parser("ask", help="питання природною мовою -> pandas")
    p.add_argument("csv")
    p.add_argument("question")
    p = sub.add_parser("explain", help="пояснення зміни метрики між двома періодами")
    p.add_argument("csv")
    p.add_argument("--date", required=True, help="колонка з датою")
    p.add_argument("--metric", required=True, help="count | sum:<col> | mean:<col> | nunique:<col>")
    p.add_argument("--a", required=True, type=_period, help="базовий період YYYY-MM-DD:YYYY-MM-DD")
    p.add_argument("--b", required=True, type=_period, help="період зі зміною YYYY-MM-DD:YYYY-MM-DD")
    p.add_argument("--dims", help="розрізи через кому (за замовчуванням – автоматично)")
    p = sub.add_parser("summary", help="executive summary з JSON-файлу метрик")
    p.add_argument("metrics_json")
    p.add_argument("--context", default="")

    args = ap.parse_args(argv)
    try:
        analyst = AIAnalyst(get_backend(args.backend))
        if args.cmd == "summary":
            text = report.render_summary(analyst.executive_summary(
                json.loads(Path(args.metrics_json).read_text(encoding="utf-8")), args.context))
        else:
            name = Path(args.csv).name
            df = profiling.load_csv(args.csv)
            if args.cmd == "eda":
                raw = pd.read_csv(args.csv, dtype=str, keep_default_na=True)
                text = report.render_eda(analyst.auto_eda(df, raw), name)
            elif args.cmd == "hypotheses":
                text = report.render_hypotheses(analyst.suggest_hypotheses(df), name)
            elif args.cmd == "ask":
                text = report.render_ask(analyst.ask(df, args.question))
            else:
                dims = [d.strip() for d in args.dims.split(",")] if args.dims else None
                text = report.render_change(analyst.explain_change(df, args.date, args.metric, args.a, args.b, dims), name)
    except (LLMError, ValueError, FileNotFoundError, pd.errors.EmptyDataError) as e:
        print(f"Помилка: {e}", file=sys.stderr)
        return 1
    text = text.replace("—", "–")  # єдиний стиль тире у звітах
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"\nЗвіт збережено: {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
