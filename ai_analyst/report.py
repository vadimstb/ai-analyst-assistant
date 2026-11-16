"""Markdown rendering of assistant results (printed to the terminal and saved with --out)."""
from __future__ import annotations

import pandas as pd

SEV = {"high": "🔴 high", "medium": "🟠 medium", "low": "🟢 low"}


def _verification(v: dict) -> str:
    line = f"\n**Перевірка чисел:** {v['verified']} з {v['numbers_checked']} чисел у відповіді LLM знайдено в даних"
    if v["unverified"]:
        line += " ⚠️ Неперевірені значення (можлива галюцінація):\n"
        line += "\n".join(f"- `{u['value']}` – …{u['context']}…" for u in v["unverified"])
    else:
        line += " ✅"
    return line + "\n"


def _table(df: pd.DataFrame, max_rows: int = 20) -> str:
    return df.head(max_rows).to_markdown(index=False)


def render_eda(res: dict, name: str) -> str:
    p, it = res["profile"], res["interpretation"]
    out = [f"# Автоматичний EDA: {name}", "",
           f"Рядків: **{p['rows']}**, колонок: **{p['columns']}**, повних дублікатів: **{p['duplicate_rows']}**", "",
           "## Колонки (розраховано pandas)", ""]
    rows = []
    for col, c in p["column_profiles"].items():
        extra = []
        if "median" in c:
            extra.append(f"медіана {c['median']}, min {c['min']}, max {c['max']}")
        if c.get("negative_values"):
            extra.append(f"від'ємних {c['negative_values']}")
        if c.get("outliers_3iqr"):
            extra.append(f"викидів {c['outliers_3iqr']}")
        if "last_regular_day" in c:
            extra.append(f"{c['min'][:10]} – {c['max'][:10]}, після {c['last_regular_day']}: {c['rows_after_last_regular_day']}")
        if "inconsistent_spellings" in c:
            extra.append(f"різні написання: {len(c['inconsistent_spellings'])} груп")
        rows.append({"колонка": col, "тип": c["dtype"], "пропуски, %": c["missing_pct"], "унікальних": c["unique"],
                     "особливості": "; ".join(extra)})
    out += [_table(pd.DataFrame(rows), 100), "", "## Інтерпретація LLM", "", it["dataset_summary"], "",
            "### Проблеми якості даних", ""]
    out += [f"- **{q['column']}** ({SEV[q['severity']]}): {q['issue']}. *Дані:* {q['evidence']}. "
            f"*Що зробити:* {q['recommendation']}" for q in it["quality_issues"]]
    out += ["", "### Розподіли", ""] + [f"- {x}" for x in it["distribution_notes"]]
    out += ["", "### Перші висновки", ""] + [f"- {x}" for x in it["first_conclusions"]]
    return "\n".join(out) + "\n" + _verification(res["verification"])


def render_hypotheses(res: dict, name: str) -> str:
    out = [f"# Гіпотези для перевірки: {name}", ""]
    for i, h in enumerate(res["hypotheses"], 1):
        out += [f"## {i}. {h['title']} ({SEV[h['priority']]})", "",
                f"- **Дані:** {h['evidence']} (факти: {', '.join(h['fact_ids']) or '–'})",
                f"- **Чому важливо:** {h['why_it_matters']}",
                f"- **Як перевірити:** {h['how_to_check']}", ""]
    return "\n".join(out) + _verification(res["verification"])


def render_ask(res: dict) -> str:
    out = [f"# Питання: {res['question']}", ""]
    for i, a in enumerate(res["attempts"][:-1] if res.get("can_answer") else res["attempts"], 1):
        out += [f"Спроба {i} (помилка, код виправлено автоматично):", "```python", a["code"], "```",
                f"`{a['error']}`", ""]
    if not res["can_answer"]:
        return "\n".join(out + [f"**Відповісти неможливо:** {res['reason']}"]) + "\n"
    out += ["## Згенерований код", "```python", res["code"], "```", "", res["explanation"], "", "## Результат", ""]
    r = res["result"]
    if isinstance(r, pd.Series):
        r = r.reset_index()
    out.append(_table(r) if isinstance(r, pd.DataFrame) else f"**{r}**")
    return "\n".join(out) + "\n"


def render_change(res: dict, name: str) -> str:
    c, e = res["change"], res["explanation"]
    out = [f"# Пояснення зміни метрики `{c['metric']}`: {name}", "",
           f"Період A {c['period_a']['from']} – {c['period_a']['to']}: **{c['period_a']['value']}**; "
           f"період B {c['period_b']['from']} – {c['period_b']['to']}: **{c['period_b']['value']}**; "
           f"зміна **{c['change']}** ({c['change_pct']} %)", "", "## Розклад по сегментах (pandas)", ""]
    for dim, rows in c["breakdown"].items():
        out += [f"**{dim}**", "", _table(pd.DataFrame(rows), 6), ""]
    out += ["## Висновок LLM", "", f"**{e['headline']}**", "", "Головні драйвери:"]
    out += [f"- {d['dimension']} = {d['segment']}: {d['evidence']}" for d in e["main_drivers"]]
    out += ["", "Гіпотези причин:"] + [f"- {x}" for x in e["cause_hypotheses"]]
    out += ["", "Що перевірити далі:"] + [f"- {x}" for x in e["next_checks"]]
    return "\n".join(out) + "\n" + _verification(res["verification"])


def render_summary(res: dict) -> str:
    s = res["summary"]
    out = [f"# {s['title']}", "", "**Що сталося**"] + [f"- {x}" for x in s["what_happened"]]
    out += ["", "**Чому це важливо**"] + [f"- {x}" for x in s["why_it_matters"]]
    out += ["", "**Що робити далі**"] + [f"- {x}" for x in s["next_steps"]]
    return "\n".join(out) + "\n" + _verification(res["verification"])
