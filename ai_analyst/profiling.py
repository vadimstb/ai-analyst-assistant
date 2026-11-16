"""Deterministic pandas computations. Every number the assistant shows comes from here, never from the LLM."""
from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd

ID_PATTERN = re.compile(r"(^id$|_id$|^id_|uuid)", re.IGNORECASE)


def load_csv(path: str) -> pd.DataFrame:
    """Reads a CSV and converts columns that are clearly numeric or dates."""
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        raise ValueError(f"Файл {path} не містить жодного рядка даних")
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = _maybe_convert(df[col])
    return df


def _maybe_convert(s: pd.Series) -> pd.Series:
    sample = s.dropna().astype(str).head(2000)
    if sample.empty:
        return s
    numeric = pd.to_numeric(sample.str.replace(",", ".", regex=False), errors="coerce")
    if numeric.notna().mean() > 0.95:
        return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce").where(s.notna())
    if sample.str.contains(r"\d{1,4}[-./]\d{1,2}[-./]\d{1,4}").mean() > 0.9:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(s, format="mixed", dayfirst=True, errors="coerce")
        if parsed.notna().mean() > 0.9:
            return parsed
    return s


def _r(x, nd=2):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return round(float(x), nd)
    return x


def is_id_column(name: str, s: pd.Series) -> bool:
    return bool(ID_PATTERN.search(name)) or (s.nunique() > 0.9 * len(s) and pd.api.types.is_integer_dtype(s))


def value_pattern(s: pd.Series) -> pd.Series:
    """Shape of values: digits -> 9, letters -> a. Shows mixed date formats etc."""
    return s.dropna().astype(str).str.replace(r"\d", "9", regex=True).str.replace(r"[^\W\d_]", "a", regex=True)


def profile_dataframe(df: pd.DataFrame, raw: pd.DataFrame | None = None) -> dict:
    """Structured EDA profile: types, missing values, distributions, anomalies."""
    raw = raw if raw is not None else df
    prof = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(raw.duplicated().sum()),
        "duplicate_rows_pct": _r(raw.duplicated().mean() * 100),
        "column_profiles": {},
    }
    for col in df.columns:
        s = df[col]
        p = {
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "missing_pct": _r(s.isna().mean() * 100),
            "unique": int(s.nunique()),
        }
        if is_id_column(col, s):
            p["role"] = "identifier"
        if pd.api.types.is_numeric_dtype(s) and p.get("role") != "identifier":
            d = s.dropna()
            if len(d):
                q1, q3 = d.quantile([0.25, 0.75])
                iqr = q3 - q1
                out = d[(d < q1 - 3 * iqr) | (d > q3 + 3 * iqr)] if iqr > 0 else d.iloc[0:0]
                p.update({
                    "min": _r(d.min()), "p25": _r(q1), "median": _r(d.median()), "mean": _r(d.mean()),
                    "p75": _r(q3), "max": _r(d.max()), "std": _r(d.std()),
                    "negative_values": int((d < 0).sum()), "zero_values": int((d == 0).sum()),
                    "outliers_3iqr": int(len(out)), "outliers_3iqr_pct": _r(len(out) / len(d) * 100),
                    "top_values": {str(k): int(v) for k, v in d.value_counts().head(5).items()},
                })
        elif pd.api.types.is_datetime64_any_dtype(s):
            d = s.dropna()
            if len(d):
                daily = d.dt.normalize().value_counts().sort_index()
                typical = daily.median()
                sparse_tail = daily[daily.index > daily[daily >= 0.2 * typical].index.max()]
                p.update({
                    "min": str(d.min()), "max": str(d.max()),
                    "rows_per_month": {str(k): int(v) for k, v in d.dt.to_period("M").value_counts().sort_index().items()},
                    "median_rows_per_day": _r(typical),
                    "last_regular_day": str(daily[daily >= 0.2 * typical].index.max().date()),
                    "rows_after_last_regular_day": int(sparse_tail.sum()),
                })
            if col in raw.columns and raw[col].dtype == object:
                p["raw_value_formats"] = {k: int(v) for k, v in value_pattern(raw[col]).value_counts().head(5).items()}
        else:
            d = s.dropna().astype(str)
            vc = d.value_counts()
            p["top_values"] = {k: int(v) for k, v in vc.head(10).items()}
            p["rare_values_lt_0_1pct"] = int((vc < 0.001 * len(df)).sum())
            norm = d.str.strip().str.lower().str.replace(r"[\s_]+", "_", regex=True)
            groups = d.groupby(norm).agg(lambda x: sorted(set(x)))
            variants = {k: v for k, v in groups.items() if len(v) > 1}
            if variants:
                p["inconsistent_spellings"] = {k: v[:6] for k, v in list(variants.items())[:8]}
            if d.str.len().max() <= 30 and p["unique"] > 1:
                p["raw_value_formats"] = {k: int(v) for k, v in value_pattern(d).value_counts().head(4).items()}
        prof["column_profiles"][col] = p
    return prof


def categorical_dims(df: pd.DataFrame, max_unique: int = 30) -> list[str]:
    dims = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s) or is_id_column(col, s):
            continue
        n = s.nunique()
        if 2 <= n <= max_unique and (s.dtype == object or n <= 10):
            dims.append(col)
    return dims


def numeric_measures(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and not is_id_column(c, df[c]) and df[c].nunique() > 10]


def date_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]


def hypothesis_facts(df: pd.DataFrame, max_facts: int = 40) -> list[dict]:
    """Candidate observations with numbers, ranked by how far a segment deviates from the whole."""
    facts: list[dict] = []
    df = df.copy()
    for col in df.columns:  # facts are computed on normalised categories ("iOS", " ios" -> "ios")
        if df[col].dtype == object:
            df[col] = df[col].str.strip().str.lower().str.replace(r"[\s_]+", "_", regex=True)
    dims = categorical_dims(df)
    measures = numeric_measures(df)
    n = len(df)

    for dim in dims:
        g = df.groupby(dim, dropna=False)
        size = g.size()
        for seg, cnt in size.items():
            if cnt < max(30, 0.01 * n):
                continue
            for m in measures:
                seg_mean, all_mean = df.loc[df[dim] == seg, m].mean(), df[m].mean()
                se = df[m].std() / np.sqrt(cnt)
                if all_mean and se and abs(seg_mean - all_mean) / se >= 3:
                    dev = seg_mean / all_mean - 1
                    facts.append({"type": "segment_mean", "dimension": dim, "segment": str(seg), "measure": m,
                                  "segment_value": _r(seg_mean), "overall_value": _r(all_mean),
                                  "relative_diff_pct": _r(dev * 100), "rows": int(cnt), "score": abs(dev)})

    # outcome rates: share of each value of a small categorical "target" inside segments of another dimension
    targets = [c for c in dims if 2 <= df[c].nunique() <= 10]
    for t in targets:
        overall = df[t].value_counts(normalize=True)
        for val, base in overall.items():
            if base < 0.003 or base > 0.97:
                continue
            for dim in dims:
                if dim == t:
                    continue
                share = df.assign(_hit=df[t] == val).groupby(dim)["_hit"].agg(["mean", "size"])
                share = share[share["size"] >= max(30, 0.01 * n)]
                for seg, row in share.iterrows():
                    if row["mean"] in (0.0, 1.0):  # structural dependency between columns, not an insight
                        continue
                    z = (row["mean"] - base) / np.sqrt(base * (1 - base) / row["size"])
                    if abs(z) < 3:  # difference within sampling noise
                        continue
                    dev = row["mean"] / base - 1
                    facts.append({"type": "segment_rate", "dimension": dim, "segment": str(seg),
                                  "outcome": f"{t} = {val}", "segment_rate_pct": _r(row["mean"] * 100),
                                  "overall_rate_pct": _r(base * 100), "relative_diff_pct": _r(dev * 100),
                                  "rows": int(row["size"]), "score": abs(dev) * min(1.0, row["size"] / (0.05 * n))})

    for dc in date_columns(df):
        d = df[dc].dropna()
        if d.empty:
            continue
        weekly = d.dt.to_period("W").value_counts().sort_index()
        if len(weekly) >= 4:
            full = weekly.iloc[1:-1] if len(weekly) > 4 else weekly
            first, last = full.iloc[: max(1, len(full) // 4)].mean(), full.iloc[-max(1, len(full) // 4):].mean()
            facts.append({"type": "trend", "date_column": dc, "first_quarter_weekly_rows": _r(first),
                          "last_quarter_weekly_rows": _r(last), "relative_diff_pct": _r((last / first - 1) * 100),
                          "score": abs(last / first - 1)})
            peak = weekly.idxmax()
            facts.append({"type": "peak_week", "date_column": dc, "week": str(peak), "rows": int(weekly.max()),
                          "median_week_rows": _r(weekly.median()),
                          "relative_diff_pct": _r((weekly.max() / weekly.median() - 1) * 100),
                          "score": weekly.max() / weekly.median() - 1})
        dow = d.dt.day_name().value_counts()
        if len(dow) == 7:
            facts.append({"type": "weekday", "date_column": dc, "max_day": dow.idxmax(), "max_rows": int(dow.max()),
                          "min_day": dow.idxmin(), "min_rows": int(dow.min()),
                          "relative_diff_pct": _r((dow.max() / dow.min() - 1) * 100),
                          "score": dow.max() / dow.min() - 1})
        for dim in dims[:6]:
            mid = d.min() + (d.max() - d.min()) / 2
            early = df.loc[df[dc] < mid, dim].value_counts(normalize=True)
            late = df.loc[df[dc] >= mid, dim].value_counts(normalize=True)
            for seg in late.index.union(early.index):
                e, l = early.get(seg, 0.0), late.get(seg, 0.0)
                if max(e, l) >= 0.02:
                    facts.append({"type": "mix_shift", "date_column": dc, "dimension": dim, "segment": str(seg),
                                  "share_first_half_pct": _r(e * 100), "share_second_half_pct": _r(l * 100),
                                  "change_pp": _r((l - e) * 100), "score": abs(l - e) * 4})

    facts.sort(key=lambda f: f["score"], reverse=True)
    out = []
    for i, f in enumerate(facts[:max_facts], 1):
        f = {k: v for k, v in f.items() if k != "score"}
        f["id"] = f"F{i}"
        out.append(f)
    return out


def metric_value(df: pd.DataFrame, metric: str) -> float:
    agg, _, col = metric.partition(":")
    if agg == "count":
        return float(len(df))
    if col not in df.columns:
        raise ValueError(f"Колонки {col!r} немає в даних")
    if agg == "sum":
        return float(df[col].sum())
    if agg == "mean":
        return float(df[col].mean())
    if agg == "nunique":
        return float(df[col].nunique())
    raise ValueError("Метрика має вигляд count | sum:<col> | mean:<col> | nunique:<col>")


def compute_change(df: pd.DataFrame, date_col: str, metric: str, period_a: tuple[str, str],
                   period_b: tuple[str, str], dims: list[str] | None = None) -> dict:
    """Metric in two periods and each segment's contribution to the change."""
    if date_col not in df.columns:
        raise ValueError(f"Колонки {date_col!r} немає в даних")
    dates = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()

    def window(p):
        return df[(dates >= pd.Timestamp(p[0])) & (dates <= pd.Timestamp(p[1]))]

    a, b = window(period_a), window(period_b)
    if a.empty or b.empty:
        raise ValueError("Один із періодів не містить жодного рядка – перевірте дати")
    va, vb = metric_value(a, metric), metric_value(b, metric)
    additive = not metric.startswith("mean")
    dims = dims or categorical_dims(df, max_unique=60)[:5]
    breakdown = {}
    for dim in dims:
        rows = []
        for seg in sorted(set(a[dim].dropna().astype(str)) | set(b[dim].dropna().astype(str))):
            sa, sb = a[a[dim].astype(str) == seg], b[b[dim].astype(str) == seg]
            ma = metric_value(sa, metric) if len(sa) else 0.0
            mb = metric_value(sb, metric) if len(sb) else 0.0
            row = {"segment": seg, "period_a": _r(ma), "period_b": _r(mb), "change": _r(mb - ma),
                   "change_pct": _r((mb / ma - 1) * 100) if ma else None,
                   "rows_a": int(len(sa)), "rows_b": int(len(sb))}
            if additive and vb != va:
                row["share_of_total_change_pct"] = _r((mb - ma) / (vb - va) * 100)
            rows.append(row)
        key = (lambda r: abs(r["change"] or 0)) if additive else (lambda r: (r["rows_a"] + r["rows_b"]))
        breakdown[dim] = sorted(rows, key=key, reverse=True)[:8]
    return {"metric": metric, "date_column": date_col,
            "period_a": {"from": period_a[0], "to": period_a[1], "value": _r(va), "rows": int(len(a))},
            "period_b": {"from": period_b[0], "to": period_b[1], "value": _r(vb), "rows": int(len(b))},
            "change": _r(vb - va), "change_pct": _r((vb / va - 1) * 100) if va else None,
            "additive_metric": additive, "breakdown": breakdown}


def schema_for_llm(df: pd.DataFrame, sample_rows: int = 5) -> dict:
    cols = {}
    for c in df.columns:
        s = df[c]
        info = {"dtype": str(s.dtype), "missing": int(s.isna().sum()), "unique": int(s.nunique())}
        if s.dtype == object and s.nunique() <= 30:
            info["values"] = [str(v) for v in s.dropna().unique()[:30]]
        elif pd.api.types.is_datetime64_any_dtype(s) and s.notna().any():
            info["range"] = [str(s.min()), str(s.max())]
        cols[c] = info
    sample = df.head(sample_rows).astype(str).to_dict(orient="records")
    return {"rows": int(len(df)), "columns": cols, "sample_rows": sample}
