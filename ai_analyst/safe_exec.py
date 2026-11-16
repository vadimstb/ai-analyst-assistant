"""Static check and restricted execution of LLM-generated pandas code."""
from __future__ import annotations

import ast

import numpy as np
import pandas as pd

FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "getattr",
                   "setattr", "delattr", "input", "breakpoint", "exit", "quit", "help", "memoryview"}
FORBIDDEN_ATTRS = {"to_csv", "to_excel", "to_pickle", "to_parquet", "to_sql", "to_json", "to_html", "to_clipboard",
                   "to_hdf", "to_feather", "to_stata", "read_csv", "read_excel", "read_pickle", "read_sql",
                   "read_json", "read_parquet", "read_html", "read_clipboard", "system", "popen", "eval", "query"}
SAFE_BUILTINS = {name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
                 for name in ["len", "range", "min", "max", "sum", "sorted", "round", "abs", "list", "dict", "set",
                              "tuple", "str", "int", "float", "bool", "enumerate", "zip", "any", "all",
                              "isinstance", "print", "reversed", "map", "filter"]}


class UnsafeCodeError(ValueError):
    pass


def check_code(code: str) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise UnsafeCodeError(f"Синтаксична помилка: {e}") from e
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise UnsafeCodeError("Імпорти заборонені – доступні лише df, pd, np")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise UnsafeCodeError(f"Заборонений виклик: {node.id}")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise UnsafeCodeError(f"Доступ до службових атрибутів заборонено: {node.attr}")
            if node.attr in FORBIDDEN_ATTRS:
                raise UnsafeCodeError(f"Операції з файлами/системою заборонені: {node.attr}")
    if not any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "result" for t in n.targets)
               for n in ast.walk(tree)):
        raise UnsafeCodeError("Код має записати відповідь у змінну result")


def run_code(code: str, df: pd.DataFrame):
    """Executes checked code on a copy of df; returns the value of `result`."""
    check_code(code)
    env = {"__builtins__": SAFE_BUILTINS, "df": df.copy(), "pd": pd, "np": np}
    exec(compile(code, "<llm-code>", "exec"), env)  # noqa: S102 – guarded by check_code + restricted builtins
    return env["result"]
