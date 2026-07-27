from __future__ import annotations

import ast
import operator
import re
from typing import Any

import numpy as np
import pandas as pd


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def formula_name(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", raw)


def evaluate_formula(formula: str, values: dict[str, Any], *, length: int | None = None):
    tree = ast.parse(formula, mode="eval")

    def evaluate(node: ast.AST):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id not in values:
                raise ValueError(f"formula input missing: {node.id}")
            return values[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            lhs = evaluate(node.left)
            rhs = evaluate(node.right)
            if isinstance(node.op, ast.Div) and np.any(np.asarray(rhs, dtype=np.float64) <= 0.0):
                raise ValueError(f"formula denominator must be positive: {formula}")
            return _BIN_OPS[type(node.op)](lhs, rhs)
        raise ValueError(f"unsupported formula expression: {formula}")

    result = evaluate(tree)
    if isinstance(result, pd.Series):
        out = result.astype(float)
    else:
        array = np.asarray(result, dtype=np.float64)
        if array.ndim == 0:
            if length is None:
                return float(array)
            out = pd.Series(float(array), index=range(length), dtype=float)
        else:
            out = array
    if not np.isfinite(np.asarray(out, dtype=np.float64)).all():
        raise ValueError(f"formula produced non-finite values: {formula}")
    return out
