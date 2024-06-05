"""Latex utility functions"""

from typing import Union

from .expr import Expr, Power


def bracketfy(expr: Expr, *, bracket="()") -> str:
    """Makes expr latex with the necessary brackets

    Conditions for putting \\left \\right:
    - has power
    - has frac

    args:
        expr: the expr to bracketfy
        bracket: must be a str of length 2
    """
    inner_latex = expr.latex()
    b1, b2 = bracket
    if expr.has(Power):
        return f"\\left{b1} {inner_latex} \\right{b2}"
    return f"{b1}{inner_latex}{b2}"


def group(expr: Union[Expr, str]) -> str:
    """wrap expr latex with curly brackets if necessary"""
    if isinstance(expr, Expr):
        expr = expr.latex()
    return expr if len(expr) == 1 else "{" + expr + "}"
