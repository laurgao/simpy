"""Latex utility functions"""

from .expr import Expr, Power


def bracketfy(expr: Expr, *, bracket="()") -> str:
    """Conditions for putting \\left \\right:
    - has power
    - has frac

    bracket must be a str of length 2
    """
    inner_latex = expr.latex()
    b1, b2 = bracket
    if expr.has(Power):
        return f"\\left{b1} {inner_latex} \\right{b2}"
    return f"{b1}{inner_latex}{b2}"
