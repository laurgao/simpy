from typing import Literal

from ..expr import Expr, Float, Power, Prod, Rat, Sum, Symbol


def debug_repr(expr: Expr, *, pedantic: Literal["always", "moderate"] = "moderate") -> str:
    """Not designed to look pretty; designed to show the structure of the expr.

    pedantic = "always" will show the non-children classes of numbers and symbols
    """
    if isinstance(expr, Power):
        return f"Power({debug_repr(expr.base)}, {debug_repr(expr.exponent)})"
        # return f'({debug_repr(expr.base)})^{debug_repr(expr.exponent)}'
    if isinstance(expr, Prod):
        return "Prod(" + ", ".join([debug_repr(t) for t in expr.terms]) + ")"
        # return " * ".join([debug_repr(t) for t in expr.terms])
    if isinstance(expr, Sum):
        return "Sum(" + ", ".join([debug_repr(t) for t in expr.terms]) + ")"
        # return " + ".join([debug_repr(t) for t in expr.terms])

    if pedantic == "always":
        if isinstance(expr, Rat):
            return f"Rat({expr.value})"
        if isinstance(expr, Symbol):
            return f"Symbol({expr.name})"
        if isinstance(expr, Float):
            return f"Float({expr.value})"

    return repr(expr)
