"""Integration table"""

from typing import List, Optional, Tuple

from .expr import Expr, Power, Rat, Symbol, cos, log, sec, sin, tan
from .utils import ExprFn, eq_with_var

x = Symbol("x")

STANDARD_TRIG_INTEGRALS: List[Tuple[Expr, ExprFn]] = [
    (sin(x), lambda x: -cos(x)),
    (cos(x), sin),
    # Integration calculator says this is a standard integral. + i haven't encountered any transform that can solve this.
    (sec(x) ** 2, tan),
    (sec(x), lambda x: log(tan(x) + sec(x))),  # not a standard integral but it's fucked so im leaving it (unless?)
]


def check_integral_table(integrand: Expr, var: Symbol) -> Optional[Expr]:
    """Checks if integrand is directly solveable from the lookup table.

    Returns None if not solveable and returns the integral otherwise.
    """
    if not integrand.contains(var):
        return integrand * var
    if isinstance(integrand, Power):
        if integrand.base == var and not integrand.exponent.contains(var):
            n = integrand.exponent
            return (1 / (n + 1)) * Power(var, n + 1) if n != -1 else log(abs(integrand.base))
        if integrand.exponent == var and not integrand.base.contains(var):
            return 1 / log(integrand.base) * integrand
    if isinstance(integrand, Symbol) and integrand == var:
        return Rat(1, 2) * Power(var, 2)

    for expr, integral in STANDARD_TRIG_INTEGRALS:
        if eq_with_var((integrand, var), (expr, x)):
            return integral(var)

    return None
