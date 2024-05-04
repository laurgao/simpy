from typing import Optional

from src.simpy.expr import Expr, Symbol, cast, debug_repr, symbols
from src.simpy.integration import integrate

x, y = symbols("x, y")

@cast
def assert_integral(integrand: Expr, expected: Expr, var: Optional[Symbol] = None):
    assert_eq_plusc(integrate(integrand, var), expected)

@cast
def assert_definite_integral(integrand: Expr, bounds: tuple, expected: Expr):
    assert_eq_strict(integrate(integrand, bounds), expected)

@cast
def assert_eq_plusc(a: Expr, b: Expr):
    """Assert a and b are equal up to a constant"""
    xs, ys = a.simplify(), b.simplify()
    ys = -ys
    if xs.expandable():
        xs = xs.expand()
    if ys.expandable():
        ys = ys.expand()
    diff = xs + ys
    diff = diff.expand().simplify() if diff.expandable() else diff.simplify()
    assert len(diff.symbols()) == 0, f"diff = {diff}"

@cast
def assert_eq_value(a: Expr, b: Expr):
    """Tests that the values of a & b are the same in spirit, regardless of how they are represented with the Expr data structures.
    """
    a = a.simplify()
    b = b.simplify()
    if a.expandable():
        a = a.expand()
    if b.expandable():
        b = b.expand()
    assert repr(a) == repr(b), f"a != b, {a} != {b}"

@cast
def assert_eq_strict(a: Expr, b: Expr):
    """Tests that the structure of a and b exprs are the same, not just their values or reprs.
    // This does the same thing as assert a == b I believe.
    """
    assert a == b, f"STRICT: {a} != {b}, \n\tDebug repr: {debug_repr(a)} != {debug_repr(b)}"


def unhashable_set_eq(a: list, b: list) -> bool:
    """Does equivalent of set(a) == set(b) but for lists containing unhashable types
    """
    for el in a:
        if el not in b:
            return False
    for el in b:
        if el not in a:
            return False
    return True