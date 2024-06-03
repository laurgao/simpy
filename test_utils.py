from typing import Optional, Tuple, Type, Union

from src.simpy.expr import (
    Expr,
    Power,
    Rat,
    SingleFunc,
    Symbol,
    TrigFunctionNotInverse,
    cast,
    cos,
    debug_repr,
    sec,
    symbols,
    tan,
)
from src.simpy.integration import integrate
from src.simpy.regex import replace_class, replace_factory

x, y = symbols("x y")


@cast
def assert_integral(integrand: Expr, expected: Expr, var: Optional[Symbol] = None, **kwargs):
    assert_eq_plusc(integrate(integrand, var, **kwargs), expected)
    # assert_eq_value(diff(expected, var), integrand) # this might be nice; idk. catch some potential diff errors.


@cast
def assert_definite_integral(integrand: Expr, bounds: tuple, expected: Expr):
    assert_eq_strict(integrate(integrand, bounds), expected)


@cast
def assert_eq_plusc(a: Expr, b: Union[Expr, Tuple[Expr, ...]], *vars):
    """Assert a and b are equal up to a constant, relative to vars.
    If no vars are given, then
    """
    if isinstance(b, tuple):
        assert any(_assert_eq_plusc(a, b_, *vars)[0] for b_ in b)
        return
    success, diff = _assert_eq_plusc(a, b, *vars)
    assert success, f"diff = {diff}"


def _assert_eq_plusc(a, b, *vars) -> Tuple[bool, Expr]:
    diff = a - b
    if len(diff.symbols()) == 0 or vars and all(var not in diff.symbols() for var in vars):
        return True, None
    diff = diff.expand().simplify() if diff.expandable() else diff.simplify()
    if not vars:
        return len(diff.symbols()) == 0, diff
    else:
        return all(var not in diff.symbols() for var in vars), diff


def _eq_value(a: Expr, b: Expr) -> Tuple[Expr, Expr]:
    a = a.simplify()
    b = b.simplify()
    if a.expandable():
        a = a.expand()
    if b.expandable():
        b = b.expand()
    return a, b


@cast
def assert_eq_value(a: Expr, b: Expr):
    """Tests that the values of a & b are the same in spirit, regardless of how they are represented with the Expr data structures."""
    if a == b:
        return
    a, b = _eq_value(a, b)
    assert a == b, f"a != b, {a} != {b}"


@cast
def assert_eq_strict(a: Expr, b: Expr):
    """Tests that the structure of a and b exprs are the same, not just their values or reprs.
    // This does the same thing as assert debug_repr(a) == debug_repr(b) I believe.
    """
    assert a == b, f"STRICT: {a} != {b}, \n\tDebug repr: {debug_repr(a)} != {debug_repr(b)}"


def unhashable_set_eq(a: list, b: list) -> bool:
    """Does equivalent of set(a) == set(b) but for lists containing unhashable types"""
    for el in a:
        if el not in b:
            return False
    for el in b:
        if el not in a:
            return False
    return True
