from typing import Optional, Tuple, Type

from src.simpy.expr import Expr, Power, Rat, SingleFunc, Symbol, TrigFunction, cast, cos, debug_repr, sec, symbols, tan
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
def assert_eq_plusc(a: Expr, b: Expr, *vars):
    """Assert a and b are equal up to a constant, relative to vars.
    If no vars are given, then
    """
    a, b = sectan(a, b)
    diff = a - b
    diff = diff.expand().simplify() if diff.expandable() else diff.simplify()
    if not vars:
        assert len(diff.symbols()) == 0, f"diff = {diff}"
    else:
        assert all(var not in diff.symbols() for var in vars), f"diff = {diff}"


def is_cls_squared(expr, cls) -> bool:
    return (
        isinstance(expr, Power)
        and isinstance(expr.base, cls)
        and isinstance(expr.exponent, Rat)
        and expr.exponent % 2 == 0
    )


def only_contains_class_squared(expr: Expr, cls: Type[SingleFunc]) -> bool:
    """Return if it's a function of cls squared.
    Which means that expr has to contain cls and all instances of cls have to be squared.
    """
    if not expr.has(cls):
        return False

    def _func(e):
        if isinstance(e, cls):
            return False
        if is_cls_squared(e, cls):
            return True
        return all(_func(c) for c in e.children())

    return _func(expr)


def sectan(a: Expr, b: Expr) -> Tuple[Expr, Expr]:
    # futureTODO: his (& similar things) should be done in simplify during subtractions.
    if not a.has(TrigFunction) or not b.has(TrigFunction):
        return a, b

    def _sectan(a, b):
        # a is sec and b is tan
        # convert a to b
        condition = lambda x: is_cls_squared(x, sec)

        def perform(e: Power):
            n = e.exponent
            return (1 + tan(e.base.inner) ** 2) ** (n / 2)

        a = replace_factory(condition, perform)(a)
        a = a.expand() if a.expandable() else a
        return a, b

    a = replace_class(a, [cos], [lambda x: 1 / sec(x)])
    b = replace_class(b, [cos], [lambda x: 1 / sec(x)])
    if only_contains_class_squared(a, sec) and only_contains_class_squared(b, tan):
        return _sectan(a, b)

    if only_contains_class_squared(b, sec) and only_contains_class_squared(a, tan):
        return _sectan(b, a)

    return a, b


def _eq_value(a: Expr, b: Expr) -> Tuple[Expr, Expr]:
    a, b = sectan(a, b)
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
