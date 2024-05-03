from src.simpy.expr import Expr, cast, debug_repr


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
def assert_eq_repr(a: Expr, b: Expr):
    a = a.simplify()
    b = b.simplify()
    if a.expandable():
        a = a.expand()
    if b.expandable():
        b = b.expand()
    assert repr(a) == repr(b), f"a != b, {a} != {b}"


@cast
def assert_eq_repr_strict(a: Expr, b: Expr):
    """Tests that the structure of a and b exprs are the same, not just their reprs"""
    assert debug_repr(a) == debug_repr(b), f"STRICT a != b, {debug_repr(a)} != {debug_repr(b)}"


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