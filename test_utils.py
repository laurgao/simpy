from src.simpy.expr import Expr, cast


@cast
def assert_eq_plusc(a: Expr, b: Expr):
    """Assert a and b are equal up to a constant"""
    xs, ys = a.simplify(), b.simplify()
    ys = -ys
    if xs.expandable():
        xs = xs.expand()
    if ys.expandable():
        ys = ys.expand()
    diff = (xs + ys).expand().simplify()
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