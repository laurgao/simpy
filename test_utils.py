from expr import Expr, Number, cast


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
    assert isinstance(diff, Number)

