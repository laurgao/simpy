from expr import Const, Expr, cast


@cast
def sassert_repr(a: Expr, b: Expr):
    xs, ys = a.simplify(), b.simplify()
    if xs.expandable():
        xs = xs.expand()
    if ys.expandable():
        ys = ys.expand()
    assert repr(xs) == repr(ys), f"{xs} != {ys} (original {a} != {b})"

