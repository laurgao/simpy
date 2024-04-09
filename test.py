from transforms import *


def assert_eq(x, y):
    assert x == y, f"{x} == {y} is False. ({x-y}).simplify() = {(x+y).simplify()}"


@cast
def sassert_repr(a, b):
    xs, ys = a.simplify(), b.simplify()
    assert repr(xs) == repr(ys), f"{xs} != {ys} (original {a} != {b})"


x, y = symbols("x y")

sassert_repr(x * 0, 0)
sassert_repr(x * 2, 2 * x)
sassert_repr(x**2, x * x)
sassert_repr(x * 2 - 2 * x, 0)
sassert_repr(((x + 1) ** 2 - (x + 1) * (x + 1)), 0)

sassert_repr(Integration.integrate(3 * x**2 - 2 * x, x), x**3 - x**2)
sassert_repr(Integration.integrate((x + 1) ** 2, x), x + x**2 + (x**3 / 3))
sassert_repr(Log(x).diff(x), 1 / x)
sassert_repr(Log(x).diff(x), 1 / x)
sassert_repr(Integration.integrate(1 / x, x), Log(x))
sassert_repr(Integration.integrate(1 / x, (x, 1, 2)), Log(2))

assert nesting(x**2, x) == 2
assert nesting(x * y**2, x) == 2
assert nesting(x * (1 / y**2 * 3), x) == 2

sassert_repr(x + (2 + y), x + 2 + y)

assert count(2, x) == 0
assert count(Tan(x + 1) ** 2 - 2 * x, x) == 2


# cos^2 + sin^2 = 1 test
expr = Sin(x) ** 2 + Cos(x) ** 2 + 3
simplified = expr.simplify()
assert_eq(simplified, 4)

expr = Sin(x - 2 * y) ** 2 + 3 + Cos(x - 2 * y) ** 2 + y**2
simplified = expr.simplify()
assert_eq(simplified, 4 + y**2)


# PullConstant test
expr = 2 * x**3
test_node = Node(expr, x)
transform = PullConstant()
assert transform.check(test_node)
transform.forward(test_node)
assert_eq(test_node.child.expr, x**3)


print("passed")
