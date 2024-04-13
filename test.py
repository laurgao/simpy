from fractions import Fraction as F

from expr import symbols
from integration import *
from test_transforms import test_lecture_example, test_x2_sqrt_1_x3


def assert_eq(x, y):
    assert x == y, f"{x} == {y} is False. ({x-y}).simplify() = {(x+y).simplify()}"


@cast
def sassert_repr(a, b):
    xs, ys = a.simplify(), b.simplify()
    assert repr(xs) == repr(ys), f"{xs} != {ys} (original {a} != {b})"


def test_polynomial_division():
    expr = x**4 * (1 + x**2) ** -1

    test_node = Node(expr, x)
    tr = PolynomialDivision()
    assert tr.check(test_node)

    tr.forward(test_node)
    sassert_repr(test_node.children[0].expr, x**2 - 1 + 1 / (1 + x**2))

    # 2nd test
    expr = x**3 / (1 - x**2)

    test_node = Node(expr, x)
    tr = PolynomialDivision()
    assert tr.check(test_node)

    tr.forward(test_node)
    ans = test_node.children[0].expr


def test_factor():
    c3, r3, c4, r4, a, b = symbols("c_3 r_3 c_4 r_4 a b")

    em_hw_expr = (
        c3 * r3 / (sqrt(a) * sqrt(b))
        - c3**2 * r3**2 / (c4 * r4 * sqrt(b) * a ** F(3 / 2))
        - c4 * r4 / (sqrt(a) * b ** F(3 / 2))
    )
    em_hw_expr = em_hw_expr.simplify()
    factored = em_hw_expr.factor()
    expected_factored = (
        1
        / (sqrt(a) * sqrt(b))
        * (c3 * r3 - c3**2 * r3**2 / (c4 * r4 * a) - c4 * r4 / b)
    )
    sassert_repr(factored, expected_factored)


if __name__ == "__main__":
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

    # new repr standards test
    expr = 1 - x**2
    assert expr.__repr__() == "1 - x^2"
    assert (2 * x).__repr__() == "2*x"
    assert (2 * (2 + x)).__repr__() == "2*(2 + x)"
    assert (2 / (2 + x)).__repr__() == "2/(2 + x)"
    assert repr(2 * (2 + x) ** (-2)) == repr(2 / (2 + x) ** 2) == "2/(2 + x)^2"

    # PolynomialDivision test
    test_polynomial_division()

    # run entire integrals
    test_lecture_example()
    test_x2_sqrt_1_x3()

    # Factor test
    test_factor()

    print("passed")
