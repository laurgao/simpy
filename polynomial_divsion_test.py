from test import sassert_repr

from transforms import *

x = symbols("x")


def test_polynomial_division():
    expr = x**4 * (1 + x**2) ** -1

    test_node = Node(expr, x)
    tr = PolynomialDivision()
    assert tr.check(test_node)

    tr.forward(test_node)
    sassert_repr(test_node.children[0].expr, x**2 - 1 + 1 / (1 + x**2))


def test_polynomial_division_2():
    expr = x**3 / (1 - x**2)

    test_node = Node(expr, x)
    tr = PolynomialDivision()
    assert tr.check(test_node)

    tr.forward(test_node)
    ans = test_node.children[0].expr
    breakpoint()


if __name__ == "__main__":
    test_polynomial_division_2()
    print("passed")
