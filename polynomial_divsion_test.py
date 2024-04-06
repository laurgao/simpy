from test import sassert_repr

from transforms import *


def test_polynomial_division():
    x = symbols("x")
    expr = x**4 * (1 + x**2) ** -1

    test_node = Node(expr, x)
    tr = PolynomialDivision()
    assert tr.check(test_node)

    tr.forward(test_node)
    sassert_repr(test_node.children[0].expr, x**2 - 1 + 1 / (1 + x**2))


if __name__ == "__main__":
    test_polynomial_division()
    print("passed")
