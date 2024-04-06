from transforms import *


def test_polynomial_division():
    x = symbols("x")
    expr = x**4 * (1 + x**2) ** -1

    test_node = Node(expr, x)
    tr = PolynomialDivision()
    assert tr.check(test_node)


if __name__ == "__main__":
    test_polynomial_division()
    print("passed")
