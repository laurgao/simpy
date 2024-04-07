from transforms import *


def test_lecture_example():

    F = Fraction
    x = symbols("x")
    expression = -5 * x**4 / (1 - x**2) ** F(5, 2)
    integral = Integration.integrate(expression, x)  # TODO auto simplify

    expected_integral = -5 * (
        -(x / (sqrt(1 - x**2))) + (F(1, 3) * x**3 / (1 - x**2) ** F(3, 2)) + ArcSin(x)
    )
    diff = (integral - expected_integral).simplify()
    assert diff == Const(0), f"diff = {diff}"


if __name__ == "__main__":
    test_lecture_example()
    print("Passed")
