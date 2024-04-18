from integration import *

F = Fraction
x = symbols("x")


def test_lecture_example():
    expression = -5 * x**4 / (1 - x**2) ** F(5, 2)
    integral = Integration.integrate(expression, x)  # TODO auto simplify

    expected_integral = -5 * (
        -(x / (sqrt(1 - x**2))) + (F(1, 3) * x**3 / (1 - x**2) ** F(3, 2)) + ArcSin(x)
    )
    diff = (integral - expected_integral).simplify()
    assert diff == Const(0), f"diff = {diff}"


def test_sin3x():
    expression = Sin(x) ** 3
    integral = Integration.integrate(expression, x)
    breakpoint()


def test_x2_sqrt_1_x3():
    expression = x**2 / sqrt(1 - x**3)
    integral = Integration.integrate(expression, x)
    expected = (-2 * sqrt(1 - x**3) / 3).simplify()

    assert integral == expected, f"{integral} != {expected}"

