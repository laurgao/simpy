from src.simpy.expr import *
from src.simpy.integration import *
from test_utils import (assert_definite_integral, assert_eq_plusc,
                        assert_eq_strict, assert_integral, x, y)

F = Fraction
x = symbols("x")

def test_basic_integrals():
    assert_definite_integral(2, (x, 5, 3), -4)
    assert_integral(x ** Fraction(7, 3), Fraction(3, 10) * x ** Fraction(10,3))
    assert_integral(3 * x**2 - 2 * x, x**3 - x**2)
    assert_integral((x + 1) ** 2, x + x**2 + (x**3 / 3))
    assert_integral(x ** 12, x ** 13 / 13)
    assert_integral(1 / x, log(x))
    assert_definite_integral(1 / x, (x, 1, 2), log(2))
    assert_integral(y, x * y, var=x)
    assert_integral(tan(y), x * tan(y), var=x)
    
def test_lecture_example():
    expression = -5 * x**4 / (1 - x**2) ** F(5, 2)
    expected_integral = -5 * (
        -(x / (sqrt(1 - x**2)))
        + (F(1, 3) * x**3 / (1 - x**2) ** F(3, 2))
        + asin(x)
    )
    assert_integral(expression, expected_integral)


# def test_sin3x():
#     expression = sin(x) ** 3
#     integral = integrate(expression, x)
#     breakpoint()


def test_x2_sqrt_1_x3():
    expression = x**2 / sqrt(1 - x**3)
    expected = -2 * sqrt(1 - x**3) / 3
    assert_integral(expression, expected)

def test_compound_angle():
    # Finds the average power of an AC circuit
    w, phi, t = symbols("w \\phi t")
    ac_power_integrand = cos(w * t - phi) * cos(w * t)
    period = 2 * pi / w
    ac_power = 1 / period * integrate(ac_power_integrand, (t, 0, period))

    expected = cos(phi) / 2
    assert_eq_plusc(ac_power, expected)


def test_sin2x():
    # tests the product-to-sum formula
    # Test integral (sin x)^2 = x / 2 - Sin(2x) / 4
    expr = sin(x) ** 2
    expected = x / 2 - sin(2 * x) / 4
    assert_integral(expr, expected)


def test_cos2x():
    # tests the product-to-sum formula
    # Test integral (cos x)^2 = Sin(2x) / 4 + x / 2
    expr = cos(x) ** 2
    expected = sin(2 * x) / 4 + x / 2
    assert_integral(expr, expected)

def test_linear_usub_with_multiple_subs():
    # Last I checked, this fails without LinearUSub
    integrand = sin(2*x) / cos(2*x)
    expected = -log(cos(2*x))/2
    assert_integral(integrand, expected)


def test_misc():
    # This integral can either be sin^2(wt) / 2w or -cos^2(wt) / 2w depending on the method used to solve it
    w, t = symbols("w t")
    expr = sin(w * t) * cos(w * t)
    expected = sin(w * t) ** 2 / (2 * w)
    assert_eq_plusc(integrate(expr, t), expected, t)

    # ln integral
    assert_eq_strict((x * log(x) - x).diff(x), log(x))
    assert_integral(log(x), x * log(x) - x)

    integrand = log(x + 6) / x**2
    expected = log(x) / 6 - log(x + 6) / x - log(x+6) / 6
    assert_integral(integrand, expected)

    # without the string search fix on InverseTrigUSub, this returns a wrong answer.
    integrand = sqrt(x ** 2 + 11)
    ans = integrate(integrand, x)
    assert ans is None


def test_integrate_with_completing_the_square():
    expr = 1 / sqrt(- x ** 2 + 10 * x + 11)
    expected = asin((x - 5) / 6)
    assert_integral(expr, expected)
