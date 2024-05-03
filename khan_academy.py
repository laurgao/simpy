"""
LOL im gonna take a bunch of integral questions from https://www.khanacademy.org/math/integral-calculus/ic-integration/ic-integration-proofs/test/ic-integration-unit-
and make sure simpy can do them
"""

from src.simpy.expr import *
from src.simpy.integration import *
from test_utils import assert_eq_plusc

x = symbols("x")

def _assert_integral(integrand, expected):
    return assert_eq_plusc(integrate(integrand), expected)

def _assert_definite_integral(integrand, bounds, expected):
    return assert_eq_plusc(integrate(integrand, bounds), expected)

def test_ex():
    integrand = 6 * e**x
    ans = integrate(integrand, (x, 6, 12))
    assert_eq_plusc(ans, 6 * e**12 - 6 * e**6)

def test_xcosx():
    """Uses integration by parts"""
    integrand = x * Cos(x)
    ans = integrate(integrand, (x, 3 * pi / 2, pi))
    assert_eq_plusc(ans, 3 * pi / 2 - 1)

def test_partial_fractions():
    integrand = (x + 8) / (x * (x + 6))
    expected_ans = Fraction(4, 3) * Log(x) - Fraction(1, 3) * Log(x + 6)
    _assert_integral(integrand, expected_ans)


def test_arcsin():
    ans = integrate(ArcSin(x), x)
    expected_ans = x * ArcSin(x) + sqrt(1 - x**2)
    assert_eq_plusc(ans, expected_ans)

    ans = integrate(ArcCos(x), x)
    expected_ans = x * ArcCos(x) - sqrt(1 - x**2)
    assert_eq_plusc(ans, expected_ans)

    ans = integrate(ArcTan(x), x)
    expected_ans = x * ArcTan(x) - Log(1 + x**2) / 2
    assert_eq_plusc(ans, expected_ans)


def test_sec2x_tan2x():
    """Uses either integration by parts with direct solve or generic sin/cos usub"""
    integrand = Sec(2*x) * Tan(2*x)
    ans = integrate(integrand, (x, 0, pi/6))
    assert ans == Fraction(1, 2)

def more_test():
    _assert_integral(4 * Sec(x) ** 2, 4 * Tan(x))
    _assert_integral(Sec(x) ** 2 * Tan(x) ** 2, Tan(x) ** 3 / 3)
    _assert_integral(5 / x - 3 * e ** x, 5 * Log(x) - 3 * e ** x)
    _assert_integral(Sec(x), Log(Sec(x) + Tan(x)))
    _assert_integral(2 * Cos(2 * x - 5), Sin(2 * x - 5)) 
    _assert_integral(3 * x ** 5 - x ** 3 + 6, 6*x - x**4/4 + x**6/2)
    _assert_integral(x ** 3 * e ** (x ** 4), (e**(x**4)/4))

    # Uses generic u-sub
    _assert_definite_integral(e ** x / (1 + e ** x), (Log(2), Log(8)), Log(9) - Log(3))

    _assert_definite_integral(8 * x / sqrt(1 - 4 * x ** 2), (0, Fraction(1,4)), 2 - sqrt(3))
    _assert_definite_integral(Sin(4*x), (0, pi/4), Fraction(1,2))

def test_expanding_big_power():
    integrand = (2 * x - 5) ** 10
    expected_ans = (2*x-5)**11/22
    _assert_integral(integrand, expected_ans)

    integrand = 3 * x ** 2 * (x ** 3 + 1) ** 6
    expected_ans = (1 + x**3)**7/7
    _assert_integral(integrand, expected_ans)

def test_polynomial_div_integrals():
    _assert_integral((x-5) / (-2 * x + 2), - x / 2 + 2 * Log(1 - x))
    _assert_integral((x ** 3 - 1)/ (x+2), x**3/3 - x**2 + 4*x- 9*Log(2 + x))
    _assert_integral((x - 1)/ (2 * x + 4), x / 2 - Fraction(3, 2) * Log(x + 2))
    
    integrand = (2 * x ** 3 + 4 * x ** 2 - 5)/ (x + 3)
    ans = integrate(integrand, x)
    # TODO: expected = ...

def test_complete_the_square_integrals():
    _assert_integral(1/(3*x**2+6*x+78), ArcTan((1 + x)/5)/15)
    _assert_integral(1/(x**2-8*x+65), ArcTan((-4 + x)/7)/7)
    _assert_integral(1/sqrt(-x**2-6*x+40), ArcSin((3 + x)/7))