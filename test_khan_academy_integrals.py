"""
LOL im gonna take a bunch of integral questions from https://www.khanacademy.org/math/integral-calculus/ic-integration/ic-integration-proofs/test/ic-integration-unit-
and make sure simpy can do them
"""

from src.simpy.expr import *
from src.simpy.integration import *
from test_utils import (assert_definite_integral, assert_eq_plusc,
                        assert_integral, x)


def test_ex():
    integrand = 6 * e**x
    ans = integrate(integrand, (x, 6, 12))
    assert_eq_plusc(ans, 6 * e**12 - 6 * e**6)

def test_xcosx():
    """Uses integration by parts"""
    integrand = x * cos(x)
    ans = integrate(integrand, (x, 3 * pi / 2, pi))
    assert_eq_plusc(ans, 3 * pi / 2 - 1)

def test_partial_fractions():
    integrand = (x + 8) / (x * (x + 6))
    expected_ans = Fraction(4, 3) * log(x) - Fraction(1, 3) * log(x + 6)
    assert_integral(integrand, expected_ans)

    integrand = (18-12*x)/(4*x-1)/(x-4)
    expected_ans = -log(4*x-1)-2*log(x-4)
    assert_integral(integrand, expected_ans)
    integrand = (2*x+3)/(x-3)/(x+3)
    expected_ans = 3*log(x-3)/2 + log(x+3)/2
    assert_integral(integrand, expected_ans)
    integrand = (x-2)/(2*x+1)/(x+3)
    expected_ans = -log(2*x+1)/2 + log(x+3)
    assert_integral(integrand, expected_ans)


def test_integration_by_parts():
    integrand = x * e ** (-x)
    expected = -e ** (-x) * (x + 1)
    assert_integral(integrand, expected)


def test_arcsin():
    ans = integrate(asin(x), x)
    expected_ans = x * asin(x) + sqrt(1 - x**2)
    assert_eq_plusc(ans, expected_ans)

    ans = integrate(acos(x), x)
    expected_ans = x * acos(x) - sqrt(1 - x**2)
    assert_eq_plusc(ans, expected_ans)

    ans = integrate(atan(x), x)
    expected_ans = x * atan(x) - log(1 + x**2) / 2
    assert_eq_plusc(ans, expected_ans)


def test_sec2x_tan2x():
    """Uses either integration by parts with direct solve or generic sin/cos usub"""
    integrand = sec(2*x) * tan(2*x)
    ans = integrate(integrand, (x, 0, pi/6))
    assert ans == Fraction(1, 2)

def test_misc():
    assert_integral(4 * sec(x) ** 2, 4 * tan(x))
    assert_integral(sec(x) ** 2 * tan(x) ** 2, tan(x) ** 3 / 3)
    assert_integral(5 / x - 3 * e ** x, 5 * log(x) - 3 * e ** x)
    assert_integral(sec(x), log(sec(x) + tan(x)))
    assert_integral(2 * cos(2 * x - 5), sin(2 * x - 5)) 
    assert_integral(3 * x ** 5 - x ** 3 + 6, 6*x - x**4/4 + x**6/2)
    assert_integral(x ** 3 * e ** (x ** 4), (e**(x**4)/4))

    # Uses generic u-sub
    assert_definite_integral(e ** x / (1 + e ** x), (log(2), log(8)), log(9) - log(3))

    assert_definite_integral(8 * x / sqrt(1 - 4 * x ** 2), (0, Fraction(1,4)), 2 - sqrt(3))
    assert_definite_integral(sin(4*x), (0, pi/4), Fraction(1,2))

def test_expanding_big_power():
    integrand = (2 * x - 5) ** 10
    expected_ans = (2*x-5)**11/22
    assert_integral(integrand, expected_ans)

    integrand = 3 * x ** 2 * (x ** 3 + 1) ** 6
    expected_ans = (1 + x**3)**7/7
    assert_integral(integrand, expected_ans)

def test_polynomial_div_integrals():
    assert_integral((x-5) / (-2 * x + 2), - x / 2 + 2 * log(1 - x))
    assert_integral((x ** 3 - 1)/ (x+2), x**3/3 - x**2 + 4*x- 9*log(2 + x))
    assert_integral((x - 1)/ (2 * x + 4), x / 2 - Fraction(3, 2) * log(x + 2))
    
    integrand = (2 * x ** 3 + 4 * x ** 2 - 5)/ (x + 3)
    ans = integrate(integrand, x)
    # TODO: expected = ...

def test_complete_the_square_integrals():
    assert_integral(1/(3*x**2+6*x+78), atan((1 + x)/5)/15)
    assert_integral(1/(x**2-8*x+65), atan((-4 + x)/7)/7)
    assert_integral(1/sqrt(-x**2-6*x+40), asin((3 + x)/7))