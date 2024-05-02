"""
LOL im gonna take a bunch of integral questions from https://www.khanacademy.org/math/integral-calculus/ic-integration/ic-integration-proofs/test/ic-integration-unit-
and make sure simpy can do them
"""

from src.simpy.expr import *
from src.simpy.integration import *
from test_utils import assert_eq_plusc

x = symbols("x")

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
    ans = integrate(integrand, x)
    expected_ans = Fraction(4, 3) * Log(x) - Fraction(1, 3) * Log(x + 6)
    assert_eq_plusc(ans, expected_ans)


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
    integrand = 4 * Sec(x) ** 2
    ans = integrate(integrand, x)
    assert ans == 4 * Tan(x)

    integrand = Sec(x) ** 2 * Tan(x) ** 2
    ans = integrate(integrand, x)
    assert ans == (Tan(x)**3/3).simplify()

    integrand = 5 / x - 3 * e ** x
    ans = integrate(integrand, x)
    assert ans == (5 * Log(x) - 3 * e ** x).simplify()

    integrand = Sec(x)
    ans = integrate(integrand, x)
    assert ans == Log(Sec(x) + Tan(x)).simplify()

    integrand = 2 * Cos(2 * x - 5)
    ans = integrate(integrand, x)
    assert ans == Sin(2 * x - 5).simplify()

    integrand = 3 * x ** 5 - x ** 3 + 6
    ans = integrate(integrand, x)
    assert ans == (6*x - x**4/4 + x**6/2).simplify()

    integrand = x ** 3 * e ** (x ** 4)
    ans = integrate(integrand, x)
    assert ans == (e**(x**4)/4).simplify()

    # Uses generic u-sub
    integrand = e ** x / (1 + e ** x)
    ans = integrate(integrand, (x, Log(2), Log(8)))
    assert_eq_plusc(ans, Log(9) - Log(3))

    integrand = 8 * x / sqrt(1 - 4 * x ** 2)
    ans = integrate(integrand, (x, 0, Fraction(1,4)))
    assert_eq_plusc(ans, 2 - sqrt(3))

    integrand = Sin(4*x)
    ans = integrate(integrand, (x, 0, pi/4))
    assert_eq_plusc(ans, Fraction(1,2))

def test_expanding_big_power():
    integrand = (2 * x - 5) ** 10
    ans = integrate(integrand, x)
    expected_ans = (2*x-5)**11/22
    assert_eq_plusc(ans, expected_ans)

    integrand = 3 * x ** 2 * (x ** 3 + 1) ** 6
    ans = integrate(integrand, x)
    expected_ans = (1 + x**3)**7/7
    const = (ans-expected_ans).expand().simplify()
    assert isinstance(const, Const)

def test_polynomial_div_integrals():
    integrand = (x-5) / (-2 * x + 2)
    ans = integrate(integrand, x)
    expected = - x / 2 + 2 * Log(1 - x)
    assert_eq_plusc(ans, expected)

    integrand = (x ** 3 - 1)/ ( x+2)
    ans = integrate(integrand, x)
    expected = x**3/3 - x**2 + 4*x- 9*Log(2 + x)
    assert_eq_plusc(ans, expected)

    integrand = (x - 1)/ (2 * x + 4)
    ans = integrate(integrand, x)
    expected = x / 2 - Fraction(3, 2) * Log(x + 2)
    assert_eq_plusc(ans, expected)
    
    integrand = (2 * x ** 3 + 4 * x ** 2 - 5)/ (x + 3)
    ans = integrate(integrand, x)
    breakpoint()
