from fractions import Fraction

import numpy as np

from khan_academy import (more_test, test_arcsin, test_ex,
                          test_expanding_big_power, test_partial_fractions,
                          test_polynomial_div_integrals, test_sec2x_tan2x,
                          test_xcosx)
from src.simpy.expr import *
from src.simpy.integration import *
from src.simpy.polynomial import to_const_polynomial
from src.simpy.transforms import PolynomialDivision, PullConstant
from test_expand import test_expand_power
from test_transforms import test_lecture_example, test_x2_sqrt_1_x3
from test_utils import assert_eq_plusc, assert_eq_repr, unhashable_set_eq


def test_to_polynomial():
    x = symbols("x")
    expr = 6 * x + x **2 
    assert np.array_equal(to_const_polynomial(expr, x), np.array([Const(0), Const(6), Const(1)]))


def test_factor():
    x = symbols("x")
    expr = 6 * x + x **2 
    expected = x * (x + 6)
    assert expr.factor() == expected


def test_polynomial_division():
    expr = x**4 * (1 + x**2) ** -1

    test_node = Node(expr, x)
    tr = PolynomialDivision()
    assert tr.check(test_node)

    tr.forward(test_node)
    assert_eq_plusc(test_node.children[0].expr, x**2 - 1 + 1 / (1 + x**2))

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
        - c3**2 * r3**2 / (c4 * r4 * sqrt(b) * a ** Fraction(3 / 2))
        - c4 * r4 / (sqrt(a) * b ** Fraction(3 / 2))
    )
    em_hw_expr = em_hw_expr.simplify()
    factored = em_hw_expr.factor()
    expected_factored = (
        1
        / (sqrt(a) * sqrt(b))
        * (c3 * r3 - c3**2 * r3**2 / (c4 * r4 * a) - c4 * r4 / b)
    )
    assert_eq_plusc(factored, expected_factored)


def test_compound_angle():
    # Finds the average power of an AC circuit
    w, phi, t = symbols("w \\phi t")
    ac_power_integrand = Cos(w * t - phi) * Cos(w * t)
    ac_power_integrand = ac_power_integrand.simplify()
    period = 2 * pi / w
    ac_power = 1 / period * integrate(ac_power_integrand, (t, 0, period))
    ac_power = ac_power.simplify()

    expected = Cos(phi) / 2
    assert_eq_plusc(ac_power, expected)


def test_sin2x():
    # tests the product-to-sum formula
    # Test integral (sin x)^2 = x / 2 - Sin(2x) / 4
    expr = Sin(x) ** 2
    integral = integrate(expr, x)
    expected = x / 2 - Sin(2 * x) / 4
    assert_eq_plusc(integral, expected)


def test_cos2x():
    # tests the product-to-sum formula
    # Test integral (cos x)^2 = Sin(2x) / 4 + x / 2
    expr = Cos(x) ** 2
    integral = integrate(expr, x)
    expected = Sin(2 * x) / 4 + x / 2
    assert_eq_plusc(integral, expected)

def test_linear_usub_with_multiple_subs():
    # Last I checked, this fails without LinearUSub
    integrand = Sin(2*x) / Cos(2*x)
    integral = integrate(integrand, x)
    expected = -Log(Cos(2*x))/2
    assert_eq_plusc(integral, expected)


def test_some_simplification():
    r1, r2, w, v = symbols("r_1 r_2 \\omega v_0")
    c = 1 / (r2 * w) * sqrt(r2 / r1 - 1)
    c = c.simplify()
    l = -r1 / (r2 * w) * sqrt(r2 / r1 - 1)
    l = l.simplify()
    r_s = r1
    x_s = (w * l).simplify()
    # z_s = r_s + x_s * 1j
    r_l = (r2 / (1 + w**2 * c**2 * r2**2)).simplify()
    x_l = (w * c * r2 / (1 + w**2 * c**2 * r2**2)).simplify()
    # z_l = r_l + x_l * 1j # simpy doesn't support complex numbers i think.

    z_eq = sqrt((r_s + r_l) ** 2 + (x_s + x_l) ** 2)
    z_eq = z_eq.simplify()
    i_0 = v / z_eq
    i_0 = i_0.simplify()
    pload = Fraction(1, 2) * r_l * i_0**2
    pload = pload.simplify()
    assert_eq_repr(
        pload, v**2 / (8 * r1)
    )  # the power dissipated through the load when z_l = conjugate(z_s)

def test_factor_const():
    expr = 2 - 2 * x
    factored = expr.factor()
    expected = 2 * (1 - x)
    assert_eq_repr(expected, factored)


def test_product_combine_like_terms():
    # this wasnt working bc the denominator "power" wasn't flattening in simplification.
    expr = (2*Sin(x)*Cos(x)**2)/(Sin(x)*Cos(x)**2)
    expr = expr.simplify()
    assert expr == 2

if __name__ == "__main__":
    x, y = symbols("x y")

    assert_eq_repr(x * 0, 0)
    assert_eq_repr(x * 2, 2 * x)
    assert_eq_repr(x**2, x * x)
    assert_eq_repr(x * 2 - 2 * x, 0)
    assert_eq_repr(((x + 1) ** 2 - (x + 1) * (x + 1)), 0)

    test_product_combine_like_terms()

    # Equality
    assert x == x
    assert x == Symbol("x")
    assert not x == y
    assert x != y
    assert not x == 2 * x
    assert (x + 2) == (x + 2)
    assert (x + 2).simplify() == (2 + x).simplify()  # ideally does this w/o simplify
    assert x + 2 == Symbol("x") + 2
    # for some reason defining __eq__ in Expr is not actually called.
    # smtn children inherits from dataclass ugh idk
    assert Const(2) == 2
    assert Const(2) != 3
    assert 2 == Const(2)
    assert 2 < Const(3)
    assert 2 <= Const(2)
    assert Const(2) == Const(2)
    assert Cos(x * 2) == Cos(x * 2)

    # const exponent simplification
    assert_eq_repr(x**0, 1)
    assert_eq_repr(x**1, x)
    assert_eq_repr(x**2, x * x)
    assert_eq_repr(x**3, x * x * x)
    assert_eq_repr(Const(2) ** 2, 4)
    assert_eq_repr(sqrt(4), 2)
    assert_eq_repr(sqrt(x**2), x)
    assert sqrt(3).__repr__() == "sqrt(3)"

    # Test nested flatten
    expr = x ** 5 + ((3 + x) + 2 * y)
    expected_terms = [x **5, 3, x, 2 * y]
    assert unhashable_set_eq(expr.terms, expected_terms)

    # Expand test
    # make sure an expandable denominator gets expanded
    assert_eq_repr((1 / (x * (x + 6))).expand(), 1 / (x**2 + x * 6))
    # make sure that a numberator with a single sum gets expanded
    assert_eq_repr(((2 + x) / Sin(x)).expand(), (2 / Sin(x) + x / Sin(x)))
    test_expand_power()

    # Factor test
    test_factor_const()
    test_factor()

    test_some_simplification()
    
    # Basic differentiation 
    assert_eq_repr((2 ** x).diff(x), Log(2) * 2 ** x)
    assert_eq_repr((x ** 7).diff(x), 7 * x ** 6)
    assert_eq_plusc(Log(x).diff(x), 1 / x)

    # Basic integrals
    assert_eq_plusc(integrate(2, (x, 5, 3)), -4)
    assert_eq_plusc(integrate(x ** Fraction(7, 3), x), Fraction(3, 10) * x ** Fraction(10,3))
    assert_eq_plusc(integrate(3 * x**2 - 2 * x, x), x**3 - x**2)
    assert_eq_plusc(integrate((x + 1) ** 2, x), x + x**2 + (x**3 / 3))

    assert_eq_plusc(integrate(x ** 12, x), x ** 13 / 13)
    assert_eq_plusc(integrate(1 / x, x), Log(x))
    assert_eq_plusc(integrate(1 / x, (x, 1, 2)), Log(2))
    assert_eq_plusc(integrate(y, x), x * y)
    assert_eq_plusc(integrate(Tan(y), x), x * Tan(y))

    assert nesting(x**2, x) == 2
    assert nesting(x * y**2, x) == 2
    assert nesting(x * (1 / y**2 * 3), x) == 2

    assert_eq_repr(x + (2 + y), x + 2 + y)

    assert count(2, x) == 0
    assert count(Tan(x + 1) ** 2 - 2 * x, x) == 2

    # cos^2 + sin^2 = 1 test
    expr = Sin(x) ** 2 + Cos(x) ** 2 + 3
    simplified = expr.simplify()
    assert_eq_repr(simplified, 4)

    expr = Sin(x - 2 * y) ** 2 + 3 + Cos(x - 2 * y) ** 2 + y**2
    simplified = expr.simplify()
    assert_eq_repr(simplified, 4 + y**2)

    # PullConstant test
    expr = 2 * x**3
    test_node = Node(expr, x)
    transform = PullConstant()
    assert transform.check(test_node)
    transform.forward(test_node)
    assert_eq_repr(test_node.child.expr, x**3)

    # new repr standards test
    expr = 1 - x**2
    assert expr.__repr__() == "1 - x^2"
    assert (2 * x).__repr__() == "2*x"
    assert (2 * (2 + x)).__repr__() == "2*(2 + x)"
    assert (2 / (2 + x)).__repr__() == "2/(2 + x)"
    assert repr(2 * (2 + x) ** (-2)) == repr(2 / (2 + x) ** 2) == "2/(2 + x)^2"
    assert repr(1 / sqrt(1 - x**2)) == "1/sqrt(1 - x^2)"
    # make sure denominator is bracketed
    assert repr(Sin(x) / (2 * x)) == "sin(x)/(2*x)"
    # make sure products with negative consts and dividing by consts are treated better
    assert repr((x / 2).simplify()) == "x/2"
    assert repr((x * Fraction(1, 2)).simplify()) == "x/2"
    assert repr((3 - 2 * x).simplify()) == "3 - 2*x"
    # make sure consts show up before pi
    assert repr((pi * 2).simplify()) == "2*pi"
    assert repr((pi * -2).simplify()) == "-2*pi"

    # This integral can either be sin^2(wt) / 2w or -cos^2(wt) / 2w depending on the method used to solve it
    w, t = symbols("w t")
    expr = Sin(w * t) * Cos(w * t)
    integral = integrate(expr, t)
    expected = (Sin(w * t) ** 2 / (2 * w)).simplify()
    expected2 = (-Cos(w * t) ** 2/ (2 * w)).simplify()
    assert repr(integral) == repr(expected) or repr(integral) == repr(expected2) # temp patch

    # PolynomialDivision test
    test_polynomial_division()
    test_to_polynomial()

    # ln integral
    assert (x * Log(x) - x).diff(x).simplify() == Log(x)
    ans = Integration._integrate(Log(x), x)
    assert_eq_plusc(ans, x * Log(x) - x)

    # run entire integrals
    test_lecture_example() 
    test_x2_sqrt_1_x3()
    test_compound_angle()
    test_ex()
    test_xcosx()
    test_partial_fractions()
    test_arcsin()
    test_linear_usub_with_multiple_subs()
    test_sec2x_tan2x()
    test_sin2x()
    test_cos2x()
    test_expanding_big_power()
    test_polynomial_div_integrals()
    more_test()

    integrand = Log(x + 6) / x**2
    integral = integrate(integrand, x)
    expected = Log(x) / 6 - Log(x + 6) / x - Log(x+6) / 6
    assert_eq_plusc(integral, expected)

    # without the string search fix on InverseTrigUSub, this returns a wrong answer.
    integrand = sqrt(x ** 2 + 11)
    ans = integrate(integrand, x)
    assert ans is None
    
    print("passed")
