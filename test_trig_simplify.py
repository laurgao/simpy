import pytest

from src.simpy import *
from src.simpy.transforms import ProductToSum, replace_factory
from test_utils import *


def test_simple():
    assert_eq_value(1/sin(x), csc(x))
    assert_eq_value(1/cos(x), sec(x))
    assert_eq_value(tan(x), sin(x)/cos(x))


"""Pythagorean simplifications"""
def test_simplify_sin2x_plus_cos2x():
    expr = sin(x) ** 2 + cos(x) ** 2 + 3
    simplified = expr.simplify()
    assert_eq_strict(simplified, 4)

    # Test when sin^2(...) -> ... is more complicated than just 'x'
    expr = (sin(x - 2 * y) ** 2 + 3 + cos(x - 2 * y) ** 2 + y**2 + cos(x))
    simplified = expr.simplify()
    assert_eq_strict(simplified, 4 + y**2 + cos(x))

    # Test when sin^2(...) and cos^2(...) share a common factor
    expr = 2*y*sin(x**3)**2 + 2*y*cos(x**3)**2
    simplified = expr.simplify()
    assert_eq_strict(simplified, 2*y)

@pytest.mark.xfail
def test_simplify_one_minus_sin_squared():
    # this one used to fail when I only allowed the entire sum to be 1 - sin^2(...)
    expr = sin(x)**2 - 5*cos(x)**2 + -1
    simplified = expr.simplify()
    assert_eq_strict(simplified, -4*cos(x))


def test_one_plus_tan_squared():
    expr = 1 + tan(x+y)** 2
    simp = expr.simplify()
    assert_eq_strict(simp, sec(x+y)**2)

    expr = 2 + 2*tan(x+y)** 2
    simp = expr.simplify()
    assert_eq_strict(simp, 2 * sec(x+y) ** 2)

    # expr = (2+y) + (2+y)*tan(x+y)** 2

"""Other things

These didn't pass when I didn't have 'heuristic simplifications'
"""

def product_to_sum(expr):
    return replace_factory(ProductToSum.condition, ProductToSum.perform)(expr)

def test_pts():
    # applying product-to-sum makes this simpler
    e1 = 2*cos(x)*sin(2*x)/3 - sin(x)*cos(2*x)/3
    e2 = sin(3*x)/6 + sin(x)/2
    e1 = product_to_sum(e1)
    e2 = product_to_sum(e2)
    assert_eq_value(e1, e2)

def test_sectan():
    # need to replace sec^2(x) with tan^2(x) + 1
    e1 = 1/(4*cos(x)**4) - 1/(3*cos(x)**6) + 1/(8*cos(x)**8)
    e2 = tan(x)**6/6 + tan(x)**8/8
    assert_eq_plusc(e1, e2)

def test_csc_squared_deriv():
    # need to write in terms of a common denominator
    e1 = sin(x)/cos(x) - csc(x)/cos(x)
    # this shit takes 2 rounds of simplification: first to -cos(x)/sin(x) then to -cot(x)
    # make sure that both rounds are done in the simplify function.
    e2 = -cot(x)
    assert_eq_strict(e1.simplify(), e2)

@pytest.mark.xfail
def test_compound_angle():
    # old solution of sin(x) ** 2 * cos(x) ** 3 simpy found when going 27 levels deep
    # im guessing needs to compound angle/double angle this one.
    e1 = sin(x)*cos(4*x)/120 + sin(x)*cos(2*x)/12 - cos(x)*sin(4*x)/30 - cos(x)*sin(2*x)/6 - sin(x)**3/6 + 3*sin(x)/8
    e2 = sin(x) ** 3 / 3 - sin(x) ** 5 / 5
    assert_eq_value(e1, e2)

