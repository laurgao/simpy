from src.simpy import *
from test_utils import *


def test_simplify_sin2x_plus_cos2x():
    expr = sin(x) ** 2 + cos(x) ** 2 + 3
    simplified = expr.simplify()
    assert_eq_strict(simplified, 4)

    # Test when sin^2(...) -> ... is more complicated than just 'x'
    expr = sin(x - 2 * y) ** 2 + 3 + cos(x - 2 * y) ** 2 + y**2
    simplified = expr.simplify()
    assert_eq_strict(simplified, 4 + y**2)

    # Test when sin^2(...) and cos^2(...) share a common factor
    expr = 2*y*sin(x**3)**2 + 2*y*cos(x**3)**2
    simplified = expr.simplify()
    assert_eq_strict(simplified, 2*y)


def test_one_plus_tan_squared():
    expr = 1 + tan(x+y)** 2
    simp = expr.simplify()
    assert_eq_strict(simp, sec(x+y)**2)

    expr = 2 + 2*tan(x+y)** 2
    simp = expr.simplify()
    assert_eq_strict(simp, 2 * sec(x+y) ** 2)

    # expr = (2+y) + (2+y)*tan(x+y)** 2