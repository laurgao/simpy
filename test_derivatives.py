from src.simpy.expr import *
from test_utils import assert_eq_strict, x


def assert_diff(a: Expr, b: Expr):
    assert_eq_strict(diff(a, x), b)
    # assert_eq_repr_strict(a.diff(x), b)

def assert_diff_(a: Expr, b: Const, c: Const):
    ans = diff(a, x).evalf({"x": b})
    assert_eq_strict(ans, c)
    # assert ans == c, f"expected {debug_repr(c)}, got {debug_repr(ans)}"


def test_basic_differentiation():
    assert_diff(2 ** x, log(2) * 2 ** x)
    assert_diff(x ** 7, 7 * x ** 6)
    assert_diff(log(x), 1/x)
    
def test_kh_derivatives():
    assert_diff_(cot(x), 5*pi/3, Fraction(-4, 3))
    assert_diff_(sec(x), 0, 0)
    assert_diff_(tan(x), 2*pi/3, 4)
    assert_diff_(sec(x), 11*pi/6, -Fraction(2,3))
    assert_diff_(csc(x), pi/2, 0)

    assert_diff_(csc(x), 3*pi/4, sqrt(2))

    assert_diff(-4*e**x-sin(x)-9, -4*e**x - cos(x))
    assert_diff(-4*x**3-sin(x), -cos(x) - 12*x**2)
    assert_diff(sqrt(x)*e**x, e**x/(2*sqrt(x)) + sqrt(x)*e**x)
    assert_diff(cos(x)/log(x), (-x*sin(x)*log(x)-cos(x))/(x*log(x)**2))
    assert_diff(sin(x)/e**x, (cos(x)-sin(x))/e**x)



    print("passed")