import pytest

from src.simpy.expr import *
from src.simpy.regex import Any_, any_, eq
from test_utils import x, y


def test_any_basic():
    assert not eq(Const(1), Const(-1))[0]
    assert eq(sin(x), sin(any_))[0]


@pytest.mark.parametrize(["sum", "expected"], [
    [tan(2*x)**2 + 1, (True, 1, 2*x)],
    [tan(2*x+y)**2 + 1, (True, 1, 2*x+y)],
    [3*tan(2*x+y)**2 + 3, (True, 3, 2*x+y)]
])
def test_any_tan(sum, expected):
    assert eq(sum, 1 + tan(any_) ** 2, True) == expected

def test_up_to_factor_on_simple_examples():
    # assert eq(2, 3, True)
    assert eq(x, x, True) == (True, 1, {})
    assert eq(x, x*2, True) == (True, 2, {})
    assert eq(tan(3), tan(3), True) == (True, 1, {})
    assert eq(tan(x), 3*tan(x), True) == (True, 3, {})

def test_factor_doesnt_overstep():
    assert not eq(1 + sin(x) ** 2, 1 + tan(any_) ** 2, True)[0]

@pytest.mark.xfail
def test_unlimited_sum():
    expr = 2 + x + sin(x) ** 2 + cos(x) ** 2
    anyterms = Any_()
    query = anyterms + sin(any_) + cos(any_)
    assert eq(expr, query)[0]


def test_up_to_sum():
    expr = -sin(x)**2 - 5**cos(x)**2 + 1
    query = 1 - sin(any_) ** 2
    ans = eq(expr, query, up_to_sum=True)
    assert ans == (True, x)

def test_up_to_sum_and_factor():
    expr = sin(x)**2 - 5**cos(x)**2 - 1
    query = 1 - sin(any_) ** 2
    ans = eq(expr, query, up_to_factor=True, up_to_sum=True)
    breakpoint()
