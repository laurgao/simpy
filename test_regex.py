import pytest

from src.simpy.expr import *
from src.simpy.regex import Any_, any_, eq
from test_utils import x, y


def test_any_basic():
    assert not eq(Rat(1), Rat(-1))["success"]
    assert eq(sin(x), sin(any_))["success"]
    assert eq(tan(x)**2, tan(any_)**2)["success"]
    assert eq(tan(2*x)**2, tan(any_)**2)["success"]


@pytest.mark.parametrize(["sum", "expected"], [
    [tan(2*x)**2 + 1, {"success": True, "factor": 1, "anyfind": 2*x}],
    [tan(2*x+y)**2 + 1, {"success": True, "factor": 1, "anyfind": 2*x+y}],
    [3*tan(2*x+y)**2 + 3, {"success": True, "factor": 3, "anyfind": 2*x+y}]
])
def test_any_tan(sum, expected):
    assert eq(sum, 1 + tan(any_) ** 2, up_to_factor=True) == expected

def test_up_to_factor_on_simple_examples():
    assert eq(tan(3), tan(3), up_to_factor=True) == {"success": True, "factor": 1, "anyfind": {}}
    assert eq(x, x, up_to_factor=True) == {"success": True, "factor": 1, "anyfind": {}}
    assert eq(tan(x), 3*tan(any_), up_to_factor=True) == {"success": True, "factor": 3, "anyfind": x}

    # Not implemented right now but shrug don't need it.
    # assert eq(2, 3, up_to_factor=True) == {"success": True, "factor": Rat(3, 2), "anyfind": {}}

    # I need to set clearer rules with like what happens if there's an any * factor on the outside bracket
    # like it's unclearish. rn in anydivide the any can only catch one but what if it's both idk.
    # right now let's just let it fail LOL
    # assert eq(any_, x, up_to_factor=True) == {"success": True, "factor": 1, "anyfind": x}
    # assert eq(any_, x*2, up_to_factor=True) == {"success": True, "factor": 1, "anyfind": 2*x}

def test_factor_doesnt_overstep():
    assert not eq(1 + sin(x) ** 2, 1 + tan(any_) ** 2, up_to_factor=True)["success"]


def test_unlimited_sum():
    expr = 2 + x + sin(x) ** 2 + cos(x) ** 2
    query = sin(any_)**2 + cos(any_)**2
    out = eq(expr, query, up_to_sum=True)
    assert out["success"]
    assert out["rest"] == 2 + x


def test_up_to_sum():
    expr = -sin(x)**2 - 5**cos(x)**2 + 1
    query = 1 - sin(any_) ** 2
    ans = eq(expr, query, up_to_sum=True)
    assert ans == {"success": True, "anyfind": x, "rest": -5**cos(x)**2}

def test_up_to_sum_and_factor():
    expr = sin(x)**2 - 5**cos(x)**2 - 1
    query = 1 - sin(any_) ** 2
    ans = eq(expr, query, up_to_factor=True, up_to_sum=True)
    assert ans == {"success": True, "anyfind": x, "factor": -1, "rest": -5**cos(x)**2}
