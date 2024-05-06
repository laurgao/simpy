import pytest

from src.simpy import *
from src.simpy.regex import any_, eq
from test_utils import x, y


@pytest.mark.parametrize(["sum", "expected"], [
    [tan(2*x)**2 + 1, (True, 1, 2*x)],
    [tan(2*x+y)**2 + 1, (True, 1, 2*x+y)],
    [3*tan(2*x+y)**2 + 3, (True, 3, 2*x+y)]
])
def test_any(sum, expected):
    assert eq(sum, 1 + tan(any_) ** 2, True) == expected

def test_up_to_factor_on_simple_examples():
    # assert eq(2, 3, True)
    assert eq(x, x, True) == (True, 1, {})
    assert eq(x, x*2, True) == (True, 2, {})
    assert eq(tan(3), tan(3), True) == (True, 1, {})
    assert eq(tan(x), 3*tan(x), True) == (True, 3, {})

def test_factor_doesnt_overstep():
    assert not eq(1 + sin(x) ** 2, 1 + tan(any_) ** 2, True)[0]