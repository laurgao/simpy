import pytest

from src.simpy import *
from src.simpy.regex import any_, eq
from test_utils import x, y


@pytest.mark.parametrize("sum", [
tan(2*x)**2 + 1, tan(2*x+y)**2 + 1

])
def test_any(sum):
    assert eq(sum, 1 + tan(any_) ** 2)