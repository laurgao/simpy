from src.simpy import *
from test_utils import *


def test_sum_subtraction_bug():
    assert latex((x - 5) / 2) == "\\frac{x - 5}{2}"
