"""latex is deployed to production now so I'll test it somewhat more rigorously."""

from src.simpy import *
from test_utils import *


def assert_latex(expr: Expr, expected_latex: str):
    result = latex(expr)
    assert result.replace(" ", "") == expected_latex.replace(" ", "")


def test_sum_subtraction_bug():
    assert latex((x - 5) / 2) == "\\frac{x - 5}{2}"


def test_readme():
    # I'll test some examples in the readme I guess.
    assert_latex(atan(x), "\\tan^{-1}\\left(x\\right)")
    assert_latex(2 * log(abs(1 - x)) - x / 2, "2 \\cdot \\ln\\left( \\left| -x + 1 \\right| \\right) - \\frac{x}{2}")
