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
    assert_latex(sec(2 * x), "\\sec(2 \\cdot x)")
    assert_latex(atan(x), "\\tan^{-1}(x)")
    assert_latex(2 * log(abs(1 - x)) - x / 2, "2 \\cdot \\ln (| -x + 1 |) - \\frac{x}{2}")


def test_big_frac():
    assert_latex(tan((2 + x) / (x + 3)), "\\tan \left(\\frac{x + 2}{x + 3} \\right)")
