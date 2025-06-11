from test_utils import assert_eq_strict, x

from simpy.expr import *
from simpy.simplify import simplify


def test_simplifies_when_expanding_is_simpler():
    expr = 2 * (2 * x + 3)
    assert_eq_strict(simplify(expr), 4 * x + 6)
