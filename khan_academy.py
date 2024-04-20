"""
LOL im gonna take a bunch of integral questions from https://www.khanacademy.org/math/integral-calculus/ic-integration/ic-integration-proofs/test/ic-integration-unit-
and make sure simpy can do them
"""

from expr import e, pi
from integration import *


@cast
def sassert_repr(a, b):
    xs, ys = a.simplify(), b.simplify()
    assert repr(xs) == repr(ys), f"{xs} != {ys} (original {a} != {b})"

x = symbols("x")

def test_ex():
    integrand = 6 * e**x
    ans = Integration.integrate(integrand, (x, 6, 12))
    sassert_repr(ans, 6 * e**12 - 6 * e**6)

def test_xcosx():
    """Uses integration by parts"""
    integrand = x * Cos(x)
    ans = Integration.integrate(integrand, (x, 3 * pi / 2, pi))
    sassert_repr(ans, 3 * pi / 2 - 1)

