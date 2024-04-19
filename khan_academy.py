"""
LOL im gonna take a bunch of integral questions from https://www.khanacademy.org/math/integral-calculus/ic-integration/ic-integration-proofs/test/ic-integration-unit-
and make sure simpy can do them
"""

from test import sassert_repr

from expr import e, pi
from integration import *

x = symbols("x")

integrand = 6 * e**x
ans = Integration.integrate(integrand, (x, 6, 12))
sassert_repr(ans, 6 * e**12 - 6 * e**6)

integrand = x * Cos(x)
ans = Integration.integrate(integrand, (x, 3 * pi / 2, pi))
