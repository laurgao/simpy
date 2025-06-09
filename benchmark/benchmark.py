import statistics
import time
from fractions import Fraction

import simpy as sp

e = sp.e

# import sympy as sp

# e = sp.E


x, w, phi = sp.symbols("x w phi")

# ALl are taken from tests.
BENCHMARKING_SUITE = [
    -5 * x**4 / (1 - x**2) ** Fraction(5, 2),
    x**2 / sp.sqrt(1 - x**3),
    (Fraction(1, 15) - Fraction(1, 360) * (x - 6)) * (1 - (40 - x) ** 2 / 875),
    sp.cos(w * x - phi) * sp.cos(w * x),
    sp.sin(2 * x) / sp.cos(2 * x),
    sp.cos(x) ** 2,
    sp.sin(x) ** 2,
    sp.sin(w * x) * sp.cos(w * x),
    1 / sp.sqrt(-(x**2) + 10 * x + 11),
    sp.log(x + 6) / x**2,
    sp.sin(x) * sp.cos(2 * x) * sp.sin(2 * x),
    6 * e**x,
    x * sp.cos(x),
    sp.asin(x),
    (x + 8) / (x * (x + 6)),
    x * e ** (-x),
    x**2 * sp.sin(sp.pi * x),
    sp.sec(2 * x) * sp.tan(2 * x),
    e**x / (1 + e**x),
    5 * sp.csc(x) ** 2,
    2 * sp.csc(x) * sp.cot(x),
    (2 * x - 5) ** 10,
    (x - 5) / (-2 * x + 2),
    sp.tan(x) ** 5 * sp.sec(x) ** 4,
    sp.tan(x) ** 4,
    sp.sin(x) ** 2 * sp.cos(x) ** 3,
    sp.sin(x) ** 5,
]


time_taken = []
for _ in range(100):
    start = time.time()

    ### CODE IN BETWEEN THESE LINES IS PROFILED ###

    for exp in BENCHMARKING_SUITE:
        ans = sp.integrate(exp, x)

    ### CODE IN BETWEEN THESE LINES IS PROFILED ###

    end = time.time()
    time_taken.append(end - start)


print(
    f"Time taken: {statistics.mean(time_taken)}, averaged across {len(time_taken)} runs with stdev {statistics.stdev(time_taken)}"
)
