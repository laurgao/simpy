import cProfile

from simpy import *
from simpy.debug.logger import Logger
from simpy.integration import Integration

logger = Logger()
Integration.logger = logger

x, w, phi = symbols("x w phi")


# ALl are taken from tests.
BENCHMARKING_SUITE = [
    -5 * x**4 / (1 - x**2) ** Rat(5, 2),
    x**2 / sqrt(1 - x**3),
    # (Rat(1, 15) - Rat(1, 360) * (x - 6)) * (1 - (40 - x) ** 2 / 875), # looks too ugly on the plot
    cos(w * x - phi) * cos(w * x),
    sin(2 * x) / cos(2 * x),
    cos(x) ** 2,
    sin(x) ** 2,
    sin(w * x) * cos(w * x),
    1 / sqrt(-(x**2) + 10 * x + 11),
    log(x + 6) / x**2,
    sin(x) * cos(2 * x) * sin(2 * x),
    6 * e**x,
    x * cos(x),
    asin(x),
    (x + 8) / (x * (x + 6)),
    x * e ** (-x),
    x**2 * sin(pi * x),
    sec(2 * x) * tan(2 * x),
    e**x / (1 + e**x),
    5 * csc(x) ** 2,
    2 * csc(x) * cot(x),
    (2 * x - 5) ** 10,
    (x - 5) / (-2 * x + 2),
    tan(x) ** 5 * sec(x) ** 4,
    tan(x) ** 4,
    sin(x) ** 2 * cos(x) ** 3,
    sin(x) ** 5,
]


# Create a Profile object
profiler = cProfile.Profile()

# Start profiling
profiler.enable()

### CODE IN BETWEEN THESE LINES IS PROFILED ###

for exp in BENCHMARKING_SUITE:
    integrate(exp, x)


### CODE IN BETWEEN THESE LINES IS PROFILED ###


profiler.disable()


logger.dump()
logger.plot()

# Print stats sorted by cumulative time
profiler.print_stats(sort="cumtime")
