from src.simpy import *
from src.simpy.simplify.product_to_sum import *
from test_utils import *


def test_sin_x_3():
    """Has been manually calculated <3"""
    product_to_sum_unit(sin(x) ** 3) == sin(x) / 2 - sin(3 * x) / 4 + sin(x) / 4


def test_cos_x_5():
    """Has been manually calculated <3"""
    product_to_sum_unit(cos(x) ** 5) == cos(5 * x) / 16 + cos(3 * x) / 16 + 3 * cos(x) / 8
