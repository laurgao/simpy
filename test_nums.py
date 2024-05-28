from src.simpy.expr import *


def test_infinity_basic_ops():
    assert 0 < inf
    assert Rat(1) == 1
    assert inf**1 == inf
    assert inf == inf + 1
    assert sqrt(inf) == inf
    assert inf == 2 * inf
    assert 192330 < inf
    assert inf > -inf
    assert -inf < -2039
    assert -inf == -inf - 1
    assert -inf == -2 * inf


def test_nums():
    assert e > 2
    assert pi <= 4
