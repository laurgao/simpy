from src.simpy.expr import *


def test_infinity_basic_ops():
    assert 0 < oo
    assert Const(1) == 1
    assert oo ** 1 == oo
    assert oo == oo + 1
    assert sqrt(oo) == oo
    assert oo == 2*oo
    assert 192330 < oo
    assert oo > -oo
    assert -oo < -2039
    assert -oo == -oo - 1
    assert -oo == -2*oo
    
def test_nums():
    assert e > 2
    assert pi <= 4