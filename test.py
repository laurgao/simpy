from main import *

def assert_eq(x, y):
    assert x == y, f"{x} == {y} is False. ({x-y}).simplify() = {(x+y).simplify()}"

@cast
def sassert_repr(a, b):
    xs, ys = a.simplify(), b.simplify()
    assert repr(xs) == repr(ys), f"{xs} != {ys} (original {a} != {b})"


x, = symbols('x')

sassert_repr(x*0, 0)
sassert_repr(x*2, 2*x)
sassert_repr(x**2, x*x)
sassert_repr(x*2 - 2*x, 0)
sassert_repr(((x+1)**2 - (x+1)*(x+1)), 0)

sassert_repr(integrate(3*x**2 - 2*x, x), x**3 - x**2)
sassert_repr(integrate((x+1)**2, x), x + x**2 + (x**3/3))
sassert_repr(Log(x).diff(x), 1/x)
sassert_repr(Log(x).diff(x), 1/x)
sassert_repr(integrate(1/x, x), Log(x))
sassert_repr(integrate(1/x, (x, 1, 2)), Log(2))

print('passed')