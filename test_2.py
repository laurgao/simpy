from main import *

x = symbols("x")
thing = 1 - x**2
assert thing.__repr__() == "(1 - x^2)"
print("Passed")
