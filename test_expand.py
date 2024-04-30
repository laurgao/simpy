from combinatorics import generate_permutations
from expr import Const, Prod, Sum, symbols


def test_expand_power():
    x = symbols("x")
    power = (x ** 3 + 1) ** 6
    prod = Prod([Sum([x ** Const(3), Const(1)])]*6)

    assert power.expand() == prod.expand()


def test_multinomial_powers():

    # Example usage
    i = 3  # Number of terms
    n = 3 # Sum of terms
    result = generate_permutations(i, n)
    print(result)
    expected_result = [
        [0, 0, 3],
        [0, 1, 2],
        [0, 2, 1],
        [0, 3, 0],
        [1, 0, 2],
        [1, 1, 1],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
        [3, 0, 0]
    ]
    # Yup it's the same.
    # TODO: make a assert