import numpy as np

from .expr import *

Polynomial = np.ndarray  # has to be 1-D array


def is_polynomial(expr: Expr, var: Symbol) -> bool:
    try:
        to_const_polynomial(expr, var)
        return True
    except AssertionError:
        return False


def _to_const_polynomial(expr: Expr, var: Symbol, answer: List[Rat] = None, multiplier: Rat = Rat(1)) -> List[Rat]:
    if answer is None:
        answer: List[Rat] = []

    def _add_answer(const: Rat, idx: int, answer: List[Rat] = answer, multiplier: Rat = multiplier):
        new_const = const * multiplier
        if idx < len(answer):
            answer[idx] += new_const
        elif idx == len(answer):
            answer.append(new_const)
        else:
            answer += [Rat(0)] * (idx - len(answer))
            answer.append(new_const)

    if not expr.contains(var):
        _add_answer(expr, 0)

    elif isinstance(expr, Sum):
        for term in expr.terms:
            _to_const_polynomial(term, var, answer)

    elif isinstance(expr, Prod):
        # has to be product of 2 terms: a constant and a power.
        assert len(expr.terms) == 2
        const, other = expr.terms
        assert isinstance(const, Rat)
        _to_const_polynomial(other, var, answer, multiplier=const)
    elif isinstance(expr, Power):
        assert expr.base == var
        assert (
            isinstance(expr.exponent, Rat)
            and expr.exponent.value == int(expr.exponent.value)
            and expr.exponent.value >= 1
        )
        _add_answer(1, int(expr.exponent.value))
    elif isinstance(expr, Symbol):
        assert expr == var
        _add_answer(Rat(1), 1)
    else:
        raise AssertionError(f"Not allowed expr for polynomial: {expr}")

    return answer


def to_const_polynomial(expr: Expr, var: Symbol) -> Polynomial:
    expr = expr.expand() if expr.expandable() else expr
    return np.array(_to_const_polynomial(expr, var))


def polynomial_to_expr(poly: Polynomial, var: Symbol) -> Expr:
    final = Rat(0)
    for i, element in enumerate(poly):
        final += element * var**i
    return final


def rid_ending_zeros(lis: Polynomial) -> Polynomial:
    num_zeros = 0
    for i in reversed(range(len(lis))):
        if lis[i] == 0:
            num_zeros += 1
        else:
            break
    new_list = lis[: len(lis) - num_zeros]
    return np.array(new_list)
