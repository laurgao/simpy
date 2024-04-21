import numpy as np

from expr import *

Polynomial = np.ndarray  # has to be 1-D array

def _to_const_polynomial(expr: Expr, var: Symbol, answer: List[Const] = None, multiplier:Const=Const(1)) -> List[Const]:
    if answer is None:
        answer: List[Const] = []
    
    def _add_answer(const: Const, idx: int, answer: List[Const]=answer, multiplier: Const=multiplier):
        new_const = (const * multiplier).simplify() # you really shouldn't need simplify here. combine like terms should happen in init.
        if idx < len(answer):
            answer[idx] += new_const
        elif idx == len(answer):
            answer.append(new_const)
        else:
            answer += [Const(0)] * (idx - len(answer))
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
        assert isinstance(const, Const)
        _to_const_polynomial(other, var, answer, multiplier=const)
    elif isinstance(expr, Power):
        assert expr.base == var
        assert isinstance(expr.exponent, Const) and expr.exponent.value == int(expr.exponent.value) and expr.exponent.value >= 1
        _add_answer(1, int(expr.exponent.value))
    elif isinstance(expr, Symbol):
        assert expr == var
        _add_answer(Const(1), 1)
    else:
        raise AssertionError(f"Not allowed expr for polynomial: {expr}")
    
    return answer
        

def to_const_polynomial(expr: Expr, var: Symbol) -> Polynomial:
    return np.array(_to_const_polynomial(expr, var))


def polynomial_to_expr(poly: Polynomial, var: Symbol) -> Expr:
    final = Const(0)
    for i, element in enumerate(poly):
        final += element * var**i
    return final.simplify()


def rid_ending_zeros(arr: Polynomial) -> Polynomial:
    lis = [el.simplify() for el in arr]
    num_zeros = 0
    for i in reversed(range(len(lis))):
        if lis[i] == 0:
            num_zeros += 1
        else:
            break
    new_list = lis[:len(lis) - num_zeros]
    return np.array(new_list)
