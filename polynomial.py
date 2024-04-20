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


def to_polynomial(expr: Expr, var: Symbol) -> Polynomial:
    # TODO: this needs to be rewritten to reuse logic between sum terms and the rest. maybe.
    if isinstance(expr, Sum):
        xyz = np.zeros(10)
        for term in expr.terms:
            if isinstance(term, Prod):
                assert len(term.terms) == 2
                const, power = term.terms
                assert isinstance(const, Const)
                if isinstance(power, Symbol):
                    xyz[1] = int(const.value)
                    continue
                assert isinstance(power, Power)
                assert power.base == var
                xyz[int(power.exponent.value)] = int(const.value)
            elif isinstance(term, Power):
                assert term.base == var
                xyz[int(term.exponent.value)] = 1
            elif isinstance(term, Symbol):
                assert term == var
                xyz[1] = 1
            elif isinstance(term, Const):
                xyz[0] = int(term.value)
            else:
                raise NotImplementedError(f"weird term: {term}")
        return rid_ending_zeros(xyz)

    if isinstance(expr, Prod):
        # has to be product of 2 terms: a constant and a power.
        assert len(expr.terms) == 2
        const, power = expr.terms
        assert isinstance(const, Const)
        if isinstance(power, Symbol):
            return np.array([0, int(const.value)])
        assert isinstance(power, Power)
        assert power.base == var
        xyz = np.zeros(int(power.exponent.value) + 1)
        xyz[-1] = const.value
        return xyz
    if isinstance(expr, Power):
        assert expr.base == var
        xyz = np.zeros(int(expr.exponent.value) + 1)
        xyz[-1] = 1
        return xyz
    if isinstance(expr, Symbol):
        assert expr == var
        return np.array([0, 1])
    if isinstance(expr, Const):
        return np.array([expr.value])

    raise NotImplementedError(f"weird expr: {expr}")


def polynomial_to_expr(poly: Polynomial, var: Symbol) -> Expr:
    final = Const(0)
    for i, element in enumerate(poly):
        final += element * var**i
    return final.simplify()


def rid_ending_zeros(arr: Polynomial) -> Polynomial:
    return np.trim_zeros(arr, "b")
