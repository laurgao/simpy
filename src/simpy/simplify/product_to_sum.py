from typing import Optional, Union

from ..expr import Expr, Power, Prod, Rat, Sum, cos, remove_const_factor, sin
from ..utils import count_symbols


def perform_on_terms(a: Union[sin, cos], b: Union[sin, cos], *, const: Optional[Expr] = None) -> Sum:
    # Dream:
    # a_, b_ = any
    # sin(a_) * sin(b_) = cos(a_-b_) - cos(a_+b_)
    # highly readable and very cool

    c = Rat(1, 2) if const is None else const / 2

    if isinstance(a, sin) and isinstance(b, cos):
        return sin(a.inner + b.inner) * c + sin(a.inner - b.inner) * c
    elif isinstance(a, cos) and isinstance(b, sin):
        return sin(a.inner + b.inner) * c - sin(a.inner - b.inner) * c
    elif isinstance(a, cos) and isinstance(b, cos):
        return cos(a.inner + b.inner) * c + cos(a.inner - b.inner) * c
    elif isinstance(a, sin) and isinstance(b, sin):
        return cos(a.inner - b.inner) * c - cos(a.inner + b.inner) * c


def pts_perf(expr: Expr) -> Optional[Expr]:
    """Returns the result of applying product-to-sum on expr, if possible
    Where expr is the product of 2 or more trig functions
    If you want to apply pts on a sum, use product_to_sum
    """
    new_expr, const = remove_const_factor(expr, include_factor=True)

    def is_valid_power(power: Power) -> bool:
        return (
            isinstance(power, Power)
            and isinstance(power.base, (sin, cos))
            and power.exponent.is_int
            and power.exponent > 1
        )

    if isinstance(new_expr, Prod):
        if len(new_expr.terms) == 2:
            t1, t2 = new_expr.terms
            if isinstance(t1, (sin, cos)) and isinstance(t2, (sin, cos)):
                return perform_on_terms(*new_expr.terms, const=const)
            if isinstance(t1, (sin, cos)) and is_valid_power(t2):
                return product_to_sum((pts_perf(t2) * t1).expand(), always_simplify=True, const=const)
            if isinstance(t2, (sin, cos)) and is_valid_power(t1):
                return product_to_sum((pts_perf(t1) * t2).expand(), always_simplify=True, const=const)
            if is_valid_power(t2) and is_valid_power(t1):
                intermediate = pts_perf(t1) * pts_perf(t2)
                return product_to_sum(intermediate.expand(), const=const, always_simplify=True)

    if is_valid_power(new_expr):
        if new_expr.exponent == 2:
            return perform_on_terms(new_expr.base, new_expr.base, const=const)
        elif new_expr.exponent % 2 == 0:
            intermediate = perform_on_terms(new_expr.base, new_expr.base) ** (new_expr.exponent / 2)
            return product_to_sum(intermediate.expand(), const=const, always_simplify=True)
        else:
            intermediate = new_expr.base * perform_on_terms(new_expr.base, new_expr.base) ** (
                (new_expr.exponent - 1) / 2
            )
            return product_to_sum(intermediate.expand(), const=const, always_simplify=True)


def product_to_sum(expr: Expr, *, always_simplify=False, const: Expr = None) -> Optional[Expr]:
    """Does applying product-to-sum on every term of a sum ... create a cancellation?

    Assumes that expr.has(TrigFunctionNotInverse) == True
    """
    if not isinstance(expr, Sum):
        return

    satisfies = [pts_perf(t) for t in expr.terms]
    if all(s is None for s in satisfies):
        return

    final_terms = []
    for boool, term in zip(satisfies, expr.terms):
        if not boool:
            final_terms.append(term)
            continue
        if isinstance(boool, Sum):
            final_terms.extend(boool.terms)

    if const is not None and const != 1:
        final_terms = [t * const for t in final_terms]
    final = Sum(final_terms)

    if always_simplify:
        return final

    # If final is simpler, return final
    if not isinstance(final, Sum):
        return final
    if len(final.terms) < len(expr.terms):
        return final

    if len(final.terms) == len(expr.terms) and count_symbols(final) < count_symbols(expr):
        # This ensures that e.g. 2*cos(x)*sin(2*x)/3 - cos(2*x)*sin(x)/3 simplifies to -2*sin(x)**3/3 + sin(x)
        return final
