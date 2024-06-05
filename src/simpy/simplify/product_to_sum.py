from typing import List, Optional, Union

from ..expr import Expr, Power, Prod, Rat, Sum, cos, remove_const_factor, sin
from ..utils import count_symbols


def _perform_on_terms(
    a: Union[sin, cos], b: Union[sin, cos], *, multiplier: Optional[Expr] = None
) -> Optional[List[Expr]]:
    # Dream:
    # a_, b_ = any
    # sin(a_) * sin(b_) = cos(a_-b_) - cos(a_+b_)
    # highly readable and very cool

    c = Rat(1, 2) if multiplier is None else multiplier / 2

    if isinstance(a, sin) and isinstance(b, cos):
        return [sin(a.inner + b.inner) * c, sin(a.inner - b.inner) * c]
    elif isinstance(a, cos) and isinstance(b, sin):
        return [sin(a.inner + b.inner) * c, -sin(a.inner - b.inner) * c]
    elif isinstance(a, cos) and isinstance(b, cos):
        return [cos(a.inner + b.inner) * c, cos(a.inner - b.inner) * c]
    elif isinstance(a, sin) and isinstance(b, sin):
        return [cos(a.inner - b.inner) * c, -cos(a.inner + b.inner) * c]


def product_to_sum_unit(expr: Expr) -> Optional[Expr]:
    """Returns the result of applying product-to-sum on expr, if possible
    Where expr is the product of 2 or more trig functions
    If you want to apply pts on a sum, use product_to_sum
    """
    result = _product_to_sum_unit(expr)
    if result is not None:
        return Sum(result)


def _product_to_sum_unit(expr: Expr, *, multiplier: Expr = None) -> Optional[List[Expr]]:
    new_expr, const = remove_const_factor(expr, include_factor=True)
    if multiplier is not None:
        const *= multiplier

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
                return _perform_on_terms(t1, t2, multiplier=const)
            elif isinstance(t1, (sin, cos)) and is_valid_power(t2):
                intermediate = _product_to_sum_unit(t2, multiplier=t1)
            elif isinstance(t2, (sin, cos)) and is_valid_power(t1):
                intermediate = _product_to_sum_unit(t1, multiplier=t2)
            elif is_valid_power(t2) and is_valid_power(t1):
                intermediate = (product_to_sum_unit(t1) * product_to_sum_unit(t2)).expand()
            else:
                return

            return _product_to_sum(intermediate, multiplier=const)

    if is_valid_power(new_expr):
        if new_expr.exponent == 2:
            return _perform_on_terms(new_expr.base, new_expr.base, multiplier=const)
        elif new_expr.exponent == 3:
            intermediate = _perform_on_terms(new_expr.base, new_expr.base, multiplier=new_expr.base)
        else:
            intermediate = _perform_on_terms(
                new_expr.base, new_expr.base, multiplier=Power(new_expr.base, new_expr.exponent - 2, skip_checks=True)
            )
        return _product_to_sum(intermediate, multiplier=const)


def _product_to_sum(sum: List[Expr], *, multiplier: Expr = None) -> Optional[List[Expr]]:
    """takes in terms and returns terms

    We call this function in intermediate steps. This prevents us from creating a bunch of unnecessary exprs in between.
    """
    results = [_product_to_sum_unit(t) for t in sum]
    if all(r is None for r in results):
        return

    final_terms = []
    for result, term in zip(results, sum):
        if not result:
            final_terms.append(term)
            continue
        final_terms.extend(result)

    if multiplier is not None and multiplier != 1:
        final_terms = [t * multiplier for t in final_terms]

    return final_terms


def product_to_sum(expr: Expr) -> Optional[Expr]:
    """The function used in simplify.
    Does applying product-to-sum on every term of a sum ... create a cancellation?

    Assumes that expr.has(TrigFunctionNotInverse) == True
    """
    if not isinstance(expr, Sum):
        return

    final_terms = _product_to_sum(expr.terms)
    if final_terms is None:
        return
    final = Sum(final_terms)

    # If final is simpler, return final
    if not isinstance(final, Sum):
        return final
    if len(final.terms) < len(expr.terms):
        return final

    if len(final.terms) == len(expr.terms) and count_symbols(final) < count_symbols(expr):
        # This ensures that e.g. 2*cos(x)*sin(2*x)/3 - cos(2*x)*sin(x)/3 simplifies to -2*sin(x)**3/3 + sin(x)
        return final
