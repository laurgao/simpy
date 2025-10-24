from typing import List, Optional, Union

from ..expr import Expr, Power, Prod, Rat, Sum, TrigFunctionNotInverse, cos, nesting, remove_const_factor, sin
from ..regex import Any_, any_, eq
from ..utils import count_symbols
from .utils import is_simpler


def _perform_on_terms(
    a: Union[sin, cos], b: Union[sin, cos], *, multiplier: Optional[Expr] = None
) -> Optional[List[Expr]]:
    """Returns the result of applying product-to-sum on a and b, if possible"""
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


def product_to_sum_unit(expr: Expr) -> Optional[Sum]:
    """Returns the result of applying product-to-sum on expr, if possible
    Where expr is the product of 2 or more trig functions
    If you want to apply pts on a sum, use product_to_sum
    """
    result = _product_to_sum_unit(expr)
    if result is not None:
        return Sum(result)


def _product_to_sum_unit(expr: Expr, *, multiplier: Expr = None) -> Optional[List[Expr]]:
    """Returns the result of applying product-to-sum on expr, if possible.
    expr is the product of 2 or more trig functions. it can be a Prod or Power
    Returns None if expr is not a product of sin and cos.
    Otherwise, returns a list of terms of a sum. This function is used in intermediate steps.
    """
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

        if not all(isinstance(t, (sin, cos)) or is_valid_power(t) for t in new_expr.terms):
            return

        if any(not isinstance(t, (sin, cos)) for t in new_expr.terms):
            new_terms = [product_to_sum_unit(t) for t in new_expr.terms if not isinstance(t, (sin, cos))]
            return _product_to_sum(Prod(new_terms).expand().terms, multiplier=const)

        return _perform_on_terms(
            new_expr.terms[0], new_expr.terms[1], multiplier=Prod(new_expr.terms[2:], skip_checks=True) * const
        )

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
    """takes in terms of a sum and returns terms of a sum after applying product-to-sum on each applicable term.
    If no terms are changed, returns None
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


# these two private functions are inter recursive.
def _double_angle_sin(num: Expr, x: Expr) -> Expr:
    if num == 2:
        final = 2 * sin(x) * cos(x)
    elif num == 3:
        final = 3 * sin(x) - 4 * sin(x) ** 3
    elif num == 4:
        final = 4 * sin(x) * cos(x) - 8 * sin(x) ** 3 * cos(x)
    elif num % 2 == 0:
        final = 2 * _double_angle_sin(num / 2, x) * _double_angle_cos(num / 2, x)
    elif num % 3 == 0:
        t = _double_angle_sin(num / 3, x)
        final = 3 * t - 4 * t**3
    else:
        return sin(num * x)
    return final


def _double_angle_cos(num: Expr, x: Expr) -> Expr:
    if num == 2:
        return 1 - 2 * sin(x) ** 2
    if num % 2 == 0:
        return 1 - 2 * _double_angle_sin(num=num / 2, x=x)
    if num == 3:
        return -3 * cos(x) + 4 * cos(x) ** 3
    if num % 3 == 0:
        t = _double_angle_cos(num / 3, x)
        return -3 * t + 4 * t**3
    return cos(num * x)


def double_angle(expr: Expr) -> Optional[Expr]:
    """Applies double angle
    Used in simplify

    Using these identities:
    sin(2x) = 2sin(x)cos(x)
    cos(2x) = 1 - 2 * sin^2(x)

    Assumes that expr.has(TrigFunctionNotInverse) == True
    """

    if not isinstance(expr, (sin, cos)):
        return

    any_even_number = Any_(
        "even_number", lambda expr: isinstance(expr, Rat) and expr.denominator == 1 and expr % 2 == 0, is_constant=True
    )
    query = any_even_number * any_
    out = eq(expr.inner, query)

    if not out["success"]:
        return

    x = out["matches"][any_.key]
    num = out["matches"]["even_number"]

    if isinstance(expr, sin):
        final = _double_angle_sin(num=num, x=x)
    else:
        final = _double_angle_cos(num=num, x=x)

    if not final.has(TrigFunctionNotInverse) or is_simpler(final, expr):
        # If final doesn't have any trig functions, it's definitely simpler.
        # this condition can def be ... improved.
        # currently im basing it off of the
        # sin(4*asin(x/2)) -> 2*x*sqrt(-x^2/4 + 1) - x^3*sqrt(-x^2/4 + 1)
        # case.
        # that is not simpler by any other metric other than it doesn't have the sin(asin) nesting.
        # nesting of 2 trig funcs is always ugly. maybe the most robust metric should just rid those ugly
        # nests.
        return final
