from typing import Iterable, List, Optional, Tuple, Type, Union

from .expr import Abs, Expr, Power, Prod, Rat, Sum, Symbol, TrigFunction, cos, cot, csc, log, nesting, sec, sin, tan
from .regex import any_, eq, general_contains, general_count, kinder_replace, kinder_replace_many, replace_class
from .utils import ExprFn


def expand_logs(expr: Expr, **kwargs) -> Expr:
    return kinder_replace(expr, _log_perform, **kwargs)


def _log_perform(expr: Expr) -> Optional[Expr]:
    if not isinstance(expr, log):
        return
    inner = expr.inner
    # IDK if this should be in simplify or if it should be like expand, in a diff function
    # like you can move stuff together or move stuff apart
    if isinstance(inner, Power):
        return log(inner.base) * inner.exponent
    if isinstance(inner, Prod):
        return Sum([log(t) for t in inner.terms])
    if isinstance(inner, Sum) and isinstance(inner.factor(), Prod):
        return Sum([log(t) for t in inner.factor().terms])

    # let's agree on some standards
    # i dont love this, can change
    if isinstance(inner, (sec, csc, cot)):
        return -1 * log(inner.reciprocal_class(inner.inner))

    # ugly patch lmfao. need better overall philosophy for this stuff.
    if isinstance(inner, Abs) and isinstance(inner.inner, Power):
        return log(abs(inner.inner.base)) * inner.inner.exponent
    if isinstance(inner, Abs) and isinstance(inner.inner, Prod):
        return Sum([log(abs(term)) for term in inner.inner.terms])
    if isinstance(inner, Abs) and isinstance(inner.inner, Sum) and isinstance(inner.inner.factor(), Prod):
        return Sum([log(abs(term)) for term in inner.inner.factor().terms])

    return log(inner)


def pythagorean_simplification(expr: Expr, **kwargs) -> Expr:
    verbose = kwargs.get("verbose", False)
    if not expr.has(TrigFunction):
        return expr if not verbose else expr, False
    return kinder_replace(expr, _pythagorean_perform, **kwargs)


def _pythagorean_perform(sum: Expr) -> Optional[Expr]:
    if not isinstance(sum, Sum):
        return

    # first check if we have anything squared before we do anything else
    # because the rest of the function, with all the `eq` calls, is expensive.
    cond = (
        lambda x: isinstance(x, Power)
        and x.exponent == 2
        and isinstance(x.base, TrigFunction)
        and x.base.is_inverse is False
    )
    if not general_contains(sum, cond):
        return

    ##------------ Simplify sin^2(...) + cos^2(...) ------------##
    def is_thing_squared(
        term: Expr, cls: Union[Type[TrigFunction], Iterable[Type[TrigFunction]]]
    ) -> Optional[Tuple[Expr, Expr]]:  # returns inner, factor
        def is_(f):
            return isinstance(f, Power) and f.exponent == 2 and isinstance(f.base, cls)

        if is_(term):
            return term.base.inner, Rat(1)

        if not isinstance(term, Prod):
            return None
        for f in term.terms:
            if is_(f):
                return f.base.inner, term / f
        return None

    def sin_cos_condition(expr: Sum):
        # make sure that a*cos^2(...) and a*sin^2(...) are both terms of this sum
        # with an optional factor that may or may not be there of a
        s = [is_thing_squared(term, sin) for term in expr.terms]
        c = [is_thing_squared(term, cos) for term in expr.terms]
        return [el for el in s if el is not None and el in c]

    both_inners = sin_cos_condition(sum)
    if len(both_inners) > 0:
        new_terms = [t for t in sum.terms if not is_thing_squared(t, (sin, cos)) in both_inners]
        new_terms += [el[1] for el in both_inners]
        return Sum(new_terms)

    ##------------ Simplify 1 +/- cls(...)^2 ------------##
    # - for this case: what if there is a constant (or variable) common factor?
    identities: List[Tuple[Expr, ExprFn]] = [
        # More common ones first
        (1 + tan(any_) ** 2, lambda x: sec(x) ** 2),
        (1 - sin(any_) ** 2, lambda x: cos(x) ** 2),
        (1 - cos(any_) ** 2, lambda x: sin(x) ** 2),
        (-1 + sec(any_) ** 2, lambda x: tan(x) ** 2),
        (1 + cot(any_) ** 2, lambda x: csc(x) ** 2),
        (-1 + csc(any_) ** 2, lambda x: cot(x) ** 2),
    ]
    for cond, perform in identities:
        result = eq(sum, cond, up_to_factor=True, up_to_sum=True)
        if result["success"]:
            factor = result["factor"]
            inner = result["matches"]
            rest = result["rest"]
            return factor * perform(inner) + rest

    # other_table = [
    #     (r"^sec\((.+)\)\^2$", r"^-tan\((.+)\)\^2$", Const(1)),
    # ]


def _pythagorean_complex_perform(sum: Expr) -> Optional[Expr]:
    if not isinstance(sum, Sum):
        return
    ##------------ More complex pythagorean simplification across terms ------------##
    new_sum = replace_class(
        sum,
        [tan, csc, cot, sec],
        [
            lambda x: sin(x) / cos(x),
            lambda x: 1 / sin(x),
            lambda x: cos(x) / sin(x),
            lambda x: 1 / cos(x),
        ],
    )
    new_sum = rewrite_as_one_fraction(new_sum)
    if sum == new_sum:
        return
    new_sum = kinder_replace(new_sum, _pythagorean_perform)
    if is_simpler(new_sum, sum):
        # assume we're doing this in the ctx of the larger simplification
        return reciprocate_trigs(new_sum)


def rewrite_as_one_fraction(sum: Expr) -> Expr:
    """Rewrites with common denominator"""
    if not isinstance(sum, Sum):
        return sum
    list_of_terms: List[Tuple[Expr, Expr]] = []
    for term in sum.terms:
        if isinstance(term, Prod):
            num, den = term.numerator_denominator
        elif isinstance(term, Power) and isinstance(term.exponent, Rat) and term.exponent.value < 0:
            num, den = Rat(1), Power(term.base, -term.exponent)
        else:
            num, den = term, Rat(1)
        list_of_terms.append((num, den))

    common_den = Rat(1)
    for _, den in list_of_terms:
        ratio = den / common_den
        common_den *= ratio

    new_nums = [num * common_den / den for num, den in list_of_terms]
    return Sum(new_nums) / common_den


def is_simpler(a, b):
    """return if a is simpler than b"""

    def count(e):
        # counts the number of symbols
        return general_count(e, lambda x: isinstance(x, Symbol))

    return count(a) < count(b)


def _reciprocate_trigs(expr: Expr) -> Optional[Expr]:
    if not isinstance(expr, Power):
        return
    # in general during integration it's not useful to apply this simplify; it's only
    # useful for simplifying the expression for the user to see & for subtractions.
    # just having this run in the integration simplify transform makes tests .5s slower (4.5 -> 5)
    b = expr.base
    x = expr.exponent
    if isinstance(x, Rat) and x < 0 and isinstance(b, TrigFunction) and not b.is_inverse:
        new_b = b.reciprocal_class(b.inner)
        return Power(new_b, -x)


def _combine_trigs(expr: Expr) -> Optional[Expr]:
    if not isinstance(expr, Prod):
        return
    # if sin(x)/cos(x) is in the product, combine it to tan(x)
    identities: List[Tuple[Expr, ExprFn]] = [
        # (sin(any_) / cos(any_), lambda x: tan(x)), # if we put this after reciprocate trigs, these won't exist.
        (sin(any_) * sec(any_), lambda x: tan(x)),
        # (cos(any_) / sin(any_), lambda x: cot(x)),
        (cos(any_) * csc(any_), lambda x: cot(x)),
    ]
    for cond, perform in identities:
        result = eq(expr, cond, up_to_factor=True)
        if result["success"]:
            factor = result["factor"]
            inner = result["matches"]
            new = factor * perform(inner)
            return new


def reciprocate_trigs(expr: Expr, **kwargs) -> Expr:
    return kinder_replace_many(
        expr,
        [
            _reciprocate_trigs,
        ],
        **kwargs,
    )


def simplify(expr: Expr) -> Expr:
    """Simplifies an expression.

    This is the general one that does all heuristics & is for aesthetics (& comparisons).
    Use more specific simplification functions in integration please.
    """
    if expr.has(TrigFunction):
        expr = trig_simplify(expr)
    if expr.has(log):
        expr = expand_logs(expr)
    return expr


def trig_simplify(expr):
    # reciprocate and combine trigs is last because sometimes the pythag complex simplification will
    # generate new trigs in the num/denom that can be simplified down.
    expr = kinder_replace_many(expr, [_pythagorean_perform, _pythagorean_complex_perform, _reciprocate_trigs])
    expr = kinder_replace_many(expr, [_combine_trigs])
    return expr
