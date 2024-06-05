from typing import Any, Iterable, List, Optional, Tuple, Type, Union

from .expr import (
    Abs,
    Expr,
    Num,
    Power,
    Prod,
    Rat,
    SingleFunc,
    Sum,
    TrigFunction,
    TrigFunctionNotInverse,
    cos,
    cot,
    csc,
    log,
    remove_const_factor,
    sec,
    sin,
    tan,
)
from .regex import any_, eq, general_contains, kinder_replace, kinder_replace_many, replace_class, replace_factory
from .utils import ExprFn, count_symbols


def expand_logs(expr: Expr, **kwargs) -> Expr:
    ans = kinder_replace(expr, _log_perform, **kwargs)
    return ans.expand() if ans.expandable() else ans


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
    return kinder_replace(expr, _pythagorean_perform, **kwargs)


def _pythagorean_perform(sum: Expr) -> Optional[Expr]:
    if not isinstance(sum, Sum):
        return
    if not sum.has(TrigFunctionNotInverse):  # It is faster to check every time
        return False

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
    """Assumes sum.has(TrigFunction is true.)"""
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

    new_sum = rewrite_as_one_fraction(new_sum, include_const_denoms=False)
    if new_sum is None:
        return
    new_sum, success = kinder_replace(new_sum, _pythagorean_perform, verbose=True)
    if success:
        # assume we're doing this in the ctx of the larger simplification
        return reciprocate_trigs(new_sum)


def rewrite_as_one_fraction(sum: Expr, include_const_denoms=True) -> Optional[Expr]:
    """Rewrites with common denominator

    Returns None if left unchanged.
    """
    if not isinstance(sum, Sum):
        return sum
    list_of_terms: List[Tuple[Expr, Expr]] = []

    has_den = False
    common_den_terms = []

    def add_common_den(factors):
        # remove dupes
        # this ensures that if two dens have the same factor, do not count it twice.
        for f in factors:
            if f not in common_den_terms:
                common_den_terms.append(f)

    for term in sum.terms:
        if isinstance(term, Prod):
            num, den = term.numerator_denominator
            if den != 1 and (include_const_denoms or len(den.symbols()) > 0):
                has_den = True
        elif isinstance(term, Power) and isinstance(term.exponent, Rat) and term.exponent.value < 0:
            num, den = Rat(1), Power(term.base, -term.exponent)
            has_den = True
        else:
            num, den = term, Rat(1)
        list_of_terms.append((num, den))

        if isinstance(den, Prod):
            add_common_den(den.terms)
        else:
            add_common_den([den])

    if has_den == False:
        return

    common_den = Prod(common_den_terms)
    new_nums = [num * common_den / den for num, den in list_of_terms]
    return Sum(new_nums) / common_den


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
    if expr.expandable():
        expr2 = expr.expand()
    else:
        expr2 = expr

    # if trig simplify has hit then it's always good :thumbsup:
    is_trig_hit = None
    if expr2.has(TrigFunctionNotInverse):
        expr2, is_trig_hit = trig_simplify(expr2)
    if expr2.has(log):
        expr2 = expand_logs(expr2)

    if is_trig_hit:
        return expr2

    if is_simpler(expr2, expr):
        return expr2
    return expr


def trig_simplify(expr):
    # reciprocate and combine trigs is last because sometimes the pythag complex simplification will
    # generate new trigs in the num/denom that can be simplified down.
    expr, is_hit_1 = kinder_replace_many(
        expr,
        [_pythagorean_perform, _pythagorean_complex_perform, _reciprocate_trigs],
        overarching_cond=lambda x: x.has(TrigFunctionNotInverse),
        verbose=True,
    )
    expr, is_hit_2 = kinder_replace_many(
        expr,
        [_combine_trigs, product_to_sum, sectan],
        overarching_cond=lambda x: x.has(TrigFunctionNotInverse),
        verbose=True,
    )
    return expr, is_hit_1 or is_hit_2


def is_cls_squared(expr, cls) -> bool:
    return (
        isinstance(expr, Power)
        and isinstance(expr.base, cls)
        and isinstance(expr.exponent, Rat)
        and expr.exponent % 2 == 0
    )


def remove_num_factor(expr):
    if not isinstance(expr, Prod):
        return expr

    x = [t for t in expr.terms if not isinstance(t, Num)]
    return Prod(x, skip_checks=True) if len(x) > 1 else x[0] if len(x) == 1 else Rat(1)


def is_class_squared(expr: Expr, cls: Type[SingleFunc]) -> bool:
    expr = remove_num_factor(expr)
    return is_cls_squared(expr, cls)


def sectan(sum: Expr) -> Optional[Expr]:
    """If a sum has some sec^n(x) and tan^n(x) where n is even, and rewriting the secs as tans with the
    pythagorean identity sec^2(x) = 1 + tan^2(x) allows cancellation of terms, then do the simplification.

    Assumes sum.has(TrigFunctionNotInverse) is already satisfied
    """
    if not isinstance(sum, Sum):
        return

    secs = []
    tans = []
    others = []
    for t in sum.terms:
        if is_class_squared(t, sec):
            secs.append(t)
        elif is_class_squared(t, tan):
            tans.append(t)
        else:
            others.append(t)

    if not secs or not tans:
        return

    # convert secs to tans
    condition = lambda x: is_cls_squared(x, sec)

    def perform(e: Power):
        n = e.exponent
        return (1 + tan(e.base.inner) ** 2) ** (n // 2)

    for s in secs:
        new = replace_factory(condition, perform)(s)
        new = new.expand() if new.expandable() else s
        assert isinstance(new, Sum)
        tans.extend(new.terms)

    new_sum = Sum(tans + others)
    if not isinstance(new_sum, Sum):
        # This must mean that we simplified the sum into one term
        return new_sum

    if len(new_sum.terms) < sum.terms:
        return new_sum


def product_to_sum(expr: Expr, *, always_simplify=False, const: Expr = None) -> Optional[Expr]:
    from .transforms import ProductToSum

    """Does applying product-to-sum on every term of a sum ... create a cancellation?

    Assumes that expr.has(TrigFunctionNotInverse) == True
    """

    if not isinstance(expr, Sum):
        return

    def perf(expr: Expr) -> bool:
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
                    return ProductToSum._perform_on_terms(*new_expr.terms, const=const)
                if isinstance(t1, (sin, cos)) and is_valid_power(t2):
                    return product_to_sum((perf(t2) * t1).expand(), always_simplify=True, const=const)
                if isinstance(t2, (sin, cos)) and is_valid_power(t1):
                    return product_to_sum((perf(t1) * t2).expand(), always_simplify=True, const=const)
                if is_valid_power(t2) and is_valid_power(t1):
                    intermediate = perf(t1) * perf(t2)
                    return product_to_sum(intermediate.expand(), const=const, always_simplify=True)

        if is_valid_power(new_expr):
            if new_expr.exponent == 2:
                return ProductToSum._perform_on_terms(new_expr.base, new_expr.base, const=const)
            elif new_expr.exponent % 2 == 0:
                breakpoint()
                intermediate = ProductToSum._perform_on_terms(new_expr.base, new_expr.base) ** (new_expr.exponent / 2)
                return product_to_sum(intermediate.expand(), const=const, always_simplify=True)
            else:
                intermediate = new_expr.base * ProductToSum._perform_on_terms(new_expr.base, new_expr.base) ** (
                    (new_expr.exponent - 1) / 2
                )
                return product_to_sum(intermediate.expand(), const=const, always_simplify=True)

        return False

    satisfies = [perf(t) for t in expr.terms]
    if all(s is False for s in satisfies):
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


def is_simpler(e1, e2) -> bool:
    """returns whether e1 is simpler than e2"""
    c1 = count_symbols(e1)
    c2 = count_symbols(e2)
    return c1 < c2
    # if c1 < c2:
    #     return True
    # if c1 == c2:
    #     return len(repr(e1)) < len(repr(e2))
    # return False
