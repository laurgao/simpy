from typing import Iterable, List, Optional, Tuple, Type, Union

from .expr import (Const, Expr, Power, Prod, Sum, TrigFunction, cos, cot, csc,
                   sec, sin, tan)
from .regex import any_, eq
from .utils import ExprFn


def trig_simplification(sum: Sum) -> Expr:
    if sum.has(TrigFunction):
        ##------------ Simplify 1 +/- cls(...)^2 ------------##
        # - for this case: what if there is a constant (or variable) common factor?
        identities: List[Tuple[Expr, ExprFn]] = [
            (1 + tan(any_) ** 2, lambda x: sec(x) ** 2),
            (1 + cot(any_) ** 2, lambda x: csc(x) ** 2),
            (1 - sin(any_) ** 2, lambda x: cos(x) ** 2),
            (1 - cos(any_) ** 2, lambda x: sin(x) ** 2),
            (1 - tan(any_) ** 2, lambda x: 1 / tan(x) ** 2), # unclear if this is even simpler...
            (1 - cot(any_) ** 2, lambda x: 1 / cot(x) ** 2)
        ]
        for condition, perform in identities:
            is_eq, factor, inner = eq(sum, condition, up_to_factor=True)
            if is_eq:
                new_sum = factor * perform(inner)
                return new_sum.simplify()

        ##------------ Simplify sin^2(...) + cos^2(...) ------------##
        def is_thing_squared(term: Expr, cls: Union[Type[TrigFunction], Iterable[Type[TrigFunction]]]) -> Optional[Tuple[Expr, Expr]]: # returns inner, factor
            def is_(f):
                return isinstance(f, Power) and f.exponent == 2 and isinstance(f.base, cls)
            if is_(term):
                return term.base.inner, Const(1)  

            if not isinstance(term, Prod):
                return None
            for f in term.terms:
                if is_(f):
                    return f.base.inner, term/f
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
            return Sum(new_terms).simplify()

        # other_table = [
        #     (r"^sec\((.+)\)\^2$", r"^-tan\((.+)\)\^2$", Const(1)),
        # ]
                        
    return sum
    