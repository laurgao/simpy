import re
from typing import Callable, Dict, Iterable, Optional, Tuple, Type, Union

from .expr import (Const, Expr, Power, Prod, Sum, Symbol, TrigFunction, cos,
                   cot, csc, sec, sin, tan)


def trig_simplification(sum: Sum) -> Expr:
    if sum.has(TrigFunction):
        # I WANT TO DO IT so that it's more robust.
        # - what if there is a constant (or variable) common factor? (i think for this i'll have to implement a .factor method)

        pythagorean_trig_identities: Dict[str, Callable[[Expr], Expr]] = {
            r"tan\((\w+)\)\^2 \+ 1": lambda x: sec(x) ** 2,
            r"cot\((\w+)\)\^2 \+ 1": lambda x: csc(x) ** 2,
            r"-sin\((\w+)\)\^2 \+ 1": lambda x: cos(x) ** 2,
            r"-cos\((\w+)\)\^2 \+ 1": lambda x: sin(x) ** 2,
            r"-tan\((\w+)\)\^2 \+ 1": lambda x: Const(1) / (tan(x) ** 2),
            r"-cot\((\w+)\)\^2 \+ 1": lambda x: Const(1) / (cot(x) ** 2),
        }

        for pattern, replacement_callable in pythagorean_trig_identities.items():
            match = re.search(pattern, sum.__repr__())
            result = match.group(1) if match else None

            if result and len(sum.terms) == 2:
                other = replacement_callable(Symbol(result)).simplify()
                return other

        # actually i dont need replace factory: we are already recursing on simplify so no more recursing here.
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
    