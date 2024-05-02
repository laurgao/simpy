"""Custom library for checking what shit exists. Replaces searching the repr with regex.

Still WIP. Ideally need to do stuff like "check if somefactor * cos(sth) + somefactor * sin(sth) exists."
"""
from typing import Callable, List

from .expr import (Const, Expr, Number, Power, Prod, SingleFunc, Sum, Symbol,
                   cast)
from .utils import ExprFn


@cast
def count(expr: Expr, query: Expr) -> int:
    if isinstance(expr, query.__class__) and expr == query:
        return 1
    return sum(count(e, query) for e in expr.children())


def replace_factory(condition: Callable[[Expr], bool], perform: ExprFn) -> ExprFn:
    # Ok honestly the factory might not be the best structure, might be better to have a single unested function that returns the replaced expr directly
    # or make a class bc they're cleaner than factories. haven't given this a ton of thought but we don't need the most perfect choice <3
    def _replace(expr: Expr) -> Expr:
        if condition(expr):
            return perform(expr)

        # find all instances of old in expr and replace with new
        if isinstance(expr, Sum):
            return Sum([_replace(e) for e in expr.terms])
        if isinstance(expr, Prod):
            return Prod([_replace(e) for e in expr.terms])
        if isinstance(expr, Power):
            return Power(base=_replace(expr.base), exponent=_replace(expr.exponent))
        # i love recursion
        if isinstance(expr, SingleFunc):
            return expr.__class__(_replace(expr.inner))

        if isinstance(expr, Const):
            return expr
        if isinstance(expr, Symbol):
            return expr
        
        raise NotImplementedError(f"replace not implemented for {expr.__class__.__name__}")

    return _replace


def replace(expr: Expr, old: Expr, new: Expr) -> Expr:
    if isinstance(expr, old.__class__) and expr == old:
        return new

    # find all instances of old in expr and replace with new
    if isinstance(expr, Sum):
        return Sum([replace(e, old, new) for e in expr.terms])
    if isinstance(expr, Prod):
        return Prod([replace(e, old, new) for e in expr.terms])
    if isinstance(expr, Power):
        return Power(
            base=replace(expr.base, old, new), exponent=replace(expr.exponent, old, new)
        )
    # i love recursion
    if isinstance(expr, SingleFunc):
        return expr.__class__(replace(expr.inner, old, new))

    if isinstance(expr, Number):
        return expr
    if isinstance(expr, Symbol):
        return expr

    raise NotImplementedError(f"replace not implemented for {expr.__class__.__name__}")


# cls here has to be a subclass of singlefunc
def replace_class(expr: Expr, cls: list, newfunc: List[Callable[[Expr], Expr]]) -> Expr:
    assert all(issubclass(cl, SingleFunc) for cl in cls), "cls must subclass SingleFunc"
    if isinstance(expr, Sum):
        return Sum([replace_class(e, cls, newfunc) for e in expr.terms])
    if isinstance(expr, Prod):
        return Prod([replace_class(e, cls, newfunc) for e in expr.terms])
    if isinstance(expr, Power):
        return Power(
            base=replace_class(expr.base, cls, newfunc),
            exponent=replace_class(expr.exponent, cls, newfunc),
        )
    if isinstance(expr, SingleFunc):
        new_inner = replace_class(expr.inner, cls, newfunc)
        for i, cl in enumerate(cls):
            if isinstance(expr, cl):
                return newfunc[i](new_inner)
        return expr.__class__(new_inner)

    if isinstance(expr, Const):
        return expr
    if isinstance(expr, Symbol):
        return expr

    raise NotImplementedError(
        f"replace_class not implemented for {expr.__class__.__name__}"
    )