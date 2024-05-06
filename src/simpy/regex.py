"""Custom library for checking what shit exists. Replaces searching the repr with regex.

Still WIP. Ideally need to do stuff like "check if somefactor * cos(sth) + somefactor * sin(sth) exists."
"""
from typing import Callable, Iterable, List, Type

from .expr import Const, Expr, Power, Prod, SingleFunc, Sum, Symbol, cast
from .utils import ExprFn


@cast
def count(expr: Expr, query: Expr) -> int:
    """Counts how many times `query` appears in `expr`. Exact matches only.
    """
    if isinstance(expr, query.__class__) and expr == query:
        return 1
    return sum(count(e, query) for e in expr.children())


def contains_cls(expr: Expr, cls: Type[Expr]) -> bool:
    if isinstance(expr, cls):
        return True

    return any([contains_cls(e, cls) for e in expr.children()])


def general_count(expr: Expr, condition: Callable[[Expr], bool]) -> int:
    """the `count` function above, except you can specify a condition rather than 
    only allowing exact matches.
    """
    if condition(expr):
        return 1
    return sum(general_count(e, condition) for e in expr.children())


def replace_factory(condition, perform) -> ExprFn:
    return replace_factory_list([condition], [perform])


def replace_factory_list(conditions: Iterable[Callable[[Expr], bool]], performs: Iterable[ExprFn]) -> ExprFn:
    """
    list of iterable conditions should be ... mutually exclusive or sth

    every time the condition returns True, you replace that expr with the output of `perform`
    hard to explain in English. read the code.
    """
    # Ok honestly the factory might not be the best structure, might be better to have a single unested function that returns the replaced expr directly
    # or make a class bc they're cleaner than factories. haven't given this a ton of thought but we don't need the most perfect choice <3
    def _replace(expr: Expr) -> Expr:
        for condition, perform in zip(conditions, performs):
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
        
        if len(expr.children()) == 0: # Number, Symbol
            return expr
        
        raise NotImplementedError(f"replace not implemented for {expr.__class__.__name__}")

    return _replace


def replace(expr: Expr, old: Expr, new: Expr) -> Expr:
    """replaces every instance of `old` (that appears in `expr`) with `new`. 
    """
    # Special case of the general replace_factory that gets used often.
    condition = lambda e: isinstance(e, old.__class__) and e == old
    perform = lambda e: new
    return replace_factory(condition, perform)(expr)


# cls here has to be a subclass of singlefunc
def replace_class(expr: Expr, cls: List[Type[SingleFunc]], newfunc: List[Callable[[Expr], Expr]]) -> Expr:
    # legacy // can be rewritten with replace_factory and put directly into transform RewriteTrig
    # bc it's not used anywhere else.
    # however it doesn't matter super much rn that everything is structured in :sparkles: prestine condition :sparkles:
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