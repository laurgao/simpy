"""Custom library for checking what shit exists. Replaces searching the repr with regex.

Still WIP. Ideally need to do stuff like "check if somefactor * cos(sth) + somefactor * sin(sth) exists."
"""
from collections import defaultdict
from dataclasses import dataclass, fields
from typing import Any, Callable, Dict, Iterable, List, Literal, Tuple, Type

from .expr import Expr, Power, Prod, Rat, SingleFunc, Sum, Symbol, cast, log
from .utils import ExprFn, OptionalExprFn, random_id


class Any_(Expr):
    def __init__(self, anykey = None, *, is_multiple_terms=False):
        if not anykey:
            anykey = random_id(10)
        self._anykey = anykey
        self._is_multiple_terms = is_multiple_terms

    @property
    def anykey(self) -> str:
        return self._anykey
    
    @property
    def is_multiple_terms(self) -> bool:
        return self._is_multiple_terms

    def __eq__(self, other):
        if isinstance(other, Any_):
            return self.anykey == other.anykey
        # if isinstance(other, Expr):
        #     return True
        # return NotImplemented
        return False
    
    def __repr__(self) -> str:
        return "any_" + self.anykey
    
    # implementing some Expr abstract methods
    def subs(self, subs: Dict[str, "Rat"]):
        raise NotImplementedError(f"Cannot evaluate {self}")
    
    def _evalf(self, subs):
        raise NotImplementedError(f"Cannot evaluate {self}")
    
    def children(self) -> List["Expr"]:
        return []

    def diff(self, var: "Symbol") -> "Expr":
        raise NotImplementedError(
            f"Cannot get the derivative of {self.__class__.__name__}"
        )

    def latex(self) -> str:
        raise NotImplementedError(f"Cannot convert {self.__class__.__name__} to latex")



any_ = Any_()

# smallTODO: make this a namedtuple
EqResult = Dict[Literal["success", "factor", "rest", "anyfind"], Any]

@cast
def eq(expr: Expr, query: Expr, *, up_to_factor = False, up_to_sum=False) -> EqResult:
    if expr.has(Any_):
        if query.has(Any_):
            raise ValueError("Can't both arguments to eq have Any_")
        expr, query = query, expr
    return Eq(expr, query, up_to_factor, up_to_sum=up_to_sum)()

# @dataclass
# class Inverse(Expr):
#     inner: Expr # = 1

#     def __eq__(self, other):
#         return isinstance(other, Inverse) and other.inner == self.inner
    
#     def __repr__(self) -> str:
#         return f"Inverse({self.inner})"

#     # implementing some Expr abstract methods
#     def subs(self, subs: Dict[str, "Rat"]):
#         raise NotImplementedError(f"Cannot evaluate {self}")
    
#     def children(self) -> List["Expr"]:
#         return self.inner

#     def diff(self, var: "Symbol") -> "Expr":
#         raise NotImplementedError(
#             f"Cannot get the derivative of {self.__class__.__name__}"
#         )

#     def latex(self) -> str:
#         raise NotImplementedError(f"Cannot convert {self.__class__.__name__} to latex")


def get_anys(expr: Expr) -> List[Any_]:
    if isinstance(expr, Any_):
        return [expr]

    str_set = set([symbol.anykey for e in expr.children() for symbol in get_anys(e)])
    return [Any_(s) for s in str_set]
            

def all_same(list_: list):
    if len(list_) == 1:
        return True
    for i in range(len(list_)-1):
        if list_[i] != list_[i+1]:
            return False
        
    return True
    

AnyfindInProgress = Dict[str, List[Expr]]
Anyfind = Dict[str, Expr]


def _anyfinds_eq(x: Anyfind, y: Anyfind):
    # assert there aren't contradictions.
    for k in x:
        if k in y and not y[k] == x[k]:
            return False
            
    for k in y:
        if k in x and not y[k] == x[k]:
            return False
            
    return True

_qanyfind = Dict[str, Dict[str, Anyfind]]
def consolidate(query_anyfinds: _qanyfind, expr: Sum) -> EqResult:
    final = defaultdict(list)
    accounted_expr_terms = []
    for query_term_repr, query_term_values in query_anyfinds.items():
        for expr_term_repr, matches in query_term_values.items():
            if all(any(_anyfinds_eq(matches2, matches) for matches2 in query_term_values2.values()) for query_term_values2 in query_anyfinds.values()):
                join_dicts(final, matches)
                accounted_expr_terms.append(expr_term_repr)


    assert all(all_same(v) for v in final.values())
    final = {k: v[0] for k, v in final.items()}
    assert all(any(_anyfinds_eq(x, final) for x in y) for y in query_anyfinds.values())
    
    rest = Sum([t for t in expr.terms if repr(t) not in accounted_expr_terms])
    return {"anyfind": final, "rest": rest, "success": True}


class Eq:
    """Create a new instance every time you run an Eq comparison.
    """
    def __init__(self, expr: Expr, query: Expr, up_to_factor: bool = False, decompose_singles=True, up_to_sum=False, _is_divide=False):
        if expr.has(Any_):
            raise ValueError("Only query can contain 'any' objects.")

        self._expr = expr
        self._query = query
        self._up_to_factor = up_to_factor
        self._decompose_singles = decompose_singles
        self._is_divide = _is_divide
        self._up_to_sum = up_to_sum

        # Mutable:
        self._anyfind: AnyfindInProgress = defaultdict(list)
        self._result: EqResult = defaultdict(str)

        if up_to_factor:
            self._anyfactor = Any_("factor_" + random_id(5))
            new_query = self._anyfactor * query
            self._query = new_query.expand() if new_query.expandable() else new_query

    def __call__(self) -> EqResult:
        """Returns (is_equal, factor, anyfinds) if up_to_factor=True and (is_equal, anyfinds) otherwise.
        """
        falsereturn = {"success": False}
        if self._up_to_sum:
            # Usually, this means that self._expr is a sum and self._query is also a sum.
            # we want to check if the query.terms is a subset of expr.terms
            if not isinstance(self._expr, Sum) or not isinstance(self._query, Sum):
                raise NotImplementedError
            if not len(self._query.terms) <= len(self._expr.terms):
                return falsereturn
            
            # i want to prioritize most exact returns first.
            # actually i want to have most exact return only fuck shit lmao

            # xx = [[self._eq(expr_term, term) for expr_term in self._expr.terms] for term in self._query.terms]
            
            # return the anyfind of each one
            # if there is like sth in each query terms where the anyfind is the same...

            def _is_sum_eq(expr: Sum, query: Sum) -> EqResult:
                query_anyfinds: _qanyfind = defaultdict(dict)
                for query_term in query.terms:
                    for expr_term in expr.terms:
                        output = Eq(expr_term, query_term, decompose_singles=False)()
                        is_eq, anyfinds = output.get("success"), output.get("anyfind")
                        if is_eq and all(any(_anyfinds_eq(x, anyfinds) for x in y) for y in query_anyfinds.values()):
                            query_anyfinds[repr(query_term)][repr(expr_term)] = anyfinds

                    if query_anyfinds[repr(query_term)] == []:
                        return {"success": False}
                
                return consolidate(query_anyfinds, expr)
            
            result = _is_sum_eq(self._expr, self._query)
            if result["success"] is False:
                return falsereturn
            self._result["rest"] = result["rest"]
            self._anyfind = result["anyfind"]

        else:
            ans = self._eq(self._expr, self._query)
            if ans is False:
                return falsereturn

            for k, v in self._anyfind.items():
                if all_same(v):
                    self._anyfind[k] = v[0]
                else:
                    return falsereturn

        ## AT THIS POINT:
        self._anyfind: Anyfind

        if self._up_to_factor:
            self._result["factor"] = self._anyfind[self._anyfactor.anykey]
            del self._anyfind[self._anyfactor.anykey]

        if self._decompose_singles and len(self._anyfind) == 1:
            self._result["anyfind"] = list(self._anyfind.values())[0]
        else:
            self._result["anyfind"] = self._anyfind
        self._result["success"] = True
        return self._result

    def _eq(self, expr: Any, query: Any) -> bool:
        if isinstance(expr, list):
            if not (isinstance(query, list) and len(expr) == len(query)):
                return False
            return all([self._eq(e, q) for e, q in zip(expr, query)])
        if expr == query:
            return True
        if not isinstance(expr, Expr) or not isinstance(query, Expr):
            return False
        if isinstance(query, Any_):
            self._anyfind[query.anykey].append(expr)
            return True
        if not query.has(Any_):
            return False
        
        if not self._is_divide:
            # You don't get to divide if we already is --- prevents inf recursion.
            one, quotient_matches = anydivide(query,expr)
            if isinstance(one, Any_):
                self._anyfind[one.anykey].append(Rat(1))
                join_dicts2(self._anyfind, quotient_matches)
                return True

            if len(one.symbols()) == 0:
                anys = get_anys(one)

                # if one contains a factor of any.
                if len(anys) == 1:
                    anyvalue = one / anys[0]
                    if get_anys(anyvalue) == []:
                        join_dicts2(self._anyfind, quotient_matches)
                        self._anyfind[anys[0].anykey].append(anyvalue)
                        return True

                # for a in anys:
                #     self._anyfind[a.anykey].append(Inverse(one))
                # return True
        
        if not expr.__class__ == query.__class__:
            return False
        return all([self._eq(getattr(expr, field.name), getattr(query, field.name)) for field in fields(expr)])


def anydivide(num: Expr, denom: Expr) -> Tuple[Expr, AnyfindInProgress]:
    """Returns: (quotient, anyfinds)
    """
    def _make_factors_list(expr: Expr) -> List[Expr]:
        if not isinstance(expr, Prod):
            return [expr]
        # I want to move all anyfactors to the end -- try to match everything else first.
        terms = []
        anys = []
        anyfactors = []
        for t in expr.terms:
            if isinstance(t, Any_) and "factor" in t.anykey:
                anyfactors.append(t)
            elif t.has(Any_):
                anys.append(t)
            else:
                terms.append(t)
        if len([t for t in expr.terms if isinstance(t, Any_)]) > 1:
            raise NotImplementedError(f"{expr} is ambiguous")
        if len(anys) > 0:
            terms.extend(anys)
        if len(anyfactors) > 0:
            terms.extend(anyfactors)
        return terms

    numfactors = _make_factors_list(num)
    denfactors = _make_factors_list(denom)
    anyfinds = defaultdict(list)
    for i in range(len(numfactors)):
        f = numfactors[i]
        for j in range(len(denfactors)):
            df = denfactors[j]
            if df is None or f is None:
                continue
            output = Eq(df, f, decompose_singles=False, _is_divide=True)()
            if output["success"]:
                join_dicts(anyfinds, output["anyfind"])
                denfactors[j] = None
                numfactors[i] = None

    new_df = [df for df in denfactors if df is not None]
    new_nf = [nf for nf in numfactors if nf is not None]
    return Prod(new_nf) / Prod(new_df), anyfinds

def join_dicts(d1: AnyfindInProgress, d2: Anyfind) -> None:
    """Mutates d1 to add the stuff in d2"""
    for k in d2.keys():
        d1[k].append(d2[k])

def join_dicts2(d1: AnyfindInProgress, d2: AnyfindInProgress) -> None:
    """Mutates d1 to add the stuff in d2"""
    for k in d2.keys():
        if k in d1:
            d1[k].extend(d2[k])
        else:
            d1[k] = d2[k]

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
        if isinstance(expr, log):
            return log(inner=_replace(expr.inner), base=_replace(expr.base))
        # i love recursion
        if isinstance(expr, SingleFunc):
            return expr.__class__(_replace(expr.inner))
        
        if len(expr.children()) == 0: # Number, Symbol
            return expr
        
        raise NotImplementedError(f"replace not implemented for {expr.__class__.__name__}")

    return _replace

def kinder_replace(expr: Expr, perform: OptionalExprFn, **kwargs) -> Expr:
    return kinder_replace_many(expr, [perform], **kwargs)

def kinder_replace_many(expr: Expr, performs: Iterable[OptionalExprFn], verbose=False, _d=False) -> Expr:
    """
    kinder, bc some queries dont have a clean condition.
    ex: checking multiple any_ matches.

    assumes mutual exclusivity.
    it' snot possible to do the thing where we process the output of one transform thru other transforms
    bc sometimes the output of one trnasfomr effects the overall expr structure which makes another
    transform possible.
    """
    is_hit = {"hi": False}

    def _replace(e: Expr) -> Expr:
        for p in performs:
            new = p(e)
            if new:
                is_hit["hi"] = True
                return new

        # find all instances of old in expr and replace with new
        if isinstance(e, Sum):
            return Sum([_replace(t) for t in e.terms])
        if isinstance(e, Prod):
            return Prod([_replace(t) for t in e.terms])
        if isinstance(e, Power):
            return Power(base=_replace(e.base), exponent=_replace(e.exponent))
        if isinstance(e, log):
            return log(inner=_replace(e.inner), base=_replace(e.base))
        # i love recursion
        if isinstance(e, SingleFunc):
            return e.__class__(_replace(e.inner))
        
        if len(e.children()) == 0: # Number, Symbol
            return e
        
        raise NotImplementedError(f"replace not implemented for {e.__class__.__name__}")

    ans = _replace(expr)
    return (ans, is_hit["hi"]) if verbose else ans



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
    if isinstance(expr, log):
        return log(
            inner=replace_class(expr.inner, cls, newfunc),
            base=replace_class(expr.base, cls, newfunc)
        )
    if isinstance(expr, SingleFunc):
        new_inner = replace_class(expr.inner, cls, newfunc)
        for i, cl in enumerate(cls):
            if isinstance(expr, cl):
                return newfunc[i](new_inner)
        return expr.__class__(new_inner)

    if len(expr.children()) == 0: # Number, Symbol
        return expr

    raise NotImplementedError(
        f"replace_class not implemented for {expr.__class__.__name__}"
    )