# TODO: tasks for the future:
# - method to convert from our expression to sympy for testing

import itertools
import random
import re
import string
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from fractions import Fraction
from functools import reduce
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np


def _cast(x):
    if type(x) == int or isinstance(x, Fraction):
        return Const(x)
    if type(x) == float and int(x) == x:  # silly patch
        return Const(int(x))
    elif isinstance(x, Expr):
        return x
    elif isinstance(x, dict):
        return {k: _cast(v) for k, v in x.items()}
    elif isinstance(x, tuple):
        return tuple(_cast(v) for v in x)
    else:
        raise NotImplementedError(f"Cannot cast {x} to Expr")


def cast(func):
    def wrapper(*args) -> "Expr":
        return func(*[_cast(a) for a in args])

    return wrapper


class Expr(ABC):
    def __post_init__(self):
        # if any field is an Expr, cast it
        # note: does not cast List[Expr]
        for field in fields(self):
            if field.type is Expr:
                setattr(self, field.name, _cast(getattr(self, field.name)))

    # should be overwritten in subclasses
    def simplify(self):
        return self

    @cast
    def __add__(self, other):
        return Sum([self, other])

    @cast
    def __radd__(self, other):
        return Sum([other, self])

    @cast
    def __sub__(self, other):
        return self + (-1 * other)

    @cast
    def __rsub__(self, other):
        return other + (-1 * self)

    @cast
    def __mul__(self, other):
        return Prod([self, other])

    @cast
    def __rmul__(self, other):
        return Prod([other, self])

    @cast
    def __pow__(self, other):
        return Power(self, other)

    @cast
    def __rpow__(self, other):
        return Power(other, self)

    @cast
    def __div__(self, other):
        return Prod([self, Power(other, -1)])

    @cast
    def __truediv__(self, other):
        return Prod([self, Power(other, -1)])

    @cast
    def __rdiv__(self, other):
        return Prod([other, Power(self, -1)])

    @cast
    def __rtruediv__(self, other):
        return Prod([other, Power(self, -1)])

    def __neg__(self):
        return -1 * self

    @cast
    def __eq__(self, other):
        return self.__repr__() == other.__repr__()

    @cast
    def __ne__(self, other):
        return not (self == other)

    # should be overloaded if necessary
    def expandable(self) -> bool:
        return False

    # overload if necessary
    def expand(self) -> "Expr":
        raise NotImplementedError(f"Cannot expand {self}")

    @cast
    @abstractmethod
    def evalf(self, subs: Dict[str, "Const"]):
        raise NotImplementedError(f"Cannot evaluate {self}")

    @abstractmethod
    def children(self) -> List["Expr"]:
        raise NotImplementedError(f"Cannot get children of {self.__class__.__name__}")

    def contains(self: "Expr", var: "Symbol"):
        is_var = isinstance(self, Symbol) and self.name == var.name
        return is_var or any(e.contains(var) for e in self.children())

    # should be overloaded
    def simplifable(self) -> bool:
        return False

    # @abstractmethod
    def diff(self, var: "Symbol"):
        raise NotImplementedError(
            f"Cannot get the derivative of {self.__class__.__name__}"
        )

    def symbols(self) -> List["Symbol"]:
        # I hate this syntax
        return [
            symbol for e in self.children() for symbol in e.symbols()
        ]  # do smtn to prevent duplicates (set doesnt work bc symbol is unhashable)


class Associative:
    def flatten(self):
        new_terms = []
        for t in self.terms:
            new_terms += t.flatten().terms if isinstance(t, self.__class__) else [t]
        return self.__class__(new_terms)

    def children(self) -> List["Expr"]:
        return self.terms


@dataclass
class Const(Expr):
    value: Fraction

    def __post_init__(self):
        assert (
            isinstance(self.value, Fraction) or type(self.value) == int
        ), f"got value={self.value} not allowed Const"
        self.value = Fraction(self.value)

    def __repr__(self):
        return str(self.value)

    @cast
    def __eq__(self, other):
        return isinstance(other, Const) and self.value == other.value

    @cast
    def __ge__(self, other):
        if not isinstance(other, Const):
            return NotImplemented
        return self.value > other.value

    @cast
    def __lt__(self, other):
        if not isinstance(other, Const):
            return NotImplemented
        return self.value < other.value

    @cast
    def evalf(self, subs: Dict[str, "Const"]):
        return self

    def diff(self, var):
        return Const(0)

    def children(self) -> List["Expr"]:
        return []


@dataclass
class Symbol(Expr):
    name: str

    def __repr__(self):
        return self.name

    @cast
    def evalf(self, subs: Dict[str, "Const"]):
        return subs.get(self.name, self)

    def diff(self, var):
        return Const(1) if self == var else Const(0)

    def __eq__(self, other):
        return isinstance(other, Symbol) and self.name == other.name

    def children(self) -> List["Expr"]:
        return []

    def symbols(self) -> List["Expr"]:
        return [self]


@dataclass
class Sum(Associative, Expr):
    terms: List[Expr]

    def simplifable(self) -> bool:
        pass

    def simplify(self):
        # TODO: this currently would not combine terms like (2+x) and (x+2)

        # simplify subexprs and flatten sub-sums
        s = Sum([t.simplify() for t in self.terms]).flatten()

        # accumulate all constants
        const = sum(t.value for t in s.terms if isinstance(t, Const))

        # return immediately if there are no non constant items
        non_constant_terms = [t for t in s.terms if not isinstance(t, Const)]
        if len(non_constant_terms) == 0:
            return Const(const)

        # otherwise, bring the constant to the front (if != 1)
        s = Sum(([] if const == 0 else [Const(const)]) + non_constant_terms)

        # accumulate all like terms
        new_terms = []
        for i, term in enumerate(s.terms):
            if term is None:
                continue

            new_coeff, non_const_factors1 = _deconstruct_prod(term)

            # check if any later terms are the same
            for j in range(i + 1, len(s.terms)):
                if s.terms[j] is None:
                    continue

                term2 = s.terms[j]
                coeff2, non_const_factors2 = _deconstruct_prod(term2)

                if non_const_factors1 == non_const_factors2:
                    new_coeff += coeff2
                    s.terms[j] = None

            new_terms.append(Prod([new_coeff] + non_const_factors1).simplify())

        # if Const(1) in new_terms and Prod([-1, Power(TrigFunction(anysymbol, "sin"), 2)]) in new_terms:
        # get rid of 1-term sums
        if len(new_terms) == 1:
            return new_terms[0]

        new_sum = Sum(new_terms)

        pattern = "\(1 \+ \(-1 \* sin\(\w+\)\^2\)\)"
        if re.search(pattern, new_sum.__repr__()) and len(new_sum.terms) == 2:
            return Power(Cos(new_sum.symbols()[0]), 2)
            # ok im  gna be dumb and j assume the sum is only this

            # for term in new_sum.terms:
            #     if isinstance(term, Const) and term.
            #     p =
            #     if re.search(term.__repr__())

        pattern = "1 \+ tan\(\w+\)\^2"
        if re.search(pattern, new_sum.__repr__()) and len(new_sum.terms) == 2:
            return Power(Sec(new_sum.symbols()[0]), 2)

        return new_sum

    @cast
    def evalf(self, subs: Dict[str, "Const"]):
        return Sum([t.evalf(subs) for t in self.terms]).simplify()

    def diff(self, var):
        return Sum([diff(e, var) for e in self.terms])

    def __repr__(self):
        return "(" + " + ".join(map(repr, self.terms)) + ")"


def _deconstruct_prod(expr: Expr) -> Tuple[Const, List[Expr]]:
    # 3*x^2*y -> (3, [x^2, y])
    # turns smtn into a constant and a list of other terms
    # assume expr is simplified
    if isinstance(expr, Prod):
        # simplifying the product puts the constants at the front
        non_const_factors = (
            expr.terms[1:] if isinstance(expr.terms[0], Const) else expr.terms
        )
        coeff = expr.terms[0] if isinstance(expr.terms[0], Const) else Const(1)
    else:
        non_const_factors = [expr]
        coeff = Const(1)
    return (coeff, non_const_factors)


def _deconstruct_power(expr: Expr) -> Tuple[Expr, Const]:
    # x^3 -> (x, 3). x -> (x, 1). 3 -> (3, 1)
    if isinstance(expr, Power):
        return (expr.base, expr.exponent)
    else:
        return (expr, Const(1))


@dataclass
class Prod(Associative, Expr):
    terms: List[Expr]

    def __repr__(self):
        return "(" + " * ".join(map(repr, self.terms)) + ")"

    @cast
    def simplify(self):
        # simplify subexprs and flatten sub-products
        new = Prod([t.simplify() for t in self.terms]).flatten()

        # accumulate all like terms
        terms = []
        for i, term in enumerate(new.terms):
            if term is None:
                continue

            base, expo = _deconstruct_power(term)

            # other terms with same base
            for j in range(i + 1, len(new.terms)):
                if new.terms[j] is None:
                    continue
                other = new.terms[j]
                base2, expo2 = _deconstruct_power(other)
                if base2 == base:  # TODO: real expr equality
                    expo += expo2
                    new.terms[j] = None

            terms.append(Power(base, expo).simplify())

        new.terms = terms

        # Check for zero
        if any(t == 0 for t in new.terms):
            return Const(0)

        # accumulate constants to the front
        const = reduce(
            lambda x, y: x * y, [t.value for t in new.terms if isinstance(t, Const)], 1
        )

        # return immediately if there are no non constant items
        non_constant_terms = [t for t in new.terms if not isinstance(t, Const)]
        if len(non_constant_terms) == 0:
            return Const(const)

        # otherwise, bring the constant to the front (if != 1)
        new.terms = ([] if const == 1 else [Const(const)]) + non_constant_terms

        return new.terms[0] if len(new.terms) == 1 else new

    @cast
    def expandable(self) -> bool:
        # a product is expandable if it contains any sums
        return any(isinstance(t, Sum) for t in self.terms) or any(
            t.expandable() for t in self.terms
        )

    def expand(self):
        # expand sub-expressions
        self = self.flatten()
        self = Prod([t.expand() if t.expandable() else t for t in self.terms])

        # expand sums that are left
        sums = [t for t in self.terms if isinstance(t, Sum)]
        other = [t for t in self.terms if not isinstance(t, Sum)]

        if not sums:
            return self

        # for every combination of terms in the sums, multiply them and add
        # (using itertools)
        expanded = []
        for terms in itertools.product(*[s.terms for s in sums]):
            expanded.append(Prod(other + list(terms)).simplify())

        return Sum(expanded).simplify()

    @cast
    def evalf(self, subs: Dict[str, "Const"]):
        return Prod([t.evalf(subs) for t in self.terms]).simplify()

    def diff(self, var):
        return Sum(
            [
                Prod([diff(e, var)] + [t for t in self.terms if t is not e])
                for e in self.terms
            ]
        )


@dataclass
class Power(Expr):
    base: Expr
    exponent: Expr

    def __repr__(self):
        return f"{self.base}^{self.exponent}"

    @cast
    def simplify(self):
        x = self.exponent.simplify()
        b = self.base.simplify()
        if x == 0:
            return Const(1)
        elif x == 1:
            return b
        elif isinstance(b, Const) and isinstance(x, Const):
            return Const(b.value**x.value)
        elif isinstance(b, Power):
            return Power(b.base, x * b.exponent).simplify()
        elif isinstance(b, Prod):
            # when you construct this new power entity you have to simplify it.
            # because what if the term raised to this exponent can be simplified?
            # ex: if you have (ab)^n where a = c^m
            return Prod([Power(term, x).simplify() for term in b.terms])
        else:
            return Power(self.base.simplify(), x)

    def expandable(self) -> bool:
        return (
            isinstance(self.exponent, Const)
            and self.exponent.value.denominator == 1
            and self.exponent.value >= 1
            and isinstance(self.base, Sum)
        )

    def expand(self) -> Expr:
        assert self.expandable(), f"Cannot expand {self}"
        return Prod([self.base] * self.exponent.value.numerator).expand()

    @cast
    def evalf(self, subs: Dict[str, "Const"]):
        return Power(self.base.evalf(subs), self.exponent.evalf(subs)).simplify()

    def children(self) -> List["Expr"]:
        return [self.base, self.exponent]


@dataclass
class SingleFunc(ABC):
    inner: Expr

    def children(self) -> List["Expr"]:
        return [self.inner]

    def simplify(self):
        inner = self.inner.simplify()
        return self.__class__(inner)


@dataclass
class Log(SingleFunc, Expr):
    inner: Expr

    def __repr__(self):
        return f"log({self.inner})"

    @cast
    def evalf(self, subs: Dict[str, "Const"]):
        inner = self.inner.evalf(subs)
        # TODO: Support floats in .evalf
        # return Const(math.log(inner.value)) if isinstance(inner, Const) else Log(inner)
        return Log(inner)

    def simplify(self):
        inner = self.inner.simplify()
        if inner == 1:
            return Const(0)

        return Log(inner)

    def diff(self, var):
        return self.inner.diff(var) / self.inner


@dataclass
class TrigFunction(SingleFunc, Expr, ABC):
    inner: Expr
    function: Literal["sin", "cos", "tan", "sec", "csc", "cot"]
    is_inverse: bool = False

    def __repr__(self):
        return f"{self.function}{'^-1' if self.is_inverse else ''}({self.inner})"

    # def simplify(self):
    #     inner = self.inner.simplify()
    #     return self.__class__(inner, self.function)


@dataclass
class Sin(TrigFunction):
    def __init__(self, inner):
        super().__init__(inner, function="sin")

    def __repr__(self):
        return super().__repr__()


@dataclass
class Cos(TrigFunction):
    def __init__(self, inner):
        super().__init__(inner, function="cos")

    def __repr__(self):
        return super().__repr__()


@dataclass
class Tan(TrigFunction):
    def __init__(self, inner):
        super().__init__(inner, function="tan")

    def __repr__(self):
        return super().__repr__()


@dataclass
class Csc(TrigFunction):
    def __init__(self, inner):
        super().__init__(inner, function="csc")

    def __repr__(self):
        return super().__repr__()


@dataclass
class Sec(TrigFunction):
    def __init__(self, inner):
        super().__init__(inner, function="sec")

    def __repr__(self):
        return super().__repr__()


@dataclass
class Cot(TrigFunction):
    def __init__(self, inner):
        super().__init__(inner, function="cot")

    def __repr__(self):
        return super().__repr__()


def symbols(symbols: str):
    return [Symbol(name=s) for s in symbols.split(" ")]


@cast
def diff(expr: Expr, var: Symbol) -> Expr:
    if hasattr(expr, "diff"):
        return expr.diff(var)
    else:
        raise NotImplementedError(f"Differentiation of {expr} not implemented")


@cast
def integrate_bounds(expr: Expr, bounds: Tuple[Symbol, Const, Const]) -> Const:
    x, a, b = bounds
    I = integrate(expr, bounds[0]).simplify()
    return (I.evalf({x.name: b}) - I.evalf({x.name: a})).simplify()


@cast
def integrate(expr: Expr, bounds: Union[Symbol, Tuple[Symbol, Const, Const]]) -> Expr:
    # - table: power rule, 1/x -> lnx, trig standard integrals
    # - safe transformations: constant out, sum, polynomial division,
    # - heuristic transformations
    if type(bounds) == tuple:
        return integrate_bounds(expr, bounds)
    else:
        var = bounds

    expr = expr.simplify()

    # start with polynomials
    if isinstance(expr, Sum):
        return Sum([integrate(e, var) for e in expr.terms])
    elif isinstance(expr, Prod):
        # if there is a constant, pull it out
        if isinstance(expr.terms[0], Const):
            return expr.terms[0] * integrate(Prod(expr.terms[1:]), var)

        # or if there is a symbol that's not the variable, pull it out
        for i, term in enumerate(expr.terms):
            if isinstance(term, Symbol) and term != var:
                return term * integrate(Prod(expr.terms[:i] + expr.terms[i + 1 :]), var)

            # or if it's a power of a symbol that's not the variable, pull it out
            if (
                isinstance(term, Power)
                and isinstance(term.base, Symbol)
                and term.base != var
            ):
                return term * integrate(Prod(expr.terms[:i] + expr.terms[i + 1 :]), var)

        # # if there are sub-sums, integrate the expansion
        if expr.expandable():
            return integrate(expr.expand(), var)

        if divisible(expr, var):
            divided = polynomial_division(expr, var)
            return integrate(divided, var)

        return heuristics(expr, var)
        raise NotImplementedError(f"Cannot integrate {expr} with respect to {var}")
    elif isinstance(expr, Power):
        if expr.base == var and isinstance(expr.exponent, Const):
            n = expr.exponent
            return (1 / (n + 1)) * Power(var, n + 1) if n != -1 else Log(expr.base)
        elif isinstance(expr.base, Symbol) and expr.base != var:
            return expr * var
        elif expr.expandable():
            return integrate(expr.expand(), var)
        else:
            return heuristics(expr, var)
            raise NotImplementedError(f"Cannot integrate {expr} with respect to {var}")
    elif isinstance(expr, Symbol):
        return Fraction(1 / 2) * Power(var, 2) if expr == var else expr * var
    elif isinstance(expr, Const):
        return expr * var
    else:
        return heuristics(expr, var)
        raise NotImplementedError(f"Cannot integrate {expr} with respect to {var}")


def divisible(expr: Prod, var: Symbol):
    # we want to check that all of the terms in the product are polynomials.
    # this means no trigfunctions and logs i guess. and that all powers are to integers.

    # none of these terms should be products because by here the product is expanded simplified and flattened

    # we also has to make sure it has at least one pos power and neg power ig ugh
    return all([is_polynomial(term, var) for term in expr.terms])


def is_polynomial(expr: Expr, var: Symbol) -> bool:
    # how to handle other symbols? treat as consts or just say no? idk.

    def _contains_singlefunc_w_inner(expr: Expr, var: Symbol) -> bool:
        if isinstance(expr, SingleFunc) and expr.inner.contains(var):
            return True

        return any([_contains_singlefunc_w_inner(e, var) for e in expr.children()])

    if _contains_singlefunc_w_inner(expr, var):
        return False

    if isinstance(expr, Const) or isinstance(expr, Symbol):
        return True
    if isinstance(expr, Power):

        if not (
            isinstance(expr.exponent, Const) and expr.exponent.value.denominator == 1
        ):
            return False
        return True
    if isinstance(expr, Sum):
        return all([is_polynomial(term, var) for term in expr.terms])

    raise NotImplementedError(
        f"is_polynomial not implemented for {expr.__class__.__name__}"
    )


def polynomial_division(expr: Prod, var: Symbol) -> Expr:
    # First, we formulate the division problem. We want a numerator and denominator. both are sums or powers or consts or symbols
    numerator = 1
    denominator = 1
    for term in expr.terms:
        b, x = _deconstruct_power(term)
        if x.value > 0:
            numerator *= term
        else:
            denominator *= Power(b, -x).simplify()

    numerator = numerator.simplify()
    denominator = denominator.simplify()
    numerator_list = to_polynomial(numerator, var)
    denominator_list = to_polynomial(denominator, var)
    quotient = np.zeros(len(numerator_list) - len(denominator_list) + 1)

    while numerator_list.size >= denominator_list.size:
        quotient_degree = len(numerator_list) - len(denominator_list)
        quotient_coeff = numerator_list[-1] / denominator_list[-1]
        quotient[quotient_degree] = quotient_coeff
        numerator_list -= np.concatenate(
            ([0] * quotient_degree, denominator_list * quotient_coeff)
        )
        numerator_list = rid_ending_zeros(numerator_list)

    remainder = polynomial_to_expr(numerator_list, var) / polynomial_to_expr(
        denominator_list, var
    )
    quotient_expr = polynomial_to_expr(quotient, var)
    answer = (quotient_expr + remainder).simplify()
    return answer


Polynomial = np.ndarray  # has to be 1-D array


def to_polynomial(expr: Expr, var: Symbol) -> Polynomial:
    if isinstance(expr, Sum):
        xyz = np.zeros(10)
        for term in expr.terms:
            if isinstance(term, Prod):
                const, power = expr.terms
                assert isinstance(const, Const)
                if isinstance(power, Symbol):
                    xyz[1] = int(const.value)
                assert isinstance(power, Power)
                assert power.base == var
                xyz[int(power.exponent.value)] = int(const.value)
            elif isinstance(term, Power):
                assert term.base == var
                xyz[int(term.exponent.value)] = 1
            elif isinstance(term, Symbol):
                assert term == var
                xyz[1] = 1
            elif isinstance(term, Const):
                xyz[0] = int(term.value)
            else:
                raise NotImplementedError(f"weird term: {term}")
        return rid_ending_zeros(xyz)

    if isinstance(expr, Prod):
        # has to be product of 2 terms: a constant and a power.
        const, power = expr.terms
        assert isinstance(const, Const)
        if isinstance(power, Symbol):
            return np.array([0, int(const.value)])
        assert isinstance(power, Power)
        assert power.base == var
        xyz = np.zeros(int(power.exponent.value) + 1)
        xyz[-1] = const.value
        return xyz
    if isinstance(expr, Power):
        assert expr.base == var
        xyz = np.zeros(int(expr.exponent.value) + 1)
        xyz[-1] = 1
        return xyz
    if isinstance(expr, Symbol):
        assert expr == var
        return np.array([0, 1])

    raise NotImplementedError(f"weird expr: {expr}")


def polynomial_to_expr(poly: Polynomial, var: Symbol) -> Expr:
    final = Const(0)
    for i, element in enumerate(poly):
        final += element * var**i
    return final.simplify()


def rid_ending_zeros(arr: Polynomial) -> Polynomial:
    return np.trim_zeros(arr, "b")


def nesting(expr: Expr, var: Symbol) -> int:
    """
    Compute the nesting amount (complexity) of an expression

    >>> nesting(x**2, x)
    2
    >>> nesting(x * y**2, x)
    2
    >>> nesting(x * (1 / y**2 * 3), x)
    2
    """

    if not expr.contains(var):
        return 0

    children = expr.children()
    if isinstance(expr, Symbol) and expr.name == var.name:
        return 1
    elif len(children) == 0:
        return 0
    else:
        return 1 + max(
            nesting(sub_expr, var) for sub_expr in children if sub_expr.contains(var)
        )


def random_id(length):
    # Define the pool of characters you can choose from
    characters = string.ascii_letters + string.digits
    # Use random.choices() to pick characters at random, then join them into a string
    random_string = "".join(random.choices(characters, k=length))
    return random_string


def generate_intermediate_var() -> Symbol:
    return symbols(f"intermediate_{random_id(10)}")[0]


@cast
def heuristics(expr: Expr, var: Symbol):
    # if it doesnt contain any x thats not in tan
    if contains(expr, Tan) and count(expr, Tan(var)) == count(expr, var):
        intermediate = generate_intermediate_var()
        # y = tanx
        new_integrand = replace(expr, Tan(var), intermediate) / (1 + intermediate**2)
        return integrate(new_integrand, intermediate)

    # if expression **contains** trig function and we the current node is not of the transform A on it
    # (or the last heuristic transform was not transform A)
    # then we want to do transform A?

    # for mvp:
    # 1. check that the expression contains a trig function.
    # 2. do a replace for all 3 sets
    # 3. simplify
    # 4. find the one with the lowest nesting
    # 5. integrate that one.
    if contains(expr, TrigFunction):
        r1 = replace_class(
            expr,
            [Tan, Csc, Cot, Sec],
            [
                lambda x: Sin(x) / Cos(x),
                lambda x: 1 / Sin(x),
                lambda x: Cos(x) / Sin(x),
                lambda x: 1 / Cos(x),
            ],
        ).simplify()
        r2 = replace_class(
            expr,
            [Sin, Cos, Cot, Sec],
            [
                lambda x: 1 / Csc(x),
                lambda x: 1 / Tan(x) / Csc(x),
                lambda x: 1 / Tan(x),
                lambda x: Tan(x) * Csc(x),
            ],
        ).simplify()
        r3 = replace_class(
            expr,
            [Sin, Cos, Tan, Csc],
            [
                lambda x: 1 / Cot(x) / Sec(x),
                lambda x: 1 / Sec(x),
                lambda x: 1 / Cot(x),
                lambda x: Cot(x) * Sec(x),
            ],
        ).simplify()

        options = [r1, r2, r3]
        results = [nesting(r, var) for r in options]
        lowest_idx_option = options[results.index(min(results))]
        return integrate(lowest_idx_option, var)

    # look for (1 + (-1 * x^2))
    s = f"(1 + (-1 * {var.name}^2))"
    if s in expr.__repr__():
        intermediate_var = generate_intermediate_var()
        # intermediate1 = sin^-1 (var)
        # ughhh idk how to do this ugh
        # like you want to create a new expression wher eyou replace every instance of the var with sin^-1???
        # psuedocode is good laura
        # how do i loop over the entire expression? and replace it all? so if it's a sum or product then you can loop over all the terms and if it's a product you loop over the ... idk just loop over .children and .children?? but you have to recreate it. ugh
        new_thing = replace(expr, var, Sin(intermediate_var)) * Cos(intermediate_var)
        new_thing = new_thing.simplify()
        # then that's a node and u store the transform and u take the integral of that.
        # Node(new_thing)
        # expr.children = [Node]
        return integrate(new_thing, intermediate_var)

    s2 = f"1 + {var.name}^2"
    if s2 in expr.__repr__():
        intermediate = generate_intermediate_var()
        dy_dx = Sec(intermediate) ** 2
        new_thing = (replace(expr, var, Tan(intermediate)) * dy_dx).simplify()
        return integrate(new_thing, intermediate)

    raise NotImplementedError(f"Cannot integrate {expr} with respect to {var}")


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

    if isinstance(expr, Const):
        return expr
    if isinstance(expr, Symbol):
        return expr

    raise NotImplementedError(f"replace not implemented for {expr.__class__.__name__}")


def contains(expr: Expr, cls) -> bool:
    if isinstance(expr, cls) or issubclass(expr.__class__, cls):
        return True

    return any([contains(e, cls) for e in expr.children()])


@cast
def count(expr: Expr, query: Expr) -> bool:
    if isinstance(expr, query.__class__) and expr == query:
        return 1
    return sum(count(e, query) for e in expr.children())


# cls here has to be a subclass of singlefunc
def replace_class(expr: Expr, cls: list, newfunc: List[Callable[[Expr], Expr]]) -> Expr:
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


if __name__ == "__main__":
    F = Fraction
    x, y = symbols("x y")
    expression = -5 * x**4 / (1 - x**2) ** F(5, 2)
    print(expression)
    integral = integrate(expression, x).simplify()  # TODO auto simplify
    print(integral)
    breakpoint()
