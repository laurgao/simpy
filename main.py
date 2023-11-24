# TODO: tasks for the future:
# - method to convert from our expression to sympy for testing

from dataclasses import dataclass, fields
from typing import List, Tuple, Dict, Union
from fractions import Fraction
from functools import reduce
import itertools


def _cast(x):
    if type(x) == int or isinstance(x, Fraction):
        return Const(x)
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


class Expr:
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

    # should be overloaded
    def expandable(self) -> bool:
        return False

    @cast
    def evalf(self, subs: Dict[str, 'Const']):
        raise NotImplementedError(f"Cannot evaluate {self}")


class Associative():
    def flatten(self):
        new_terms = []
        for t in self.terms:
            new_terms += t.terms if isinstance(t, self.__class__) else [t]
        return self.__class__(new_terms)


@dataclass
class Const(Expr):
    value: Fraction

    def __post_init__(self):
        assert isinstance(self.value, Fraction) or type(self.value) == int, f"got value={self.value} not allowed Const"
        self.value = Fraction(self.value)

    def __repr__(self):
        return str(self.value)

    @cast
    def __eq__(self, other):
        return isinstance(other, Const) and self.value == other.value

    @cast
    def __ne__(self, other):
        return not (self == other)

    @cast
    def evalf(self, subs: Dict[str, 'Const']):
        return self

    def diff(self, var):
        return Const(0)


@dataclass
class Symbol(Expr):
    name: str

    def __repr__(self):
        return self.name
    
    @cast
    def evalf(self, subs: Dict[str, 'Const']):
        return subs.get(self.name, self)

    def diff(self, var):
        return Const(1) if self.name == var.name else Const(0)


@dataclass
class Sum(Expr, Associative):
    terms: List[Expr]

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
            for j in range(i+1, len(s.terms)):
                if s.terms[j] is None:
                    continue

                term2 = s.terms[j]
                coeff2, non_const_factors2 = _deconstruct_prod(term2)

                if (non_const_factors1 == non_const_factors2):
                    new_coeff += coeff2
                    s.terms[j] = None

            new_terms.append(Prod([new_coeff] + non_const_factors1).simplify())

        # get rid of 1-term sums
        return new_terms[0] if len(new_terms) == 1 else Sum(new_terms)

    @cast
    def evalf(self, subs: Dict[str, 'Const']):
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
        non_const_factors = expr.terms[1:] if isinstance(expr.terms[0], Const) else expr.terms
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
class Prod(Expr, Associative):
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
            for j in range(i+1, len(new.terms)):
                if new.terms[j] is None:
                    continue
                other = new.terms[j]
                base2, expo2 = _deconstruct_power(other)
                if base2 == base: # TODO: real expr equality
                    expo += expo2
                    new.terms[j] = None
            
            terms.append(Power(base, expo).simplify())

        new.terms = terms

        # Check for zero
        if any(t == 0 for t in new.terms):
            return Const(0)

        # accumulate constants to the front
        const = reduce(lambda x,y: x*y, [t.value for t in new.terms if isinstance(t, Const)], 1)

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
        return any(isinstance(t, Sum) for t in self.terms) or any(t.expandable() for t in self.terms)


    @cast
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
    def evalf(self, subs: Dict[str, 'Const']):
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
            return Const(b.value ** x.value)
        else:
            return Power(self.base.simplify(), x)

    def expandable(self) -> bool:
        return isinstance(self.exponent, Const) and self.exponent.value.denominator == 1 and self.exponent.value >= 1

    def expand(self) -> Expr:
        assert self.expandable(), f"Cannot expand {self}"
        return Prod([self.base] * self.exponent.value.numerator).expand()

    @cast
    def evalf(self, subs: Dict[str, 'Const']):
        return Power(self.base.evalf(subs), self.exponent.evalf(subs)).simplify()


@dataclass
class Log(Expr):
    inner: Expr

    def __repr__(self):
        return f"log({self.inner})"

    def diff(self, var):
        return self.inner.diff(var) / self.inner 
    

def symbols(symbols: str):
    return [Symbol(name=s) for s in symbols.split(" ")]


@cast
def diff(expr: Expr, var: Symbol) -> Expr:
    if hasattr(expr, 'diff'):
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

        # if there are sub-sums, integrate the expansion
        if expr.expandable():
            return integrate(expr.expand(), var)
        
        raise NotImplementedError(f"Cannot integrate {expr}")
    elif isinstance(expr, Power):
        if expr.base == var and isinstance(expr.exponent, Const):
            n = expr.exponent
            return (1 / (n + 1)) * Power(var, n + 1) if n != -1 else Log(expr.base)
        elif expr.expandable():
            return integrate(expr.expand(), var)
        else:
            raise NotImplementedError(f"Cannot integrate {expr}")
    elif isinstance(expr, Symbol):
        return Fraction(1 / 2) * Power(var, 2) if expr == var else expr * var
    elif isinstance(expr, Const):
        return expr * var
    else:
        raise NotImplementedError(f"Cannot integrate {expr}")


if __name__ == "__main__":
    x, y = symbols("x y")

    F = Fraction
    I1 = integrate((x/90 * (x-5)**2 / 350), (x, 5, 6))
    I2 = integrate((F(1, 15) - F(1, 360) * (x-6))*(x-5)**2 / 350, (x, 6, 15))
    I3 = integrate((F(1, 15) - F(1, 360) * (x-6))*(1 - (40-x)**2/875), (x, 15, 30))
    
    print(I1, I2, I3)
    final = (I1 + I2 + I3).simplify()
    print(final)
    print(float(final.value))
