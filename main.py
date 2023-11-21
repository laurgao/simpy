# TODO: tasks for the future:
# - method to convert from our expression to sympy for testing

from dataclasses import dataclass, fields
from typing import List, Tuple
from fractions import Fraction
from functools import reduce
import itertools

def _cast(x):
    if type(x) == int or isinstance(x, Fraction):
        return Const(x)
    elif isinstance(x, Expr):
        return x
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


@dataclass
class Symbol(Expr):
    name: str

    def __repr__(self):
        return self.name


@dataclass
class Sum(Expr):
    terms: List[Expr]

    @cast
    def simplify(self):
        s = Sum([t.simplify() for t in self.terms if t.simplify() != 0])

        # flatten sub-sums
        if all(isinstance(t, (Sum, Symbol, Const)) for t in s.terms):
            new = []
            for term in s.terms:
                new += term.terms if isinstance(term, Sum) else [term]
            s = Sum(new)

        # accumulate all constants
        const = sum(t.value for t in s.terms if isinstance(t, Const))
        if const != 0:
            s = Sum([Const(const)] + [t for t in s.terms if not isinstance(t, Const)])


        new_terms = []

        for i, term in enumerate(s.terms):
            if term is None:
                continue

            new_coeff, non_const_factors1 = _deconstruct(term)

            # check if any later terms are the same
            for j in range(i+1, len(s.terms)):
                if s.terms[j] is None:
                    continue

                term2 = s.terms[j]
                coeff2, non_const_factors2 = _deconstruct(term2)

                if (non_const_factors1 == non_const_factors2):
                    new_coeff += coeff2
                    s.terms[j] = None

            new_terms.append(Prod([new_coeff] + non_const_factors1).simplify())

        # get rid of 1-term sums
        return new_terms[0] if len(new_terms) == 1 else Sum(new_terms)


    def __repr__(self):
        return "(" + " + ".join(map(repr, self.terms)) + ")"


def _deconstruct(expr: Expr) -> Tuple[Const, List[Expr]]:
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


@dataclass
class Prod(Expr):
    terms: List[Expr]

    def __repr__(self):
        return "(" + " * ".join(map(repr, self.terms)) + ")"

    @cast
    def simplify(self):
        if any(t == 0 for t in self.terms):
            return 0
        else:
            p = Prod([t.simplify() for t in self.terms if t.simplify() != 1])

            # flatten sub-products
            if all(isinstance(t, (Prod, Symbol, Const)) for t in p.terms):
                new = []
                for t in p.terms:
                    new += t.terms if isinstance(t, Prod) else [t]
                p = Prod(new)

            # accumulate constants
            const = reduce(lambda x,y: x*y, [t.value for t in p.terms if isinstance(t, Const)], 1)
            if const != 1:
                p = Prod([Const(const)] + [t for t in p.terms if not isinstance(t, Const)])

            return p.terms[0] if len(p.terms) == 1 else p  # no 1-term prod

    @cast
    def expand(self):
        # simplify first in order to flatten
        self = self.simplify()

        sums = [t for t in self.terms if isinstance(t, Sum)]
        other = [t for t in self.terms if not isinstance(t, Sum)]

        if not sums:
            return self

        # for every combination of terms in the sums, multiply them and add
        # (using itertools)
        expanded = []
        for terms in itertools.product(*[s.terms for s in sums]):
            expanded.append(Prod(terms).simplify())

        return (Prod(other) * Sum(expanded)).simplify() if other else Sum(expanded).simplify()


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

    @cast
    def expand(self):
        assert isinstance(self.exponent, Const) and self.exponent.value.denominator == 1 and self.exponent.value >= 1, f"Cannot expand {self}"

        return Prod([self.base] * self.exponent.value.numerator).expand()



def symbols(symbols: str):
    return [Symbol(name=s) for s in symbols.split(" ")]


@cast
def diff(expr: Expr, var: Symbol) -> Expr:
    if isinstance(expr, Sum):
        return Sum([diff(e, var) for e in expr.terms])
    elif isinstance(expr, Symbol):
        return Const(1) if expr.name == var.name else Const(0)
    elif isinstance(expr, Prod):
        return Sum(
            [
                Prod([diff(e, var)] + [t for t in expr.terms if t is not e])
                for e in expr.terms
            ]
        )
    elif isinstance(expr, Const):
        return Const(0)
    else:
        raise NotImplementedError(f"Differentiation of {expr} not implemented")


@cast
def integrate(expr: Expr, var: Symbol):
    # - table: power rule, 1/x -> lnx, trig standard integrals
    # - safe transformations: constant out, sum, polynomial division,
    # - heuristic transformations

    # start with polynomials
    if isinstance(expr, Sum):
        return Sum([integrate(e, var) for e in expr.terms])
    elif isinstance(expr, Prod) and any([isinstance(e.simplify(), Const) for e in expr.terms]):
        # if there is a constant, pull it out
        coeff = 1
        new_terms = []
        for e in expr.terms:
            e = e.simplify()
            if isinstance(e, Const):
                coeff *= e.value
            else:
                new_terms.append(e)

        if coeff == 0:
            return Const(0)
        return coeff * integrate(Prod(new_terms).simplify(), var)
    elif isinstance(expr, Power):
        if expr.base == var and isinstance(expr.exponent, Const):
            n = expr.exponent
            return (1 / (n + 1)) * Power(var, n + 1)
        else:
            raise NotImplementedError(f"Cannot integrate {expr}")
    elif isinstance(expr, Symbol):
        return (1 / 2) * Power(var, 2) if expr == var else expr * var
    elif isinstance(expr, Const):
        return expr * var
    else:
        raise NotImplementedError(f"Cannot integrate {expr}")


# lisp expressions
# (+ 3 2 (* 2 3)) = 12
# (+ 3 2 (* 2 3) (/ 4 2)) = 14

if __name__ == "__main__":
    x, y = symbols("x y")
    # print((x + y) * 3)
    # print(diff((x + y) * 3, x).simplify())
    # print(integrate((x + 2)**2, x))
    print(((x + 2)**3).expand())
