from dataclasses import dataclass
from typing import List


def _cast(x):
    if type(x) == int:
        return Const(x)
    elif isinstance(x, Expr) or type(x) == bool:
        return x
    else:
        raise NotImplementedError(f"Cannot cast {x} to Expr")


def cast(func):
    def wrapper(*args) -> "Expr":
        return _cast(func(*[_cast(a) for a in args]))

    return wrapper


class Expr:
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


@dataclass
class Const(Expr):
    value: int

    def __post_init__(self):
        assert type(self.value) == int, f"got value={self.value} not allowed Const"

    def __repr__(self):
        return repr(self.value)

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
        return s.terms[0] if len(s.terms) == 1 else s  # no 1-term sums

        # todo: combine like terms

    def __repr__(self):
        return "(" + " + ".join(map(repr, self.terms)) + ")"


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
            return p.terms[0] if len(p.terms) == 1 else p  # no 1-term prod


@dataclass
class Power(Expr):
    base: Expr
    exponent: Expr

    def __repr__(self):
        return f"({self.base}^{self.exponent})"

    def simplify(self):
        x = self.exponent.simplify()
        if x == 0:
            return Const(1)
        elif x == 1:
            return self.base.simplify()
        else:
            return Power(self.base.simplify(), x)

        # todo: expand


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
    pass


def _integrate_power(expr: Expr, var: Symbol):
    "Integrate smtn using the power rule"
    if not (
        isinstance(expr, Power)
        and expr.base == var
        and isinstance(expr.exponent, Const)
    ):
        raise ValueError(f"Cannot integrate {expr} using power rule")

    n = expr.exponent.value
    return (1 / (n + 1)) * Power(var, Const(n + 1))


# lisp expressions
# (+ 3 2 (* 2 3)) = 12
# (+ 3 2 (* 2 3) (/ 4 2)) = 14

if __name__ == "__main__":
    x, y = symbols("x y")
    print((x + y) * 3)
    print(diff((x + y) * 3, x).simplify())
