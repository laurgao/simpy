"""RULES OF EXPRs:

1. Exprs shall NOT be mutated in place after __post_init__.
For example, if I put a Const into a numpy array, I don't want to have to copy it. i can trust that its value stays the same forever.

Note on equality: if you call (expr1 == expr2), it returns true/false based on **the structure of the expr is the same.** rather than based on their values
being equal. if you wanna check equality of values, try simplifying, expanding, etc.

This currently operates on fractions only. I personally love keeping it rational and precise
so I don't really mind the lack of float support, but I can see why you'd be annoyed.
~~but i spent so much time making the fractional arithmetic perfect -- see Power.__new__() for instance

"""

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from fractions import Fraction
from functools import cmp_to_key, reduce
from typing import Callable, Dict, List, Literal, Optional, Tuple, Type, Union

from .combinatorics import generate_permutations, multinomial_coefficient


def nesting(expr: "Expr", var: Optional["Symbol"] = None) -> int:
    """
    Compute the nesting amount (complexity) of an expression
    If var is provided, only count the nesting of the subexpression containing var

    >>> nesting(x**2, x)
    2
    >>> nesting(x * y**2, x)
    2
    >>> nesting(x * (1 / y**2 * 3), x)
    2
    """

    if var is not None and not expr.contains(var):
        return 0

    # special case
    if isinstance(expr, Prod) and expr.terms[0] == Const(-1) and len(expr.terms) == 2:
        return nesting(expr.terms[1], var)

    if isinstance(expr, Symbol) and (var is None or expr.name == var.name):
        return 1
    elif len(expr.children()) == 0:
        return 0
    else:
        return 1 + max(nesting(sub_expr, var) for sub_expr in expr.children())



def _cast(x):
    if x is None or x is True or x is False:
        # let it slide because it's probably intentional
        return x

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
    elif isinstance(x, list):
        return [_cast(v) for v in x]
    else:
        raise NotImplementedError(f"Cannot cast {x} to Expr")


def cast(func):
    def wrapper(*args, **kwargs) -> "Expr":
        return func(*[_cast(a) for a in args], **{k: _cast(v) for k, v in kwargs.items()})

    return wrapper


class Expr(ABC):
    def __post_init__(self):
        # if any field is an Expr, cast it
        for field in fields(self):
            if field.type is Expr or field.type is List[Expr]:
                setattr(self, field.name, _cast(getattr(self, field.name)))

    # should be overwritten in subclasses
    def simplify(self) -> "Expr":
        return self
    
    @cast
    def __add__(self, other) -> "Expr":
        return Sum([self, other])

    @cast
    def __radd__(self, other) -> "Expr":
        return Sum([other, self])

    @cast
    def __sub__(self, other) -> "Expr":
        return self + (-1 * other)

    @cast
    def __rsub__(self, other) -> "Expr":
        return other + (-1 * self)

    @cast
    def __mul__(self, other) -> "Expr":
        return Prod([self, other])

    @cast
    def __rmul__(self, other) -> "Expr":
        return Prod([other, self])

    @cast
    def __pow__(self, other) -> "Expr":
        return Power(self, other)

    @cast
    def __rpow__(self, other) -> "Expr":
        return Power(other, self)

    @cast
    def __div__(self, other) -> "Expr":
        return Prod([self, Power(other, -1)])

    @cast
    def __truediv__(self, other) -> "Expr":
        return Prod([self, Power(other, -1)])

    @cast
    def __rdiv__(self, other) -> "Expr":
        return Prod([other, Power(self, -1)])

    @cast
    def __rtruediv__(self, other) -> "Expr":
        return Prod([other, Power(self, -1)])

    def __neg__(self) -> "Expr":
        return -1 * self

    # should be overloaded if necessary
    def expandable(self) -> bool:
        if not self.children():
            return False
        return any([c.expandable() for c in self.children()])

    # overload if necessary
    def expand(self) -> "Expr":
        """Subclasses: this function should raise AssertionError if self.expandable() is false.
        """
        if self.expandable():
            raise NotImplementedError(f"Expansion of {self} not implemented")
        raise AssertionError(f"Cannot expand {self}")

    @cast
    @abstractmethod
    def evalf(self, subs: Dict[str, "Const"]):
        raise NotImplementedError(f"Cannot evaluate {self}")

    @abstractmethod
    def children(self) -> List["Expr"]:
        raise NotImplementedError(f"Cannot get children of {self.__class__.__name__}")

    def contains(self: "Expr", var: "Symbol") -> bool:
        is_var = isinstance(self, Symbol) and self.name == var.name
        return is_var or any(e.contains(var) for e in self.children())
    
    def has(self, cls: Type["Expr"]) -> bool:
        from .regex import contains_cls
        return contains_cls(self, cls)

    @abstractmethod
    def diff(self, var: "Symbol") -> "Expr":
        raise NotImplementedError(
            f"Cannot get the derivative of {self.__class__.__name__}"
        )

    def symbols(self) -> List["Symbol"]:
        # I hate this syntax
        str_set = set([symbol.name for e in self.children() for symbol in e.symbols()])
        return [Symbol(name=s) for s in str_set]

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError(f"Cannot represent {self.__class__.__name__}")

    @abstractmethod
    def latex(self) -> str:
        raise NotImplementedError(f"Cannot convert {self.__class__.__name__} to latex")

    @cast    
    def __mod__(self, other) -> "Const":
        return NotImplemented 


@dataclass
class Associative:
    terms: List[Expr]

    def __post_init__(self):
        # Calls super because: If we have Sum(Associative, Expr) and Sum.__post_init__()
        # will call Associative.__post_init__(). the super in Associative will call Expr.__post_init__()
        # (This is according to chatgpt and I haven't confirmed it yet.)
        super().__post_init__()
        self._flatten()
        self._sort()

    @staticmethod
    def _flatten_terms(terms: List[Expr], cls: Type["Associative"]):
        """Utility function for flattening a list.
        """
        new_terms = []
        for t in terms:
            if isinstance(t, cls):
                t._flatten()
                new_terms += t.terms
            else:
                new_terms += [t]
        return new_terms


    def _flatten(self) -> None:
        """Put any sub-products/sub-sums into the parent
        ex: (x * 3) * y -> x * 3 * y
        Can go down as many levels deep as necessary.
        """
        # no need to flatten in simplify because we can then always assume ALL Prod and Sum class objects are flattened
        self.terms = self._flatten_terms(self.terms, self.__class__)

    def children(self) -> List["Expr"]:
        return self.terms

    def _sort(self) -> None:
        def _nesting(expr: "Expr") -> int:
            """Like the public nesting except constant multipliers don't count
            to prevent fucky reprs for things like 
            1 + x^3 + 3*x^2
            """
            expr = remove_const_factor(expr)
            if isinstance(expr, Symbol):
                return 1
            if len(expr.symbols()) == 0:
                return 0
            return 1 + max(nesting(sub_expr) for sub_expr in expr.children())


        def _compare(a: Expr, b: Expr) -> int:
            """Returns -1 if a < b, 0 if a == b, 1 if a > b.
            The idea is you sort first by nesting, then by power, then by the term alphabetical
            """

            def _deconstruct_const_power(expr: Expr) -> Const:
                expr = remove_const_factor(expr)
                if isinstance(expr, Power) and isinstance(expr.exponent, Const):
                    return expr.exponent
                return Const(1)

            n = _nesting(a) - _nesting(b)
            if n != 0:
                return n
            power = (
                _deconstruct_const_power(a).value - _deconstruct_const_power(b).value
            )
            if power != 0:
                return power
            return 1 if a.__repr__() > b.__repr__() else -1

        key = cmp_to_key(_compare)
        self.terms = sorted(self.terms, key=key)


class Number(ABC):
    def diff(self, var) -> "Const":
        return Const(0)

    def children(self) -> List["Expr"]:
        return []

    @cast
    def subs(self, subs: Dict[str, "Const"]):
        return self


@dataclass
class Const(Number, Expr):
    value: Fraction

    def __post_init__(self):
        assert (
            isinstance(self.value, (int, Fraction)) or int(self.value) == self.value
        ), f"got value={self.value} not allowed Const"

        if not isinstance(self.value, Fraction):
            self.value = Fraction(self.value)

    def __repr__(self) -> str:
        return str(self.value)

    ### These are unnecessary because it would get simplified to this by Sum/Prod anyways (it's tested---remove these, all tests pass)
    # but I'm doing them for some sense of "efficiency" (not really, isn't actually an efficiency bottleneck at all.)
    @cast
    def __add__(self, other) -> "Expr":
        if isinstance(other, Const):
            return Const(self.value + other.value)
        return super().__add__(other)
    
    @cast
    def __radd__(self, other) -> "Expr":
        if isinstance(other, Const):
            return Const(other.value + self.value)
        return super().__radd__(other)

    @cast
    def __sub__(self, other) -> "Expr":
        if isinstance(other, Const):
            return Const(self.value - other.value)
        return super().__sub__(other)
    
    @cast
    def __rsub__(self, other) -> "Expr":
        if isinstance(other, Const):
            return Const(other.value - self.value)
        return super().__rsub__(other)

    def __neg__(self):
        return Const(-self.value)
    ###

    @cast
    def __eq__(self, other):
        return isinstance(other, Const) and self.value == other.value

    @cast
    def __ge__(self, other):
        return isinstance(other, Const) and self.value >= other.value

    @cast
    def __gt__(self, other):
        return isinstance(other, Const) and self.value > other.value

    @cast
    def __le__(self, other):
        return isinstance(other, Const) and self.value <= other.value

    @cast
    def __lt__(self, other):
        return isinstance(other, Const) and self.value < other.value
    
    @cast
    def evalf(self, subs: Dict[str, "Const"]):
        return self

    def latex(self) -> str:
        if self.value.denominator == 1:
            return f"{self.value}"
        return (
            "\\frac{"
            + str(self.value.numerator)
            + "}{"
            + str(self.value.denominator)
            + "}"
        )

    def abs(self) -> "Const":
        return Const(abs(self.value))

    def reciprocal(self) -> "Const":
        return Const(Fraction(self.value.denominator, self.value.numerator))

    @cast    
    def __mod__(self, other) -> "Const":
        if isinstance(other, Const):  
            return Const(self.value % other.value)
        else:
            return NotImplemented



@dataclass
class Pi(Number, Expr):
    @cast
    def evalf(self, subs: Dict[str, "Const"]):
        # return 3.141592653589793
        return self

    def __repr__(self) -> str:
        return "pi"

    def latex(self) -> str:
        return "\\pi"


@dataclass
class E(Number, Expr):
    @cast
    def evalf(self, subs: Dict[str, "Const"]):
        return self
        # return 2.718281828459045

    def __repr__(self) -> str:
        return "e"

    def latex(self) -> str:
        return "e"

    def __eq__(self, other) -> bool:
        return isinstance(other, E)
    

@dataclass
class Infinity(Number, Expr):
    @cast
    def evalf(self, subs: Dict[str, "Const"]):
        return self

    def __repr__(self) -> str:
        return "inf"

    def latex(self) -> str:
        return "\\infty"


pi = Pi()
e = E()
infinity = Infinity() # This isn't used for anything yet.


@dataclass
class Symbol(Expr):
    name: str

    def __repr__(self) -> str:
        return self.name

    @cast
    def evalf(self, subs: Dict[str, "Const"]):
        return subs.get(self.name, self)

    def diff(self, var) -> Const:
        return Const(1) if self == var else Const(0)

    def __eq__(self, other):
        return isinstance(other, Symbol) and self.name == other.name

    def children(self) -> List["Expr"]:
        return []

    def symbols(self) -> List["Expr"]:
        return [self]

    def latex(self) -> str:
        return self.name


@dataclass
class Sum(Associative, Expr):
    def __new__(cls, terms: List[Expr]) -> "Expr":
        """When a sum is initiated:
        - terms are converted to expr 
        - flatten
        - accumulate like terms & constants
        - sort
        """
        terms = cls._flatten_terms(terms, cls)
        # accumulate all like terms
        new_terms = []
        for i, term in enumerate(terms):
            if term is None:
                continue
            if isinstance(term, Const):
                new_terms.append(term)
                continue

            new_coeff, non_const_factors1 = _deconstruct_prod(term)

            # check if any later terms are the same
            for j in range(i + 1, len(terms)):
                term2 = terms[j]
                if term2 is None:
                    continue

                coeff2, non_const_factors2 = _deconstruct_prod(term2)

                if non_const_factors1 == non_const_factors2:
                    new_coeff += coeff2
                    terms[j] = None

            new_terms.append(Prod([new_coeff] + non_const_factors1))

        # accumulate all constants
        const = sum(t.value for t in new_terms if isinstance(t, Const))
        non_constant_terms = [t for t in new_terms if not isinstance(t, Const)]

        final_terms = ([Const(const)] if const != 0 else []) + non_constant_terms
        if len(final_terms) == 0:
            return Const(0)
        if len(final_terms) == 1:
            return final_terms[0]
         
        instance = super().__new__(cls)
        instance.terms = final_terms
        return instance
    
    def __init__(self, terms: List[Expr]):
        # Overrides the shit that does self.terms = terms because i've already set terms
        # in __new__.
        self.__post_init__()

    def _sort(self):
        # Sums are typically written from largest complexity to smallest (whereas for products it's the opposite)
        super()._sort()
        self.terms = list(reversed(self.terms))

    def expand(self) -> Expr:
        assert self.expandable(), f"Cannot expand {self}"
        return Sum([t.expand() if t.expandable() else t for t in self.terms])

    def simplify(self) -> "Expr":
        new_sum = Sum([t.simplify() for t in self.terms])
        if not isinstance(new_sum, Sum):
            return new_sum

        from .simplify import trig_simplification
        return trig_simplification(new_sum)

    @cast
    def evalf(self, subs: Dict[str, "Const"]):
        return Sum([t.evalf(subs) for t in self.terms])

    def diff(self, var) -> "Sum":
        return Sum([diff(e, var) for e in self.terms])

    def __repr__(self) -> str:
        ongoing_str = ""
        for i, term in enumerate(self.terms):
            if i == 0:
                ongoing_str += f"{term}"
            elif isinstance(term, Prod) and term.is_subtraction:
                ongoing_str += f" - {(term * -1)}"
            else:
                ongoing_str += f" + {term}"

        return ongoing_str

    def latex(self) -> str:
        ongoing_str = ""
        for i, term in enumerate(self.terms):
            if i == 0:
                ongoing_str += term.latex()
            elif isinstance(term, Prod) and term.is_subtraction:
                ongoing_str += f" - {(term * -1).latex()}"
            else:
                ongoing_str += f" + {term.latex()}"

        return ongoing_str

    def factor(self) -> Union["Prod", "Sum"]:
        # TODO: this feels like not the most efficient algo
        # assume self is simplified please
        # If there is a factor that is common to all terms, factor it out.
        # If there is a factor that is common to some terms, let's just ignore it.
        # TODO: doesn't factor ex. quadratics into 2 binomials. implement some form of multi-term polynomial factoring at some point
        # (needed for partial fractions)

        def _df(term: Expr) -> Tuple[Const, Optional[List[Tuple[Expr, int, bool]]]]:
            """Deconstruct a term into its factors.

            Returns: Number, List[(factor, abs(exponent), sign(exponent))]
            """
            if isinstance(term, Prod):
                num, terms = _deconstruct_prod(term)
                return num, [_df(f)[1][0] for f in terms]
            if isinstance(term, Power) and isinstance(
                term.exponent, Const
            ):  # can't be prod bc it's simplified
                return Const(1), [[term.base, term.exponent.abs(), term.exponent.value > 0]]
            if isinstance(term, Const):
                return term, [[term, Const(1), True]]
            return Const(1), [[term, Const(1), True]]

        dfs = [_df(term) for term in self.terms]
        factors_per_term = [d[1] for d in dfs]
        coeffs = [d[0] for d in dfs]
        common_factors = factors_per_term[0]

        for this_terms_factors in factors_per_term[1:]:
            for i, cfactor in enumerate(common_factors):
                if cfactor is None:
                    continue
                is_in_at_least_1 = False
                for tfactor in this_terms_factors:
                    if (
                        cfactor[0].__repr__() == tfactor[0].__repr__()
                        and cfactor[2] == tfactor[2]
                    ):
                        cfactor[1] = min(cfactor[1], tfactor[1])
                        is_in_at_least_1 = True
                        break
                if not is_in_at_least_1:
                    common_factors[i] = None

        # Factor coeffs
        common_coeff = coeffs[0].abs()
        for c in coeffs[1:]:
            x: Const = (c / common_coeff)
            y: Const = (common_coeff / c)
            if x.value.denominator == 1 or y.value.denominator == 1:
                common_coeff = min(c.abs(), common_coeff.abs())
            else:
                common_coeff = None
                break
        is_negative = all(c < 0 for c in coeffs)
        if is_negative and common_coeff:
            common_coeff *= -1

        common_factors = [f for f in common_factors if f is not None]

        def _makeprod(terms: List[Tuple[Expr, int, bool]]):
            return (
                Const(1)
                if len(terms) == 0
                else Prod([Power(t[0], t[1] * (1 if t[2] else -1)) for t in terms])
            )

        common_expr = _makeprod(common_factors)
        if common_coeff:
            common_expr *= common_coeff

        # factor out the common factors
        new_terms = []
        for term in self.terms:
            new_terms.append(term / common_expr)

        return (common_expr * Sum(new_terms))


def _deconstruct_prod(expr: Expr) -> Tuple[Const, List[Expr]]:
    # 3*x^2*y -> (3, [x^2, y])
    # turns smtn into a constant and a list of other terms
    if isinstance(expr, Prod):
        non_const_factors = [term for term in expr.terms if not isinstance(term, Const)]
        const_factors = [term for term in expr.terms if isinstance(term, Const)]
        coeff = Prod(const_factors) if const_factors else Const(1)
    else:
        non_const_factors = [expr]
        coeff = Const(1)
    return (coeff, non_const_factors)


def deconstruct_power(expr: Expr) -> Tuple[Expr, Const]:
    # x^3 -> (x, 3). x -> (x, 1). 3 -> (3, 1)
    if isinstance(expr, Power):
        return (expr.base, expr.exponent)
    else:
        return (expr, Const(1))


@dataclass
class Prod(Associative, Expr):
    def __new__(cls, terms: List[Expr]) -> "Expr":
        # We need to flatten BEFORE we accumulate like terms
        # ex: Prod(x, Prod(Power(x, -1), y))
        initial_terms = cls._flatten_terms(terms, cls)

        # accumulate all like terms
        new_terms = []
        for i, term in enumerate(initial_terms):
            if term is None:
                continue

            base, expo = deconstruct_power(term)

            # other terms with same base
            for j in range(i + 1, len(initial_terms)):
                if initial_terms[j] is None:
                    continue
                other = initial_terms[j]
                base2, expo2 = deconstruct_power(other)
                if base2 == base:
                    expo += expo2
                    initial_terms[j] = None
            
            # Decision: if exponent here is 0, we don't include it.
            # We don't wait for simplify to remove it.
            if expo == 0:
                continue
            new_terms.append(Power(base, expo) if expo != 1 else base)
        
        # accumulate constants to the front
        const = reduce(
            lambda x, y: x * y, [t.value for t in new_terms if isinstance(t, Const)], 1
        )
        if const == 0:
            return Const(0)

        non_constant_terms = [t for t in new_terms if not isinstance(t, Const)]
        new_terms = ([] if const == 1 else [Const(const)]) + non_constant_terms
        
        if len(new_terms) == 0:
            return Const(1)
        if len(new_terms) == 1:
            return new_terms[0]

        instance = super().__new__(cls) 
        instance.terms = new_terms
        return instance

    def __init__(self, terms: List[Expr]):
        # terms are already set in __new__
        self.__post_init__()

    def simplify(self) -> "Expr":
        return Prod([t.simplify() for t in self.terms])

    def __repr__(self) -> str:
        def _term_repr(term):
            if isinstance(term, Sum):
                return "(" + repr(term) + ")"
            return repr(term)

        # special case for subtraction:
        if self.is_subtraction:
            new_prod = self * -1
            if not isinstance(new_prod, Prod):
                return f"-{_term_repr(new_prod)}"
            return "-" + new_prod.__repr__()

        numerator, denominator = self.numerator_denominator
        if denominator != Const(1):

            def _x(prod: Prod):
                if not isinstance(prod, Prod):
                    return _term_repr(prod)
                if len(prod.terms) == 1:
                    return _term_repr(prod.terms[0])
                return "(" + repr(prod) + ")"

            return _x(numerator) + "/" + _x(denominator)

        return "".join(map(_term_repr, self.terms))

    def latex(self) -> str:
        def _term_latex(term: Expr):
            if isinstance(term, Sum):
                return "\\left(" + term.latex() + "\\right)"
            return term.latex()

        # special case for subtraction:
        if self.is_subtraction:
            new = self * -1
            if not isinstance(new, Prod):
                return "-" + _term_latex(new)
            return "-" + new.latex()

        numerator, denominator = self.numerator_denominator
        if denominator != Const(1):
            # don't need brackets around num/denom bc the frac bar handles it.
            return "\\frac{" + numerator.latex() + "}{" + denominator.latex() + "}"

        return "".join(map(_term_latex, self.terms))

    @property
    def numerator_denominator(self) -> Tuple[Expr, Expr]:
        denominator = []
        numerator = []
        for term in self.terms:
            # handle consts seperately
            if isinstance(term, Const):
                if term.value.numerator != 1:
                    numerator.append(Const(term.value.numerator))
                if term.value.denominator != 1:
                    denominator.append(Const(term.value.denominator))
                continue

            b, x = deconstruct_power(term)
            if isinstance(x, Const) and x.value < 0:
                denominator.append(b if x == Const(-1) else Power(b, -x))
            else:
                numerator.append(term)

        num_expr = Prod(numerator) if len(numerator) > 0 else Const(1)
        denom_expr = Prod(denominator) if len(denominator) > 0  else Const(1)
        return [num_expr, denom_expr]

    @property
    def is_subtraction(self):
        return isinstance(self.terms[0], Const) and self.terms[0].value < 0

    @cast
    def expandable(self) -> bool:
        # a product is expandable if it contains any sums in the numerator
        # OR if it contains sums in the denominator AND the denominator has another term other than the sum
        # (so, a singular sum in a numerator is expandable but a single sum in the denominator isn't.)
        num, denom = self.numerator_denominator
        num_expandable = any(isinstance(t, Sum) for t in num.terms) if isinstance(num, Prod) else isinstance(num, Sum)
        denom_expandable = any(isinstance(t, Sum) for t in denom.terms) if isinstance(denom, Prod) else False
        has_sub_expandable = any(t.expandable() for t in self.terms)
        return num_expandable or denom_expandable or has_sub_expandable


    def expand(self):
        assert self.expandable(), f"Cannot expand {self}"
        # expand sub-expressions
        num, denom = self.numerator_denominator
        if denom.expandable():
            denom = denom.expand()
            
        # now we assume denom is good and we move on with life as usual
        if not isinstance(num, Prod):
            expanded_terms = [num.expand() if num.expandable() else num]
        else:
            expanded_terms = [t.expand() if t.expandable() else t for t in num.terms]
        sums = [t for t in expanded_terms if isinstance(t, Sum)]
        other = [t for t in expanded_terms if not isinstance(t, Sum)]
        if not sums:
            return (Prod(expanded_terms) / denom)

        # for every combination of terms in the sums, multiply them and add
        # (using itertools)
        final_sum_terms = []
        for terms in itertools.product(*[s.terms for s in sums]):
            final_sum_terms.append(Prod(other + list(terms)) / denom)
        
        return Sum(final_sum_terms)

    @cast
    def evalf(self, subs: Dict[str, "Const"]):
        return Prod([t.evalf(subs) for t in self.terms])

    def diff(self, var) -> Sum:
        return Sum(
            [
                Prod([diff(e, var)] + [t for t in self.terms if t is not e])
                for e in self.terms
            ]
        )



def debug_repr(expr: Expr) -> str:
    """Not designed to look pretty; designed to show the structure of the expr.
    """
    if isinstance(expr, Power):
        return f'Power({debug_repr(expr.base)}, {debug_repr(expr.exponent)})'
        # return f'({debug_repr(expr.base)})^{debug_repr(expr.exponent)}'
    if isinstance(expr, Prod):
        return "Prod(" + ", ".join([debug_repr(t) for t in expr.terms]) + ")"
        # return " * ".join([debug_repr(t) for t in expr.terms])
    if isinstance(expr, Sum):
        return "Sum(" + ", ".join([debug_repr(t) for t in expr.terms]) + ")"
        # return " + ".join([debug_repr(t) for t in expr.terms])
    else:
        return repr(expr)


@dataclass
class Power(Expr):
    base: Expr
    exponent: Expr

    def __repr__(self) -> str:
        isfraction = lambda x: isinstance(x, Const) and x.value.denominator != 1
        def _term_repr(term):
            if isinstance(term, Sum) or isinstance(term, Prod) or isfraction(term):
                return "(" + repr(term) + ")"
            return repr(term)

        # represent negative powers as reciprocals
        if self.exponent == Const(-1):
            return "1/" + _term_repr(self.base)
        if isinstance(self.exponent, Const) and self.exponent < 0:
            new_power = Power(self.base, -self.exponent)
            return "1/" + repr(new_power)

        # special case for sqrt
        if self.exponent == Const(Fraction(1, 2)):
            return _repr(self.base, "sqrt")

        return f"{_term_repr(self.base)}^{_term_repr(self.exponent)}"

    def latex(self) -> str:
        def _term_latex(term: Expr):
            if isinstance(term, Sum) or isinstance(term, Prod):
                return "\\left(" + term.latex() + "\\right)"
            return term.latex()

        # special case for sqrt
        if self.exponent == Const(Fraction(1, 2)):
            return "\\sqrt{" + self.base.latex() + "}"
        if self.exponent == Const(Fraction(-1, 2)):
            return "{\\sqrt{" + self.base.latex() + "}" + "}^{-1}"

        return "{" + _term_latex(self.base) + "}^{" + _term_latex(self.exponent) + "}"

    def __new__(cls, base: Expr, exponent: Expr) -> "Expr":
        b = _cast(base)
        x = _cast(exponent)

        default_return = super().__new__(cls)
        default_return.base = b
        default_return.exponent = x

        if x == 0:
            # Ok this is up to debate, but since python does 0**0 = 1 I'm gonna do it too.
            # "0**0 represents the empty product (the number of sets of 0 elements that can be chosen from a set of 0 elements), which by definition is 1. This is also the same reason why anything else raised to the power of 0 is 1."
            return Const(1)
        if x == 1:
            return b
        if b == 0:
            return Const(0)
        if b == 1:
            return Const(1)
        if isinstance(b, Const) and isinstance(x, Const):
            try:
                return Const(b.value**x.value)
            except:
                pass
            
            # Rewriting for consistency between same values.
            if b.value.numerator == 1:
                return Power(b.reciprocal(), -x)
            elif b.value.denominator != 1 and x < 0:
                return Power(b.reciprocal(), -x)
            
            if x.value.denominator % 2 == 0 and b.value.numerator < 0:
                # Cannot be simplified further.
                return default_return
            
            # check if one of num^exponent or denom^exponent is integer
            n, d = abs(b.value.numerator), b.value.denominator
            if x < 0:
                n, d = d, n
            xvalue = x.abs().value
            xn = n**xvalue  # exponentiated numerator
            xd = d**xvalue
            isint = lambda a: int(a) == a and a != 1
            if not isint(xn) and not isint(xd): # Cannot be simplified further
                return default_return

            # the answer will be a product
            terms = []
            if isint(xn):
                terms.append(Const(xn))
                terms.append(Power(d, -xvalue)) # will this circular? no it won't it should get caught under line after `if not isint(xn) and not...`
            else:
                # isint(xd)
                terms.append(Const(Fraction(1, int(xd))))
                terms.append(Power(n, xvalue))
            if b.value.numerator < 0:
                terms.append(Const(-1))
            
            return Prod(terms)
        if isinstance(b, Power):
            # Have to call the class here. In the case that x*b.exponent = 1, this will have to re-simplfiy
            # through calling the constructor.
            return Power(b.base, x * b.exponent)
        if isinstance(b, Prod):
            # when you construct this new power entity you have to simplify it.
            # because what if the term raised to this exponent can be simplified?
            # ex: if you have (ab)^n where a = c^m
            return Prod([Power(term, x) for term in b.terms])
        if isinstance(x, log) and b == x.base:
            return x.inner
        if isinstance(x, Prod):
            for i, t in enumerate(x.terms):
                if isinstance(t, log) and t.base == b:
                    rest = Prod(x.terms[:i] + x.terms[i+1:])
                    return Power(t.inner, rest) # Just in case this needs to be simplified, we will pass this through the constructor
                    # return Power(Power(b, t), rest)
                    # Power(b, t) will be simplified to t.inner
        return default_return
    
    def __init__(self, base: Expr, exponent: Expr):
        self.__post_init__()
    
    def simplify(self) -> "Expr":
        return Power(self.base.simplify(), self.exponent.simplify())
    
    def _power_expandable(self) -> bool:
        return (
            isinstance(self.exponent, Const)
            and self.exponent.value.denominator == 1
            and self.exponent.value > 1
            and isinstance(self.base, Sum)
        )

    def expandable(self) -> bool:
        return self._power_expandable() or self.base.expandable() or self.exponent.expandable()

    def expand(self) -> Expr:
        assert self.expandable(), f"Cannot expand {self}"
        b = self.base.expand() if self.base.expandable() else self.base
        x = self.exponent.expand() if self.exponent.expandable() else self.exponent
        new = Power(b, x)
        if not isinstance(new, Power) or not new._power_expandable():
            return new

        expanded = []
        n = new.exponent.value.numerator
        i = len(new.base.terms)
        permutations = generate_permutations(i, n)
        for permutation in permutations:
            new_term = [Power(t, p) for t, p in zip(new.base.terms, permutation)]
            coefficient = multinomial_coefficient(permutation, n)
            expanded.append(Prod([Const(coefficient)] + new_term))
        return Sum(expanded)

    @cast
    def evalf(self, subs: Dict[str, "Const"]):
        return Power(self.base.evalf(subs), self.exponent.evalf(subs))

    def children(self) -> List["Expr"]:
        return [self.base, self.exponent]

    def diff(self, var) -> Expr:
        if not self.exponent.contains(var):
            return self.exponent * self.base ** (self.exponent - 1) * self.base.diff(var)
        if not self.base.contains(var):
            return log(self.base) * self * self.exponent.diff(var)
        raise NotImplementedError(
            "Power.diff not implemented for functions with var in both base and exponent."
        )


@dataclass
class SingleFunc(Expr):
    inner: Expr

    @property
    @abstractmethod
    def _label(self) -> str:
        raise NotImplementedError("Label not implemented")

    def children(self) -> List["Expr"]:
        return [self.inner]

    def simplify(self) -> "Expr":
        inner = self.inner.simplify()
        return self.__class__(inner)

    def __repr__(self) -> str:
        return _repr(self.inner, self._label)

    def latex(self) -> str:
        return "\\text{" + self._label + "}\\left(" + self.inner.latex() + "\\right)"

    @cast
    def evalf(self, subs: Dict[str, "Const"]):
        inner = self.inner.evalf(subs)
        # TODO: Support floats in .evalf
        # return Const(math.log(inner.value)) if isinstance(inner, Const) else Log(inner)
        return self.__class__(inner)
    
    def expand(self) -> Expr:
        assert self.inner.expandable(), f"Cannot expand {self}"
        return self.__class__(self.inner.expand())


def _repr(inner: Expr, label: str) -> str:
    inner_repr = inner.__repr__()
    if inner_repr[0] == "(" and inner_repr[-1] == ")":
        return f"{label}{inner_repr}"
    return f"{label}({inner_repr})"


class log(SingleFunc):
    inner: Expr
    base: Expr = e  # currently there isn't a way to set this yet from public api
    # if we make log a dataclass, the __repr__ no longer subclsases from SingleFun

    @property
    def _label(self):
        if self.base == e:
            return "ln"
        elif isinstance(self.base, Const) and self.base.value.denominator == 1:
            return "log" + self.base
        else:
            return f"log[self.base]"
        
    def latex(self) -> str:
        # Have informally tested this; does the job.
        if self.base == e:
            return "\\ln\\left( " + self.inner.latex() + " \\right)"
        else:
            return "\\log_{" + self.base.latex() + "}\\left( " + self.inner.latex() + " \\right)"

    def __new__(cls, inner: Expr, base: Expr = e):
        if inner == 1:
            return Const(0)
        if inner == base:
            return Const(1)
        if isinstance(inner, Power) and inner.base == base:
            return inner.exponent
        
        return super().__new__(cls)

    def simplify(self) -> Expr:
        inner = self.inner.simplify()
        
        # IDK if this should be in simplify or if it should be like expand, in a diff function
        # like you can move stuff together or move stuff apart
        if isinstance(inner, Power):
            return (log(inner.base) * inner.exponent).simplify()
        if isinstance(inner, Prod):
            return Sum([log(t) for t in inner.terms]).simplify()
        if isinstance(inner, Sum) and isinstance(inner.factor(), Prod):
            return Sum([log(t) for t in inner.factor().terms]).simplify()
        
        # let's agree on some standards
        # i dont love this, can change
        if isinstance(inner, (sec, csc, cot)):
            return -1 * log(inner.reciprocal_class(inner.inner)).simplify()

        return log(inner)

    def diff(self, var) -> Expr:
        return self.inner.diff(var) / self.inner


@cast
def sqrt(x: Expr) -> Expr:
    return x ** Const(Fraction(1, 2))


class classproperty:
    """python 3.11 no longer supports
    @classmethod
    @property

    so we make our own :)
    """
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls):
        return self.func(cls)


class staticproperty:
    """
    @staticmethod
    @property
    """
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls):
        return self.func()


TrigStr = Literal["sin", "cos", "tan", "sec", "csc", "cot"]

class TrigFunction(SingleFunc, ABC):
    _SPECIAL_KEYS = ["0", "1/6", "1/4", "1/3", "1/2", "2/3", "3/4", "5/6", "1", "7/6", "5/4", "4/3", "3/2", "5/3", "7/4", "11/6"]

    @abstractmethod
    @classproperty
    def func(cls) -> TrigStr:
        pass

    @abstractmethod
    @classproperty
    def is_inverse(cls) -> bool:
        pass

    @staticproperty
    def _double_simplification_dict() -> Dict[str, Callable[[Expr], Expr]]:
        # Making this a property instead of a class attribute or a global var because it requires computation.
        return {
            "sin acos": lambda x: sqrt(1 - x**2),
            "sin atan": lambda x: x / sqrt(1 + x**2),
            "cos asin": lambda x: sqrt(1 - x**2),  # same as sin acos
            "cos atan": lambda x: 1 / sqrt(1 + x**2),
            "tan asin": lambda x: x / sqrt(1 - x**2),
            "tan acos": lambda x: sqrt(1 - x**2) / x,
            # Arcsecant
            "sin asec": lambda x: sqrt(x**2 - 1) / x,  # Since sin(asec(x)) = sqrt(x^2 - 1) / x
            "tan asec": lambda x: sqrt(x**2 - 1),  # tan(asec(x)) = sqrt(x^2 - 1)
            # Arccosecant
            "cos acsc": lambda x: sqrt(1 - 1 / x**2),  # cos(acsc(x)) = sqrt(1 - 1/x^2)
            "tan acsc": lambda x: 1 / sqrt(x**2 - 1),  # tan(acsc(x)) = 1/sqrt(x^2 - 1)
            # Arccotangent
            "sin acot": lambda x: 1 / sqrt(1 + x**2),  # sin(acot(x)) = 1/sqrt(1 + x^2)
            "cos acot": lambda x: x / sqrt(1 + x**2),  # cos(acot(x)) = x/sqrt(1 + x^2)
        }

    @classproperty
    def reciprocal_class(cls) -> Type["TrigFunction"]:
        if cls.is_inverse:
            raise ValueError(f"Inverse trig function {cls.__name__} does not have a reciprocal")
        else:
            raise NotImplementedError(f"reciprocal_class not implemented for {cls.__name__}")

    @classproperty
    @abstractmethod
    def special_values(cls) -> Dict[str, Expr]:
        pass

    @property
    def _label(self) -> str:
        return f"{'a' if self.is_inverse else ''}{self.func}"

    def __new__(cls, inner: Expr) -> "Expr":
        # 1. Check if inner is a special value
        pi_coeff = inner / pi
        if isinstance(pi_coeff, Const):
            pi_coeff = pi_coeff % 2
            if str(pi_coeff.value) in cls._SPECIAL_KEYS:
                return cls.special_values[str(pi_coeff.value)]
        
        # 2. Check if inner is trigfunction
        # things like sin(cos(x)) cannot be more simplified.
        if isinstance(inner, TrigFunction) and inner.is_inverse != cls.is_inverse:
            # sin(asin(x)) -> x
            if inner.func == cls.func:
                return inner.inner

            # sin(acsc(x)) -> 1/x
            if inner.is_inverse and inner.func == cls.reciprocal_class.func or not inner.is_inverse and inner.reciprocal_class.func == cls.func:
                return (1 / inner.inner)

            if not cls.is_inverse:
                if cls.func in ["sin", "cos", "tan"]:
                    callable_ = cls._double_simplification_dict[
                        f"{cls.func} {inner._label}"
                    ]
                    return callable_(inner.inner)

                else:
                    callable_ = cls._double_simplification_dict[
                        f"{cls.reciprocal_class.__name__} {inner._label}"
                    ]
                    return (1 / callable_(inner.inner))

            # not supporting stuff like asin(cos(x)) sorry.

        return super().__new__(cls)


class sin(TrigFunction):
    func = "sin"

    @classproperty
    def reciprocal_class(cls):
        return csc

    @classproperty    
    def special_values(cls):
        return {
            "0": Const(0),
            "1/6": Const(Fraction(1,2)),
            "1/4": 1/sqrt(2),
            "1/3": sqrt(3)/2,
            "1/2": Const(1),
            "2/3": sqrt(3)/2,
            "3/4": 1/sqrt(2),
            "5/6": Const(Fraction(1,2)),
            "1": Const(0),
            "7/6": -Const(Fraction(1,2)),
            "5/4": -1/sqrt(2),
            "4/3": -sqrt(3)/2,
            "3/2": -Const(1),
            "5/3": -sqrt(3)/2,
            "7/4": -1/sqrt(2),
            "11/6": -Const(Fraction(1,2)),
        }

    def diff(self, var) -> Expr:
        return cos(self.inner) * self.inner.diff(var)
    
    def __new__(cls, inner: Expr) -> Expr:
        if isinstance(inner, Prod) and inner.is_subtraction:
            return -sin(-inner)
        return super().__new__(cls, inner)


class cos(TrigFunction):
    func = "cos"

    @classproperty
    def reciprocal_class(cls):
        return sec
    
    @classproperty
    def special_values(cls):
        return {
            "0": Const(1),
            "1/6": sqrt(3)/2,
            "1/4": 1/sqrt(2),
            "1/3": Const(Fraction(1,2)),
            "1/2": Const(0),
            "2/3": -Const(Fraction(1,2)),
            "3/4": -1/sqrt(2),
            "5/6": -sqrt(3)/2,
            "1": Const(-1),
            "7/6": -sqrt(3)/2,
            "5/4": -1/sqrt(2),
            "4/3": -Const(Fraction(1,2)),
            "3/2": -Const(0),
            "5/3": Const(Fraction(1,2)),
            "7/4": 1/sqrt(2),
            "11/6": sqrt(3)/2,
        }

    def diff(self, var: Symbol) -> Expr:
        return -sin(self.inner) * self.inner.diff(var)
    
    def __init__(self, inner):
        pass
    
    def __new__(cls, inner: Expr) -> "Expr":
        # bruh this is so complicated because doing cos(new, -inner) just sets inner as the original inner because of passing
        # the same args down to init.
        new = super().__new__(cls, inner)
        if not isinstance(new, cos):
            return new
        new.inner = inner
        if isinstance(inner, Prod) and inner.is_subtraction:
            new.inner = -inner
        return new


class tan(TrigFunction):
    func = "tan"

    @classproperty
    def reciprocal_class(cls):
        return cot
    
    def __new__(cls, inner: Expr) -> Expr:
        if isinstance(inner, Prod) and inner.is_subtraction:
            return -tan(-inner)
        return super().__new__(cls, inner)

    @classproperty
    def special_values(cls):
        return {k: sin.special_values[k] / cos.special_values[k] for k in cls._SPECIAL_KEYS}

    def diff(self, var) -> Expr:
        return sec(self.inner) ** 2 * self.inner.diff(var)


class csc(TrigFunction):
    func = "csc"
    reciprocal_class = sin

    @classproperty
    def special_values(cls):
        return {k: 1 / sin.special_values[k] for k in cls._SPECIAL_KEYS}

    def diff(self, var) -> Expr:
        return (1 / sin(self.inner)).diff(var)


class sec(TrigFunction):
    func = "sec"
    reciprocal_class = cos

    @classproperty
    def special_values(cls):
        return {k: 1 / cos.special_values[k] for k in cls._SPECIAL_KEYS}

    def diff(self, var) -> Expr:
        return sec(self.inner) * tan(self.inner) * self.inner.diff(var)


class cot(TrigFunction):
    reciprocal_class = tan
    func = "cot"

    @classproperty
    def special_values(cls):
        return {k: cos.special_values[k] / sin.special_values[k] for k in cls._SPECIAL_KEYS}

    def diff(self, var) -> Expr:
        return (1 / tan(self.inner)).diff(var)


class asin(TrigFunction):
    func = "sin"
    is_inverse = True

    def diff(self, var):
        return 1 / sqrt(1 - self.inner**2) * self.inner.diff(var)


class acos(TrigFunction):
    func = "cos"
    is_inverse = True

    def diff(self, var):
        return -1 / sqrt(1 - self.inner**2) * self.inner.diff(var)


class atan(TrigFunction):
    func = "tan"
    is_inverse = True

    def diff(self, var):
        return 1 / (1 + self.inner**2) * self.inner.diff(var)


def symbols(symbols: str) -> Union[Symbol, List[Symbol]]:
    symbols = [Symbol(name=s) for s in symbols.split(" ")]
    return symbols if len(symbols) > 1 else symbols[0]


@cast
def diff(expr: Expr, var: Symbol) -> Expr:
    if hasattr(expr, "diff"):
        return expr.diff(var)
    else:
        raise NotImplementedError(f"Differentiation of {expr} not implemented")


# Problem: can't move this to regex or use general_count because regex imports from this file
# hmmmmm

def remove_const_factor(expr: Expr) -> Expr:
    if isinstance(expr, Prod):
        new_terms = [t for t in expr.terms if len(t.symbols()) > 0]
        # simplifying manually because don't wanna invoke the recursive stuff by calling
        # simplify
        if len(new_terms) == 0:
            return Const(1)
        if len(new_terms) == 1:
            return new_terms[0]
        return Prod(new_terms)
    return expr
