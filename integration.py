# structures for integration w transforms

import random
import string
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

import linalg
from expr import (ArcCos, ArcSin, ArcTan, Const, Cos, Cot, Csc, Expr, Log,
                  Power, Prod, Sec, Sin, SingleFunc, Sum, Symbol, Tan,
                  TrigFunction, cast, contains_cls, count, deconstruct_power,
                  e, nesting, sqrt, symbols)
from polynomial import (Polynomial, polynomial_to_expr, rid_ending_zeros,
                        to_const_polynomial, to_polynomial)

ExprFn = Callable[[Expr], Expr]
Number = Union[Fraction, int]


@dataclass
class Node:
    expr: Expr
    var: Symbol  # variable that THIS EXPR is integrated by.
    transform: Optional["Transform"] = None  # the transform that led to this node
    parent: Optional["Node"] = None  # None for root node only
    children: Optional[List["Node"]] = (
        None  # smtn smtn setting it to [] by default causes errors
    )
    type: Literal["AND", "OR", "UNSET", "SOLUTION", "FAILURE"] = "UNSET"
    solution: Optional[Expr] = (
        None  # only for SOLUTION nodes (& their parents when we go backwards)
    )
    # failure = can't proceed forward.

    def __post_init__(self):
        self.expr = self.expr.simplify()
        if self.children is None:
            self.children = []

    def __repr__(self):
        num_children = len(self.children) if self.children else 0
        return f"Node({self.expr.__repr__()}, {self.var}, transform {self.transform.__class__.__name__}, {num_children} children, {self.type})"

    @property
    def leaves(self) -> List["Node"]:
        # Returns the leaves of the tree (all nodes without children)
        if not self.children:
            return [self]

        return [leaf for child in self.children for leaf in child.leaves]

    @property
    def unfinished_leaves(self) -> List["Node"]:
        # Leaves to work on :)
        return [leaf for leaf in self.leaves if not leaf.is_finished]

    @property
    def root(self) -> "Node":
        if not self.parent:
            return self
        return self.parent.root

    @property
    def distance_from_root(self) -> int:
        if not self.parent:
            return 0
        return 1 + self.parent.distance_from_root

    @property
    def is_solved(self) -> bool:
        # Returns True if all leaves WITH AND NODES are solved and all OR nodes are solved
        # if limit is reached, return False
        if self.type == "SOLUTION":
            return True

        if not self.children:
            return False

        if self.type == "AND" or self.type == "UNSET":
            # UNSET should only have one child.
            return all([child.is_solved for child in self.children])

        if self.type == "OR":
            return any([child.is_solved for child in self.children])

    @property
    def is_failed(self) -> bool:
        # Is not solveable if one branch is not solveable and it has no "OR"s

        if self.type == "FAILURE":
            return True

        if not self.children:
            # if it has no children and it's not "FAILURE", it means this node is an unfinished leaf (or a solution).
            return False

        if self.type == "OR":
            return all([child.is_failed for child in self.children])

        return any([child.is_failed for child in self.children])

    @property
    def is_finished(self) -> bool:
        return self.is_solved or self.is_failed

    @property
    def unfinished_children(self) -> List["Node"]:
        if not self.children:
            return []
        return [child for child in self.children if not child.is_finished]

    @property
    def child(self) -> Optional["Node"]:
        # this is only here because when debugging im lazy and want to type node.child instead of node.children[0]
        if not self.children:
            return None
        return self.children[0]


class Transform(ABC):
    "An integral transform -- base class"
    # forward and backward modify the nodetree directly. check is a pure function

    def __init__(self):
        pass

    @abstractmethod
    def forward(self, node: Node) -> None:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def backward(self, node: Node) -> None:
        if not node.solution:
            raise ValueError("Node has no solution")

    @abstractmethod
    def check(self, node: Node) -> bool:
        raise NotImplementedError("Not implemented")


class SafeTransform(Transform, ABC):
    pass


class PullConstant(SafeTransform):
    _constant: Expr = None
    _non_constant_part: Expr = None

    def check(self, node: Node) -> bool:
        expr = node.expr
        var = node.var
        if isinstance(expr, Prod):
            # if there is a constant, pull it out
            # # or if there is a symbol that's not the variable, pull it out
            for i, term in enumerate(expr.terms):
                if var not in term.symbols():
                    self._constant = term
                    self._non_constant_part = Prod(
                        expr.terms[:i] + expr.terms[i + 1 :]
                    ).simplify()
                    return True

        return False

    def forward(self, node: Node):
        node.children = [Node(self._non_constant_part, node.var, self, node)]

    def backward(self, node: Node) -> None:
        super().backward(node)
        node.parent.solution = (self._constant * node.solution).simplify()


class PolynomialDivision(SafeTransform):
    _numerator: Polynomial = None
    _denominator: Polynomial = None

    def check(self, node: Node) -> bool:
        # This is so messy we can honestly just do catching in the `to_polynomial`
        expr = node.expr
        # currently we don't support division of polynomials with multiple variables
        if expr.symbols() != [node.var]:
            return False
        if not isinstance(expr, Prod):
            return False

        ## Don't contain any SingleFunc with inner containing var
        def _contains_singlefunc_w_inner(expr: Expr) -> bool:
            if isinstance(expr, SingleFunc) and expr.inner.contains(node.var):
                return True

            return any([_contains_singlefunc_w_inner(e) for e in expr.children()])

        if _contains_singlefunc_w_inner(expr):
            return False

        ## Make sure each factor is a polynomial
        for factor in expr.terms:

            def _is_polynomial(expression: Expr):
                if isinstance(expression, Power):
                    if not (
                        isinstance(expression.exponent, Const)
                        and expression.exponent.value.denominator == 1
                    ):
                        return False
                    return True
                if isinstance(expression, Const) or isinstance(expression, Symbol):
                    return True

                if isinstance(expression, Sum):
                    return all([_is_polynomial(term) for term in expression.terms])

                if isinstance(expression, Prod):
                    # You'd need this if it's a term in a sum
                    return isinstance((expression / node.var).simplify(), Const)

                raise NotImplementedError(f"Not implemented: {expression}")

            if not _is_polynomial(factor):
                return False

        ## Make sure numerator and denominator are good
        numerator = Const(1)
        denominator = Const(1)
        for factor in expr.terms:
            b, x = deconstruct_power(factor)
            if x.value > 0:
                numerator *= factor
            else:
                denominator *= Power(b, -x).simplify()

        numerator = numerator.simplify()
        if denominator == 1:
            # there is nothing to divide. this is not a division.
            return False
        denominator = denominator.simplify()
        try:
            numerator_list = to_polynomial(numerator, node.var)
            denominator_list = to_polynomial(denominator, node.var)
        except AssertionError:
            return False

        if len(numerator_list) < len(denominator_list):
            return False

        self._numerator = numerator_list
        self._denominator = denominator_list
        return True

    def forward(self, node: Node):
        var = node.var
        quotient = np.zeros(len(self._numerator) - len(self._denominator) + 1)

        while self._numerator.size >= self._denominator.size:
            quotient_degree = len(self._numerator) - len(self._denominator)
            quotient_coeff = self._numerator[-1] / self._denominator[-1]
            quotient[quotient_degree] = quotient_coeff
            self._numerator -= np.concatenate(
                ([0] * quotient_degree, self._denominator * quotient_coeff)
            )
            self._numerator = rid_ending_zeros(self._numerator)

        remainder = polynomial_to_expr(self._numerator, var) / polynomial_to_expr(
            self._denominator, var
        )
        quotient_expr = polynomial_to_expr(quotient, var)
        answer = (quotient_expr + remainder).simplify()
        node.children = [Node(answer, var, self, node)]

    def backward(self, node: Node) -> None:
        super().backward(node)
        node.parent.solution = node.solution


class Expand(SafeTransform):
    def forward(self, node: Node):
        node.children = [Node(node.expr.expand(), node.var, self, node)]

    def check(self, node: Node) -> bool:
        return node.expr.expandable()

    def backward(self, node: Node) -> None:
        super().backward(node)
        node.parent.solution = node.solution


class Additivity(SafeTransform):
    def forward(self, node: Node):
        node.type = "AND"
        node.children = [Node(e, node.var, self, node) for e in node.expr.terms]

    def check(self, node: Node) -> bool:
        return isinstance(node.expr, Sum)

    def backward(self, node: Node) -> None:
        super().backward(node)

        # For this to work, we must have a solution for each sibling.
        if not all([child.solution for child in node.parent.children]):
            raise ValueError(f"Additivity backward for {node} failed")

        node.parent.solution = Sum(
            [child.solution for child in node.parent.children]
        ).simplify()


def _get_last_heuristic_transform(node: Node):
    if isinstance(node.transform, (PullConstant, Expand, Additivity)):
        # We'll let polynomial division go because it changes things sufficiently that
        # we actually sorta make progress towards the integral.
        # PullConstant and Additivity are like fake, they dont make any substantial changes.
        # Expand is also like, idk, if we do A and then expand we dont rlly wanna do A again.

        # Alternatively, we could just make sure that the last transform didnt have the same
        # key. (but no the lecture example has B tan then polydiv then C tan)

        # Idk this thing rn is a lil messy and there might be a better way to do it.
        return _get_last_heuristic_transform(node.parent)
    return node.transform


# Let's just add all the transforms we've used for now.
# and we will make this shit good and generalized later.
class TrigUSub2(Transform):
    """
    u-sub of a trig function
    this is the weird u-sub where if u=sinx, dx != du/cosx but dx = du/sqrt(1-u^2)
    ex: integral of f(tanx) -> integral of f(u) / 1 + y^2, sub u = tanx
    -> dx = du/(1+x^2)
    """

    _variable_change = None

    _key: str = None
    # {label: trigfn class, derivative of inverse trigfunction}
    _table: Dict[str, Tuple[ExprFn, ExprFn]] = {
        "sin": (Sin, lambda var: 1 / sqrt(1 - var**2)),  # Asin(x).diff(x)
        "cos": (Cos, lambda var: -1 / sqrt(1 - var**2)),
        "tan": (Tan, lambda var: 1 / (1 + var**2)),
    }

    def forward(self, node: Node):
        intermediate = generate_intermediate_var()
        expr = node.expr
        # y = tanx
        cls, dy_dx = self._table[self._key]
        new_integrand = replace(expr, cls(node.var), intermediate) * dy_dx(intermediate)
        new_node = Node(new_integrand, intermediate, self, node)
        node.children.append(new_node)

        self._variable_change = cls(node.var)

    def check(self, node: Node) -> bool:
        # Since B and C essentially undo each other, we want to make sure that the last
        # heuristic transform wasn't C.

        t = _get_last_heuristic_transform(node)
        if isinstance(t, InverseTrigUSub):
            return False

        for k, v in self._table.items():
            cls, dy_dx = v
            count_ = count(node.expr, cls(node.var))
            if count_ >= 1 and count_ == count(node.expr, node.var):
                self._key = k
                return True

        return False

    def backward(self, node: Node) -> None:
        super().backward(node)
        node.parent.solution = replace(
            node.solution, node.var, self._variable_change
        ).simplify()


class RewriteTrig(Transform):
    """Rewrites trig functions in terms of (sin, cos), (tan, sec), and (cot, csc)"""

    def forward(self, node: Node):
        expr = node.expr
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

        stuff = [r1, r2, r3]
        for thing in stuff:
            if thing.__repr__() == expr.__repr__():
                stuff.remove(thing)
        node.children += [Node(option, node.var, self, node) for option in stuff]
        node.type = "OR"

    def check(self, node: Node) -> bool:
        # make sure that this node didn't get here by this transform
        t = _get_last_heuristic_transform(node)
        if isinstance(t, RewriteTrig):
            return False

        expr = node.expr
        return contains_cls(expr, TrigFunction)

    def backward(self, node: Node) -> None:
        super().backward(node)
        node.parent.solution = node.solution


class InverseTrigUSub(Transform):
    _key = None
    _variable_change = None

    # {label: class, search query, dy_dx, variable_change}
    _table: Dict[str, Tuple[ExprFn, Callable[[str], str], ExprFn, ExprFn]] = {
        "sin": (Sin, lambda name: f"1 - {name}^2", lambda var: Cos(var), ArcSin),
        "tan": (Tan, lambda name: f"1 + {name}^2", lambda var: Sec(var) ** 2, ArcTan),
    }

    def forward(self, node: Node):
        intermediate = generate_intermediate_var()
        cls, q, dy_dx, var_change = self._table[self._key]
        dy_dx = dy_dx(intermediate)
        new_thing = (replace(node.expr, node.var, cls(intermediate)) * dy_dx).simplify()
        node.children.append(Node(new_thing, intermediate, self, node))

        self._variable_change = var_change(node.var)

    def check(self, node: Node) -> bool:
        t = _get_last_heuristic_transform(node)
        if isinstance(t, TrigUSub2):
            # If it just went through B, C is guaranteed to have a match.
            # going through C will just undo B.
            return False

        for k, v in self._table.items():
            query = v[1]
            if query(node.var.name) in node.expr.__repr__():
                self._key = k
                return True

        return False

    def backward(self, node: Node) -> None:
        super().backward(node)
        node.parent.solution = replace(
            node.solution, node.var, self._variable_change
        ).simplify()


class PolynomialUSub(Transform):
    """check that x^n-1 is a term and every other instance of x is x^n
    you're gonna replace u=x^n
    ex: x/sqrt(1-x^2)
    """

    _variable_change = None  # x^n

    def check(self, node: Node) -> bool:
        if not isinstance(node.expr, Prod):
            return False

        rest: Expr = None
        n = None
        for i, term in enumerate(node.expr.terms):
            # yes you can assume it's an expanded simplified product. so no terms
            # are Prod or Sum.
            # so x^n-1 must be exactly a term with no fluff. :)
            if (
                isinstance(term, Power)
                and term.base == node.var
                and not term.exponent.contains(node.var)
            ):
                n = (term.exponent + 1).simplify()
                rest = Prod(node.expr.terms[:i] + node.expr.terms[i + 1 :]).simplify()
                break

            if term == node.var:
                n = 2
                rest = Prod(node.expr.terms[:i] + node.expr.terms[i + 1 :]).simplify()
                break

        if n is None:
            return False
        if n == 0:
            # How are you gonna sub u = x^0 = 1, du = 0 dx
            return False

        self._variable_change = Power(node.var, n).simplify()  # x^n
        count_ = count(node.expr, self._variable_change)
        return count_ > 0 and count_ == count(rest, node.var)

    def forward(self, node: Node) -> None:
        intermediate = generate_intermediate_var()
        dx_dy = self._variable_change.diff(node.var)
        new_integrand = replace(node.expr, self._variable_change, intermediate) / dx_dy
        new_integrand = new_integrand.simplify()
        node.children.append(Node(new_integrand, intermediate, self, node))

    def backward(self, node: Node) -> None:
        super().backward(node)
        node.parent.solution = replace(
            node.solution, node.var, self._variable_change
        ).simplify()


class LinearUSub(Transform):
    """u-substitution for smtn like f(ax+b)
    u = ax+b
    du = a dx
    \int f(ax+b) dx = 1/a \int f(u) du
    """

    _variable_change: Expr = None

    def check(self, node: Node) -> bool:
        if count(node.expr, node.var) < 1: # just to cover our bases bc the later thing assume it appears >=1 
            return False

        def _is_a_linear_sum_or_prod(expr: Expr) -> bool:
            if isinstance(expr, Sum):
                return all(
                    [
                        not c.contains(node.var)
                        or not (c / node.var).simplify().contains(node.var)
                        for c in expr.terms
                    ]
                )
            if isinstance(expr, Prod):
                # Is a constant multiple of var
                return not (expr / node.var).simplify().contains(node.var)
            return False

        def _check(e: Expr) -> bool:
            if not e.contains(node.var):
                return True
            if _is_a_linear_sum_or_prod(e):
                if self._variable_change is not None:
                    return e == self._variable_change
                self._variable_change = e
                return True
            if not e.children():
                # This must mean that we contain the var and it has no children, which means that we are the var.
                # have to do this because all([]) = True. Covering my bases.
                return False
            else:
                return all([_check(child) for child in e.children()])

        return _check(node.expr)

    def forward(self, node: Node) -> None:
        intermediate = generate_intermediate_var()
        du_dx = self._variable_change.diff(node.var)
        new_integrand = replace(node.expr, self._variable_change, intermediate) / du_dx
        new_integrand = new_integrand.simplify()
        node.children.append(Node(new_integrand, intermediate, self, node))

    def backward(self, node: Node) -> None:
        super().backward(node)
        node.parent.solution = replace(
            node.solution, node.var, self._variable_change
        ).simplify()


class CompoundAngle(Transform):
    """Compound angle formulae"""

    def check(self, node: Node) -> bool:
        def _check(e: Expr) -> bool:
            if (
                isinstance(e, (Sin, Cos))
                and isinstance(e.inner, Sum)
                and len(e.inner.terms) == 2  # for now lets j do 2 terms
            ):
                return True
            else:
                return any([_check(child) for child in e.children()])

        return _check(node.expr)

    def forward(self, node: Node) -> None:
        condition = (
            lambda expr: isinstance(expr, (Sin, Cos))
            and isinstance(expr.inner, Sum)
            and len(expr.inner.terms) == 2
        )

        def _perform(expr: Union[Sin, Cos]) -> Expr:
            a, b = expr.inner.terms
            if isinstance(expr, Sin):
                return Sin(a) * Cos(b) + Cos(a) * Sin(b)
            elif isinstance(expr, Cos):
                return Cos(a) * Cos(b) - Sin(a) * Sin(b)

        new_integrand = _replace_factory(condition, _perform)(node.expr)

        node.children.append(Node(new_integrand, node.var, self, node))

    def backward(self, node: Node) -> None:
        super().backward(node)
        node.parent.solution = node.solution


class SinUSub(Transform):
    """u-substitution for if sinx cosx exists in the outer product"""

    # TODO: generalize this in some form? to other trig fns maybe?
    # - generalize to if the sin is in a power but the cos is under no power.
    # like transform D but for trigfns
    _sin: Sin = None
    _cos: Cos = None
    _variable_change: Expr = None

    def check(self, node: Node) -> bool:
        if not isinstance(node.expr, Prod):
            return False

        def is_constant_product_of_var(expr, var):
            if expr == var:
                return True
            if not (expr / var).simplify().contains(var):
                return True
            return False

        # sins = [term.inner for term in node.expr.terms if isinstance(term, Sin)]
        # coses = [term.inner for term in node.expr.terms if isinstance(term, Cos)]
        sins: List[Sin] = []
        coses: List[Cos] = []
        for term in node.expr.terms:
            if isinstance(term, Sin):
                if not is_constant_product_of_var(term.inner, node.var):
                    continue

                sins.append(term)

                for cos in coses:
                    if term.inner == cos.inner:
                        self._sin = term
                        self._cos = cos
                        return True

            if isinstance(term, Cos):
                if not is_constant_product_of_var(term.inner, node.var):
                    continue

                coses.append(term)

                for sin in sins:
                    if term.inner == sin.inner:
                        self._sin = sin
                        self._cos = term
                        return True

        return False

    def forward(self, node: Node) -> None:
        intermediate = generate_intermediate_var()
        dy_dx = self._sin.diff(node.var)
        new_integrand = (
            replace(
                node.expr,
                self._sin,
                intermediate,
            )
            / dy_dx
        )
        node.children.append(Node(new_integrand, intermediate, self, node))
        self._variable_change = self._sin

    def backward(self, node: Node) -> None:
        super().backward(node)
        node.parent.solution = replace(
            node.solution, node.var, self._variable_change
        ).simplify()


class ProductToSum(Transform):
    """product to sum identities for sin & cos"""

    _a: Expr = None
    _b: Expr = None

    def check(self, node: Node) -> bool:
        if isinstance(node.expr, Prod):
            if len(node.expr.terms) == 2 and all(
                isinstance(term, (Sin, Cos)) for term in node.expr.terms
            ):
                self._a, self._b = node.expr.terms
                return True

        if isinstance(node.expr, Power):
            if isinstance(node.expr.base, (Sin, Cos)) and node.expr.exponent == 2:
                self._a = self._b = node.expr.base
                return True

        return False

    def forward(self, node: Node) -> None:
        a, b = self._a, self._b
        if isinstance(a, Sin) and isinstance(b, Cos):
            temp = Sin(a.inner + b.inner) + Sin(a.inner - b.inner)
        elif isinstance(a, Cos) and isinstance(b, Sin):
            temp = Sin(a.inner + b.inner) - Sin(a.inner - b.inner)
        elif isinstance(a, Cos) and isinstance(b, Cos):
            temp = Cos(a.inner + b.inner) + Cos(a.inner - b.inner)
        elif isinstance(a, Sin) and isinstance(b, Sin):
            temp = Cos(a.inner - b.inner) - Cos(a.inner + b.inner)

        new_integrand = temp / 2
        node.children.append(Node(new_integrand, node.var, self, node))

    def backward(self, node: Node) -> None:
        super().backward(node)
        node.parent.solution = node.solution


class ByParts(Transform):
    _stuff: List[Tuple[Expr, Expr, Expr, Expr]] = None

    """Integration by parts"""

    def __init__(self):
        self._stuff = []

    def check(self, node: Node) -> bool:
        if not isinstance(node.expr, Prod):
            if isinstance(node.expr, SingleFunc) and node.expr.inner == node.var: # Special case for Log, ArcSin, etc.
                # TODO universalize it more?? how to make it more universal without making inf loops?
                dv = 1
                v = node.var
                u = node.expr
                du = u.diff(node.var)
                self._stuff.append((u, du, v, dv))
                return True
            return False
        if not len(node.expr.terms) == 2:
            return False

        def _check(u: Expr, dv: Expr) -> bool:
            du = u.diff(node.var)
            v = _check_if_solveable(dv, node.var)
            if v is None:
                return False

            self._stuff.append((u, du, v, dv))
            return True

        a, b = node.expr.terms
        return _check(a, b) or _check(b, a)

    def forward(self, node: Node) -> None:
        # This is tricky bc you have 2 layers of children here.
        for u, du, v, dv in self._stuff:
            child1 = (u * v).simplify()
            integrand2 = (du * v * -1).simplify()

            tr = ByParts()
            tr._stuff = [(u, du, v, dv)]

            ### special case where you can jump directly to the solution wheeee
            factor = (integrand2 / node.expr).simplify()
            if not factor.contains(node.var):
                solution = child1 / (1 - factor)
                node.children.append(Node(node.expr, node.var, tr, node, type="SOLUTION", solution=solution))
                return
            ### 
            
            funky_node = Node(node.expr, node.var, tr, node, type="AND")
            funky_node.children = [
                Node(
                    node.expr,
                    node.var,
                    Additivity(),
                    funky_node,
                    type="SOLUTION",
                    solution=child1,
                ),
                Node(integrand2, node.var, Additivity(), funky_node),
            ]
            node.children.append(funky_node)

    def backward(self, node: Node) -> None:
        super().backward(node)
        node.parent.solution = node.solution


class PartialFractions(Transform):
    _new_integrand: Expr = None
    def check(self, node: Node) -> bool:
        # First, make sure that this is a fraction
        if not isinstance(node.expr, Prod):
            return False
        num, denom = node.expr.numerator_denominator
        if denom == Const(1):
            return False
        
        # and that both numerator and denominator are polynomials
        try:
            numerator_list = to_const_polynomial(num, node.var)
            denominator_list = to_const_polynomial(denom, node.var)
        except AssertionError:
            return False
        
        # numerator has to be a smaller order than the denom
        if len(numerator_list) > len(denominator_list):
            return False
        
        # The denominator has to be a product
        if not isinstance(denom, (Prod, Sum)):
            return False
        if isinstance(denom, Sum):
            new = denom.factor()
            if not isinstance(new, Prod):
                return False
            denom = new

        # ok im stupid so im gonna only do the case for 2 factors for now
        # shouldnt be hard to generalize
        if len(denom.terms) != 2:
            return False
        
        d1, d2 = denom.terms

        # Make sure that it's not the case that one of the denominator factors is just a constant.
        if not (d1.contains(node.var) and d2.contains(node.var)):
            return False

        d1_list = to_const_polynomial(d1, node.var)
        d2_list = to_const_polynomial(d2, node.var)

        matrix = np.array([d2_list, d1_list]).T
        inv = linalg.invert(matrix)
        if inv is None:
            return False
        ans = inv @ numerator_list
        self._new_integrand = ans[0] / d1 + ans[1] / d2
        return True
    
    def forward(self, node: Node) -> None:
        node.children.append(Node(self._new_integrand, node.var, self, node))
    
    def backward(self, node: Node) -> None:
        super().backward(node)
        node.parent.solution = node.solution


def _replace_factory(condition: Callable[[Expr], bool], perform: ExprFn) -> ExprFn:
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

    return _replace


# Leave RewriteTrig, InverseTrigUSub near the end bc they are deprioritized
# and more fucky
HEURISTICS = [
    PolynomialUSub,
    LinearUSub,
    CompoundAngle,
    SinUSub,
    ProductToSum,
    TrigUSub2,
    ByParts,
    RewriteTrig,
    InverseTrigUSub,
]
SAFE_TRANSFORMS = [Additivity, PullConstant, Expand, PolynomialDivision, PartialFractions]

STANDARD_TRIG_INTEGRALS = {
    "sin(x)": lambda x: -Cos(x),
    "cos(x)": Sin,
    "sec(x)^2": Tan, # Integration calculator says this is a standard integral. + it doesn't get solved on simpy if i dont add this.
    "sec(x)": lambda x: Log(Tan(x) + Sec(x)) # not a standard integral but it's fucked so im leaving it (unless?)
}

def _check_if_solveable(integrand: Expr, var: Symbol) -> Optional[Expr]:
    if not integrand.contains(var):
        return integrand * var
    if isinstance(integrand, Power):
        if integrand.base == var and isinstance(integrand.exponent, Const):
            n = integrand.exponent
            return (1 / (n + 1)) * Power(var, n + 1) if n != -1 else Log(integrand.base)
        if (
            integrand.base == e
            and not (integrand.exponent / var).simplify().contains(var)
            and not integrand.base.contains(var)
        ):
            return (var / integrand.exponent) * integrand
    if isinstance(integrand, Symbol) and integrand == var:
        return Fraction(1 / 2) * Power(var, 2)
    if repr(integrand) in STANDARD_TRIG_INTEGRALS:
        # jank but i give up on life
        return STANDARD_TRIG_INTEGRALS[repr(replace(integrand, var, Symbol("x")))](var)
    
    return None

def _check_if_node_solvable(node: Node):
    answer = _check_if_solveable(node.expr, node.var)
    if answer is None:
        return

    node.type = "SOLUTION"
    node.solution = answer.simplify()


def _cycle(node: Node) -> Optional[Union[Node, Literal["SOLVED"]]]:
    # 1. APPLY ALL SAFE TRANSFORMS
    _integrate_safely(node)

    # now we have a tree with all the safe transforms applied
    # 2. LOOK IN TABLE
    for leaf in node.unfinished_leaves:
        _check_if_node_solvable(leaf)

    if len(node.unfinished_leaves) == 0:
        return "SOLVED"

    # 3. APPLY HEURISTICS
    next_node = node.unfinished_leaves[0]  # random lol
    _integrate_heuristically(next_node)

    next_next_node = _get_next_node_post_heuristic(next_node)
    return next_next_node


def _get_next_node_post_heuristic(node: Node) -> Node:

    if len(node.unfinished_leaves) == 0:
        if node.is_failed:
            # if on the first cycle, there are zero safe transforms AND no heuristics then
            # node here would be the root.
            if node.parent is None:
                return None

            # we want to go back and solve the parent
            parent = node.parent
            while len(parent.children) == 1 or parent.type == "AND":
                if parent.parent is None:
                    # we've reached root.
                    # this means... we can't solve this integral.
                    return None
                parent = parent.parent
            # now parent is the lowest OR node with multiple children
            return _get_next_node_post_heuristic(parent)
        else:
            # This happens when we use integration by parts and the heuristic finds a whole ass solution
            return "SOLVED"

    if len(node.unfinished_leaves) == 1:
        return node.unfinished_leaves[0]

    if len(node.unfinished_leaves) > 1:
        return _nesting_node(node)


# a recursive function.
# find the simplest problem to work on that makes progress.
def _nesting_node(node: Node) -> Node:
    if len(node.unfinished_children) == 1:
        return _nesting_node(node.unfinished_children[0])

    if len(node.unfinished_children) == 0:
        return node  # base case ???
        raise ValueError("nesting_node on a solved node?")

    is_2nd_lowest_parent = all(
        [not child.unfinished_children for child in node.unfinished_children]
    )
    fn = min if node.type == "OR" else max
    if is_2nd_lowest_parent:
        return _get_node_with_best_nesting(node.unfinished_children, fn)

    childrens_best_nodes = [_nesting_node(c) for c in node.unfinished_children]
    return _get_node_with_best_nesting(childrens_best_nodes, fn)


def _get_node_with_best_nesting(
    nodes: List[Node], fn: Callable[[List[Node]], Node]
) -> Node:
    results = [nesting(node.expr, node.var) for node in nodes]
    best_value = fn(results)
    return nodes[results.index(best_value)]


class Integration:
    """
    Keeps track of integration work as we go
    """

    def _integrate_bounds(
        expr: Expr, bounds: Tuple[Symbol, Const, Const], verbose: bool
    ) -> Expr:
        x, a, b = bounds
        integral = Integration._integrate(expr, bounds[0], verbose)
        return (integral.evalf({x.name: b}) - integral.evalf({x.name: a})).simplify()

    @cast
    @staticmethod
    def integrate(
        expr: Expr,
        bounds: Union[Symbol, Tuple[Symbol, Const, Const]],
        verbose: bool = False,
    ) -> Expr:
        """Returns None if the integral cannot be solved."""
        if type(bounds) == tuple:
            return Integration._integrate_bounds(expr, bounds, verbose)
        else:
            return Integration._integrate(expr, bounds, verbose)

    @staticmethod
    def _integrate(
        integrand: Expr, var: Symbol, verbose: bool = False 
    ) -> Expr:
        root = Node(integrand, var)
        curr_node = root
        while True:
            answer = _cycle(curr_node)

            if root.is_finished:
                break

            if answer == "SOLVED":
                # just do any other thing in root
                curr_node = _get_next_node_post_heuristic(root)
            else:
                curr_node = answer

        if root.is_failed:
            breakpoint()
            warnings.warn(f"Failed to integrate {integrand} wrt {var}")
            return None

        # now we have a solved tree or a failed tree
        # we can go back and get the answer
        solved_leaves = [leaf for leaf in root.leaves if leaf.is_solved]
        for leaf in solved_leaves:
            # GO backwards on each leaf until it errors out, then go backwards on the next leaf.
            curr = leaf
            while True:
                if curr.parent is None:  # is root
                    break
                try:
                    curr.transform.backward(curr)
                    curr = curr.parent
                except ValueError:
                    break

        if root.solution is None:
            raise ValueError("something went wrong while going backwards...")

        if verbose:
            _print_success_tree(root)
        return root.solution


def _integrate_safely(node: Node):
    for transform in SAFE_TRANSFORMS:
        tr = transform()
        if tr.check(node):
            tr.forward(node)
            break
    for child in node.children:
        _integrate_safely(child)


def _integrate_heuristically(node: Node):
    for transform in HEURISTICS:
        tr = transform()
        if tr.check(node):
            tr.forward(node)

    if not node.children:
        node.type = "FAILURE"
        return

    if len(node.children) > 1:
        node.type = "OR"


def _print_tree(root: Node) -> None:
    print(f"[{root.distance_from_root}] {root.expr} ({root.transform.__class__.__name__})")
    if not root.children:
        print(root.type)
        print("")
        return
    for child in root.children:
        _print_tree(child)


def _print_success_tree(root: Node) -> None:
    if not root.is_solved:
        return
    print(f"[{root.distance_from_root}] {root.expr} ({root.transform.__class__.__name__})")
    if not root.children:
        print("")
        return
    for child in root.children:
        _print_success_tree(child)


def random_id(length):
    # Define the pool of characters you can choose from
    characters = string.ascii_letters + string.digits
    # Use random.choices() to pick characters at random, then join them into a string
    random_string = "".join(random.choices(characters, k=length))
    return random_string


def generate_intermediate_var() -> Symbol:
    return symbols(f"u_{random_id(10)}")


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
