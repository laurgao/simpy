# structures for integration w transforms

import random
import string
from abc import ABC, abstractmethod
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np

from .expr import (Const, Expr, Power, Prod, Sum, Symbol, TrigFunction, asin,
                   atan, cos, cot, csc, log, remove_const_factor, sec, sin,
                   sqrt, symbols, tan)
from .linalg import invert
from .polynomial import (Polynomial, polynomial_to_expr, rid_ending_zeros,
                         to_const_polynomial)
from .regex import (count, general_count, replace, replace_class,
                    replace_factory)
from .utils import ExprFn, random_id

Number_ = Union[Fraction, int]


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
        """Subclasses must call super().backward(...) before doing its own logic."""
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

        ## Make sure numerator and denominator are both polynomials
        numerator, denominator = expr.numerator_denominator
        if denominator == 1:
            return False
        try:
            numerator_list = to_const_polynomial(numerator, node.var)
            denominator_list = to_const_polynomial(denominator, node.var)
        except AssertionError:
            return False

        # You can divide if they're same order
        if len(numerator_list) < len(denominator_list):
            return False

        self._numerator = numerator_list
        self._denominator = denominator_list
        return True

    def forward(self, node: Node):
        var = node.var
        quotient = [Const(0)] * (len(self._numerator) - len(self._denominator) + 1)
        quotient = np.array(quotient)

        while self._numerator.size >= self._denominator.size:
            quotient_degree = len(self._numerator) - len(self._denominator)
            quotient_coeff = (self._numerator[-1] / self._denominator[-1]).simplify()
            quotient[quotient_degree] = quotient_coeff
            self._numerator -= np.concatenate(
                ([Const(0)] * quotient_degree, self._denominator * quotient_coeff)
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
        node.children.append(Node(node.expr.expand(), node.var, self, node))

    def check(self, node: Node) -> bool:
        # If the last heuristic was completing the square,
        # Please give it a break.
        t = _get_last_heuristic_transform(node)
        if isinstance(t, CompleteTheSquare):
            return False

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
    if isinstance(node.transform, (PullConstant, Additivity)):
        # We'll let polynomial division go because it changes things sufficiently that
        # we actually sorta make progress towards the integral.
        # PullConstant and Additivity are like fake, they dont make any substantial changes.
        # Expand is also like, idk, if we do A and then expand we dont rlly wanna do A again.

        # Alternatively, we could just make sure that the last transform didnt have the same
        # key. (but no the lecture example has B tan then polydiv then C tan)

        # Idk this thing rn is a lil messy and there might be a better way to do it.

        # 05/13/2024: no more expand bc it's not even a "safe transform" anymore lwk.
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
        "sin": (sin, lambda var: 1 / sqrt(1 - var**2)),  # Asin(x).diff(x)
        "cos": (cos, lambda var: -1 / sqrt(1 - var**2)),
        "tan": (tan, lambda var: 1 / (1 + var**2)),
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
            [tan, csc, cot, sec],
            [
                lambda x: sin(x) / cos(x),
                lambda x: 1 / sin(x),
                lambda x: cos(x) / sin(x),
                lambda x: 1 / cos(x),
            ],
        ).simplify()
        r2 = replace_class(
            expr,
            [sin, cos, cot, sec],
            [
                lambda x: 1 / csc(x),
                lambda x: 1 / tan(x) / csc(x),
                lambda x: 1 / tan(x),
                lambda x: tan(x) * csc(x),
            ],
        ).simplify()
        r3 = replace_class(
            expr,
            [sin, cos, tan, csc],
            [
                lambda x: 1 / cot(x) / sec(x),
                lambda x: 1 / sec(x),
                lambda x: 1 / cot(x),
                lambda x: cot(x) * sec(x),
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
        return expr.has(TrigFunction)

    def backward(self, node: Node) -> None:
        super().backward(node)
        node.parent.solution = node.solution


class InverseTrigUSub(Transform):
    _key = None
    _variable_change = None

    # {label: class, search query, dy_dx, variable_change}
    _table: Dict[str, Tuple[ExprFn, Callable[[str], str], ExprFn, ExprFn]] = {
        "sin": (sin, lambda symbol: 1 - symbol ** 2, lambda var: cos(var), asin),
        "tan": (tan, lambda symbol: 1 + symbol ** 2, lambda var: sec(var) ** 2, atan),
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
            query = v[1](node.var)
            if count(node.expr, query) > 0 or count(node.expr, (query * -1).simplify()) > 0:
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

    _variable_change: Expr = None # writes u in terms of x
    _inverse_var_change: ExprFn = None # this will write x in terms of u

    def check(self, node: Node) -> bool:
        if count(node.expr, node.var) < 1: # just to cover our bases bc the later thing assume it appears >=1 
            return False

        def _is_a_linear_sum_or_prod(expr: Expr) -> Optional[Tuple[Expr, Optional[ExprFn]]]:
            """Returns what u should be in terms of x. & None if it's not.
            """
            if isinstance(expr, Sum):
                if all(
                    [
                        not c.contains(node.var)
                        or not (c / node.var).simplify().contains(node.var)
                        for c in expr.terms
                    ]
                ):
                    return expr, None
            if isinstance(expr, Prod):
                # Is a constant multiple of var
                if not (expr / node.var).simplify().contains(node.var):
                    return expr, None
                # or is constant multiple of var**const
                # ex: if we have x**2/36, we want to rewrite it as (x/6)**2 and sub u=x/6
                # if we have sth like (x+3)**2/36 we will just let the sum catch it and do 2 linearusubs in sequence :shrug:
                if expr.is_subtraction:
                    expr = (-expr).simplify()
                xyz = remove_const_factor(expr)
                is_const_multiple_of_power = isinstance(xyz, Power) and xyz.base == node.var and not xyz.exponent.contains(node.var)
                if not is_const_multiple_of_power:
                    return None
                coeff = (expr / xyz).simplify()
                if coeff == Const(1):
                    return None
                coeff_abs = coeff.abs() if isinstance(coeff, Const) else coeff
                inner_coeff = Power(coeff_abs, 1 / xyz.exponent).simplify()
                return inner_coeff * node.var, lambda u: u / inner_coeff
            return None

        def _check(e: Expr) -> bool:
            if not e.contains(node.var):
                return True
            result = _is_a_linear_sum_or_prod(e)
            if result is not None:
                u, u_inverse = result
                if self._variable_change is not None:
                    return u == self._variable_change
                self._variable_change = u

                # If u_inverse exists, set it.
                # it must be the same as any prev u_inverse because the same u implies the same
                # u_inverse
                if u_inverse is not None:
                    self._inverse_var_change = u_inverse
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

        # We have to account for the case where self._variable_change doesn't appear directly
        # in the integrand.
        if self._inverse_var_change is not None:
            new = self._inverse_var_change(intermediate)
            new_integrand = replace(node.expr, node.var, new)
        else:
            new_integrand = replace(node.expr, self._variable_change, intermediate)
        new_integrand /= du_dx
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
                isinstance(e, (sin, cos))
                and isinstance(e.inner, Sum)
                and len(e.inner.terms) == 2  # for now lets j do 2 terms
            ):
                return True
            else:
                return any([_check(child) for child in e.children()])

        return _check(node.expr)

    def forward(self, node: Node) -> None:
        condition = (
            lambda expr: isinstance(expr, (sin, cos))
            and isinstance(expr.inner, Sum)
            and len(expr.inner.terms) == 2
        )

        def _perform(expr: Union[sin, cos]) -> Expr:
            a, b = expr.inner.terms
            if isinstance(expr, sin):
                return sin(a) * cos(b) + cos(a) * sin(b)
            elif isinstance(expr, cos):
                return cos(a) * cos(b) - sin(a) * sin(b)

        new_integrand = replace_factory(condition, _perform)(node.expr)

        node.children.append(Node(new_integrand, node.var, self, node))

    def backward(self, node: Node) -> None:
        super().backward(node)
        node.parent.solution = node.solution


class SinUSub(Transform):
    """u-substitution for if sinx cosx exists in the outer product"""

    # TODO: generalize this in some form? to other trig fns maybe?
    # - generalize to if the sin is in a power but the cos is under no power.
    # like transform D but for trigfns
    _sin: sin = None
    _cos: cos = None
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
        sins: List[sin] = []
        coses: List[cos] = []
        for term in node.expr.terms:
            if isinstance(term, sin):
                if not is_constant_product_of_var(term.inner, node.var):
                    continue

                sins.append(term)

                for cos_expr in coses:
                    if term.inner == cos_expr.inner:
                        self._sin = term
                        self._cos = cos_expr
                        return True

            if isinstance(term, cos):
                if not is_constant_product_of_var(term.inner, node.var):
                    continue

                coses.append(term)

                for sin_expr in sins:
                    if term.inner == sin_expr.inner:
                        self._sin = sin_expr
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
                isinstance(term, (sin, cos)) for term in node.expr.terms
            ):
                self._a, self._b = node.expr.terms
                return True

        if isinstance(node.expr, Power):
            if isinstance(node.expr.base, (sin, cos)) and node.expr.exponent == 2:
                self._a = self._b = node.expr.base
                return True

        return False

    def forward(self, node: Node) -> None:
        a, b = self._a, self._b
        if isinstance(a, sin) and isinstance(b, cos):
            temp = sin(a.inner + b.inner) + sin(a.inner - b.inner)
        elif isinstance(a, cos) and isinstance(b, sin):
            temp = sin(a.inner + b.inner) - sin(a.inner - b.inner)
        elif isinstance(a, cos) and isinstance(b, cos):
            temp = cos(a.inner + b.inner) + cos(a.inner - b.inner)
        elif isinstance(a, sin) and isinstance(b, sin):
            temp = cos(a.inner - b.inner) - cos(a.inner + b.inner)

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
            if (isinstance(node.expr, log) or isinstance(node.expr, TrigFunction) and node.expr.is_inverse) and node.expr.inner == node.var: # Special case for Log, ArcSin, etc.
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
            
            # if -u'v = node.expr, it means means you get 0 * integral(node.expr) = uv
            # which is invalid
            integrand2 = (du * v * -1)
            factor = (integrand2 / node.expr).simplify()
            if factor == 1:
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
        inv = invert(matrix)
        if inv is None:
            return False
        if numerator_list.size == 1:
            numerator_list = np.array([numerator_list[0], 0])
        ans = inv @ numerator_list
        self._new_integrand = ans[0] / d1 + ans[1] / d2
        return True
    
    def forward(self, node: Node) -> None:
        node.children.append(Node(self._new_integrand, node.var, self, node))
    
    def backward(self, node: Node) -> None:
        super().backward(node)
        node.parent.solution = node.solution

class GenericUSub(Transform):
    _u: Expr = None
    _variable_change: Expr = None
    def check(self, node: Node) -> bool:
        if not isinstance(node.expr, Prod):
            return False
        for i, term in enumerate(node.expr.terms):
            integral = _check_if_solveable(term, node.var)
            if integral is None:
                continue
            integral = remove_const_factor(integral)
            # assume term appears only once in integrand
            # because node.expr is simplified
            rest = Prod(node.expr.terms[:i] + node.expr.terms[i+1:])
            if count(rest, integral) == count(rest, node.var) / count(integral, node.var):
                self._u = integral
                return True
            
        return False
    
    def forward(self, node: Node) -> None:
        intermediate = generate_intermediate_var()
        du_dx = self._u.diff(node.var)
        new_integrand = replace((node.expr/du_dx).simplify(), self._u, intermediate)
        node.children.append(Node(new_integrand, intermediate, self, node))
        self._variable_change = self._u

    def backward(self, node: Node) -> None:
        super().backward(node)
        node.parent.solution = replace(
            node.solution, node.var, self._variable_change
        ).simplify()


class CompleteTheSquare(Transform):
    def check(self, node: Node) -> bool:
        # Completing the square is only useful if you can do InverseTrigUSub after it.
        # So we only look for these two cases:
        # 1 / quadratic
        # 1 / sqrt(quadratic)
        def condition(expr: Expr) -> bool:
            # 1 / xyz should be epxressed as a power so i omit prod check
            if not isinstance(expr, Power):
                return False
            if expr.exponent != Fraction(-1,2) and expr.exponent != -1:
                return False
            try:
                poly = to_const_polynomial(expr.base, node.var)
            except AssertionError:
                return False
            
            # hmm completing the square could work on non-quadratics in some cases no?
            # but I'll just limit it to quadratics for now
            return poly.size == 3 and poly[1] != 0
            # poly[1] = 0 implies that there's no bx term
            # which means that there's no square to complete.
            # the result will just be the same as the original.
    
        return general_count(node.expr, condition) > 0


    def forward(self, node: Node) -> None:
        # replace all quadratics with their completed-the-square form
        def condition(expr: Expr) -> bool:
            try:
                poly = to_const_polynomial(expr, node.var)
            except AssertionError:
                return False
            return poly.size == 3
        
        def perform(expr: Expr) -> Expr:
            poly = to_const_polynomial(expr, node.var)
            norm_factor = poly[-1]
            normalized = poly / norm_factor
            const = normalized[1] / 2
            diff = normalized[0] - const ** 2
            new_expr = (((node.var + const) / sqrt(diff)) ** 2 + 1) * diff * norm_factor

            return new_expr.simplify()
        
        new_expr = replace_factory(condition, perform)(node.expr)
        node.children.append(Node(new_expr, node.var, self, node))


    def backward(self, node: Node) -> None:
        super().backward(node)
        node.parent.solution = node.solution



# Leave RewriteTrig, InverseTrigUSub near the end bc they are deprioritized
# and more fucky
HEURISTICS: List[Type[Transform]] = [
    PolynomialUSub,
    CompoundAngle,
    SinUSub,
    ProductToSum,
    TrigUSub2,
    ByParts,
    RewriteTrig,
    InverseTrigUSub,
    CompleteTheSquare,
    GenericUSub
]
SAFE_TRANSFORMS: List[Type[Transform]] = [
    Additivity, PullConstant, PartialFractions, 
    PolynomialDivision,
    Expand, # expanding a fraction is not safe bc it destroys partialfractions. but if you put it after polynomial division & partial fractions, it doesn't cause any issues. more robust solution is to refactor & put expanding a fraction seperately as a heuristic transform, but idt this is necessary right now.
    LinearUSub,
]


def generate_intermediate_var() -> Symbol:
    return symbols(f"u_{random_id(10)}")



STANDARD_TRIG_INTEGRALS: Dict[str, ExprFn] = {
    "sin(x)": lambda x: -cos(x),
    "cos(x)": sin,
    "sec(x)^2": tan, # Integration calculator says this is a standard integral. + i haven't encountered any transform that can solve this.
    "sec(x)": lambda x: log(tan(x) + sec(x)) # not a standard integral but it's fucked so im leaving it (unless?)
}

def _check_if_solveable(integrand: Expr, var: Symbol) -> Optional[Expr]:
    """outputs a SIMPLIFIED expr"""
    if not integrand.contains(var):
        return (integrand * var).simplify()
    if isinstance(integrand, Power):
        if integrand.base == var and not integrand.exponent.contains(var):
            n = integrand.exponent
            return ((1 / (n + 1)) * Power(var, n + 1)).simplify() if n != -1 else log(integrand.base).simplify()
        if integrand.exponent == var and not integrand.base.contains(var):
            return (1 / log(integrand.base) * integrand).simplify()
    if isinstance(integrand, Symbol) and integrand == var:
        return( Fraction(1 / 2) * Power(var, 2)).simplify()

    silly_key = repr(replace(integrand, var, Symbol("x"))) # jank but does the job
    if silly_key in STANDARD_TRIG_INTEGRALS:
        return STANDARD_TRIG_INTEGRALS[silly_key](var).simplify()

    return None