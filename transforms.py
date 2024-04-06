# structures for integration w transforms

from typing import Optional

from .main import *

# feels like var should be global of some sorts for each integration call. being passed down each by each feels sad.
# idk im overthinking it.


@dataclass
class Node:
    expr: Expr
    var: Symbol  # variable that THIS EXPR is integrated by.
    transform: Optional["Transform"]  # the transform that led to this node
    parent: Optional["Node"]  # None for root node only
    children: Optional[List["Node"]] = (
        None  # smtn smtn setting it to [] by default causes errors
    )
    type: Literal["AND", "OR", "UNSET", "SOLUTION", "FAILURE"] = "UNSET"
    # failure = can't proceed forward.

    @property
    def leaves(self) -> List["Node"]:
        # Returns the leaves of the tree (all nodes without children)
        if len(self.children) == 0:
            return [self]

        return [leaf for child in self.children for leaf in child.leaves()]

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
    def is_solved(self) -> bool:
        # Returns True if all leaves WITH AND NODES are solved and all OR nodes are solved
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
        # TODO
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
    def unsolved_children(self) -> List["Node"]:
        return [child for child in self.children if not child.is_solved]

    # @property
    # def grouped_unsolved_leaves(self) -> list:
    #     # returns lists of like
    #     # based on AND/OR
    #     # like i want to group based on "groups you need to solve in order to solve the problem"


class Transform(ABC):
    "An integral transform -- base class"

    def __init__(self):
        pass

    def forward(self, node: Node, var: Symbol):
        raise NotImplementedError("Not implemented")

    def backward(self, node: Node, var: Symbol):
        raise NotImplementedError("Not implemented")

    def check(self, node: Node, var: Symbol) -> bool:
        raise NotImplementedError("Not implemented")


class PullConstant(Transform):

    constant: Expr = None
    non_constant_part: Expr = None

    def check(self, node: Node, var: Symbol) -> bool:
        expr = node.expr
        if isinstance(expr, Prod):
            # if there is a constant, pull it out
            if isinstance(expr.terms[0], Const):
                self.constant = expr.terms[0]
                self.non_constant_part = Prod(expr.terms[1:]).simplify()
                return True

            # or if there is a symbol that's not the variable, pull it out
            for i, term in enumerate(expr.terms):
                is_nonvar_symbol = isinstance(term, Symbol) and term != var
                is_nonvar_power = (
                    isinstance(term, Power)
                    and isinstance(term.base, Symbol)
                    and term.base != var
                )
                if is_nonvar_symbol or is_nonvar_power:
                    self.constant = term
                    self.non_constant_part = Prod(
                        expr.terms[:i] + expr.terms[i + 1 :]
                    ).simplify()
                    return True

        return False

    def forward(self, node: Node, var: Symbol):
        node.children = [Node(self.non_constant_part, self, node)]


class PolynomialDivision(Transform):
    numerator: Polynomial = None
    denominator: Polynomial = None

    def check(self, node: Node, var: Symbol) -> bool:
        # This is so messy we can honestly just do catching in the `to_polynomial`
        expr = node.expr
        # currently we don't support division of polynomials with multiple variables
        if expr.symbols() != [var]:
            return False
        if not isinstance(expr, Prod):
            return False

        ## Don't contain any SingleFunc with inner containing var
        def _contains_singlefunc_w_inner(expr: Expr, var: Symbol) -> bool:
            if isinstance(expr, SingleFunc) and expr.inner.contains(var):
                return True

            return any([_contains_singlefunc_w_inner(e, var) for e in expr.children()])

        if _contains_singlefunc_w_inner(expr, var):
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
                if isinstance(expr, Const) or isinstance(expr, Symbol):
                    return True

                if isinstance(expr, Sum):
                    return all([_is_polynomial(term, var) for term in expr.terms])

                raise NotImplementedError(f"Not implemented: {expression}")

            if not _is_polynomial(factor):
                return False

        ## Make sure numerator and denominator are good
        numerator = 1
        denominator = 1
        for factor in expr.terms:
            b, x = deconstruct_power(factor)
            if x.value > 0:
                numerator *= factor
            else:
                denominator *= Power(b, -x).simplify()

        numerator = numerator.simplify()
        denominator = denominator.simplify()

        try:
            numerator_list = to_polynomial(numerator, var)
            denominator_list = to_polynomial(denominator, var)
        except AssertionError:
            return False

        if len(numerator_list) < len(denominator_list):
            return False

        self.numerator = numerator_list
        self.numerator = denominator_list
        return True

    def forward(self, node: Node, var: Symbol):
        quotient = np.zeros(len(self.numerator) - len(self.denominator) + 1)

        while self.numerator.size >= self.denominator.size:
            quotient_degree = len(self.numerator) - len(self.denominator)
            quotient_coeff = self.numerator[-1] / self.denominator[-1]
            quotient[quotient_degree] = quotient_coeff
            self.numerator -= np.concatenate(
                ([0] * quotient_degree, self.denominator * quotient_coeff)
            )
            self.numerator = rid_ending_zeros(self.numerator)

        remainder = polynomial_to_expr(self.numerator, var) / polynomial_to_expr(
            self.denominator, var
        )
        quotient_expr = polynomial_to_expr(quotient, var)
        answer = (quotient_expr + remainder).simplify()
        node.children = [Node(answer, self, node)]


class Expand(Transform):
    def forward(self, node: Node, var: Symbol):
        node.children = [Node(node.expr.expand(), self, node)]

    def check(node: Node, var: Symbol) -> bool:
        return node.expr.expandable()


class Additivity(Transform):
    def forward(self, node: Node, var: Symbol):
        node.type = "AND"
        node.children = [
            Node("UNSET", e, [], node, Additivity) for e in node.expr.terms
        ]

    def check(self, node: Node, var: Symbol) -> bool:
        return isinstance(node, Sum)


# Let's just add all the transforms we've used for now.
# and we will make this shit good and generalized later.
class B_Tan(Transform):
    def forward(self, node: Node, var: Symbol):
        intermediate = generate_intermediate_var()
        expr = node.exprarea
        # y = tanx
        new_integrand = replace(expr, Tan(var), intermediate) / (1 + intermediate**2)
        new_node = Node(new_integrand, self, node)
        node.children = [new_node]

    def check(node: Node, var: Symbol) -> bool:
        expr = node.expr
        return contains(expr, Tan) and count(expr, Tan(var)) == count(
            expr, var
        )  # ugh everything is so sus


class A(Transform):
    def forward(self, node: Node, var: Symbol):
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
        node.children = [Node(option, self, node) for option in stuff]
        node.type = "OR"

    def check(node: Node, var: Symbol) -> bool:
        expr = node.expr
        return contains(expr, TrigFunction)


class C_Sin(Transform):
    def forward(self, node: Node, var: Symbol):
        intermediate_var = generate_intermediate_var()
        new_thing = replace(node.expr, var, Sin(intermediate_var)) * Cos(
            intermediate_var
        )
        new_thing = new_thing.simplify()

        # then that's a node and u store the transform and u take the integral of that.
        node.children = [Node(new_thing, self, node)]

    def check(node: Node, var: Symbol) -> bool:
        s = f"(1 + (-1 * {var.name}^2))"
        return s in node.expr.__repr__()  # ugh unclean


class C_Tan(Transform):
    def forward(self, node: Node, var: Symbol):
        intermediate = generate_intermediate_var()
        dy_dx = Sec(intermediate) ** 2
        new_thing = (replace(node.expr, var, Tan(intermediate)) * dy_dx).simplify()
        node.children = [Node(new_thing, self, node)]
        # TODO: I NEED TO STORE THAT THE EXPR SHOULD NOW BE INTEGRATED WRT INTERMEDIATE VAR!!!!

    def check(node: Node, var: Symbol) -> bool:
        s2 = f"1 + {var.name}^2"
        return s2 in node.expr.__repr__()


HEURISTICS = [B_Tan, A, C_Sin, C_Tan]
SAFE_TRANSFORMS = [Additivity, PullConstant, Expand, PolynomialDivision]


def check_if_solvable(node: Node, var: Symbol):
    expr = node.expr
    answer = None
    if isinstance(expr, Power):
        if expr.base == var and isinstance(expr.exponent, Const):
            n = expr.exponent
            answer = (1 / (n + 1)) * Power(var, n + 1) if n != -1 else Log(expr.base)
        elif isinstance(expr.base, Symbol) and expr.base != var:
            answer = expr * var

    elif isinstance(expr, Symbol):
        answer = Fraction(1 / 2) * Power(var, 2) if expr == var else expr * var
    elif isinstance(expr, Const):
        answer = expr * var

    if answer is None:
        return

    node.children = [Node(answer, parent=node, type="SOLUTION")]


def cycle(node: Node, var: Symbol):
    # 1. APPLY ALL SAFE TRANSFORMS
    integrate_safely(node, var)

    # now we have a tree with all the safe transforms applied
    # 2. LOOK IN TABLE
    for leaf in node.unfinished_leaves:
        check_if_solvable(leaf, var)

    if len(node.unfinished_leaves) == 0:
        return "SOLVED"

    # 3. APPLY HEURISTICS
    next_node = node.unfinished_leaves[0]  # random lol
    integrate_heuristically(next_node)

    next_next_node = _get_next_node_post_heuristic(next_node)
    # if next_next_node is None:
    #     return "FAILURE"
    return next_next_node


def _get_next_node_post_heuristic(node: Node) -> Node:

    if len(node.unfinished_leaves) == 0:
        if node.type == "FAILURE":
            # this means node.type == FAILURE (it better be based on our structure)
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
            # node.type = "SUCCESS"
            raise NotImplementedError("TODO _get_next_node for success nodes")

    if len(node.unfinished_leaves) == 1:
        return node.unfinished_leaves[0]

    if len(node.unfinished_leaves) > 1:
        # if node.type == "AND":

        #     # if none of node's children has a child
        #     # if all([not child.children for child in node.children]):

        #     # choose a random one lmfao
        #     raise NotImplementedError(
        #         "_get_next_node for AND nodes"
        #     )  # not gonna do this here tbh bc we call this function
        #     # exclusively for post-heuristics. this can change later if we have heuristics that aren't pure "OR"
        #     return node.unfinished_leaves[0]

        # ok now node.type = "OR"

        # do least nested one
        # if there are ANDs that are children of this node,
        # do the one
        return _nesting_node(node)


# a recursive function.
def _nesting_node(node: Node) -> Node:
    if len(node.unsolved_children) == 1:
        return _nesting_node(node.unsolved_children[0])

    if len(node.unsolved_children) == 0:
        return node  # base case ???
        raise ValueError("nesting_node on a solved node?")

    is_2nd_lowest_parent = all(
        [not child.unsolved_children for child in node.unsolved_children]
    )
    fn = min if node.type == "OR" else max
    if is_2nd_lowest_parent:
        return _get_node_with_best_nesting(node.unsolved_children, fn)

    childrens_best_nodes = [_nesting_node(c) for c in node.unsolved_children]
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

    @staticmethod
    def integrate(integrand: Expr, var: Symbol):

        root = Node(integrand)
        curr_node = root
        while True:
            answer = cycle(curr_node)

            if root.is_finished:
                break

            if answer == "SOLVED":
                # just do any other thing in root
                curr_node = _get_next_node_post_heuristic(root)
            else:
                curr_node = answer

        if root.is_failed:
            raise NotImplementedError(f"Failed to integrate {integrand} wrt {var}")

        breakpoint()

        # now we have a solved tree or a failed tree
        # we can go back and get the answer
        ...


def integrate_safely(node: Node, var: Symbol):
    for transform in SAFE_TRANSFORMS:
        tr = transform()
        if tr.check(node, var):
            tr.forward(node, var)
            for child in node.children:
                integrate_safely(child, var)


def integrate_heuristically(node: Node, var: Symbol):
    for heuristic_transform in HEURISTICS:
        if heuristic_transform.check(node, var):
            heuristic_transform.forward(node, var)

    if len(node.children) > 1:
        node.type = "OR"

    if len(node.children) == 0:
        node.type = "FAILURE"


if __name__ == "__main__":
    F = Fraction
    x, y = symbols("x y")
    expression = -5 * x**4 / (1 - x**2) ** F(5, 2)
    print(expression)
    integral = Integration.integrate(expression, x)  # TODO auto simplify
    # print(integral)
    # breakpoint()