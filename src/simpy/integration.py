"""Transversing integration tree logic.
"""

import time
import warnings
from collections import defaultdict
from typing import Callable, Dict, List, Literal, Tuple, Union

from .expr import Expr, Optional, Rat, Symbol, cast, nesting
from .regex import replace, replace_factory
from .transforms import HEURISTICS, SAFE_TRANSFORMS, Node, _check_if_solveable


def _check_if_node_solvable(node: Node):
    answer = _check_if_solveable(node.expr, node.var)
    if answer is None:
        return

    node.type = "SOLUTION"
    node.solution = answer


# a recursive function.
# find the simplest problem to work on that makes progress.
def _get_best_leaf(root: Node) -> Node:
    if len(root.unfinished_children) == 1:
        return _get_best_leaf(root.unfinished_children[0])

    if len(root.unfinished_children) == 0:
        return root  # base case ???

    is_2nd_lowest_parent = all(not child.unfinished_children for child in root.unfinished_children)
    fn = min if root.type == "OR" else max
    if is_2nd_lowest_parent:
        return _get_node_with_best_nesting(root.unfinished_children, fn)

    childrens_best_nodes = [_get_best_leaf(c) for c in root.unfinished_children]
    return _get_node_with_best_nesting(childrens_best_nodes, fn)


def _get_node_with_best_nesting(nodes: List[Node], fn: Callable[[List[Node]], Node]) -> Node:
    results = [nesting(node.expr, node.var) for node in nodes]
    best_value = fn(results)
    return nodes[results.index(best_value)]


@cast
def integrate(
    expr: Expr,
    bounds: Optional[Union[Symbol, Tuple[Expr, Expr], Tuple[Symbol, Expr, Expr]]] = None,
    **kwargs,
) -> Optional[Expr]:
    """
    Integrates an expression.

    Args:
        expr: the integrand
        bounds: (var, a, b) where var is the variable of integration and a, b are the integration bounds.
            can omit var if integrand contains exactly one symbol. omit a, b for an indefinite integral.
    kwargs:
        debug: prints the integration tree + enables the python debugger right before returning.
            you can use the python debugger to trace back the integration tree & find out what went wrong.
            if you don't know what this means, don't worry about it -- this is mostly for developers and
            particularly nerdy adventurers.
        debug_hardcore
        breadth_first

        Examples of valid uses:
            integrate(x**2)
            integrate(x*y, x)
            integrate(3*x + Tan(x), (2, pi))
            integrate(x*y+3*x, (x, 3, 4))

    Returns:
        The solution to the integral.
        None if the integral cannot be solved.
    """
    # If variable of integration isn't specified, set it.
    if bounds is None:
        vars = expr.symbols()
        if len(vars) != 1:
            raise ValueError(f"Please specify the variable of integration for {expr}")
        bounds = vars[0]
    elif isinstance(bounds, tuple) and len(bounds) == 2:
        vars = expr.symbols()
        if len(vars) != 1:
            raise ValueError(f"Please specify the variable of integration for {expr}")
        bounds = (vars[0], bounds[0], bounds[1])

    integration = Integration(**kwargs)
    if isinstance(bounds, Symbol):
        return integration.integrate(expr, bounds)
    if isinstance(bounds, tuple):
        return integration.integrate_bounds(expr, bounds)
    else:
        raise ValueError(f"Invalid bounds: {bounds}")


class Integration:
    """
    Keeps track of integration work as we go
    """

    # tweakable params
    DEPTH_FIRST_MAX_NESTING = 7  # chosen somewhat-arbitarily: on may 10th, it lead to lowest time spent on my tests.

    def __init__(self, *, debug: bool = False, debug_hardcore: bool = False, breadth_first=False):
        self._debug = debug
        self._debug_hardcore = debug_hardcore
        self._breadth_first = breadth_first
        self._timeout = 2  # seconds

    def integrate_bounds(self, expr: Expr, bounds: Tuple[Symbol, Expr, Expr]) -> Optional[Expr]:
        """Performs definite integral."""
        x, a, b = bounds
        integral = self.integrate(expr, bounds[0], final=False)
        if integral is None:
            return None
        return (integral.subs({x.name: b}) - integral.subs({x.name: a})).simplify()

    @staticmethod
    def integrate_without_heuristics(integrand: Expr, var: Symbol) -> Optional[Expr]:
        """Performs indefinite integral. used for byparts checking if dv is integrateable."""
        root = Node(integrand, var)
        _integrate_safely(root)
        for leaf in root.unfinished_leaves:
            _check_if_node_solvable(leaf)
            if leaf.type != "SOLUTION":
                leaf.type = "FAILURE"

        if root.is_failed:
            return None
        Integration._go_backwards(root)

        # Unsure if not simplifying here could cause problems down the line
        # but it seems to work fine for now.
        return root.solution

    def _cycle(self, node: Node) -> Optional[Union[Node, Literal["SOLVED"]]]:
        # 1. APPLY ALL SAFE TRANSFORMS
        # _integrate_safely(node)

        # now we have a tree with all the safe transforms applied
        # 2. LOOK IN TABLE
        for leaf in node.unfinished_leaves:
            _check_if_node_solvable(leaf)

        if len(node.unfinished_leaves) == 0:
            return "SOLVED"

        # 3. APPLY HEURISTICS
        next_node = node.unfinished_leaves[0]  # random lol
        _integrate_heuristically(next_node)

        # 4. FIND BEST NEXT NODE BASED ON NESTING
        return self._get_next_node_post_heuristic(next_node)

    def _get_next_node_post_heuristic(self, node: Node) -> Node:
        if self._breadth_first:
            return self._get_next_node_post_heuristic_breadth_first(node)
        else:
            return self._get_next_node_post_heuristic_depth_first(node)

    def _get_next_node_post_heuristic_breadth_first(self, node: Node) -> Node:
        root = node.root
        if root.unfinished_leaves == 0:
            if root.is_failed:
                return None
            return "SOLVED"

        if len(root.unfinished_leaves) == 1:
            return root.unfinished_leaves[0]

        return _get_best_leaf(root)

    def _get_next_node_post_heuristic_depth_first(self, node: Node) -> Node:
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
                return self._get_next_node_post_heuristic_depth_first(parent)
            else:
                # This happens when we use integration by parts and the heuristic finds a whole ass solution
                return "SOLVED"

        if len(node.unfinished_leaves) == 1:
            ans = node.unfinished_leaves[0]
        else:
            ans = _get_node_with_best_nesting(node.unfinished_leaves, min)
        if self._check_if_depth_first_bad(ans):
            return self._get_next_node_post_heuristic_breadth_first(ans)
        return ans

    def _check_if_depth_first_bad(self, ans: Node) -> bool:
        # setting this alone makes csc^2 get solved when depth first!!
        # if nesting is greater than 5 i honestly don't see how it can get solved super easily??
        # byparts should not be depth first.

        if nesting(ans.expr) >= self.DEPTH_FIRST_MAX_NESTING:
            return True
        from .transforms import Additivity, ByParts, Expand, PullConstant, _get_last_heuristic_transform

        t = _get_last_heuristic_transform(ans, (Expand, PullConstant, Additivity))
        if isinstance(t, ByParts):
            return True
        return False

    def integrate(self, integrand: Expr, var: Symbol, final=True) -> Optional[Expr]:
        """Performs indefinite integral."""
        root = Node(integrand.simplify(), var)
        _integrate_safely(root)
        curr_node = root
        start = time.time()
        while True:
            answer = self._cycle(curr_node)

            current = time.time()
            if current - start >= self._timeout:
                break
            if root.is_finished:
                break
            if answer == "SOLVED":
                # just do any other thing in root
                curr_node = self._get_next_node_post_heuristic_breadth_first(root)
            else:
                curr_node = answer

        if not root.is_solved:
            message = f"Failed to integrate {integrand} wrt {var}"
            if not root.is_failed:
                message += ", TIMED OUT"
            warnings.warn(message)
            if self._debug:
                _print_tree(root)
                breakpoint()
            return None

        # now we have a solved tree we can go back and get the answer
        self._go_backwards(root)

        if self._debug:
            _print_success_tree(root)
            breakpoint()
        return root.solution.simplify() if final else root.solution

    @staticmethod
    def _go_backwards(root: Node):
        """Mutates the tree in place/returns nothing.

        At the end, root.solution should be existant.
        """
        if not root.is_solved:
            raise ValueError("Cannot go backwards on an unsovled tree.")

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

    if node.is_solved:
        return

    if not node.children:
        node.type = "FAILURE"
        return

    if len(node.children) > 1:
        node.type = "OR"

    for child in node.children:
        _integrate_safely(child)


def _print_tree(root: Node, show_stale=True, show_failures=True, show_solution=False, _vardict=None) -> None:
    if _vardict is None:
        _vardict = defaultdict(str)

    if (not show_stale) and root.is_stale:
        return
    if (not show_failures) and root.is_failed:
        return

    def _exprify(expr: Expr):
        if root.var.name.startswith("u_"):
            if not _vardict[root.var.name]:
                new_name = "u_" + str(len(_vardict))
                _vardict[root.var.name] = Symbol(new_name)
            return replace(expr, root.var, _vardict[root.var.name])
        else:
            _vardict[root.var.name] = root.var
            return expr

    if root.is_filler:
        repr_expr = "filler"
    else:
        expr = _exprify(root.expr)
        repr_expr = repr(expr)

    x = (
        ""
        if (not root.children and (root.is_solved or root.is_failed))
        else (" (solved)" if root.is_solved else " (failed)" if root.is_failed else " (stale)" if root.is_stale else "")
    )
    varchange = (
        ""
        if not hasattr(root.transform, "_variable_change")
        else " ("
        + _vardict[root.var.name].name
        + "="
        + repr(_replaceall(root.transform._variable_change, _vardict))
        + ")"
    )
    ending = " (" + root.type + ")" if not root.children else ""
    n = "{" + str(nesting(root.expr)) + "}"

    def _wrap(string: str, num: int) -> str:
        if len(string) > num:
            string = string[: num - 3] + "..."
            spaces = ""
        else:
            spaces = " " * (num - len(string))
        return string + spaces

    repr_expr = _wrap(repr_expr, 50)
    solution = _wrap(repr(_exprify(root.solution)), 50) if (show_solution and root.is_solved) else ""
    distance = _wrap(f"[{root.distance_from_root}]", 4)

    print(f"{distance} {n} {repr_expr}  {solution}  ({root.transform.__class__.__name__}){x}{varchange}{ending}")
    if not root.children:
        print("")
        return
    for child in root.children:
        _print_tree(child, show_stale, show_failures, show_solution, _vardict=_vardict)


def _replaceall(expr: Expr, vardict: Dict[str, Symbol]) -> Expr:
    def condition(expr):
        return isinstance(expr, Symbol)

    def perform(expr):
        return vardict[expr.name]

    return replace_factory(condition, perform)(expr)


def _print_success_tree(root: Node) -> None:
    _print_tree(root, False, False)


def _print_solution_tree(root: Node) -> None:
    _print_tree(root, False, False, show_solution=True)
