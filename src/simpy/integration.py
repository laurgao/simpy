import warnings
from typing import Callable, List, Literal, Tuple, Union

from .expr import Const, Expr, Optional, Symbol, cast, nesting
from .transforms import HEURISTICS, SAFE_TRANSFORMS, Node, _check_if_solveable


def _check_if_node_solvable(node: Node):
    answer = _check_if_solveable(node.expr, node.var)
    if answer is None:
        return

    node.type = "SOLUTION"
    node.solution = answer

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
    
    return _get_leaf_with_best_nesting(node)
    
    # you are making this way too complicated
    if len(node.unfinished_leaves) > 1:
        return _nesting_node(node)

def _get_leaf_with_best_nesting(node: Node) -> Node:
    return _get_node_with_best_nesting(node.unfinished_leaves, min)


# a recursive function.
# find the simplest problem to work on that makes progress.
# def _nesting_node(node: Node) -> Node:
#     if len(node.unfinished_children) == 1:
#         return _nesting_node(node.unfinished_children[0])

#     if len(node.unfinished_children) == 0:
#         return node  # base case ???

#     is_2nd_lowest_parent = all(
#         [not child.unfinished_children for child in node.unfinished_children]
#     )
#     fn = min if node.type == "OR" else max
#     if is_2nd_lowest_parent:
#         return _get_node_with_best_nesting(node.unfinished_children, fn)

#     childrens_best_nodes = [_nesting_node(c) for c in node.unfinished_children]
#     return _get_node_with_best_nesting(childrens_best_nodes, fn)


def _get_node_with_best_nesting(
    nodes: List[Node], fn: Callable[[List[Node]], Node]
) -> Node:
    results = [nesting(node.expr, node.var) for node in nodes]
    best_value = fn(results)
    return nodes[results.index(best_value)]


@cast
def integrate(
    expr: Expr,
    bounds: Optional[Union[Symbol, Tuple[Expr, Expr], Tuple[Symbol, Expr, Expr]]] = None,
    verbose: bool = False,
    debug: bool = False,
) -> Optional[Expr]:
    """
    Integrates an expression.

    Args:
        expr: the integrand
        bounds: (var, a, b) where var is the variable of integration and a, b are the integration bounds.
            can omit var if integrand contains exactly one symbol. omit a, b for an indefinite integral.
        verbose: prints the integration tree + enables the python debugger right before returning.
            you can use the python debugger to trace back the integration tree & find out what went wrong.
            if you don't know what this means, don't worry about it -- this is mostly for developers and
            particularly nerdy adventurers.

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


    integration = Integration(verbose=verbose, debug=debug)
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

    def __init__(self, *, verbose: bool = False, debug: bool = False):
        self._verbose = verbose
        self._debug = debug
    
    def integrate_bounds(
        self, expr: Expr, bounds: Tuple[Symbol, Expr, Expr]
    ) -> Optional[Expr]:
        """Performs definite integral.
        """
        x, a, b = bounds
        integral = self.integrate(expr, bounds[0])
        if integral is None:
            return None
        return (integral.evalf({x.name: b}) - integral.evalf({x.name: a})).simplify()
    
    @staticmethod
    def integrate_without_heuristics(integrand: Expr, var: Symbol) -> Optional[Expr]:
        """Performs indefinite integral. used for byparts checking if dv is integrateable.
        """
        root = Node(integrand, var)
        _integrate_safely(root)
        for leaf in root.unfinished_leaves:
            _check_if_node_solvable(leaf)
            if leaf.type != "SOLUTION":
                leaf.type = "FAILURE"

        if root.is_failed:
            return None
        Integration._go_backwards(root)
        return root.solution.simplify()

    def integrate(
        self, integrand: Expr, var: Symbol 
    ) -> Optional[Expr]:
        """Performs indefinite integral.
        """
        root = Node(integrand, var)
        curr_node = root
        while True:
            answer = _cycle(curr_node)
            if self._debug:
                breakpoint()
            if root.is_finished:
                break
            if answer == "SOLVED":
                # just do any other thing in root
                curr_node = _get_next_node_post_heuristic(root)
            else:
                curr_node = answer

        if root.is_failed:
            warnings.warn(f"Failed to integrate {integrand} wrt {var}")
            if self._verbose:
                _print_tree(root)
                breakpoint()
            return None

        # now we have a solved tree we can go back and get the answer
        self._go_backwards(root)

        if self._verbose:
            _print_success_tree(root)
            breakpoint()
        return root.solution.simplify()
    
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

    if not node.children:
        node.type = "FAILURE"
        return

    if len(node.children) > 1:
        node.type = "OR"


def _print_tree(root: Node) -> None:
    if root.is_stale:
        print('stale')
        return
    x = " (solved)" if root.is_solved else ''
    print(f"[{root.distance_from_root}] {root.expr} ({root.transform.__class__.__name__}){x}")
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


def _print_solution_tree(root: Node) -> None:
    if not root.is_solved:
        return
    
    varchange = None if not hasattr(root.transform, "_variable_change") else root.transform._variable_change
    
    num = 40
    spaces = " " * (num - len(repr(root.expr)))
    print(f"[{root.distance_from_root}] {root.expr}{spaces}{root.solution} ({root.transform.__class__.__name__}, {varchange})")
    if not root.children:
        print("")
        return
    for child in root.children:
        _print_solution_tree(child)
