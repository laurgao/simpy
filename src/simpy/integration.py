import warnings
from typing import Callable, List, Literal, Tuple, Union

from expr import Const, Expr, Optional, Symbol, cast, nesting
from transforms import HEURISTICS, SAFE_TRANSFORMS, Node, _check_if_solveable


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

    if len(node.unfinished_leaves) > 1:
        return _nesting_node(node)


# a recursive function.
# find the simplest problem to work on that makes progress.
def _nesting_node(node: Node) -> Node:
    if len(node.unfinished_children) == 1:
        return _nesting_node(node.unfinished_children[0])

    if len(node.unfinished_children) == 0:
        return node  # base case ???

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
        return root.solution.simplify()


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


def _print_solution_tree(root: Node) -> None:
    if not root.is_solved:
        return
    
    varchange = None if not hasattr(root.transform, "_variable_change") else root.transform._variable_change
        
    print(f"[{root.distance_from_root}] {root.solution} ({root.transform.__class__.__name__}, {varchange})")
    if not root.children:
        print("")
        return
    for child in root.children:
        _print_solution_tree(child)