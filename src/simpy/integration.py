import time
import warnings
from collections import defaultdict
from typing import Callable, Dict, List, Literal, Tuple, Union

from .expr import Const, Expr, Optional, Symbol, cast, nesting
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

    is_2nd_lowest_parent = all(
        [not child.unfinished_children for child in root.unfinished_children]
    )
    fn = min if root.type == "OR" else max
    if is_2nd_lowest_parent:
        return _get_node_with_best_nesting(root.unfinished_children, fn)

    childrens_best_nodes = [_get_best_leaf(c) for c in root.unfinished_children]
    return _get_node_with_best_nesting(childrens_best_nodes, fn)


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
    **kwargs
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

    def __init__(self, *, debug: bool = False, debug_hardcore: bool = False, breadth_first = True):
        self._debug = debug
        self._debug_hardcore = debug_hardcore
        self._breadth_first = breadth_first
        self._timeout = 2 # seconds
    
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
    
    def _cycle(self, node: Node) -> Optional[Union[Node, Literal["SOLVED"]]]:
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
            return node.unfinished_leaves[0]
        
        return _get_node_with_best_nesting(node.unfinished_leaves, min)


    def integrate(
        self, integrand: Expr, var: Symbol 
    ) -> Optional[Expr]:
        """Performs indefinite integral.
        """
        root = Node(integrand, var)
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



def _print_tree(root: Node, show_stale=False, show_failures=False, _vardict = None) -> None:
    if _vardict is None:
        _vardict = defaultdict(str)

    # if (not show_stale) and root.is_stale:
    #     return
    # if (not show_failures) and root.is_failed:
    #     return

    if root.var.name.startswith("u_"):
        if not _vardict[root.var.name]:
            new_name = "u_" + str(len(_vardict))
            _vardict[root.var.name] = Symbol(new_name)
        expr = replace(root.expr, root.var, _vardict[root.var.name])
    else:
        _vardict[root.var.name] = root.var
        expr = root.expr

    x = " (solved)" if root.is_solved else ''
    varchange = '' if not hasattr(root.transform, "_variable_change") else " (" + _vardict[root.var.name].name + "=" + repr(_replaceall(root.transform._variable_change, _vardict)) + ")"
    ending = " (" + root.type + ")" if not root.children else ''
    n = '{' + str(nesting(expr)) + '}'
    print(f"[{root.distance_from_root}] {n} {expr} ({root.transform.__class__.__name__}){x}{varchange}{ending}")
    if not root.children:
        print("")
        return
    for child in root.children:
        _print_tree(child, show_stale, show_failures, _vardict=_vardict)


def _replaceall(expr: Expr, vardict: Dict[str, Symbol]) -> Expr:
    def condition(expr):
        return isinstance(expr, Symbol)
    def perform(expr):
        return vardict[expr.name]
    return replace_factory(condition, perform)(expr)
    


def _print_success_tree(root: Node) -> None:
    _print_tree(root, False, False)


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
