from collections import defaultdict
from typing import Dict

from ..expr import Expr, Symbol, nesting
from ..regex import replace, replace_factory
from ..transforms import Node


def print_tree(root: Node, show_stale=True, show_failures=True, show_solution=False, _vardict=None, func=print) -> None:
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

    func(f"{distance} {n} {repr_expr}  {solution}  ({root.transform.__class__.__name__}){x}{varchange}{ending}")
    if not root.children:
        func("")
        return
    for child in root.children:
        print_tree(child, show_stale, show_failures, show_solution, _vardict=_vardict, func=func)


def _replaceall(expr: Expr, vardict: Dict[str, Symbol]) -> Expr:
    def condition(expr):
        return isinstance(expr, Symbol)

    def perform(expr):
        return vardict[expr.name]

    return replace_factory(condition, perform)(expr)


def print_success_tree(root: Node) -> None:
    print_tree(root, False, False)


def print_solution_tree(root: Node, func=print) -> None:
    print_tree(root, False, False, show_solution=True, func=func)
