# structures for integration w transforms

from .main import *


@dataclass
class Node:
    type: Literal["AND", "OR", "UNSET"]
    expr: Expr
    children: List["Node"]
    parent: Optional["Node"]  # None for root node only
    transform: "Transform"

    def leaves(self) -> List["Node"]:
        if len(self.children) == 0:
            return [self]

        return [leaf for child in self.children for leaf in child.leaves()]


def nesting(expr: Expr, var: Symbol) -> int:
    """
    Compute the nesting amount (complexity) of an expression

    >>> nesting(x**2, x)
    2
    >>> nesting(x * y**2, x)
    2
    >>> nesting(x * (1 / y**2 * 3), x)
    2
    """

    children = expr.children()
    if isinstance(expr, Symbol) and expr.name == var.name:
        return 1
    elif len(children) == 0:
        return 0
    else:
        return 1 + max(
            nesting(sub_expr, var) for sub_expr in children if sub_expr.contains(var)
        )


class Transform:
    "An integral transform -- base class"

    @staticmethod
    def forward(node: Node):
        raise NotImplementedError("Not implemented")

    @staticmethod
    def backward():
        raise NotImplementedError("Not implemented")


class Simplify(Transform):
    def forward(node: Node):
        simplified = node.expr.simplify()
        if node.expr.__repr__ != simplified.__repr__:
            node.children = [Node("UNSET", simplified, [], node, Simplify)]


class Additivity(Transform):
    def forward(node: Node):
        if isinstance(node.expr, Sum):
            node.type = "AND"
            node.children = [
                Node("UNSET", e, [], node, Additivity) for e in node.expr.children()
            ]


def get_safe_transform(expr: Expr):
    pass


heuristic_transforms = [
    # f(tan x) -> f(y) / (1 + y**2) [after y = tan x]
]


safe_transforms = [
    # I[f(x) + g(x)] -> I[f(x)] + I[g(x)]
    # expansion
    # expression simplification
]


class Integration:
    """
    Keeps track of integration work as we go
    """

    def integrate(integrand: Expr):

        root = Node(type="UNSET", expr=integrand, children=[], parent=None)

        solved = False
        while not solved:
            node = None
            for leaf in root.leaves():
                if not node or nesting(leaf) < nesting(node):
                    node = leaf

            # (for first iter node is root)

            # perfom safe transforms.
            # 1. simplify
            # 2. expand
            # 3. pull out constant
            # 4. sum
            # 5. polydiv

            integrate_safely(node)

            # apply all heuristic transformations and append them all to the tree
            node.type = "OR"
            for heuristic_transform in heuristic_transforms:
                node.children.append(heuristic_transform(node))


@cast
def integrate_safely(node: Node):
    simplified = node.expr.simplify()
    if node.expr.__repr__ != simplified.__repr__:
        node.children = [Node("UNSET", simplified, [], node, Simplify)]

    if isinstance(node.expr, Sum):
        node.type = "AND"
        node.children = [
            Node("UNSET", e, [], node, Additivity) for e in node.expr.children()
        ]

    node.expr
