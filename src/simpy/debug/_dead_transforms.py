from ..transforms import *


class CompoundAngle(Transform):
    """Compound angle formulae"""

    def check(self, node: Node) -> bool:
        if super().check(node) is False:
            return False

        def _check(e: Expr) -> bool:
            if (
                isinstance(e, (sin, cos))
                and isinstance(e.inner, Sum)
                and len(e.inner.terms) == 2  # for now lets j do 2 terms
            ):
                return True
            else:
                return any(_check(child) for child in e.children())

        return _check(node.expr)

    def forward(self, node: Node) -> None:
        condition = (
            lambda expr: isinstance(expr, (sin, cos)) and isinstance(expr.inner, Sum) and len(expr.inner.terms) == 2
        )

        def _perform(expr: Union[sin, cos]) -> Expr:
            a, b = expr.inner.terms
            if isinstance(expr, sin):
                return sin(a) * cos(b) + cos(a) * sin(b)
            elif isinstance(expr, cos):
                return cos(a) * cos(b) - sin(a) * sin(b)

        new_integrand = replace_factory(condition, _perform)(node.expr)

        node.add_child(Node(new_integrand, node.var, self, node))

    def backward(self, node: Node) -> None:
        super().backward(node)
        node.parent.solution = node.solution


class SinUSub(USub):
    """u-substitution for if sinx cosx exists in the outer product"""

    # TODO: generalize this in some form? to other trig fns maybe?
    # - generalize to if the sin is in a power but the cos is under no power.
    # like transform D but for trigfns
    _sin: sin = None
    _cos: cos = None

    def check(self, node: Node) -> bool:
        if super().check(node) is False:
            return False

        if not isinstance(node.expr, Prod):
            return False

        def is_constant_product_of_var(expr, var):
            if expr == var:
                return True
            if not (expr / var).contains(var):
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
        new_integrand = replace(node.expr, self._sin, intermediate) / dy_dx
        # should be done in check, here is a patch.
        if new_integrand.contains(node.var):
            return
        node.add_child(Node(new_integrand, intermediate, self, node))
        self._u = self._sin
