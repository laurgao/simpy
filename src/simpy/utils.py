from typing import Callable

from .expr import Expr

ExprFn = Callable[[Expr], Expr]
# Useless file since Expr can't import this 