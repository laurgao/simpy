from typing import Callable

from .expr import Expr

ExprFn = Callable[[Expr], Expr]
