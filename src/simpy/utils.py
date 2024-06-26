import random
import string
from typing import Callable, Optional, Tuple

from .expr import Expr, Symbol

ExprFn = Callable[[Expr], Expr]
OptionalExprFn = Callable[[Expr], Optional[Expr]]
ExprCondition = Callable[[Expr], bool]


def random_id(length):
    """Generates a random string of length `length` using ascii letters and digits."""
    # Define the pool of characters you can choose from
    characters = string.ascii_letters + string.digits
    # Use random.choices() to pick characters at random, then join them into a string
    random_string = "".join(random.choices(characters, k=length))
    return random_string


def eq_with_var(a: Tuple[Expr, Symbol], b: Tuple[Expr, Symbol]) -> bool:
    """Checks if 2 expressions are equivalent up to symbol.

    More efficient way of doing replace(a_expr, a_var, b_var) == b_expr
    """
    a_expr, a_var = a
    b_expr, b_var = b

    def _recursive_call(e1: Expr, e2: Expr) -> bool:
        if type(e1) != type(e2):
            return False
        if isinstance(e1, Symbol):
            if e1 == a_var:
                return e2 == b_var
            return e1 == e2
        c1 = e1.children()
        c2 = e2.children()
        if c1 == [] or c2 == []:
            return e1 == e2
        if len(c1) != len(c2):
            return False
        return all(_recursive_call(x, y) for x, y in zip(c1, c2))

    return _recursive_call(a_expr, b_expr)


def count_symbols(expr: Expr) -> int:
    """Counts the number of symbols in an expression."""
    from .regex import general_count

    return general_count(expr, lambda x: isinstance(x, Symbol))
