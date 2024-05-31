import random
import string
from typing import Callable, Optional

from .expr import Expr, Symbol

ExprFn = Callable[[Expr], Expr]
OptionalExprFn = Callable[[Expr], Optional[Expr]]
ExprCondition = Callable[[Expr], bool]


def random_id(length):
    # Define the pool of characters you can choose from
    characters = string.ascii_letters + string.digits
    # Use random.choices() to pick characters at random, then join them into a string
    random_string = "".join(random.choices(characters, k=length))
    return random_string


def is_simpler(a, b):
    """return if a is simpler than b

    currently unused. leaving case for potential future purposes.
    """
    from .regex import general_count

    def count(e):
        # counts the number of symbols
        return general_count(e, lambda x: isinstance(x, Symbol))

    return count(a) < count(b)
