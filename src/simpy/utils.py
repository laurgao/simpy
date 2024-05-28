import random
import string
from typing import Callable, Optional

from .expr import Expr

ExprFn = Callable[[Expr], Expr]
OptionalExprFn = Callable[[Expr], Optional[Expr]]


def random_id(length):
    # Define the pool of characters you can choose from
    characters = string.ascii_letters + string.digits
    # Use random.choices() to pick characters at random, then join them into a string
    random_string = "".join(random.choices(characters, k=length))
    return random_string
