from ..expr import nesting
from ..utils import count_symbols


def is_simpler(e1, e2) -> bool:
    """returns whether e1 is simpler than e2"""
    c1 = count_symbols(e1)
    c2 = count_symbols(e2)
    if c1 < c2:
        return True

    if c1 == c2:
        return nesting(e1) < nesting(e2)

    return False
