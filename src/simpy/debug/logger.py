import time
from typing import Dict, NamedTuple

from ..expr import Expr
from ..transforms import Node
from .utils import print_solution_tree, print_tree


class Datum(NamedTuple):
    expr: Expr
    time_spent: float
    root: Node


class Logger:
    """Keeps track of time spent on integration.

    Maybe in the future we'll be fancy and also keep track of the integration tree better & time
    spent on each step.
    """

    _data: Dict[str, Datum] = None

    def __init__(self):
        self._data = {}

    def log(self, expr: Expr, time_spent: float, root: Node):
        self._data[str(expr)] = Datum(expr, time_spent, root)

    @property
    def data(self) -> Dict[str, float]:
        return self._data

    def sort(self):
        """sorts the data by time spent on each integral, from most time to least time."""
        self._data = dict(sorted(self._data.items(), key=lambda x: x[1].time_spent, reverse=True))

    def dump(self):
        self.sort()

        with open("integration_log.txt", "w") as f:
            for k, v in self._data.items():
                f.write(f"{k}: {v.time_spent}\n")

            # For the one with the most time spent, print the tree.
            f.write("\n\n\n")
            f.write("Most time spent: \n")
            print_solution_tree(self._data[list(self._data.keys())[0]].root, func=lambda x: f.write(f"{x}\n"))
            f.write("\n\n\n")
            print_tree(self._data[list(self._data.keys())[0]].root, func=lambda x: f.write(f"{x}\n"))

    def plot(self):
        import matplotlib.pyplot as plt

        self.sort()
        x = list(self._data.keys())
        y = [v.time_spent for v in self._data.values()]
        plt.bar(x, y)
        plt.savefig("integration_log.png")


def log_time(func):
    """Decorator to cast all arguments to Expr."""

    def wrapper(*args, **kwargs) -> "Expr":
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        return result

    return wrapper
