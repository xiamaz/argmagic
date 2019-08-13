from typing import Dict, List
from argmagic import argmagic


def hello_fmt(values: Dict[str, List[int]]):
    """Print the sum per category.

    Args:
        values: Dict containing lists of int per category.
    """

    print(", ".join(f"{k}: {sum(v)}" for k, v in values.items()))


if __name__ == "__main__":
    argmagic(hello_fmt)
