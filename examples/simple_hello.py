import argmagic


def hello(name: str):
    """
    Say hello to name.

    Args:
        name: Your name.

    Raises:
        Nothing.

    Returns:
        Nothing.
    """
    print("Hello", name)


if __name__ == "__main__":
    argmagic.argmagic(hello)
