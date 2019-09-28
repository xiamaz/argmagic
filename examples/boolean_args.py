from argmagic import argmagic


def bool_hello(name: str, valid: bool):
    if valid:
        print(name, "is valid")
    else:
        print(name, "is invalid")


argmagic(bool_hello)
