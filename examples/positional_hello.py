from argmagic import argmagic


def main(name: str, other: str):
    print("Hello", name, "I am", other)


argmagic(main, positional=["name"])
