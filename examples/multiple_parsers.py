from argmagic import argmagic


def main(hello: str):
    """Test"""
    print("Hello", hello)


def other(bello: str, strello: int):
    """Yolo"""
    print("Bello", bello, strello + 10)


argmagic([main, other])
