import argmagic
import unittest
from typing import Union, List, Tuple, Dict


def simple():
    """
    This is a function.
    """


def simple_1arg(hello: str):
    """This will print hello.

    Args:
        hello: Your name.
    """
    print("Hello", hello)


def simple_complex_desc():
    """ This will print hello and do more stuff such as:
        - cook your meals
        - start a startup
        - go to the moon

    Returns:
        Nothing
    """
    # but the docstring lied



class ParseDocstringTestCase(unittest.TestCase):

    def test_simple(self):
        cases = [
            ("simple", simple, {
                "description": "This is a function.",
                "args": {},
            }),
            ("simple_1arg", simple_1arg, {
                "description": "This will print hello.",
                "args": {
                    "hello": "Your name."
                }
            }),
            ("simple_complex_desc", simple_complex_desc, {
                "description": """This will print hello and do more stuff such as:
    - cook your meals
    - start a startup
    - go to the moon""",
                "args": {},
            }),
        ]
        for name, function, result in cases:
            with self.subTest(name=name):
                docstrings = argmagic.parse_docstring(function)
                self.assertDictEqual(docstrings, result)


class InferTypefunTestCase(unittest.TestCase):

    def test_primitive(self):
        strparser = argmagic.infer_typefun(str)
        self.assertEqual(strparser(0), "0")

        intparser = argmagic.infer_typefun(int)
        self.assertEqual(intparser("0"), 0)

        with self.assertRaises(ValueError):
            self.assertEqual(intparser("None"), 1)

        with self.assertRaises(ValueError):
            self.assertEqual(intparser("1.56"), 1)

        unionparser = argmagic.infer_typefun(Union[int, float])
        self.assertEqual(unionparser("1"), 1)
        self.assertEqual(unionparser("1.6"), 1.6)

    def test_nested_primitive(self):
        tests = [
            (
                "string tuple",
                argmagic.infer_typefun(Tuple[str, str]),
                [
                    ("(hello, world)", ("hello", "world")),
                    ("(hello, [test, a, b, c])", ("hello", "[test, a, b, c]")),
                ],
            ),
            (
                "string tuple list",
                argmagic.infer_typefun(Tuple[str, List[str]]),
                [
                    (
                        "(hello, [test, a, b, c])",
                        ("hello", ["test", "a", "b", "c"])
                    ),
                ],
            ),
            (
                "string dict list",
                argmagic.infer_typefun(Dict[str, List[str]]),
                [
                    (
                        "{hello: [test, a, b, c]}",
                        {"hello": ["test", "a", "b", "c"]}
                    ),
                ],
            ),
        ]
        for name, typefun, cases in tests:
            for testvalue, expected in cases:
                with self.subTest(name=name, value=testvalue):
                    self.assertEquals(typefun(testvalue), expected)
