import unittest
from typing import Union, List, Tuple, Dict
from argparse import ArgumentParser

import argmagic


def no_docstring():
    pass


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
            ("no_docstring", no_docstring, {
                "description": "",
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
                    self.assertEqual(typefun(testvalue), expected)


class CreateArgparseParserTestCase(unittest.TestCase):

    def test_simple(self):
        function_info = {
            "name": "simple", "description": "testdesc", "args": {
                "name": {
                    "doc": "A simple name",
                    "typefun": str,
                },
                "number": {
                    "doc": "A simple number",
                    "typefun": int,
                }
            }
        }
        parser = argmagic.create_argparse_parser(function_info, use_flags=False)
        self.assertEqual(parser.prog, "simple")
        self.assertEqual(parser.description, "testdesc")

        optionals = parser._get_optional_actions()
        dests = [o.dest for o in optionals]
        helps = [o.help for o in optionals]
        for name, arg in function_info["args"].items():
            self.assertIn(arg["doc"], helps)
            self.assertIn(name, dests)

        args = parser.parse_args(["--name", "testname", "--number", "6"])
        self.assertEqual(args.name, "testname")
        self.assertEqual(args.number, 6)


class ExtractArgsTestCase(unittest.TestCase):
    def test_merging(self):
        cases = [
            ({"a": 1, "b": 4}, {"b": None}, {"a": 1, "b": 4}),
            ({"b": 4}, {"a": None}, {"a": None, "b": 4}),
            ({"b": None}, {"a": None}, {"a": None, "b": None})
        ]
        for dict_a, dict_b, expected in cases:
            result = argmagic.extract_args(dict_a, dict_b)
            self.assertDictEqual(result, expected)


class ArgsValidationTestCase(unittest.TestCase):
    def test_validate(self):
        parser = ArgumentParser()
        function_info = {
            "args": {
                "a": {
                    "required": True
                },
                "b": {
                    "required": False
                },
            }
        }

        cases = [
            ({"a": None, "b": None}, False),
            ({"a": None, "b": 2}, False),
            ({"a": 1, "b": 2}, True),
            ({"a": 1, "b": None}, True),
        ]
        for args, valid in cases:
            if not valid:
                with self.assertRaises(SystemExit):
                    argmagic.validate_args(parser, function_info, args)
            else:
                argmagic.validate_args(parser, function_info, args)
