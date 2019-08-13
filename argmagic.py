"""
Automagically generate a CLI interface based on function type hints and
docstring.

Either wrap your function in argmagic or fill a python dataclass structure.
"""
import os
import enum
import inspect
from collections import defaultdict
from argparse import ArgumentParser
import typing
from typing import Union, Callable, Any


ENV_TEXT = """All arguments are also available as environment variables in
all-upper-case."""


class DocstringState(enum.Enum):
    """Possible sections in a docstring."""

    DESCRIPTION = 1
    ARGS = 2
    RAISES = 3
    RETURNS = 4


def parse_docstring(obj: Any) -> dict:
    """Parse the given obj. For now only support Google-style docstrings."""
    args_lines = defaultdict(list)
    cur_section = DocstringState.DESCRIPTION
    current_arg = None
    indent_level = None
    description = []

    for line in inspect.getdoc(obj).split("\n"):
        stripped = line.strip()
        if stripped == "Args:":
            cur_section = DocstringState.ARGS
            continue
        elif stripped == "Raises:":
            cur_section = DocstringState.RAISES
        elif stripped == "Returns:":
            cur_section = DocstringState.RETURNS

        if cur_section == DocstringState.DESCRIPTION:
            description.append(line)
        elif cur_section == DocstringState.ARGS:
            indent = len(line) - len(line.lstrip())
            if indent_level is None:
                indent_level = indent

            if current_arg is None or indent == indent_level:
                current_arg, desc = line.split(":", 1)
                current_arg = current_arg.strip()
                args_lines[current_arg].append(desc.strip())
            else:
                args_lines[current_arg].append(line.strip())

    args_strings = {arg: "\n".join(lines) for arg, lines in args_lines.items()}

    return {
        "description": "\n".join(description),
        "args": args_strings
    }


def make_union_parser(argfuns):
    def union_parse(token):
        value = None
        for fun in argfuns:
            try:
                value = fun(token)
                break
            except ValueError:
                continue
        return value
    return union_parse


def make_tuple_parser(argfuns):
    def tuple_parse(input_token):
        tokens = [s.strip() for s in input_token.split(",")]
        value = tuple(
            typefun(token) for typefun, token in zip(argfuns, tokens))
        return value
    return tuple_parse


def make_list_parser(argfun):
    def list_parse(input_token):
        value = [argfun(s.strip()) for s in input_token.split(",")]
        return value
    return list_parse


def make_dict_parser(keyfun, valfun):
    def dict_parse(input_token):
        value = {
            keyfun(rawkey.strip()): valfun(rawval.strip())
            for rawkey, rawval in
            [token.split(":", 1) for token in input_token.split(",")]
        }
        return value
    return dict_parse


def make_type_parser(argfun):
    def type_parse(input_token):
        if input_token is None:
            raise ValueError(f"Input should be {argfun} but was {input_token}")
        return argfun(input_token)
    return type_parse


def infer_typefun(typehint):
    typefun = None
    # try to use directly if not a typehint generic
    if not isinstance(typehint, typing._GenericAlias):
        typefun = make_type_parser(typehint)
    elif typehint.__origin__ == typing.Union:
        union_args = [infer_typefun(arg) for arg in typehint.__args__]
        typefun = make_union_parser(union_args)
    elif typehint.__origin__ == typing.Tuple:
        tuple_fields = [infer_typefun(arg) for arg in typehint.__args__]
        typefun = make_tuple_parser(tuple_fields)
    elif typehint.__origin__ == typing.List:
        listfun = infer_typefun(typehint.__args__[0])
        typefun = make_list_parser(listfun)
    elif typehint.__origin__ == typing.Dict:
        key_fun, val_fun = [infer_typefun(arg) for arg in typehint.__args__]
        typefun = make_dict_parser(key_fun, val_fun)
    else:
        raise NotImplementedError(f"{typehint} not implemented")

    return typefun


def get_function_info(target: Callable) -> dict:
    name = str(target.__name__)
    docstrings = parse_docstring(target)

    arg_info = {}
    sig_params = inspect.signature(target).parameters
    for name, param in sig_params.items():
        typehint = param.annotation
        if param.default is inspect.Parameter.empty:
            typehint = Union[typehint, None]
        typefun = infer_typefun(typehint)
        arg_info[name] = {
            "doc": docstrings["args"].get(name, ""),
            "typehint": typehint,
            "typefun": typefun,
        }

    return {
        "name": name,
        "description": docstrings["description"],
        "args": arg_info
    }


def parsermagic(function_info: dict, usage=""):
    """Create an argparse parser and parse it and return parsed arguments.
    Returns:
        Dictionary containing parsed values for arguments.
    """

    parser = ArgumentParser(
        prog=function_info["name"],
        description=function_info["description"])

    for name, arg_info in function_info["args"].items():
        parser.add_argument(
            f"--{name}",
            help=arg_info["doc"],
            type=arg_info["typefun"])

    args = parser.parse_args()
    parser_args = {name: getattr(args, name) for name in function_info["args"]}
    return parser_args


def envmagic(function_info: dict):
    env_args = {}
    for name, arg_info in function_info["args"].items():
        raw = os.environ.get(name.upper(), None)
        env_args[name] = arg_info["typefun"](raw)
    return env_args


def extract_args(env_args: dict, parser_args: dict) -> dict:
    """Combine arguments from env variables and parser results."""
    target_args = env_args.copy()
    for name, arg in parser_args:
        target_args[name] = arg
    return target_args


def argmagic(target: Callable, environment=True):
    """Generate a parser based on target signature and execute it."""

    function_info = get_function_info(target)

    if environment:
        env_args = envmagic(function_info)
        usage_text = ENV_TEXT
    else:
        env_args = {}
        usage_text = ""

    parser_args = parsermagic(function_info, usage=usage_text)

    target_args = extract_args(env_args, parser_args)

    return target(**target_args)
