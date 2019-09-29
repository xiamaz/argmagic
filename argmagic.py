"""
Automagically generate a CLI interface based on function type hints and
docstring.

Either wrap your function in argmagic or fill a python dataclass structure.
"""
import os
import json
import enum
import inspect
from collections import defaultdict
from argparse import ArgumentParser
import typing
from typing import Union, Callable, Any, Iterable


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
    description_lines = []

    docstring = inspect.getdoc(obj)
    if docstring is None:
        return {"description": "", "args": {}}

    for line in docstring.split("\n"):
        stripped = line.strip()
        if stripped == "Args:":
            cur_section = DocstringState.ARGS
            continue
        elif stripped == "Raises:":
            cur_section = DocstringState.RAISES
        elif stripped == "Returns:":
            cur_section = DocstringState.RETURNS

        if cur_section == DocstringState.DESCRIPTION:
            description_lines.append(line)
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
    description = "\n".join(description_lines).rstrip("\n")

    return {
        "description": description,
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


def remove_start_end(token, start, end):
    """Remove the enclosing tags"""
    token = token.strip()
    if not (token.startswith(start) and token.endswith(end)):
        raise ValueError
    return token[1:-1]


NESTING_BEGIN = ["(", "{", "["]
NESTING_END = [")", "}", "]"]


def tokenize(line, sep=",", num=None):
    """Tokenize using the current sep, but do not split inside nesting tags."""
    tokens = []
    start = 0
    nesting_level = 0
    splits = 0
    for i, char in enumerate(line):
        if char in NESTING_BEGIN:
            nesting_level += 1

        if char in NESTING_END:
            nesting_level -= 1
        if nesting_level == 0 and char == sep:
            tokens.append(line[start:i])
            start = i + 1
            splits += 1
        if num is not None and splits == num:
            break
    tokens.append(line[start:])
    return tokens


def make_tuple_parser(argfuns):

    def tuple_parse(input_token):
        input_token = remove_start_end(input_token, "(", ")")
        tokens = tokenize(input_token, ",")
        value = tuple(
            typefun(token.strip()) for typefun, token in zip(argfuns, tokens)
        )
        return value

    return tuple_parse


def make_list_parser(argfun, start="[", end="]"):

    def list_parse(input_token):
        input_token = remove_start_end(input_token, "[", "]")

        tokens = tokenize(input_token, ",")

        value = [argfun(token.strip()) for token in tokens]
        return value

    return list_parse


def make_dict_parser(keyfun, valfun):

    def dict_parse(input_token):
        input_token = remove_start_end(input_token, "{", "}")

        kvtokens = tokenize(input_token, ",")

        dictvalue = {}
        for kvtoken in kvtokens:
            rawkey, rawval = tokenize(input_token, ":", 1)
            key = keyfun(rawkey)
            value = valfun(rawval)
            dictvalue[key] = value

        return dictvalue

    return dict_parse


def is_simple_bool(typehint) -> bool:
    """Check if typehint is bool or Union[None, bool]."""
    if typehint is bool:
        return True
    if isinstance(typehint, typing._GenericAlias) and bool in typehint.__args__:
        return True
    return False


def make_type_parser(argfun):
    def bool_parse(input_token):
        return argfun(int(input_token))

    def type_parse(input_token):
        if input_token is None:
            raise ValueError(f"Input should be {argfun} but was {input_token}")
        return argfun(input_token)

    if argfun is bool:
        return bool_parse

    return type_parse


def infer_typefun(typehint):
    typefun = None
    # try to use directly if not a typehint generic
    if not isinstance(typehint, typing._GenericAlias):
        typefun = make_type_parser(typehint)
    elif typehint.__origin__ == typing.Union:
        union_args = [infer_typefun(arg) for arg in typehint.__args__]
        typefun = make_union_parser(union_args)
    elif typehint.__origin__ in (typing.Tuple, tuple):
        tuple_fields = [infer_typefun(arg) for arg in typehint.__args__]
        typefun = make_tuple_parser(tuple_fields)
    elif typehint.__origin__ in (typing.List, list):
        listfun = infer_typefun(typehint.__args__[0])
        typefun = make_list_parser(listfun)
    elif typehint.__origin__ in (typing.Dict, dict):
        key_fun, val_fun = [infer_typefun(arg) for arg in typehint.__args__]
        typefun = make_dict_parser(key_fun, val_fun)
    else:
        print(typehint.__origin__)
        raise NotImplementedError(f"{typehint} not implemented")

    return typefun


def get_function_info(target: Callable) -> dict:
    name = str(target.__name__)
    docstrings = parse_docstring(target)

    arg_info = {}
    sig_params = inspect.signature(target).parameters
    for name, param in sig_params.items():
        typehint = param.annotation
        required = False
        if param.default is inspect.Parameter.empty:
            typehint = Union[typehint, None]
            required = True
        typefun = infer_typefun(typehint)
        arg_info[name] = {
            "doc": docstrings["args"].get(name, ""),
            "typehint": typehint,
            "typefun": typefun,
            "required": required,
        }

    return {
        "name": name,
        "description": docstrings["description"],
        "args": arg_info
    }


def add_argument(
        parser: ArgumentParser,
        name: str,
        arg_info: dict,
        use_flags: bool):
    if use_flags and is_simple_bool(arg_info["typehint"]):
        parser.add_argument(
            f"--{name}",
            help=arg_info["doc"],
            action="store_true")
    else:
        parser.add_argument(
            f"--{name}",
            help=arg_info["doc"],
            type=arg_info["typefun"])


def create_argparse_parser(
        function_info: dict,
        use_flags: bool,
        usage: str = "",
        parser: ArgumentParser = None,
        positional: Iterable[str] = ()):
    """Create an argparse parser."""
    if parser is None:
        parser = ArgumentParser(
            prog=function_info["name"],
            description=function_info["description"])
    else:
        parser = parser.add_argument_group(
            title=function_info["name"],
            description=function_info["description"])

    for name, arg_info in function_info["args"].items():
        if name in positional:
            continue

        add_argument(parser, name, arg_info, use_flags=use_flags)

    for name in positional:
        add_argument(
            parser, name, function_info["args"][name], use_flags=use_flags)
    return parser


def parse_env_args(function_info: dict) -> dict:
    env_args = {}
    for name, arg_info in function_info["args"].items():
        try:
            raw = os.environ[name.upper()]
            env_args[name] = arg_info["typefun"](raw)
        except KeyError:
            continue
    return env_args


def extract_args(env_args: dict, parser_args: dict) -> dict:
    """Combine arguments from env variables and parser results."""
    target_args = env_args.copy()
    for name, arg in parser_args.items():
        if arg is not None or name not in target_args:
            target_args[name] = arg
    return target_args


def validate_args(
        parser: ArgumentParser,
        function_info: dict,
        target_args: dict) -> None:
    """Check whether required args are included."""
    for name, arg_info in function_info["args"].items():
        if arg_info["required"] and target_args[name] is None:
            parser.error(f"{name} is required but not given")


def argmagic(
        target: Callable,
        positional: Iterable = (),
        environment: bool = True,
        use_flags: bool = False,
        parser: ArgumentParser = None):
    """Generate a parser based on target signature and execute it."""

    function_info = get_function_info(target)

    if environment:
        env_args = parse_env_args(function_info)
        usage_text = ENV_TEXT
    else:
        env_args = {}
        usage_text = ""

    parser = create_argparse_parser(
        function_info,
        usage=usage_text,
        parser=parser,
        use_flags=use_flags,
        positional=positional)

    args = parser.parse_args()
    parser_args = {name: getattr(args, name) for name in function_info["args"]}

    target_args = extract_args(env_args, parser_args)

    validate_args(parser, function_info, target_args)

    return target(**target_args)
