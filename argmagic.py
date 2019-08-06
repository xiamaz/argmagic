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
        elif stripped  == "Returns:":
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


def parse_function_typehints(function: Callable) -> dict:
    """Parse typehints for the given function and return a dict mapping
    parameters to type hints."""
    sig_params = inspect.signature(function).parameters
    hints = {name: param.annotation for name, param in sig_params.items()}
    return hints


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


def infer_typefun(typehint):
    typefun = None
    # try to use directly if not a typehint generic
    if not isinstance(typehint, typing._GenericAlias):
        typefun = typehint
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


def add_target(
        target: Callable,
        parser: ArgumentParser, docstrings: dict = None):
    """Add target to general parser."""
    if docstrings is None:
        docstrings = parse_docstring(target)
    arg_funs = {}
    args = parse_function_typehints(target)
    for arg, typehint in args.items():
        arg_doc = docstrings["args"].get(arg, "")
        argopt = f"--{arg}"
        typefun = infer_typefun(typehint)
        parser.add_argument(argopt, help=arg_doc, type=typefun)
        arg_funs[arg] = typefun

    return arg_funs


def argmagic_group(target: Callable, parser: ArgumentParser):
    name = str(target.__name__)
    docstrings = parse_docstring(target)

    target_group = parser.add_argument_group(
        title=name, description=docstrings["description"])

    add_target(target, target_group, docstrings=docstrings)


def argmagic(target: Callable, environment=True):
    """Generate a parser based on target signature and execute it."""
    name = str(target.__name__)
    docstrings = parse_docstring(target)
    parser = ArgumentParser(
        prog=name,
        description=docstrings["description"])

    args = add_target(target, parser, docstrings=docstrings)

    parsed_args = parser.parse_args()
    env_args = {
        arg: args[arg](raw) if raw else None
        for arg, raw in
        [(arg, os.environ.get(arg.upper(), None)) for arg in args]
    }
    cli_args = {arg: getattr(parsed_args, arg) for arg in args}
    target_args = {}
    for arg in args:
        if env_args[arg] is None and cli_args[arg] is None:
            continue
        if cli_args[arg] is None:
            target_args[arg] = env_args[arg]
        else:
            target_args[arg] = cli_args[arg]
    return target(**target_args)


def hello(name: str = None):
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
    result = argmagic(hello)
