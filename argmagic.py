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
from typing import Union, Callable, Any, Iterable, Dict

from dataclasses import dataclass


FUNCTION_ENV_NOTICE = "Env vars for parameters are formatted as FUNNAME_PARAM"
ENV_NOTICE = "Env var setting params by UPPERCASE_NAME"


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
    if hasattr(typehint, "__args__") and bool in typehint.__args__:
        return True
    return False


def make_type_parser(argfun):
    def bool_parse(input_token):
        return argfun(int(input_token))

    def type_parse(input_token):
        if input_token is None:
            raise ValueError(f"Input should be {argfun} but was {input_token}")
        return argfun(input_token)

    def pass_parse(input_token):
        return input_token

    if argfun is inspect._empty:
        return pass_parse

    if argfun is bool:
        return bool_parse

    return type_parse


def infer_typefun(typehint):
    typefun = None
    # try to use directly if not a typehint generic
    if not hasattr(typehint, "__origin__"):
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


@dataclass
class ArgumentInformation:
    typehint: "Typehint"
    default: Any
    docstring: str = ""
    converter: Callable = None
    required: bool = False

    def __post_init__(self):
        if self.default is inspect.Parameter.empty:
            self.typehint = Union[self.typehint, None]
            self.required = True

        if self.converter is None:
            self.converter = infer_typefun(self.typehint)


def get_function_info(target: Callable) -> Dict[str, ArgumentInformation]:
    function_name = str(target.__name__)
    docstrings = parse_docstring(target)

    arg_info = {}
    sig_params = inspect.signature(target).parameters
    for name, param in sig_params.items():
        docstring = docstrings["args"].get(function_name, "")
        arg_info[name] = ArgumentInformation(
            param.annotation, param.default, docstring=docstring)

    return {
        "name": function_name,
        "description": docstrings["description"],
        "args": arg_info
    }


def add_argument(
        parser: ArgumentParser,
        name: str,
        arg_info: ArgumentInformation,
        use_flags: bool):
    if use_flags and is_simple_bool(arg_info.typehint):
        parser.add_argument(
            name, help=arg_info.docstring, action="store_true")
    else:
        parser.add_argument(
            name, help=arg_info.docstring, type=arg_info.converter)


def add_arguments_to_parser(
        parser: ArgumentParser,
        arguments: dict,
        use_flags: bool,
        positional: Iterable[str] = ()):
    """Create an argparse parser."""
    for name, arg_info in arguments.items():
        if name in positional:
            continue

        add_argument(parser, f"--{name}", arg_info, use_flags=use_flags)

    for name in positional:
        add_argument(
            parser, name, arguments[name], use_flags=use_flags)
    return parser


def parse_env(env_names, arguments: dict) -> dict:
    env_args = {}
    for env_name, arg_name in env_names.items():
        arg_info = arguments[arg_name]
        try:
            raw = os.environ[env_name]
            env_args[arg_name] = arg_info.converter(raw)
        except KeyError:
            continue
    return env_args


def merge_args(env_args: dict, parser_args: dict) -> dict:
    """Combine arguments from env variables and parser results."""
    target_args = env_args.copy()
    for name, arg in parser_args.items():
        if arg is not None or name not in target_args:
            target_args[name] = arg
    return target_args


class FunctionInformation:
    def __init__(
            self,
            function: Callable,
            positional: Iterable[str] = (),
            prefix=False):
        self.function = function
        self._info = get_function_info(function)

        self.positional = positional

        self.prefix = self._info["name"]
        self.description = self._info["description"]

        self.args = self._info["args"]
        self.env_args = {
            f"{self.prefix}_{n}".upper() if prefix else n.upper(): n
            for n in self.args
        }

    def add_parser(self, subparsers=None, use_flags=False):
        if subparsers is None:
            parser = ArgumentParser(
                prog=self.prefix,
                epilog=ENV_NOTICE,
                description=self.description)
        else:
            parser = subparsers.add_parser(
                self.prefix,
                help=self.description,
                description=self.description)

        parser.set_defaults(fun=self)

        add_arguments_to_parser(
            parser,
            self.args,
            use_flags=use_flags,
            positional=self.positional
        )
        return parser

    def parse_env_cli(self, parsed, environment=True):
        """Get args from env and cli."""
        cli_args = {name: getattr(parsed, name) for name in self.args}
        if environment:
            env_args = parse_env(self.env_args, self.args)
        else:
            env_args = {}

        all_args = merge_args(env_args, cli_args)
        return all_args


def validate_args(
        parser: ArgumentParser,
        funtion_info: FunctionInformation,
        parsed_args: dict) -> None:
    """Check whether required args are included."""
    for name, arg_info in funtion_info.args.items():
        if arg_info.required and parsed_args[name] is None:
            parser.error(f"{name} is required but not given")


def argmagic(
        target,
        positional: Iterable = (),
        environment: bool = True,
        use_flags: bool = False,
        description: str = "",
        parser: ArgumentParser = None,
        args: list = None):
    """Generate a parser based on target signature and execute it."""
    try:
        infos = [
            FunctionInformation(t, positional=positional, prefix=True)
            for t in target
        ]
        parser = ArgumentParser(
            description=description,
            epilog=FUNCTION_ENV_NOTICE,
        )
        subparsers = parser.add_subparsers(help="Available subcommands")
        for info in infos:
            info.add_parser(subparsers, use_flags=use_flags)
    except TypeError:
        info = FunctionInformation(
            target, positional=positional, prefix=False)
        parser = info.add_parser(use_flags=use_flags)

    args = parser.parse_args(args)

    # get the selected active function to get args for
    if not hasattr(args, "fun"):
        parser.print_help()
        parser.exit(1)

    active = args.fun
    function_args = active.parse_env_cli(args, environment=environment)

    validate_args(parser, active, function_args)

    return active.function(**function_args)
