# Argmagic

Automatically generate argparse based env-var/CLI-interface from a given function.

Example:

Given a function with a docstring and type hints.

```python
def hello(name: str):
    '''
    Say hello to name.

    Args:
        name: Your name.

    Raises:
        Nothing.

    Returns:
        Nothing.
    '''
    print('Hello', name)
```

Create a CLI interface:

```python
argmagic(hello)
```

Argmagic will call the function with all parameters filled from CLI arguments.

```sh
$ ./hello.py -h
usage: hello [-h] [--name NAME]

Say hello to name.

optional arguments:
  -h, --help   show this help message and exit
  --name NAME  Your name.
```

Additionally all specified parameters can also be defined via environment
variables.

```sh
$ NAME=test hello.py
Hello test
```

These can then again be overriden by CLI arguments.

```sh
$ NAME=test hello.py --name something
Hello something
```
