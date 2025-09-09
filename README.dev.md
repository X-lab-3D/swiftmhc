# `swiftmhc` developer documentation

If you're looking for user documentation, go [here](README.md).

## Linting and formatting

We use [ruff](https://docs.astral.sh/ruff/) for linting, sorting imports and formatting code. The configurations of `ruff` are set in [pyproject.toml](pyproject.toml) file.

Running the linters and formatters requires an activated virtual environment with the development tools installed.

```shell
# Lint all files in the current directory.
ruff check .

# Lint all files in the current directory, and fix any fixable errors.
ruff check . --fix

# Format all files in the current directory
ruff format .

# Format a single python file
ruff format filename.py
```

## Static typing

We use [inline type annotation](https://typing.readthedocs.io/en/latest/source/libraries.html#how-to-provide-type-annotations) for static typing rather than stub files (i.e. `.pyi` files).

Since Python 3.12 is used as dev environment, you may see various typing issues at runtime. Here is [a guide to solve the potential runtime issues](https://mypy.readthedocs.io/en/stable/runtime_troubles.html).

By default, we use `from __future__ import annotations` at module level to stop evaluating annotations at function definition time (see [PEP 563](https://peps.python.org/pep-0563/)), which would solve most of compatibility issues between different Python versions. Make sure you're aware of the [caveats](https://mypy.readthedocs.io/en/stable/runtime_troubles.html#future-annotations-import-pep-563).

We use [Mypy](http://mypy-lang.org/) as static type checker:

```
# install mypy
pip install mypy

# run mypy
mypy swiftmhc
```

Mypy configurations are set in [pyproject.toml](pyproject.toml) file.

For more info about static typing and mypy, see:
- [Static typing with Python](https://typing.readthedocs.io/en/latest/index.html#)
- [Mypy doc](https://mypy.readthedocs.io/en/stable/)


## Versioning

Updating the version of the swiftmhc package is done with make command `update-version`, e.g.

```shell
make update-version CURRENT_VERSION=0.0.1 NEW_VERSION=0.0.2
```

This command will update the version in the following files:
- `swiftmhc/__version__.py`
- `pyproject.toml`
- `CITATION.cff`
