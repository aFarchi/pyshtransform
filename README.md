# pyshtransform

[![PyPI version](https://badge.fury.io/py/pyshtransform.svg)](https://badge.fury.io/py/pyshtransform)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation status](https://app.readthedocs.org/projects/pyshtransform/badge/?version=latest)](https://app.readthedocs.org/projects/pyshtransform/badge/?version=latest)

`pyshtransform` is a python toolbox that implements spherical harmonics transformations.

- [Documentation](https://pyshtransform.readthedocs.io)
- [Source code](https://github.com/aFarchi/pyshtransform)
- [Issue tracker](https://github.com/aFarchi/pyshtransform/issues)

## Installation

Install using `pip`:

    $ pip install pyshtransform

## Usage

...

## Acknowledgements

This package contains some functions inspired by [SHTOOLS](https://shtools.github.io/SHTOOLS/).

## Todo-list

- write documentation

## Snippets

Code linting and formatting:
```sh
pixi run ruff check
pixi run ruff format
```

Static type checking:
```sh
pixi run mypy src/
```

Run test suite:
```sh
pixi run pytest -vs tests/
```

Generate api doc:
```sh
pixi run sphinx-apidoc -f -o docs/source/api/ src/
```

Generate doc:
```sh
pixi run sphinx-build -M clean docs/source/ docs/build/
pixi run sphinx-build -M html docs/source/ docs/build/
```

