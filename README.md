# cr_nematic_structure
[![License: GPL 2.0](https://img.shields.io/github/license/jonaspleyer/cr_nematic_structure?style=flat-square)](https://opensource.org/license/gpl-2-0/)
[![Test](https://img.shields.io/github/actions/workflow/status/jonaspleyer/cr_nematic_structure/test.yml?label=Test&style=flat-square)](https://github.com/jonaspleyer/cr_nematic_structure/actions)
[![CI](https://img.shields.io/github/actions/workflow/status/jonaspleyer/cr_nematic_structure/CI.yml?label=CI&style=flat-square)](https://github.com/jonaspleyer/cr_nematic_structure/actions)
[![PyPI - Version](https://img.shields.io/pypi/v/cr_nematic_structure?style=flat-square)]()

## Installation
Use [maturin](https://github.com/PyO3/maturin) to build the project.
The following instructions are for nix-like operating systems.
Please use the resources at [python.org](https://python.org/) to adjust them for your needs.
First we create a virtual environment and activate it.

```
python3 -m venv .venv
source .venv/bin/activate
```

If you have not yet used maturin, install it.
We recommend that you use the [uv](https://github.com/astral-sh/uv) package manager for dependency
management.

```
uv pip install maturin
```

To install `cr_nematic_structure`, you can either install it directly from pypi.org

```
uv pip install cr_nematic_structure
```

or by cloning the github repository.

```
git clone https://github.com/jonaspleyer/cr_nematic_structure
cd cr_nematic_structure
maturin develop -r --uv
```

Now you are ready to use `cr_nematic_structure`.
If you modify the source code, you have rerun the last command in order to install the updated
version.
