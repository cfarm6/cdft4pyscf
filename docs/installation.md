# Installation

## Requirements

- Python 3.14+
- A working C/C++ toolchain for scientific Python dependencies
- Optional: CUDA + supported GPU if you want to use the `gpu` backend

## Install with Pixi (recommended for development)

From the repository root:

```bash
pixi install
pixi shell -e dev
```

This installs the package in editable mode and the development toolchain.

For optional GPU support with Pixi, use the GPU environments:

```bash
pixi shell -e gpu
# or for development tools + GPU stack
pixi shell -e dev-gpu
```

## Install with `pip`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For GPU support, install the optional `gpu` extra:

```bash
pip install -e ".[gpu]"
```

## Docs environment

The project includes a dedicated Pixi docs environment with Zensical:

```bash
pixi run -e docs docs-serve
```

Use this command to preview docs locally while editing.
