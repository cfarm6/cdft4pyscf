# Examples

## Constrained DFT script

An end-to-end example is available at:

- `examples/033_constrained_dft.py`

Run it from the repository root:

```bash
pixi run -e dev python examples/033_constrained_dft.py
```

## What the example demonstrates

- Defining atom-list regions
- Applying both regional electron-number and total net-charge constraints
- Configuring solver tolerances and verbosity
- Reading final energy, constraint values, and optimized Lagrange multipliers
