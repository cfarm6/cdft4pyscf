# Examples

## Constrained DFT scripts

End-to-end examples are available at:

- `examples/h2o_electron_number.py`
- `examples/atcne_net_charge.py`
- `examples/fa-aq_example.py`

Run one from the repository root:

```bash
pixi run -e dev python examples/h2o_electron_number.py
```

## What the example demonstrates

- Defining atom-list regions
- Applying both regional electron-number and total net-charge constraints
- Configuring solver tolerances and verbosity
- Reading final energy, constraint values, and optimized Lagrange multipliers
