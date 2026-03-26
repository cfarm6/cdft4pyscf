# cdft4pyscf

[License](https://github.com/cfarm6/cdft4pyscf/blob/master/LICENSE)
[Powered by: Pixi](https://pixi.sh)
[Code style: ruff](https://github.com/astral-sh/ruff)
[Typing: ty](https://github.com/astral-sh/ty)
[GitHub Workflow Status](https://github.com/cfarm6/cdft4pyscf/actions/)
[Codecov](https://codecov.io/gh/cfarm6/cdft4pyscf)

## Description

This package adds support for performing constrained density functional theory (cDFT) calculations using the PySCF and the GPU4PySCF. Constraints, regions, and property outputs are fully typed with PyDantic dataclasses to ensure type safety and documentation. The package is designed to be used in a workflow-based manner, with the user providing the constraints, regions, and property outputs they want to calculate. The package will then handle the rest of the calculation, including the solving for the constrained potential $V_c$.

### Constraints

The following types of constraints are supported:

- Number of electrons (non-integer values are supported)
- Net charge

### Regions

The currently supported regions are:

- Atom lists

### Population Analysis

The current supported methods for population analysis are:

#### Löwdin

The Löwdin population analysis for constrained region, $C$, is done using the following equation:

```math
\begin{align*}
    N_c & = \sum\limits_{\mu \in C}(\mathbf{S}^{1/2}\mathbf{P}\mathbf{S}^{1/2})_{\mu\mu} \\
    & = \sum\limits_{\mu \in C}\sum\limits_{\nu\lambda}S_{\mu\nu}^{1/2}P_{\nu\lambda}S_{\lambda\mu}^{1/2} \\
    & = \sum\limits_{\nu\lambda}P_{\nu\lambda}\sum\limits_{\mu \in C}S_{\mu\nu}^{1/2}S_{\lambda\mu}^{1/2} \\
    &  = \text{Tr}(\mathbf{P}\mathbf{w_c^L})
\end{align*}
```

where $\mathbf{S}^{1/2}$ is the square root of the overlap matrix and $\mathbf{w_c^L}$ is the Löwdin weight matrix defined as:

```math
w_{c\lambda\nu}^L = \sum\limits_{\lambda \in C}S_{\lambda\mu}^{1/2}S_{\mu\nu}^{1/2}
```


### Solving For $`V_c`$

Solving for $V_c$ in the cDFT is done using the direct optimization approach explained in section 2.2 of [1]. The constraint is satisfied by first solving the Kohn-Sham equations for the constrained density matrix $\rho_c$ and then using the constrained density matrix to solve for $V_c$. For using DIIS to solve the KS equations the process from [1] is as follows:

1. Construct the current Fock matrix, $\mathbf{F}$ from the current density matrix, $\mathbf{P}$.
2. Use the optimal $V_c$ from the last iteration to build the constrained Fock matrix $\mathbf{F}_c = \mathbf{F} + V_c \mathbf{w}_c$.
3. Determine the DIIS linear coefficients $\mathbf{d}^i$, and replace the current Fock matrix with $\mathbf{F}^\ast = \sum\limits_{i}^{n}\mathbf{d}^i \mathbf{F}_c^i$.
4. Fix $\mathbf{F}^\ast$, and optimize $V_c$ again until the constraints are satisfied.
5. Obtain the new density matrix from $\mathbf{F}^\ast$ and the optimized $V_c$. The new $\mathbf{P}$ and the optimized $V_c$ are then fed into the next iteration, and the above steps are repeated until convergence.

## References

[1] Q. Wu and T. Van Voorhis, “Constrained Density Functional Theory and Its Application in Long-Range Electron Transfer,” J. Chem. Theory Comput., vol. 2, no. 3, pp. 765–774, May 2006, doi: 10.1021/ct0503163.

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [jevandezande/pixi-cookiecutter](https://github.com/jevandezande/pixi-cookiecutter) project template.
