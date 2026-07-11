# Neural Operators for Nonlinear Heat Conduction

A structured, article-style study of physics-informed and data-driven **neural operators**
for 1-D nonlinear heat conduction, built as a Jupyter Book and hosted on GitHub Pages.

## Structure

**I. Foundations — Solvers and Neural Networks**
- `1D_Thermal_PINN_FEM_FDM` — finite-element / finite-difference solvers and a PINN baseline.

**II. Steady-State Neural Operators**
- `FNO_STEADY_STATE` — the Fourier Neural Operator baseline.
- `FNO_loss_comparison` — one FNO under three losses; concludes with the weighted-residual (FOL) loss.
- `architecture_comparison` — FNO vs DeepONet vs iFOL vs Transformer; concludes with FNO.

**III. Transient Neural Operators**
- `FNO_transient_data_driven` — data-driven transient operator.
- `FNO_transient_weighted_residual` — weighted-residual (label-free) transient operator.
- `transient_comparison` — mean rollout error per step, data-driven vs weighted-residual.

## Build & deploy

```bash
pip install -r requirements.txt
jupyter-book build .
ghp-import -n -p -f _build/html      # or just push to main; the GitHub Action rebuilds
```

Figures are published from the outputs stored in each notebook (`execute_notebooks: off`).
See `CLAUDE.md` for project conventions and the open to-do list.
