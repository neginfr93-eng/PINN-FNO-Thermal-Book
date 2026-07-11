# CLAUDE.md — PINN-FNO-Thermal-Book

Project context for Claude Code. Read this at the start of every session.

## What this project is

A scientific Jupyter Book documenting physics-informed and data-driven neural-operator
methods for **nonlinear heat conduction**. Hosted on GitHub Pages at
`https://neginfr93-eng.github.io/PINN-FNO-Thermal-Book/`. The goal is a publication-quality,
article-style resource: rigorous explanations, LaTeX derivations, conceptual diagrams, and
clean code — not just annotated scripts.

## Tech stack

- **Jupyter Book** for site generation (`_config.yml`, `_toc.yml`).
- **JAX + optax** for the models; **matplotlib** for figures.
- **GitHub Pages** for hosting.
- Build: `jupyter-book build .`   Deploy: `ghp-import -n -p -f _build/html`

## Book structure (authoritative — see _toc.yml)

**I. Foundations — Solvers and Neural Networks**
- `1D_Thermal_PINN_FEM_FDM` — FEM & finite-difference solvers, then MLP/PINN.

**II. Steady-State Neural Operators**
- `FNO_STEADY_STATE` — the Fourier Neural Operator baseline.
- `fno_loss_comparison` — same FNO under 3 losses (MSE residual / weighted residual / data-driven); **conclusion: choose the weighted-residual (FOL) loss**, label-free and most accurate.
- `03_architecture_comparison` — FNO vs DeepONet vs iFOL vs Transformer, 3 capacity tiers; **conclusion: choose FNO**; Transformer excluded from the pipeline on runtime grounds.

**III. Transient Neural Operators**
- `01_FNO_transient` — data-driven transient FNO.
- `02_FOL_transient_physics_informed` — weighted-residual (FOL) transient FNO.
- `transient_comparison` — ONE plot: mean absolute error per rollout step, data-driven vs weighted-residual (replaces box plots with per-step means).
- Future work: a transient **Transformer** chapter.

## Writing conventions

- Each chapter follows a research-paper arc: Problem statement -> Reference solver (FEM) ->
  Architecture -> Training objective -> Results (in-distribution) -> Results (out-of-distribution) -> Discussion.
- Scientific chapter titles (no "Step 1", "Step 2").
- Article-style markdown with LaTeX. Conceptual SVG diagrams live in `figs/` and are embedded
  in the notebooks (base64) so they render even if the file is moved.
- Results prose explains **how to read** a figure rather than hard-coding numbers, so it stays
  correct after re-runs.

## Physics (shared across chapters)

Nonlinear conductivity `k(x,T) = alpha(x) * (0.5 + T^2)` (transient) or `k0(x)*(1+beta*T^2)` (steady).
Reference solutions from finite elements: backward-Euler implicit stepping (transient),
Newton-Raphson (steady). Weak-form / weighted-residual (Galerkin) loss = FEM residual as the
training signal, no labels.

## Status

Done and in the repo (from prior work):
- `01_FNO_transient`, `02_FOL_transient_physics_informed`, `03_architecture_comparison`,
  `fno_loss_comparison`, `transient_comparison`, `_toc.yml`, `figs/*.svg`.
- Fixed: `spectral_layer` missing-paren bug; corrected FNO hyperparameter table
  (MODES=8, EPOCHS=10001, cosine-decayed LR=1e-3).

Pending:
- Enhance the two existing chapters `1D_Thermal_PINN_FEM_FDM` and `FNO_STEADY_STATE`
  (scientific titles, corrected text, a diagram each) — apply the same treatment as the others.
- `transient_comparison` uses `EPOCHS=6000` for speed; raise to 10001 to match the other transient chapters.

## Working notes for Claude

- Notebooks are the deliverable; keep existing computed outputs where possible, and prefer
  re-execution on build (`execute_notebooks: force` or `cache` in `_config.yml`) since seeds are fixed.
- When editing a notebook, preserve code cells; add/adjust markdown and fix bugs only.
- Show diffs and confirm before large rewrites.
