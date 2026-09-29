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

**I. Foundations**
- `1D_Thermal_PINN_FEM_FDM` — FDM, FEM and PINN on a constant-k0 rod with an analytic solution.
- `benchmark_problem` — the shared rod k=alpha(x)(0.5+T^2): equations, FEM discretization, Newton
  solvers, fine references + derivation, discretization floor (measured), datasets/splits, OOD sets,
  metrics, notation, roadmap. Only loads cached references + one coarse FEM solve (fast to run).

**II. Steady-State Neural Operators**
- `FNO_STEADY_STATE` — supervised FNO baseline alpha -> T; OOD; transient->steady consistency.
- `FNO_loss_comparison` — MSE residual / weighted residual / data-driven; **choose weighted residual**.
- `architecture_comparison` — FNO vs DeepONet vs iFOL vs Transformer, 3 tiers; **choose FNO**.

**III. Transient Neural Operators**
- `FNO_transient_data_driven` — supervised transient FNO (all 19 transitions).
- `FNO_transient_weighted_residual` — label-free WR, T0 only (from thesis `..._T0only.ipynb`).
- `FNO_transient_PITI` — tangent-learning PITI, Euler/RK4 at inference, MODES ablation
  (from thesis `fno_piti_physics_dualdt_ablation_noLC.ipynb`).
- `transient_comparison` — trains nothing; DD vs WR vs PITI in-dist (with measured floor) and OOD.

**IV. Synthesis**
- `conclusions.md` — findings table (numbers from executed chapters), limitations, outlook.
  Update its numbers if any chapter is re-run with changes.

## Writing conventions

- Each chapter follows a research-paper arc: Problem statement -> Reference solver (FEM) ->
  Architecture -> Training objective -> Results (in-distribution) -> Results (out-of-distribution) -> Discussion.
  Sections are numbered continuously; every chapter ends with a Discussion/Conclusion that links
  to the next chapter; cross-chapter references are markdown links to the .ipynb, never "Chapter N".
- Scientific chapter titles (no "Step 1", "Step 2").
- Article-style markdown with LaTeX. Conceptual SVG diagrams live in `figs/` and are embedded
  in the notebooks (base64) so they render even if the file is moved.
- Results prose explains **how to read** a figure rather than hard-coding numbers, so it stays
  correct after re-runs.

## Physics (shared across chapters)

One rod for Parts II and III: `k(x,T) = alpha(x) * (0.5 + T^2)`, T(0)=1, T(1)=0 (Part I's
`1D_Thermal_PINN_FEM_FDM` keeps its own k0(1+beta*T) problem because it checks against an analytic
solution). Reference solutions from finite elements: backward-Euler stepping (transient) or the
steady residual K(T)T=0 (steady), both **fully implicit**, solved with Newton-Raphson. Weak-form / weighted-residual (Galerkin) loss = FEM residual as the training
signal, no labels.

## Transient ground truth (single source of truth)

All transient chapters import `fem_ground_truth.py` (copied from the thesis folder
`transient_fno_with_femseprate/fem_ground_truth/`). Shared setup: N_NODES=64, MODES=4, DT=0.02,
1000 samples split 800/100/100 (ROLLOUT_TEST_START=900). Reference caches in `checkpoints/`:
`field_pool.npz`, `fine_reference.npz` (N=256, dt=0.001, 400 steps; 825 MB) and
`ood_fine_reference.pkl` (50 shapes, N=512) — the two large ones are git-ignored.
Execution order: data-driven, weighted-residual, PITI (overlays WR's `compare_wr.npz`), comparison.
Steady chapters (Part II) use the module's steady section (`solve_steady_fem`, `get_steady_reference`,
`derive_steady_dataset`, `derive_steady_ood`): same 1000 alpha fields, solved once at N=511 (float64),
cached in `checkpoints/steady_reference.npz` (~8 MB, committed); OOD = 50 unseen alpha shapes
(Layered/Inclusion/Sawtooth/HF-Rand) x 6 grids. `FNO_STEADY_STATE` Section 6 checks that a long
transient run converges to the steady solution.

## Status

Done and in the repo (from prior work):
- `01_FNO_transient`, `02_FOL_transient_physics_informed`, `03_architecture_comparison`,
  `fno_loss_comparison`, `transient_comparison`, `_toc.yml`, `figs/*.svg`.
- Fixed: `spectral_layer` missing-paren bug; corrected FNO hyperparameter table
  (MODES=8, EPOCHS=10001, cosine-decayed LR=1e-3).
- All three transient chapters upgraded to a **fully-implicit** FEM reference solver
  (`k(x,T)` evaluated at the unknown `T_next`, solved with Newton-Raphson) — previously all
  three used a semi-implicit/lagged-coefficient scheme (`k` evaluated at the known `T_n`).
  The weighted-residual chapter's `compute_weak_residual` now evaluates the coefficient at
  the predicted `T_next` to match, keeping `stop_gradient` on the residual.
- All three transient chapters' FNO switched to a **Ṫ-parameterization**: the network
  predicts the time-derivative, stepped forward with one explicit-Euler update
  (`T_next = T_curr + DT*Ṫ`), replacing the previous direct increment prediction.
- `FNO_transient_data_driven` and `FNO_transient_weighted_residual` now use real **early
  stopping** (held-out validation, patience, best-checkpoint selection) instead of a fixed
  `EPOCHS` count, and save their best checkpoint to `checkpoints/*.npz`.
- 2026-09-29: transient part rebuilt on the shared `fem_ground_truth.py` reference. WR chapter
  replaced by the T0-only thesis notebook; new `FNO_transient_PITI` chapter; data-driven ported
  (N=64, MODES=4, 800/100/100 split, OOD = fem_ground_truth's 50 shapes x 6 resolutions);
  `transient_comparison` now compares DD vs WR vs PITI (Euler/RK4). All four executed with the
  `miniconda3/envs/faircomp` env (env_python312's matplotlib crashes on draw).
- Data-driven Section 9: the old "OOD collapses under long training" finding does NOT reproduce on
  the shared reference (OOD error falls monotonically), so the DATA_MAX_EPOCHS=12000 cap was removed.
- PITI MODES ablation: only the endpoints (MODES=4 stable/best, MODES=16 diverges) are robust
  across runs; MODES=2/8 behaviour flipped between JAX versions -- documented in the chapter.

Pending:
- Enhance the two existing chapters `1D_Thermal_PINN_FEM_FDM` and `FNO_STEADY_STATE`
  (scientific titles, corrected text, a diagram each) — apply the same treatment as the others.

## Working notes for Claude

- Notebooks are the deliverable; keep existing computed outputs where possible, and prefer
  re-execution on build (`execute_notebooks: force` or `cache` in `_config.yml`) since seeds are fixed.
- When editing a notebook, preserve code cells; add/adjust markdown and fix bugs only.
- Show diffs and confirm before large rewrites.
