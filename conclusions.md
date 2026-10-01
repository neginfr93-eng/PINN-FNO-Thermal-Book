# Summary, Limitations and Outlook

*What the book set out to answer, what the experiments showed, and what remains open.*

---

## 1. The question

Can a neural operator learn the solution map of a nonlinear, heterogeneous heat-conduction problem accurately enough to replace repeated finite-element solves? Can it do so from the physics alone, without labelled solutions? And does it generalize to inputs and grids it was never trained on? All experiments use one benchmark, a rod with conductivity $k(x,T)=\alpha(x)(0.5+T^2)$ (see [The Benchmark Problem](benchmark_problem.ipynb)). Every model is graded against the same fine, fully implicit finite-element reference.

**Hyperparameters.** The hyperparameters were selected in a dedicated study, and the best configuration was then held fixed across the models being compared. Each comparison therefore changes only the quantity under study (the training loss, the architecture, or the training signal), so differences in the results can be attributed to it.

## 2. Findings, chapter by chapter

The values below are taken from the executed chapters. Errors are measured against the fine reference on the 100 held-out samples, or on the 300 out-of-distribution (OOD) evaluations: 50 unseen shapes on 6 grids from $N=64$ to $256$.

**Foundations.**

| Chapter | Result |
| :--- | :--- |
| [Classical solvers and PINNs](1D_Thermal_PINN_FEM_FDM.ipynb) | On a problem with an exact solution, FEM (≈$10^{-7}$) and FDM (≈$10^{-6}$) far outperform a small PINN (≈$10^{-2}$). A PINN also solves only one instance per training run. |
| [The benchmark problem](benchmark_problem.ipynb) | The discretization floor on the training grid is ≈$5\times10^{-5}$ (steady). In the transient case it is $5.4\times10^{-3}$ at the first step of $\Delta t=0.02$, falling to $2.1\times10^{-3}$ at step 19. |

**Steady state.**

| Chapter | Result |
| :--- | :--- |
| [Steady-state FNO](FNO_STEADY_STATE.ipynb) | Relative $L^2$ error ≈0.2% on held-out materials. OOD error stays within about 4× of it; sawtooth materials are hardest. The steady reference is exactly the long-time limit of the transient problem. |
| [Loss comparison](FNO_loss_comparison.ipynb) | Weighted residual ≈0.02%, data-driven ≈0.03%, MSE residual ≈0.09% (relative $L^2$). The label-free weighted residual is also best out of distribution. |
| [Architecture comparison](architecture_comparison.ipynb) | iFOL (small and medium tiers) and the Transformer (large tier) lead in distribution. The FNO has the lowest OOD error of the affordable models at every tier and trains 8–16× faster than the Transformer. |

**Transient.**

| Chapter | Result |
| :--- | :--- |
| [Data-driven transient FNO](FNO_transient_data_driven.ipynb) | Accurate rollouts in distribution; OOD error several times larger. Longer training improves OOD accuracy too, so no epoch cap is needed. |
| [Weighted-residual transient FNO](FNO_transient_weighted_residual.ipynb) | Trained without labels and on initial states only, it follows the reference over all 19 steps, with a first-step error close to the floor. |
| [PITI-FNO](FNO_transient_PITI.ipynb) | A $\Delta t$-free loss: one set of weights runs at $\Delta t=0.001$–$0.02$ with Euler or RK4, but diverges at $\Delta t=0.1$. Accuracy is about half that of WR at $\Delta t=0.02$. |
| [Transient comparison](transient_comparison.ipynb) | In distribution: data-driven < WR < PITI. Out of distribution: **WR best** (≈3×10⁻² mean), then data-driven (≈5×10⁻²), then PITI (≈8×10⁻²). |

## 3. Conclusions

1. **Labels are not required.** The finite-element residual, used in its weighted (Galerkin) form with a stop-gradient, trains neural operators without a single labelled solution. In the steady case it is the most accurate objective. In the transient case, the fully discrete form (WR) gives up little in-distribution accuracy relative to supervised training.
2. **Which residual is weighted matters for generalization.** Out of distribution, the weighted-residual loss built on the *fully discrete* equations beats supervised training, in the steady and in the transient problem alike (WR in the transient case). PITI uses the same weighted-residual form, but on the *semi-discrete* equation $M\dot T + K(T)T = 0$, and has the largest out-of-distribution error of the three transient operators. Being label-free is therefore not enough on its own: what helps is a residual that contains the full implicit time step, not just the instantaneous rate. Super-resolution costs every model about the same; the separation between training signals comes from unseen shapes.
3. **The FNO is the practical architecture.** It gives the best out-of-distribution accuracy per unit of training cost. Architectures that are more accurate in distribution (iFOL, Transformer) either generalize worse or cost an order of magnitude more.
4. **Where the time step lives is a design choice with a price.** Building the implicit step into the loss (WR) gives stable, accurate rollouts but ties the model to one $\Delta t$. Learning the time derivative (PITI) frees $\Delta t$, but inherits the stiffness of heat conduction at inference and loses accuracy.
5. **Grade against a converged reference.** Solving the reference once on a fine mesh and interpolating it, instead of re-solving on each coarse grid, separates model error from discretization error. It revealed a first-step floor of about $5\times10^{-3}$ on the transient training grid, which the best operators nearly reach.

## 4. Limitations

- **One training run per configuration.** No results are averaged over random seeds. The PITI MODES ablation shows that run-to-run variation can change qualitative conclusions for intermediate settings.
- **One dimension and modest material contrast.** The in-distribution materials vary by a factor of about 1.9, and all problems are 1-D.
- **Single-precision training.** The networks run in float32; only the references are float64.

## 5. Outlook

- A **transient Transformer** operator, excluded here on runtime grounds, as a dedicated study.
- **Seed-averaged** ablations, and a direct analysis of the learned operator's Jacobian eigenvalues to explain its stability limits.
- **Implicit or stabilized integrators** for PITI at inference, to recover large time steps.
- **Higher material contrast and 2-D domains**, where the advantage of label-free, resolution-independent training should matter most.
