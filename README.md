# Neural Operators for Nonlinear Heat Conduction

Welcome. This book is a hands-on, article-style introduction to solving **nonlinear
heat-conduction problems** with modern scientific machine learning — from classical
solvers and physics-informed neural networks through to **neural operators** that learn
the solution map itself.

Every chapter pairs the mathematics with runnable code and figures, so you can read it as
a tutorial, a reference, or a worked case study.

## What you will learn

- How a 1-D nonlinear heat-conduction problem is posed and solved with the finite-difference
  and finite-element methods, and how **physics-informed neural networks (PINNs)** turn a
  differential equation into a loss.
- What a **neural operator** is, and how the **Fourier Neural Operator (FNO)** learns a map
  between functions rather than a single solution.
- How the choice of **training objective** (data-driven vs. physics-based weighted residual
  vs. physics-informed time integration) changes accuracy and generalization.
- How different **architectures** (FNO, DeepONet, iFOL, Transformer) compare under a fair,
  controlled study.
- How these ideas extend from steady state to **time-dependent (transient)** problems, and how
  to evaluate operators fairly against a converged finite-element reference.

## How the book is organized

The book follows one line of argument. Each chapter answers one question and hands its
answer to the next.

**Part I — Foundations.** Classical solvers (finite differences, finite elements) and a
physics-informed neural network on a model problem with an exact solution. Then *The Benchmark
Problem*: the heterogeneous, nonlinear rod studied in the rest of the book, its finite-element
discretization, the reference solutions, datasets, out-of-distribution test sets and error metrics.

**Part II — Steady-State Neural Operators.** Can an FNO learn the steady map from material to
temperature? Which training loss should it use (answer: the label-free weighted residual)? Which
operator architecture (answer: the FNO)?

**Part III — Transient Neural Operators.** The same rod in time: an operator trained on FEM
data; one trained on the label-free weighted residual from initial states only; and a
physics-informed time-integrated operator (PITI) that learns the time derivative, so that the
time step is chosen only at inference. A final chapter compares all three.

**Part IV — Synthesis.** The findings of all chapters, their limitations, and open questions.

Parts II and III study the same rod: the steady state of Part II is exactly where the transient
problem of Part III ends up after a long time. Every chapter of Parts II and III is graded
against the same fully-implicit finite-element reference, computed by one shared module
(`fem_ground_truth.py`).

## How each chapter is written

Chapters follow a research-paper structure: motivation and problem, method, experimental setup,
results in distribution, results out of distribution, and a closing discussion that links to the
next chapter. Figures come with a short guide on how to read them.

Use the table of contents on the left to navigate. Chapters are largely self-contained, but
reading in order builds the ideas from the ground up.

## Who it is for

Students and researchers who know a little calculus and Python and want a practical,
example-driven path into physics-informed learning and neural operators. No prior experience
with neural operators is assumed.
