# Neural Operators for Nonlinear Heat Conduction

Welcome. This book is a hands-on, article-style introduction to solving **nonlinear
heat-conduction problems** with modern scientific machine learning — from classical
solvers and physics-informed neural networks through to **neural operators** that learn
the solution map itself.

Every chapter pairs the mathematics with runnable code and figures, so you can read it as
a tutorial, a reference, or a worked case study.

## What you will learn

- How a 1-D nonlinear heat-conduction problem is posed and solved with the finite-difference
  and finite-element methods.
- How **physics-informed neural networks (PINNs)** turn a differential equation into a loss.
- What a **neural operator** is, and how the **Fourier Neural Operator (FNO)** learns a map
  between functions rather than a single solution.
- How the choice of **training objective** (data-driven vs. physics-based weighted residual)
  changes accuracy and generalization.
- How different **architectures** (FNO, DeepONet, iFOL, Transformer) compare under a fair,
  controlled study.
- How these ideas extend from steady state to **time-dependent (transient)** problems.

## How the book is organized

**Part I — Foundations.** The governing physics, classical solvers (finite difference and
finite element), and a physics-informed neural network baseline.

**Part II — Steady-State Neural Operators.** The Fourier Neural Operator; a comparison of
training losses that motivates the weighted-residual objective; and a controlled comparison
of neural-operator architectures.

**Part III — Transient Neural Operators.** Operators that advance the temperature field in
time — trained on data and trained physics-informed — and a direct comparison of the two.

Use the table of contents on the left to navigate. Chapters are largely self-contained, but
reading in order builds the ideas from the ground up.

## Who it is for

Students and researchers who know a little calculus and Python and want a practical,
example-driven path into physics-informed learning and neural operators. No prior experience
with neural operators is assumed.
