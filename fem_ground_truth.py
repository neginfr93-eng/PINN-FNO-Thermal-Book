"""
Reusable ground-truth generator for the 1D nonlinear transient heat-conduction
problem used across the FNO_transient_weighted_residual* notebooks:

    rho_cp * dT/dt = d/dx( k(x,T) * dT/dx ),   k(x,T) = alpha(x) * (0.5 + T^2)

on x in [0, 1], Dirichlet BCs T(0)=1, T(1)=0. Discretized with linear FEM in
space and a fully-implicit backward-Euler step in time: k(x,T) is evaluated
at the UNKNOWN T_next, so each step is nonlinear and is solved with
Newton-Raphson using an exact Jacobian (jax.jacfwd) of the residual on the
free (interior) nodes.

Resolution-generic: mesh size is read from the input array's shape, not a
hardcoded constant, so the same functions serve training-resolution and
super-resolution (OOD) use alike.

Typical use as a ground-truth oracle from any notebook/script:

    from fem_ground_truth import generate_T0_alpha, fem_rollout, batch_fem_rollout

    alpha, T0  = generate_T0_alpha(key_k, key_t, n_nodes=64)          # in-distribution sample
    true_traj  = fem_rollout(T0, alpha, steps=19)                     # (steps+1, n_nodes)
    true_batch = batch_fem_rollout(T0_batch, alpha_batch, steps=19)   # (batch, steps+1, n_nodes)

    from fem_ground_truth import build_ood_scenarios
    ood = build_ood_scenarios(n_total=50, res_choices=(64, 96, 128))  # out-of-distribution stress test

dt / rho_cp / newton_iters default to this module's DEFAULT_* constants;
pass them explicitly to any function to override without editing this file.
compute_weak_residual is also exposed since it's useful beyond ground-truth
generation (e.g. as a physics-residual training/monitoring signal).

Three independent pieces, usable on their own or together:
  1. FEM solver          -- assemble_LM, solve_transient_fem_step,
                             compute_weak_residual, fem_rollout, batch_fem_rollout
  2. In-distribution T0/alpha sampler (Yamazaki et al. 2025, Eng. w/ Computers
     41:1-29, Sect. 3.3) -- generate_T0_alpha (+ its sub-generators)
  3. Out-of-distribution scenario builder (shape families x resolutions)
                          -- build_ood_scenarios (+ its shape functions)

On top of these, get_dataset(path, n_samples, max_steps) is a disk-cached
"ground truth for every sample" oracle: the first call solves the full
Newton-Raphson trajectory for all n_samples and writes it to `path`; every
call after that (this run or a future one) just reads the file back, no
re-solving:

    from fem_ground_truth import get_dataset
    data = get_dataset("checkpoints/fem_dataset_cache.npz", n_samples=1000, max_steps=19)
    data["T0"], data["alpha"], data["generator"], data["trajectory"]  # (1000,19+1,64)
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import os
import pickle
import time

DEFAULT_DT = 0.02
DEFAULT_RHO_CP = 5.0
DEFAULT_NEWTON_ITERS = 4
DEFAULT_N_NODES = 64  # this project's training resolution
DEFAULT_REFERENCE_N_NODES = 512  # fine reference resolution, ~0.2% rel. L2 from N=256 for smooth in-distribution fields

@jax.jit
def assemble_LM(T_ref, alpha_x, dt=DEFAULT_DT, rho_cp=DEFAULT_RHO_CP):
    """Vectorized FEM assembly. Returns LHS = M + dt*K(T_ref) and RHS = M @ T_ref
    (M does not depend on T_ref; only K does, through the nonlinear conductivity
    k(x,T_ref)). Mesh size is read from T_ref.shape, not a global constant, so
    this assembles correctly at any node count."""
    n_nodes = T_ref.shape[0]
    n_elem = n_nodes - 1
    dx = 1.0 / n_elem
    k_nodal = alpha_x * (0.5 + T_ref**2)
    k_avg = 0.5 * (k_nodal[:-1] + k_nodal[1:])                 # (n_elem,)
    e = jnp.arange(n_elem)
    rows = jnp.stack([e, e, e + 1, e + 1], axis=1).reshape(-1)  # 4 local dofs / element
    cols = jnp.stack([e, e + 1, e, e + 1], axis=1).reshape(-1)
    k_loc = (k_avg[:, None] / dx) * jnp.array([1.0, -1.0, -1.0, 1.0])[None, :]
    m_loc = (rho_cp * dx / 6.0) * jnp.broadcast_to(jnp.array([2.0, 1.0, 1.0, 2.0]), (n_elem, 4))
    K_mat = jnp.zeros((n_nodes, n_nodes)).at[rows, cols].add(k_loc.reshape(-1))
    M_mat = jnp.zeros((n_nodes, n_nodes)).at[rows, cols].add(m_loc.reshape(-1))
    return M_mat + dt * K_mat, M_mat @ T_ref


@partial(jax.jit, static_argnames=("newton_iters",))
def solve_transient_fem_step(T_n, alpha_x, dt=DEFAULT_DT, rho_cp=DEFAULT_RHO_CP,
                              newton_iters=DEFAULT_NEWTON_ITERS):
    """One fully-implicit backward-Euler step. Newton-Raphson refines the free
    (interior) nodes over `newton_iters` iterations using an exact Jacobian of
    the residual (jax.jacfwd). Resolution-generic (mesh size read from
    T_n.shape), same reasoning as assemble_LM above."""
    n_nodes = T_n.shape[0]
    free = jnp.arange(1, n_nodes - 1)
    RHS = assemble_LM(T_n, alpha_x, dt, rho_cp)[1]

    def residual(Tf):
        T_full = jnp.zeros(n_nodes).at[0].set(1.0).at[-1].set(0.0).at[free].set(Tf)
        LHS = assemble_LM(T_full, alpha_x, dt, rho_cp)[0]
        return (LHS @ T_full - RHS)[free]

    def newton(Tf, _):
        F = residual(Tf)
        J = jax.jacfwd(residual)(Tf)
        Tf_new = Tf - jnp.linalg.solve(J, F)
        return Tf_new, Tf_new

    T_next_free, _ = jax.lax.scan(newton, T_n[free], None, length=newton_iters)
    return jnp.zeros(n_nodes).at[0].set(1.0).at[-1].set(0.0).at[free].set(T_next_free)


@jax.jit
def compute_weak_residual(T_next, T_curr, alpha_x, dt=DEFAULT_DT, rho_cp=DEFAULT_RHO_CP):
    """Weak-form residual LHS(T_next)@T_next - RHS(T_curr), on interior nodes.
    Conductivity is evaluated at the predicted T_next (matching the
    fully-implicit ground truth above), not lagged at T_curr."""
    LHS = assemble_LM(T_next, alpha_x, dt, rho_cp)[0]
    RHS = assemble_LM(T_curr, alpha_x, dt, rho_cp)[1]
    n_nodes = T_next.shape[0]
    free = jnp.arange(1, n_nodes - 1)
    return (LHS @ T_next - RHS)[free]


@partial(jax.jit, static_argnames=("steps", "newton_iters"))
def fem_rollout(T_init, alpha_prof, steps, dt=DEFAULT_DT, rho_cp=DEFAULT_RHO_CP,
                 newton_iters=DEFAULT_NEWTON_ITERS):
    """Ground-truth trajectory for a single (T_init, alpha) pair.
    Returns (steps+1, n_nodes): T_init followed by `steps` implicit FEM steps."""
    def step(T_curr, _):
        val = solve_transient_fem_step(T_curr, alpha_prof, dt, rho_cp, newton_iters)
        return val, val
    return jnp.vstack([T_init, jax.lax.scan(step, T_init, None, length=steps)[1]])


@partial(jax.jit, static_argnames=("steps", "newton_iters"))
def batch_fem_rollout(T0_batch, alpha_batch, steps, dt=DEFAULT_DT, rho_cp=DEFAULT_RHO_CP,
                       newton_iters=DEFAULT_NEWTON_ITERS):
    """Batched ground-truth trajectories (Newton-Raphson per step, vmapped
    across the batch). Returns (batch, steps+1, n_nodes)."""
    def step(T_curr, _):
        T_next = jax.vmap(lambda t, a: solve_transient_fem_step(t, a, dt, rho_cp, newton_iters))(T_curr, alpha_batch)
        return T_next, T_next
    _, traj = jax.lax.scan(step, T0_batch, None, length=steps)
    traj = jnp.transpose(traj, (1, 0, 2))
    return jnp.concatenate([T0_batch[:, None, :], traj], axis=1)


# =============================================================================
# In-distribution T0 / alpha sample generator (Yamazaki et al. 2025, Eng. w/
# Computers 41:1-29, Sect. 3.3): training-style initial-temperature and
# conductivity fields drawn from a mixture of a random Fourier series, a
# Gaussian random process, and a constant field. Eq. (19) in the paper is 2D
# (x,y); the y-dependent terms are dropped here for a 1D domain, keeping the
# same per-term c/A/B/C structure and Table 2 ranges.
# =============================================================================
NSUM = 6   # NOT the paper's 50: summing that many random 1D sinusoids collapses
           # via the law of large numbers -- every draw regresses toward the same
           # mean shape (measured ~4x less per-sample spread at NSUM=50 vs NSUM=4).
           # The paper's Eq. (19) is 2D, where sin(Cx)cos(Dy)-type cross terms keep
           # contributing spatially-localized variation even after 50 are summed;
           # our 1D reduction has no such cross-term structure to preserve variety.
FOURIER_C_RANGE = (0.0, 1.5)          # per-term offset c_i (paper's Table 2)
FOURIER_AMP_RANGE = (0.5, 1.5)        # per-term base amplitude, BEFORE the decay below
FOURIER_FREQ_RANGE = (0.5, 8.0)       # random continuous frequency (paper-style, not fixed harmonics)
FOURIER_DECAY = 0.8                   # amplitude ~ base_amp / freq**DECAY
T0_GENERATOR_NAMES = ["Fourier series", "Gaussian random process", "Constant field"]


def _minmax_norm(v):
    lo, hi = jnp.min(v), jnp.max(v)
    return (v - lo) / (hi - lo + 1e-8)


@partial(jax.jit, static_argnames=("n_nodes",))
def generate_fourier_fn(key, n_nodes=DEFAULT_N_NODES):
    """Hybrid of the paper's Eq. (19) random-Fourier generator and a colored-
    noise (1/f) generator: keeps the paper's per-term offset c_i and RANDOM
    CONTINUOUS frequency (not fixed harmonics), but borrows an explicit random
    phase and a frequency-dependent amplitude decay (amp = base_amp /
    freq**DECAY). The decay damps whichever terms happen to draw a high
    frequency, which is what lets NSUM go up to 6 without the NSUM=50
    collapse. Min-max normalized to [0,1]."""
    x = jnp.linspace(0, 1, n_nodes)
    kc, ka, kf, kph = jax.random.split(key, 4)
    c = jax.random.uniform(kc, (NSUM,), minval=FOURIER_C_RANGE[0], maxval=FOURIER_C_RANGE[1])
    base_amp = jax.random.uniform(ka, (NSUM,), minval=FOURIER_AMP_RANGE[0], maxval=FOURIER_AMP_RANGE[1])
    freq = jax.random.uniform(kf, (NSUM,), minval=FOURIER_FREQ_RANGE[0], maxval=FOURIER_FREQ_RANGE[1])
    phase = jax.random.uniform(kph, (NSUM,), minval=0.0, maxval=2 * jnp.pi)
    amp = base_amp / (freq ** FOURIER_DECAY)
    terms = c[:, None] + amp[:, None] * jnp.sin(freq[:, None] * jnp.pi * x[None, :] + phase[:, None])
    return _minmax_norm(jnp.sum(terms, axis=0))


@partial(jax.jit, static_argnames=("n_nodes",))
def generate_grf_fn(key, n_nodes=DEFAULT_N_NODES):
    """Gaussian random process generator: zero-mean GP with a squared-
    exponential kernel over the grid, Cholesky-sampled, with a randomized
    length scale for variety. Min-max normalized to [0,1] as in the paper
    (Sect. 3.3, second generator)."""
    x = jnp.linspace(0, 1, n_nodes)
    k_len, k_z = jax.random.split(key, 2)
    length_scale = jax.random.uniform(k_len, (), minval=0.05, maxval=0.4)
    d2 = (x[:, None] - x[None, :]) ** 2
    cov = jnp.exp(-0.5 * d2 / length_scale**2) + 1e-6 * jnp.eye(n_nodes)
    L = jnp.linalg.cholesky(cov)
    z = jax.random.normal(k_z, (n_nodes,))
    return _minmax_norm(L @ z)


@partial(jax.jit, static_argnames=("n_nodes",))
def generate_constant_fn(key, n_nodes=DEFAULT_N_NODES):
    """Constant-temperature generator (paper's third generator)."""
    c = jax.random.uniform(key, ())
    return jnp.full((n_nodes,), c)


@partial(jax.jit, static_argnames=("n_nodes",))
def generate_T0_signal(key, n_nodes=DEFAULT_N_NODES):
    """Mixture matching the paper's ~40% Fourier / 50% Gaussian / 10% constant
    split of training samples, recentered from [0,1] to [-1,1] to match the
    perturbation-amplitude convention used in generate_T0_alpha below. Also
    reused there for the conductivity field alpha -- it's a generic
    diverse-smooth-field mixture, not something inherently temperature-specific."""
    k_choice, k_gen = jax.random.split(key, 2)
    choice = jax.random.choice(k_choice, 3, p=jnp.array([0.4, 0.5, 0.1]))
    raw = jax.lax.switch(
        choice,
        [lambda k: generate_fourier_fn(k, n_nodes),
         lambda k: generate_grf_fn(k, n_nodes),
         lambda k: generate_constant_fn(k, n_nodes)],
        k_gen,
    )
    return raw * 2.0 - 1.0


@jax.jit
def t0_generator_choice(key):
    """Which of the 3 T0 generators produced a given key's sample -- replays
    generate_T0_signal's own key-split so it exactly matches, bit-for-bit.
    Purely diagnostic: lets downstream plots label a sample's provenance
    instead of guessing from its shape. Index into T0_GENERATOR_NAMES."""
    k_choice, _ = jax.random.split(key, 2)
    return jax.random.choice(k_choice, 3, p=jnp.array([0.4, 0.5, 0.1]))


@partial(jax.jit, static_argnames=("n_nodes",))
def generate_T0_alpha(key_k, key_t, n_nodes=DEFAULT_N_NODES):
    """One in-distribution (T0, alpha) sample -- NO Newton-Raphson
    time-stepping, just the initial condition and material field. T0 is a
    linear ramp (satisfying the T(0)=1, T(1)=0 Dirichlet BCs) plus a
    zero-at-the-boundaries perturbation; alpha uses the same Fourier/GRF/
    constant mixture, passed through a sigmoid into [0.1, 0.5]."""
    alpha = jax.nn.sigmoid(generate_T0_signal(key_k, n_nodes)) * 0.4 + 0.1
    T0_signal = generate_T0_signal(key_t, n_nodes)
    ramp = jnp.linspace(1.0, 0.0, n_nodes)
    mask = jnp.sin(jnp.pi * jnp.linspace(0, 1, n_nodes))
    T0 = ramp + (T0_signal * mask)
    T0 = T0.at[0].set(1.0).at[-1].set(0.0)  # sin(pi) is only ~1e-8 in float32, not exact -- pin BCs exactly
    return alpha, T0


# =============================================================================
# Out-of-distribution scenario builder: shape families (deterministic) plus a
# randomized high-frequency filler, crossed with a list of resolutions, for
# stress-testing a trained model outside its training distribution (novel
# shapes, and/or super-resolution).
# =============================================================================
DEFAULT_OOD_RES_CHOICES = (DEFAULT_N_NODES, 96, 128, 160, 192, 256)


def get_staircase_T0(n, transition_width=0.02, key=None):
    """Smoothed step (tanh transition, not a hard jump) -- still a steep,
    OOD-worthy edge at every training-scale resolution (transition_width=0.02
    spans ~1.3 elements at N_GRID=64, so it's still under-resolved there,
    same stress-test intent as a true step), but bounded-gradient, so linear
    FEM converges at its normal rate here instead of the reduced rate a true
    discontinuity forces. That's what let this shape share one common
    (dt_ref, reference_n_nodes) with every other case instead of needing its
    own finer settings -- see fem_self_convergence_study.py's spatial study,
    where the old hard-jump version was still ~7e-3 unconverged at N=512
    while every other shape had long since dropped under tolerance.

    key=None (default) reproduces the original single fixed shape (edge at
    x=0.5, transition_width as given). Pass a PRNGKey to get one randomized
    variant instead: transition center in [0.3, 0.7], width in [0.01, 0.05]
    (transition_width is then ignored)."""
    x = jnp.linspace(0, 1, n)
    center = 0.5
    if key is not None:
        k1, k2 = jax.random.split(key)
        center = jax.random.uniform(k1, (), minval=0.3, maxval=0.7)
        transition_width = jax.random.uniform(k2, (), minval=0.01, maxval=0.05)
    return 0.1 + 0.4 * (1.0 + jnp.tanh((x - center) / transition_width))


def get_pyramid_T0(n, key=None):
    """key=None (default) reproduces the original fixed pyramid (peak at
    x=0.5). Pass a PRNGKey for one randomized variant: peak position in
    [0.25, 0.75] (still 0 at both boundaries, 1 at the peak)."""
    x = jnp.linspace(0, 1, n)
    peak = 0.5 if key is None else jax.random.uniform(key, (), minval=0.25, maxval=0.75)
    return jnp.where(x < peak, x / peak, (1.0 - x) / (1.0 - peak))


def get_asym_T0(n, key=None):
    """key=None (default) reproduces the original fixed shape (freqs 2/4,
    amps 4/2). Pass a PRNGKey for one randomized variant: freqs/amps drawn
    from the same ranges that produced the original's asymmetry."""
    x = jnp.linspace(0, 1, n)
    f1, f2, a1, a2 = 2.0, 4.0, 4.0, 2.0
    if key is not None:
        k1, k2, k3, k4 = jax.random.split(key, 4)
        f1 = jax.random.uniform(k1, (), minval=1.0, maxval=4.0)
        f2 = jax.random.uniform(k2, (), minval=2.0, maxval=8.0)
        a1 = jax.random.uniform(k3, (), minval=2.0, maxval=6.0)
        a2 = jax.random.uniform(k4, (), minval=1.0, maxval=3.0)
    raw = a1 * jnp.sin(f1 * jnp.pi * x) + a2 * jnp.cos(f2 * jnp.pi * x)
    return jnp.clip(0.5 + raw / 12.0, 0.0, 1.0)


def get_random_T0(key, n, max_freq=18, decay=0.8):
    freqs = jnp.arange(1, max_freq + 1)
    k1, k2 = jax.random.split(key)
    amps = jax.random.normal(k1, (max_freq,)) / (freqs ** decay)
    phases = jax.random.uniform(k2, (max_freq,), minval=0, maxval=2 * jnp.pi)
    x = jnp.linspace(0, 1, n)
    raw = jnp.sum(amps[:, None] * jnp.sin(freqs[:, None] * jnp.pi * x[None, :] + phases[:, None]), axis=0)
    return (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)


def build_ood_scenarios(n_total, res_choices=DEFAULT_OOD_RES_CHOICES, alpha_const=0.25, seed=101):
    """3 fixed shape families (Staircase/Pyramid/Asymmetric) x len(res_choices)
    resolutions come first, then HF-Rand fills the rest with fresh random
    draws cycling through resolutions. Each scenario has its own resolution,
    so they can't be stacked into one batched array -- grade them individually
    with fem_rollout / a resolution-generic model rollout.

    Returns a list of (label, family, res, T0, alpha) tuples. T0 has the hard
    Dirichlet BCs (T0[0]=1, T0[-1]=0) applied; alpha is a constant field."""
    fams = [("Staircase", get_staircase_T0), ("Pyramid", get_pyramid_T0), ("Asymmetric", get_asym_T0)]
    scen = []
    for res in res_choices:
        for fname, fn in fams:
            scen.append((fname, res, fn(res)))
    key = jax.random.PRNGKey(seed)
    i = 0
    while len(scen) < n_total:
        key, kk = jax.random.split(key)
        res = res_choices[i % len(res_choices)]
        scen.append((f"HF-Rand#{i}", res, get_random_T0(kk, res)))
        i += 1
    scen = scen[:n_total]

    built = []
    for fname, res, T0_raw in scen:
        T0 = T0_raw.at[0].set(1.0).at[-1].set(0.0)
        alpha_local = jnp.full((res,), alpha_const)
        built.append((f"{fname}@{res}", fname, res, T0, alpha_local))
    return built


# =============================================================================
# Disk-cached CONVERGED ground truth for the OOD stress test. build_ood_scenarios
# solves a fresh FEM trajectory AT each scenario's own (often coarse) resolution
# -- fine for building the scenario's T0/alpha, but NOT a fair "ground truth" to
# grade a model against across resolutions: a coarse solve has its own real
# discretization error (e.g. ~1% for Staircase at N_GRID=64), which then gets
# conflated with the model's own error. precompute_ood_reference_dataset instead
# solves each UNIQUE underlying shape ONCE at a fine reference_n_nodes mesh, then
# interpolates that one converged trajectory down onto every resolution actually
# needed -- so every resolution is graded against the SAME true answer.
# get_ood_reference_dataset is the entry point most callers want.
# =============================================================================
def precompute_ood_reference_dataset(path, res_choices=DEFAULT_OOD_RES_CHOICES, n_total=50,
                                      reference_n_nodes=513, alpha_const=0.25, steps=19,
                                      dt=DEFAULT_DT, rho_cp=DEFAULT_RHO_CP,
                                      newton_iters=DEFAULT_NEWTON_ITERS, seed=101):
    """Mirrors build_ood_scenarios' exact scenario list (same families, same
    n_total, same resolution cycling for HF-Rand), but every scenario's
    true_traj comes from interpolating ONE fine reference_n_nodes solve down
    to that scenario's resolution, not from solving fresh at that resolution.
    The 3 fixed families are solved once each (shared across every resolution
    they appear at); each HF-Rand draw is solved once (used at its one
    assigned resolution). Always recomputes -- use get_ood_reference_dataset
    for the load-if-cached, else-compute version.

    Saves (pickle, since resolutions are ragged and can't share one array
    shape) a list of dicts, one per scenario: label, family, res, T0, alpha,
    true_traj (shape (steps+1, res))."""
    fams = [("Staircase", get_staircase_T0), ("Pyramid", get_pyramid_T0), ("Asymmetric", get_asym_T0)]

    scenarios = []  # (label, family, res, shape_fn, key_or_None)
    for res in res_choices:
        for fname, fn in fams:
            scenarios.append((f"{fname}@{res}", fname, res, fn, None))
    key = jax.random.PRNGKey(seed)
    i = 0
    while len(scenarios) < n_total:
        key, kk = jax.random.split(key)
        res = res_choices[i % len(res_choices)]
        scenarios.append((f"HF-Rand#{i}@{res}", f"HF-Rand#{i}", res, get_random_T0, kk))
        i += 1
    scenarios = scenarios[:n_total]

    ref_x = jnp.linspace(0, 1, reference_n_nodes)
    alpha_ref = jnp.full((reference_n_nodes,), alpha_const)
    fine_traj_cache = {}   # family name -> (steps+1, reference_n_nodes), shared by the 3 fixed families

    def fine_trajectory(fname, fn, shape_key):
        if fname in fine_traj_cache:
            return fine_traj_cache[fname]
        T0_raw = fn(reference_n_nodes) if shape_key is None else fn(shape_key, reference_n_nodes)
        T0_ref = T0_raw.at[0].set(1.0).at[-1].set(0.0)
        traj_ref = fem_rollout(T0_ref, alpha_ref, steps, dt, rho_cp, newton_iters)
        fine_traj_cache[fname] = traj_ref
        return traj_ref

    built = []
    for label, fname, res, fn, shape_key in scenarios:
        traj_ref = fine_trajectory(fname, fn, shape_key)
        this_x = jnp.linspace(0, 1, res)
        true_traj = jnp.stack([jnp.interp(this_x, ref_x, traj_ref[t]) for t in range(traj_ref.shape[0])])

        T0_raw = fn(res) if shape_key is None else fn(shape_key, res)
        T0 = T0_raw.at[0].set(1.0).at[-1].set(0.0)
        alpha = jnp.full((res,), alpha_const)

        built.append({
            "label": label, "family": fname, "res": res,
            "T0": np.asarray(T0), "alpha": np.asarray(alpha),
            "true_traj": np.asarray(true_traj),
        })

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    config = dict(res_choices=tuple(res_choices), n_total=n_total, reference_n_nodes=reference_n_nodes,
                  alpha_const=alpha_const, steps=steps, dt=dt, rho_cp=rho_cp,
                  newton_iters=newton_iters, seed=seed)
    with open(path, "wb") as f:
        pickle.dump({"scenarios": built, "config": config}, f)
    return built


def load_ood_reference_dataset(path):
    """Read back a dataset written by precompute_ood_reference_dataset. Raises
    FileNotFoundError with a clear message if `path` doesn't exist yet."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} doesn't exist yet -- call get_ood_reference_dataset(path, ...) "
            "once to generate and cache it."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def get_ood_reference_dataset(path, res_choices=DEFAULT_OOD_RES_CHOICES, n_total=50,
                               reference_n_nodes=513, alpha_const=0.25, steps=19,
                               dt=DEFAULT_DT, rho_cp=DEFAULT_RHO_CP,
                               newton_iters=DEFAULT_NEWTON_ITERS, seed=101, force_recompute=False):
    """The entry point: if a cache matching this exact config already exists
    at `path`, load it (no solving). Otherwise solve it once (which needs a
    working FEM solve at reference_n_nodes -- expensive, run this from a
    dedicated script, ideally with jax_enable_x64=True since reference_n_nodes
    is typically well past float32's precision floor) and save it there.
    Returns the list of scenario dicts (see precompute_ood_reference_dataset)."""
    if not force_recompute and os.path.exists(path):
        cached = load_ood_reference_dataset(path)
        cfg = cached["config"]
        if (cfg["res_choices"] == tuple(res_choices) and cfg["n_total"] == n_total
                and cfg["reference_n_nodes"] == reference_n_nodes and cfg["alpha_const"] == alpha_const
                and cfg["steps"] == steps and cfg["seed"] == seed):
            return cached["scenarios"]
        print(f"{path} exists but was cached with a different config -- recomputing.")
    return precompute_ood_reference_dataset(path, res_choices, n_total, reference_n_nodes, alpha_const,
                                             steps, dt, rho_cp, newton_iters, seed)


# =============================================================================
# General OOD ground truth, solvable at ANY (n_nodes, dt) -- the OOD analogue of
# precompute_fine_trajectory / derive_dataset above. get_ood_reference_dataset
# already debiases SPACE correctly (solves once at reference_n_nodes, interpolates
# down) but is pinned to a single dt=DEFAULT_DT. The functions below add the same
# dt-generalization the in-distribution side got: solve each UNIQUE shape ONCE at a
# fine reference mesh AND a fine reference dt, then derive any (n_nodes, dt) from
# that single solve via exact interpolation/subsampling -- no re-solving, ever,
# for any resolution or dt any script asks for.
# =============================================================================
def _split_evenly(n_total, n_groups):
    """n_total split into n_groups whole-number counts as evenly as possible --
    the first (n_total % n_groups) groups get one extra. E.g. (50, 4) -> [13,13,12,12]."""
    base, rem = divmod(n_total, n_groups)
    return [base + 1 if i < rem else base for i in range(n_groups)]


def precompute_ood_fine_trajectories(path, n_total=50, dt_ref=0.0002, steps_ref=1900,
                                      reference_n_nodes=DEFAULT_REFERENCE_N_NODES, alpha_const=0.25,
                                      seed=101, rho_cp=DEFAULT_RHO_CP, newton_iters=DEFAULT_NEWTON_ITERS,
                                      chunk_size=8):
    """Solve the ONE canonical fine trajectory for n_total OOD scenarios, split as
    evenly as possible across the 4 shape families -- Staircase (random transition
    center/width), Pyramid (random peak position), Asymmetric (random freqs/amps),
    and HF-Rand (already random) -- at a fine reference mesh AND fine reference dt.
    n_total need not be a multiple of 4: e.g. n_total=50 gives 13/13/12/12. Every
    (n_nodes, dt) any script needs is derived from this ONE set of solves via
    derive_ood_dataset below, with no further FEM solving. Always recomputes -- use
    get_ood_fine_trajectories for the load-if-cached, else-compute version.

    dt_ref should be fine enough that every dt you'll ever need is an exact multiple
    of it (see subsample_to_dt) -- matching precompute_fine_trajectory's dt_ref (e.g.
    0.0002) lets in-distribution and OOD ground truth share the same time grid.

    Saved as a pickle (labels/families aren't a uniform array) with keys: labels,
    families, T0, alpha, trajectory (all shape (n_shapes, ...)), n_nodes, dt,
    max_steps, alpha_const, seed, n_total."""
    fams = [("Staircase", get_staircase_T0), ("Pyramid", get_pyramid_T0), ("Asymmetric", get_asym_T0)]
    n_staircase, n_pyramid, n_asym, n_hf_rand = _split_evenly(n_total, 4)
    counts = {"Staircase": n_staircase, "Pyramid": n_pyramid, "Asymmetric": n_asym}
    labels, families, T0_list = [], [], []
    key = jax.random.PRNGKey(seed)
    for fname, fn in fams:
        for i in range(counts[fname]):
            key, kk = jax.random.split(key)
            labels.append(f"{fname}#{i}"); families.append(fname)
            T0_list.append(fn(reference_n_nodes, key=kk).at[0].set(1.0).at[-1].set(0.0))

    for i in range(n_hf_rand):
        key, kk = jax.random.split(key)
        labels.append(f"HF-Rand#{i}"); families.append("HF-Rand")
        T0_list.append(get_random_T0(kk, reference_n_nodes).at[0].set(1.0).at[-1].set(0.0))

    all_T0 = jnp.stack(T0_list)
    all_alpha = jnp.tile(jnp.full((reference_n_nodes,), alpha_const)[None, :], (len(labels), 1))
    trajectory = _chunked_batch_fem_rollout(all_T0, all_alpha, steps_ref, dt_ref, rho_cp, newton_iters, chunk_size)

    result = {"labels": labels, "families": families, "T0": np.asarray(all_T0), "alpha": np.asarray(all_alpha),
              "trajectory": np.asarray(trajectory), "n_nodes": reference_n_nodes, "dt": dt_ref,
              "max_steps": steps_ref, "alpha_const": alpha_const, "seed": seed, "n_total": n_total}

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(result, f)
    return result


def load_ood_fine_trajectories(path):
    """Read back a pool written by precompute_ood_fine_trajectories. Raises
    FileNotFoundError with a clear message if `path` doesn't exist yet."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} doesn't exist yet -- call get_ood_fine_trajectories(path, ...) "
            "once to generate and cache it."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def get_ood_fine_trajectories(path, n_total=50, dt_ref=0.0002, steps_ref=1900,
                               reference_n_nodes=DEFAULT_REFERENCE_N_NODES, alpha_const=0.25, seed=101,
                               rho_cp=DEFAULT_RHO_CP, newton_iters=DEFAULT_NEWTON_ITERS, force_recompute=False,
                               chunk_size=8):
    """The one-time-expensive OOD entry point: if a cache matching this exact config
    already exists at `path`, load it (no solving). Otherwise solve it once (see
    precompute_ood_fine_trajectories) and save it there."""
    if not force_recompute and os.path.exists(path):
        data = load_ood_fine_trajectories(path)
        if (data.get("n_total") == n_total and abs(data["dt"] - dt_ref) < 1e-12
                and data["max_steps"] == steps_ref and data["n_nodes"] == reference_n_nodes
                and data["alpha_const"] == alpha_const and data["seed"] == seed):
            return data
        print(f"{path} exists but was cached with a different config -- recomputing.")
    return precompute_ood_fine_trajectories(path, n_total, dt_ref, steps_ref, reference_n_nodes,
                                             alpha_const, seed, rho_cp, newton_iters, chunk_size)


def derive_ood_dataset(fine_data, n_nodes_target, dt_target):
    """Derive OOD ground truth at ANY (n_nodes_target, dt_target) from the fine
    trajectories above -- exact time stride-subsampling + spatial interpolation,
    no re-solving, mirroring derive_dataset for the in-distribution case. Returns a
    list of dicts in the same shape as build_ood_scenarios/get_ood_reference_dataset
    (label, family, res, T0, alpha, true_traj), so it's a drop-in replacement
    anywhere those are already consumed."""
    traj_ref = jnp.asarray(fine_data["trajectory"])           # (n_shapes, steps_ref+1, n_nodes_ref)
    dt_ref = float(fine_data["dt"])
    traj_sub = subsample_to_dt(traj_ref, fine_dt=dt_ref, target_dt=dt_target)
    trajectory = resample_field(traj_sub, n_nodes_target)      # (n_shapes, steps_target+1, n_nodes_target)
    alpha = resample_field(jnp.asarray(fine_data["alpha"]), n_nodes_target)

    out = []
    for i, label in enumerate(fine_data["labels"]):
        out.append({
            "label": f"{label}@{n_nodes_target}", "family": fine_data["families"][i], "res": n_nodes_target,
            "T0": np.asarray(trajectory[i, 0, :]), "alpha": np.asarray(alpha[i]),
            "true_traj": np.asarray(trajectory[i]),
        })
    return out


# =============================================================================
# Disk-cached "ground truth for every sample" dataset. This is the expensive
# part -- the full Newton-Raphson trajectory for every sample, not just T0/
# alpha -- so it's the part worth saving to disk instead of resolving on
# every run. get_dataset is the one function most callers want.
# =============================================================================
def precompute_and_cache_dataset(path, n_samples, max_steps, n_nodes=DEFAULT_N_NODES,
                                  seed_k=42, seed_t=7, dt=DEFAULT_DT, rho_cp=DEFAULT_RHO_CP,
                                  newton_iters=DEFAULT_NEWTON_ITERS):
    """Generate n_samples in-distribution (T0, alpha) pairs and their full FEM
    ground-truth trajectory, then write everything to `path` (.npz). Always
    recomputes -- use get_dataset for the load-if-cached, else-compute version."""
    keys_k = jax.random.split(jax.random.PRNGKey(seed_k), n_samples)
    keys_t = jax.random.split(jax.random.PRNGKey(seed_t), n_samples)
    all_alpha, all_T0 = jax.vmap(generate_T0_alpha, in_axes=(0, 0, None))(keys_k, keys_t, n_nodes)
    all_generator = jax.vmap(t0_generator_choice)(keys_t)
    trajectory = batch_fem_rollout(all_T0, all_alpha, max_steps, dt, rho_cp, newton_iters)

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    np.savez(
        path,
        T0=np.asarray(all_T0), alpha=np.asarray(all_alpha),
        generator=np.asarray(all_generator), trajectory=np.asarray(trajectory),
        n_samples=n_samples, max_steps=max_steps, n_nodes=n_nodes,
        dt=dt, rho_cp=rho_cp, newton_iters=newton_iters, seed_k=seed_k, seed_t=seed_t,
    )
    return {"T0": all_T0, "alpha": all_alpha, "generator": all_generator, "trajectory": trajectory}


def load_dataset(path):
    """Read back a dataset written by precompute_and_cache_dataset. Raises
    FileNotFoundError with a clear message if `path` doesn't exist yet."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} doesn't exist yet -- call get_dataset(path, n_samples, max_steps) "
            "once to generate and cache it."
        )
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def get_dataset(path, n_samples, max_steps, n_nodes=DEFAULT_N_NODES, seed_k=42, seed_t=7,
                 dt=DEFAULT_DT, rho_cp=DEFAULT_RHO_CP, newton_iters=DEFAULT_NEWTON_ITERS,
                 force_recompute=False):
    """The ground-truth entry point: if a cache matching this exact config
    already exists at `path`, load it (no solving). Otherwise solve it once
    and save it there for next time. Returns a dict with keys T0, alpha,
    generator, trajectory -- trajectory has shape (n_samples, max_steps+1, n_nodes)."""
    if not force_recompute and os.path.exists(path):
        data = load_dataset(path)
        if (int(data["n_samples"]) == n_samples and int(data["max_steps"]) == max_steps
                and int(data["n_nodes"]) == n_nodes and int(data["seed_k"]) == seed_k
                and int(data["seed_t"]) == seed_t):
            return data
        print(f"{path} exists but was cached with a different config -- recomputing.")
    return precompute_and_cache_dataset(path, n_samples, max_steps, n_nodes, seed_k, seed_t,
                                         dt, rho_cp, newton_iters)


# =============================================================================
# Universal, resolution-independent reference: ONE fixed pool of in-distribution
# (T0, alpha) fields, generated once at a fine reference_n_nodes mesh, then
# resampled DOWN to whatever (n_nodes, dt, steps) any given script actually needs.
#
# Why this exists: get_dataset above solves generate_T0_alpha fresh at whatever
# n_nodes you pass it -- fine if only one script, one resolution ever calls it,
# but not fair for comparing FNOs trained/tested at DIFFERENT resolutions (or
# different notebooks with different dt scales) against each other, since each
# fresh solve's own discretization error gets conflated with the model's error
# (same reasoning as precompute_ood_reference_dataset above, now applied to the
# in-distribution generator too).
#
# The extra wrinkle here that OOD doesn't have: generate_T0_alpha's Gaussian-
# random-process branch samples its field via a Cholesky factorization built ON
# THE MESH ITSELF -- the same key at two different n_nodes gives two UNRELATED
# random fields, not two resolutions of the same field (see
# fem_convergence_study_indist.py's docstring, which worked around this by
# filtering GRF samples out of its convergence sweep entirely). Fixing the field
# ONCE at a fine mesh here, and only ever interpolating it DOWN afterwards,
# sidesteps that problem completely: every resolution gets a real, well-defined
# coarsening of the exact same underlying field, GRF branch included.
#
# Typical use from any script:
#
#     from fem_ground_truth import get_field_pool, get_resolution_dataset, subsample_to_dt
#
#     get_field_pool("checkpoints/field_pool.npz", n_samples=1000)   # once, cheap (no FEM solve)
#
#     big = get_resolution_dataset("checkpoints/res64_dt02.npz", "checkpoints/field_pool.npz",
#                                   n_nodes=64, dt=0.02, steps=19)
#     fine = get_resolution_dataset("checkpoints/res64_dt0002.npz", "checkpoints/field_pool.npz",
#                                    n_nodes=64, dt=0.0002, steps=1900)
#
#     # Two scripts sharing pool_path always start from the SAME T0/alpha fields,
#     # no matter what resolution/dt each one asks for.
# =============================================================================


def resample_field(field, n_nodes_target):
    """Interpolate a 1D field on [0,1] (or a batch of them, any leading shape) from
    its native node count down (or up) to n_nodes_target. Only meaningful as "the
    same case, different resolution" when going from a FINER native resolution to a
    coarser target -- the reverse just interpolates, it doesn't invent new detail."""
    field = jnp.asarray(field)
    n_native = field.shape[-1]
    x_native = jnp.linspace(0.0, 1.0, n_native)
    x_target = jnp.linspace(0.0, 1.0, n_nodes_target)
    interp_1d = lambda f: jnp.interp(x_target, x_native, f)
    if field.ndim == 1:
        return interp_1d(field)
    flat = field.reshape(-1, n_native)
    out = jax.vmap(interp_1d)(flat)
    return out.reshape(field.shape[:-1] + (n_nodes_target,))


def precompute_field_pool(path, n_samples, reference_n_nodes=DEFAULT_REFERENCE_N_NODES,
                           seed_k=42, seed_t=7):
    """Generate and disk-cache ONE fixed pool of in-distribution (T0, alpha) pairs at
    a fine reference_n_nodes mesh -- input fields only, no FEM solving (cheap, seconds
    not minutes). This is the single fixed pool every (n_nodes, dt) combo should be
    solved from via get_resolution_dataset below, instead of calling generate_T0_alpha
    fresh at each resolution. Always recomputes -- use get_field_pool for the
    load-if-cached, else-compute version."""
    keys_k = jax.random.split(jax.random.PRNGKey(seed_k), n_samples)
    keys_t = jax.random.split(jax.random.PRNGKey(seed_t), n_samples)
    all_alpha, all_T0 = jax.vmap(generate_T0_alpha, in_axes=(0, 0, None))(keys_k, keys_t, reference_n_nodes)
    all_generator = jax.vmap(t0_generator_choice)(keys_t)

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    np.savez(path, T0=np.asarray(all_T0), alpha=np.asarray(all_alpha), generator=np.asarray(all_generator),
              n_samples=n_samples, reference_n_nodes=reference_n_nodes, seed_k=seed_k, seed_t=seed_t)
    return {"T0": all_T0, "alpha": all_alpha, "generator": all_generator}


def load_field_pool(path):
    """Read back a pool written by precompute_field_pool. Raises FileNotFoundError
    with a clear message if `path` doesn't exist yet."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} doesn't exist yet -- call get_field_pool(path, n_samples) once "
            "to generate and cache it."
        )
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def get_field_pool(path, n_samples, reference_n_nodes=DEFAULT_REFERENCE_N_NODES,
                    seed_k=42, seed_t=7, force_recompute=False):
    """The fixed-pool entry point: if a cache matching this exact config already
    exists at `path`, load it. Otherwise generate it once and save it there."""
    if not force_recompute and os.path.exists(path):
        data = load_field_pool(path)
        if (int(data["n_samples"]) == n_samples and int(data["reference_n_nodes"]) == reference_n_nodes
                and int(data["seed_k"]) == seed_k and int(data["seed_t"]) == seed_t):
            return data
        print(f"{path} exists but was cached with a different config -- recomputing.")
    return precompute_field_pool(path, n_samples, reference_n_nodes, seed_k, seed_t)


# -----------------------------------------------------------------------------
# get_resolution_dataset (an earlier version of this idea) resampled T0/alpha
# down to the target n_nodes and then SOLVED FEM FRESH AT THAT RESOLUTION --
# which only fixes cross-resolution INPUT consistency (the GRF problem above).
# The resulting trajectory still carried that target resolution's own spatial
# discretization error, exactly the problem precompute_ood_reference_dataset
# already avoids for OOD (solve once at the fine mesh, interpolate the SOLVED
# TRAJECTORY down -- never re-solve at the coarser target). Replaced below by
# the same solve-once-interpolate-down pattern, now for in-distribution data.
# -----------------------------------------------------------------------------
def _chunked_batch_fem_rollout(T0_batch, alpha_batch, steps, dt, rho_cp, newton_iters, chunk_size):
    """Same result as batch_fem_rollout, computed sample-chunk by sample-chunk instead
    of vmapping the whole batch at once. solve_transient_fem_step's exact Jacobian
    (jax.jacfwd) already needs O(n_nodes^2) memory for a SINGLE sample; batch_fem_rollout
    vmaps that across the whole batch, so a large batch (e.g. 1000 samples at n_nodes=513)
    can ask for far more device memory than any GPU has in one shot (observed: ~2.17TB
    for batch=1000). Chunking bounds peak memory to chunk_size samples' worth, at the
    cost of chunk_size being small enough to fit -- same total FLOPs and identical
    results either way, just computed in smaller pieces. chunk_size=1 is always safe
    (matches the proven-working single-sample fem_rollout path) but slower; raise it as
    high as your GPU's memory allows for speed."""
    n = T0_batch.shape[0]
    n_chunks = -(-n // chunk_size)   # ceil division
    chunks = []
    t_start = time.perf_counter()
    for i, start in enumerate(range(0, n, chunk_size)):
        end = min(start + chunk_size, n)
        traj = batch_fem_rollout(T0_batch[start:end], alpha_batch[start:end], steps, dt, rho_cp, newton_iters)
        chunks.append(np.asarray(traj))   # pull off device now so the next chunk's memory is freed
        elapsed = time.perf_counter() - t_start
        done = i + 1
        eta = elapsed / done * (n_chunks - done)
        print(f"    chunk {done}/{n_chunks}  ({end}/{n} samples)  "
              f"elapsed {elapsed/60:.1f} min  ETA {eta/60:.1f} min", flush=True)
    return np.concatenate(chunks, axis=0)


def precompute_fine_trajectory(path, pool_path, dt_ref, steps_ref, n_samples=None,
                                rho_cp=DEFAULT_RHO_CP, newton_iters=DEFAULT_NEWTON_ITERS, chunk_size=8):
    """Solve the ONE canonical fine trajectory everything downstream derives from:
    the fixed field pool's (T0, alpha) -- already at the pool's fine reference mesh,
    no resampling needed here -- rolled out at a fine reference (dt_ref, steps_ref).
    This is the single expensive solve in the whole pipeline (fine mesh x fine dt x
    all samples) -- run it once, ideally with real compute (GPU/cluster), then every
    (n_nodes, dt) any script needs comes from derive_dataset below on this cached
    result, with NO further FEM solving, ever. Always recomputes -- use
    get_reference_trajectory for the load-if-cached, else-compute version.

    dt_ref should be a fine enough dt that every dt you'll ever need downstream is an
    exact integer multiple of it (see subsample_to_dt) -- e.g. dt_ref=0.0002 divides
    evenly into 0.002, 0.02, etc.

    chunk_size bounds peak GPU memory (see _chunked_batch_fem_rollout) -- lower it if
    you hit RESOURCE_EXHAUSTED, raise it for speed if your GPU has room to spare."""
    pool = load_field_pool(pool_path)
    all_T0 = jnp.asarray(pool["T0"]); all_alpha = jnp.asarray(pool["alpha"])
    all_generator = pool["generator"]
    if n_samples is not None:
        all_T0, all_alpha, all_generator = all_T0[:n_samples], all_alpha[:n_samples], all_generator[:n_samples]
    trajectory = _chunked_batch_fem_rollout(all_T0, all_alpha, steps_ref, dt_ref, rho_cp, newton_iters, chunk_size)

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    np.savez(path, T0=np.asarray(all_T0), alpha=np.asarray(all_alpha), generator=np.asarray(all_generator),
              trajectory=np.asarray(trajectory), n_samples=all_T0.shape[0], max_steps=steps_ref,
              n_nodes=all_T0.shape[-1], dt=dt_ref, rho_cp=rho_cp, newton_iters=newton_iters, pool_path=pool_path)
    return {"T0": all_T0, "alpha": all_alpha, "generator": all_generator, "trajectory": trajectory,
            "n_nodes": all_T0.shape[-1], "dt": dt_ref, "max_steps": steps_ref}


def get_reference_trajectory(path, pool_path, dt_ref, steps_ref, n_samples=None,
                              rho_cp=DEFAULT_RHO_CP, newton_iters=DEFAULT_NEWTON_ITERS,
                              force_recompute=False, chunk_size=8):
    """The one-time-expensive entry point: if a cache matching this exact config
    already exists at `path`, load it (no solving). Otherwise solve the full fine
    trajectory once (see precompute_fine_trajectory) and save it there."""
    if not force_recompute and os.path.exists(path):
        data = load_dataset(path)
        if (int(data["max_steps"]) == steps_ref and abs(float(data["dt"]) - dt_ref) < 1e-12
                and (n_samples is None or int(data["n_samples"]) == n_samples)):
            return data
        print(f"{path} exists but was cached with a different config -- recomputing.")
    return precompute_fine_trajectory(path, pool_path, dt_ref, steps_ref, n_samples, rho_cp, newton_iters, chunk_size)


def derive_dataset(reference_data, n_nodes_target, dt_target):
    """Derive a (T0, alpha, trajectory) dataset at ANY (n_nodes_target, dt_target)
    from the ONE canonical fine trajectory (get_reference_trajectory above) -- by
    exact time stride-subsampling (dt_target must be an exact multiple of the
    reference's own dt -- see subsample_to_dt) and spatial interpolation
    (resample_field). Solves NOTHING: this never re-runs Newton-Raphson, it only
    re-views the single already-accurate reference solve, so the result is exactly
    as spatially/temporally accurate as that one reference, no matter how coarse
    n_nodes_target/dt_target are -- this is what actually makes the ground truth
    resolution/dt-independent, not just consistent in its inputs."""
    traj_ref = jnp.asarray(reference_data["trajectory"])            # (n_samples, steps_ref+1, n_nodes_ref)
    dt_ref = float(reference_data["dt"])
    traj_sub = subsample_to_dt(traj_ref, fine_dt=dt_ref, target_dt=dt_target)
    trajectory = resample_field(traj_sub, n_nodes_target)           # (n_samples, steps_target+1, n_nodes_target)
    alpha = resample_field(jnp.asarray(reference_data["alpha"]), n_nodes_target)
    return {"T0": np.asarray(trajectory[:, 0, :]), "alpha": np.asarray(alpha),
            "generator": reference_data["generator"], "trajectory": np.asarray(trajectory),
            "n_nodes": n_nodes_target, "dt": dt_target, "max_steps": trajectory.shape[1] - 1}


def subsample_to_dt(trajectory, fine_dt, target_dt):
    """Pull out the states of a fine trajectory that land on a coarser target_dt
    time grid, by exact integer stride -- only valid when target_dt is an exact
    multiple of fine_dt. Raises rather than silently interpolating: interpolating
    in time would add a second, hidden error source on top of whatever's being
    graded. trajectory: (..., steps+1, n_nodes)."""
    ratio = target_dt / fine_dt
    stride = round(ratio)
    if abs(stride - ratio) > 1e-9:
        raise ValueError(
            f"target_dt={target_dt} is not an exact multiple of fine_dt={fine_dt} "
            f"(ratio={ratio}) -- pick a fine_dt that divides target_dt evenly, or "
            "explicitly time-interpolate instead of using this helper."
        )
    return trajectory[..., ::stride, :]



# =============================================================================
# STEADY STATE of the same rod: d/dx( k(x,T) dT/dx ) = 0, k(x,T) = alpha(x)*(0.5+T^2),
# T(0)=1, T(1)=0 -- i.e. the t -> infinity limit of the transient problem above,
# with the time term dropped. In FEM form K(T) T = 0 on the free nodes, solved with
# Newton-Raphson (exact Jacobian, jax.jacfwd), reusing assemble_LM's stiffness
# assembly so the steady and transient problems share one discretization.
#
# The steady answer depends on alpha only (every initial state relaxes to the same
# steady field), so the steady dataset is alpha(x) -> T(x). Like the transient side,
# the ground truth is solved ONCE on a fine mesh and interpolated down:
#
#     from fem_ground_truth import get_steady_reference, derive_steady_dataset, derive_steady_ood
#     ref = get_steady_reference("checkpoints/steady_reference.npz", "checkpoints/field_pool.npz")
#     data = derive_steady_dataset(ref, n_nodes_target=64)      # {"alpha": (1000,64), "T": (1000,64)}
#     ood  = derive_steady_ood(ref, n_nodes_target=128)         # list of dicts, 50 unseen alpha shapes
#
# Additions only -- nothing above this line was changed, so every transient result
# built on this module is unaffected.
# =============================================================================
DEFAULT_STEADY_NEWTON_ITERS = 25
DEFAULT_STEADY_REFERENCE_N_NODES = 511   # 2*255+1: nests the field pool's 256-node grid exactly,
                                         # so the pool's alpha is represented without re-interpolation
# alpha range of the in-distribution pool: generate_T0_alpha maps a [-1,1] signal through
# sigmoid(.)*0.4+0.1. The OOD alpha shapes below use the same range, so they are novel in
# SHAPE only, not in magnitude.
STEADY_ALPHA_MIN = float(jax.nn.sigmoid(-1.0) * 0.4 + 0.1)
STEADY_ALPHA_MAX = float(jax.nn.sigmoid(1.0) * 0.4 + 0.1)


def steady_stiffness(T, alpha_x):
    """K(T) alone (no mass matrix, no dt): assemble_LM's LHS with rho_cp=0, dt=1."""
    return assemble_LM(T, alpha_x, dt=1.0, rho_cp=0.0)[0]


@jax.jit
def compute_steady_residual(T, alpha_x):
    """Steady residual K(T) T on the interior nodes -- zero for the exact FEM solution."""
    free = jnp.arange(1, T.shape[0] - 1)
    return (steady_stiffness(T, alpha_x) @ T)[free]


def steady_residual_flux(T, alpha_x):
    """The same K(T) T on the interior nodes, written element-by-element instead of via
    the dense assembled matrix: with element flux g_e = k_avg_e (T_{e+1}-T_e)/dx and
    k_avg_e the element average of alpha*(0.5+T^2) (exactly assemble_LM's k_avg),
    (K T)_i = g_{i-1} - g_i. O(n) to evaluate, so its exact Jacobian (jacfwd) is cheap
    even on the fine reference mesh; compute_steady_residual is the dense-matrix check."""
    dx = 1.0 / (T.shape[0] - 1)
    k_nodal = alpha_x * (0.5 + T**2)
    g = 0.5 * (k_nodal[:-1] + k_nodal[1:]) * (T[1:] - T[:-1]) / dx
    return g[:-1] - g[1:]


@partial(jax.jit, static_argnames=("newton_iters",))
def solve_steady_fem(alpha_x, newton_iters=DEFAULT_STEADY_NEWTON_ITERS):
    """Steady FEM solution for one alpha field (any resolution, read from alpha_x.shape).
    Newton-Raphson from the linear ramp, exact Jacobian of the free-node residual."""
    n_nodes = alpha_x.shape[0]
    free = jnp.arange(1, n_nodes - 1)

    def full(Tf):
        return jnp.zeros(n_nodes).at[0].set(1.0).at[-1].set(0.0).at[free].set(Tf)

    def residual(Tf):
        return steady_residual_flux(full(Tf), alpha_x)

    def newton(Tf, _):
        Tf_new = Tf - jnp.linalg.solve(jax.jacfwd(residual)(Tf), residual(Tf))
        return Tf_new, None

    Tf, _ = jax.lax.scan(newton, jnp.linspace(1.0, 0.0, n_nodes)[free], None, length=newton_iters)
    return full(Tf)


# ---- out-of-distribution alpha shapes (same magnitude range as training, unseen shapes) ----
STEADY_OOD_FAMILIES = ["Layered", "Inclusion", "Sawtooth", "HF-Rand"]


def _to_alpha_range(v):
    return STEADY_ALPHA_MIN + (STEADY_ALPHA_MAX - STEADY_ALPHA_MIN) * v


def get_layered_alpha(key, n):
    """2-5 material layers with random values, joined by steep tanh interfaces
    (width 0.005-0.015: sharp, but bounded-gradient so FEM converges normally)."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    n_layers = int(jax.random.randint(k1, (), 2, 6))
    edges = jnp.sort(jax.random.uniform(k2, (n_layers - 1,), minval=0.1, maxval=0.9))
    levels = jax.random.uniform(k3, (n_layers,))
    width = jax.random.uniform(k4, (), minval=0.005, maxval=0.015)
    x = jnp.linspace(0, 1, n)
    v = jnp.full((n,), levels[0])
    for i in range(n_layers - 1):
        v = v + (levels[i + 1] - levels[i]) * 0.5 * (1.0 + jnp.tanh((x - edges[i]) / width))
    return _to_alpha_range(v)


def get_inclusion_alpha(key, n):
    """1-3 narrow Gaussian inclusions (width 0.015-0.04), each either much more or much
    less conductive than a uniform background."""
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    n_inc = int(jax.random.randint(k1, (), 1, 4))
    centers = jax.random.uniform(k2, (n_inc,), minval=0.15, maxval=0.85)
    widths = jax.random.uniform(k3, (n_inc,), minval=0.015, maxval=0.04)
    signs = jnp.where(jax.random.uniform(k4, (n_inc,)) < 0.5, -1.0, 1.0)
    base = jax.random.uniform(k5, (), minval=0.35, maxval=0.65)
    x = jnp.linspace(0, 1, n)
    bumps = jnp.sum(signs[:, None] * jnp.exp(-0.5 * ((x[None, :] - centers[:, None]) / widths[:, None]) ** 2), axis=0)
    return _to_alpha_range(jnp.clip(base + 0.6 * bumps, 0.0, 1.0))


def get_sawtooth_alpha(key, n):
    """Periodic sawtooth with 3-8 teeth (sharp drops every period), random phase."""
    k1, k2 = jax.random.split(key)
    teeth = jax.random.uniform(k1, (), minval=3.0, maxval=8.0)
    phase = jax.random.uniform(k2, ())
    x = jnp.linspace(0, 1, n)
    return _to_alpha_range(jnp.mod(teeth * x + phase, 1.0))


def get_hf_alpha(key, n, max_freq=18, decay=0.8):
    """Random Fourier field with flat spectral decay -> far more high-frequency content
    than the training pool (same construction as get_random_T0, mapped to the alpha range)."""
    return _to_alpha_range(get_random_T0(key, n, max_freq=max_freq, decay=decay))


_STEADY_OOD_FNS = {"Layered": get_layered_alpha, "Inclusion": get_inclusion_alpha,
                   "Sawtooth": get_sawtooth_alpha, "HF-Rand": get_hf_alpha}


def precompute_steady_reference(path, pool_path, reference_n_nodes=DEFAULT_STEADY_REFERENCE_N_NODES,
                                n_ood=50, ood_seed=303, newton_iters=DEFAULT_STEADY_NEWTON_ITERS):
    """Solve the steady ground truth once on the fine mesh and cache it (.npz):
    - in-distribution: every alpha of the fixed field pool (the SAME alpha fields the
      transient chapters use), interpolated onto the fine mesh;
    - OOD: n_ood unseen alpha shapes split evenly over STEADY_OOD_FAMILIES.
    Run with jax_enable_x64=True for a float64 reference. Always recomputes -- use
    get_steady_reference for the load-if-cached version."""
    pool = load_field_pool(pool_path)
    alpha = resample_field(jnp.asarray(pool["alpha"]), reference_n_nodes)
    T = jax.lax.map(lambda a: solve_steady_fem(a, newton_iters), alpha)

    labels, families, ood_alpha = [], [], []
    key = jax.random.PRNGKey(ood_seed)
    for fam, count in zip(STEADY_OOD_FAMILIES, _split_evenly(n_ood, len(STEADY_OOD_FAMILIES))):
        for i in range(count):
            key, kk = jax.random.split(key)
            labels.append(f"{fam}#{i}"); families.append(fam)
            ood_alpha.append(_STEADY_OOD_FNS[fam](kk, reference_n_nodes))
    ood_alpha = jnp.stack(ood_alpha)
    ood_T = jax.lax.map(lambda a: solve_steady_fem(a, newton_iters), ood_alpha)

    res_in = jax.lax.map(lambda ta: compute_steady_residual(*ta), (T, alpha))        # dense-matrix check
    res_ood = jax.lax.map(lambda ta: compute_steady_residual(*ta), (ood_T, ood_alpha))
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    np.savez(path, alpha=np.asarray(alpha), T=np.asarray(T), generator=np.asarray(pool["generator"]),
             ood_alpha=np.asarray(ood_alpha), ood_T=np.asarray(ood_T),
             ood_labels=np.asarray(labels), ood_families=np.asarray(families),
             max_residual=float(jnp.max(jnp.abs(jnp.concatenate([res_in.ravel(), res_ood.ravel()])))),
             n_nodes=reference_n_nodes, n_samples=alpha.shape[0], n_ood=n_ood, ood_seed=ood_seed,
             newton_iters=newton_iters, pool_path=pool_path)
    return load_dataset(path)


def get_steady_reference(path, pool_path, reference_n_nodes=DEFAULT_STEADY_REFERENCE_N_NODES,
                         n_ood=50, ood_seed=303, newton_iters=DEFAULT_STEADY_NEWTON_ITERS,
                         force_recompute=False):
    """Load the cached steady reference if it matches this config, else solve and cache it."""
    if not force_recompute and os.path.exists(path):
        data = load_dataset(path)
        if (int(data["n_nodes"]) == reference_n_nodes and int(data["n_ood"]) == n_ood
                and int(data["ood_seed"]) == ood_seed):
            return data
        print(f"{path} exists but was cached with a different config -- recomputing.")
    return precompute_steady_reference(path, pool_path, reference_n_nodes, n_ood, ood_seed, newton_iters)


def derive_steady_dataset(ref, n_nodes_target):
    """In-distribution steady data at any resolution, by interpolating the fine solve
    (no re-solving): {"alpha", "T", "generator"} with shapes (n_samples, n_nodes_target)."""
    return {"alpha": np.asarray(resample_field(jnp.asarray(ref["alpha"]), n_nodes_target)),
            "T": np.asarray(resample_field(jnp.asarray(ref["T"]), n_nodes_target)),
            "generator": np.asarray(ref["generator"])}


def derive_steady_ood(ref, n_nodes_target):
    """The OOD alpha shapes and their steady ground truth at any resolution, as a list of
    dicts (label, family, res, alpha, T) -- the steady analogue of derive_ood_dataset."""
    alpha = np.asarray(resample_field(jnp.asarray(ref["ood_alpha"]), n_nodes_target))
    T = np.asarray(resample_field(jnp.asarray(ref["ood_T"]), n_nodes_target))
    return [{"label": f"{str(lbl)}@{n_nodes_target}", "family": str(fam), "res": n_nodes_target,
             "alpha": alpha[i], "T": T[i]}
            for i, (lbl, fam) in enumerate(zip(ref["ood_labels"], ref["ood_families"]))]
