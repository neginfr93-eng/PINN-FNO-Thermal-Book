import jax.numpy as jnp

# Resolution, split and dt are shared with the weighted-residual and PITI chapters.
N_NODES = 64
N_GRID = N_NODES - 1
WIDTH = 64
MODES = 4
BATCH_SIZE = 32
N_SAMPLES = 1000
TRAIN_SAMPLES = 800
VAL_SAMPLES = 100                                   # held out from training, used for early stopping
ROLLOUT_TEST_START = TRAIN_SAMPLES + VAL_SAMPLES    # last 100: never seen by training or early stopping
MAX_STEPS = 20
DT = 0.02
RHO_CP = 5.0
GRID = jnp.linspace(0, 1, N_NODES)

# FEM ground truth: fem_ground_truth.py's cached fine reference (N=256, dt=0.001, 400 steps,
# fully-implicit Newton-Raphson), interpolated down to (N_NODES, DT) -- never re-solved here.
POOL_PATH = "checkpoints/field_pool.npz"
REFERENCE_PATH = "checkpoints/fine_reference.npz"
OOD_REFERENCE_PATH = "checkpoints/ood_fine_reference.pkl"
REF_DT, REF_STEPS = 0.001, 400

# Early stopping (no fixed epoch count): train up to MAX_EPOCHS, stopping sooner if
# held-out validation MSE stalls/worsens for PATIENCE consecutive checks.
MAX_EPOCHS = 20000      # same budget as the weighted-residual and PITI chapters
# Optional cap on checkpoint selection (training always runs to MAX_EPOCHS). Set to
# MAX_EPOCHS, i.e. no cap: Section 9 shows OOD error keeps falling with training.
DATA_MAX_EPOCHS = MAX_EPOCHS
CHECK_EVERY = 200
PATIENCE = 15
