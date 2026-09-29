import jax
import jax.numpy as jnp
from config import WIDTH, MODES, DT

def spectral_layer(x, w_spec, w_skip):
    n = x.shape[0]
    x_ft = jnp.fft.rfft(x, axis=0)
    out_ft = jnp.zeros_like(x_ft, dtype=jnp.complex64).at[:MODES].set(
        jnp.einsum('mi, mij -> mj', x_ft[:MODES], w_spec))
    return jax.nn.gelu(jnp.fft.irfft(out_ft, n=n, axis=0) + jnp.dot(x, w_skip))

def fno_model(params, x_norm, T_curr):
    # Predicts the time-derivative T-dot, not the raw increment; one explicit-Euler
    # step of size DT turns that rate into the next state: T_next = T_curr + DT*Tdot.
    x = jax.nn.gelu(jnp.dot(x_norm, params[0]) + params[1])
    for i in range(2, 10, 2):
        x = spectral_layer(x, params[i], params[i+1])
    x = jax.nn.gelu(jnp.dot(x, params[10]) + params[11])
    T_dot = (jnp.dot(x, params[12]) + params[13]).squeeze()
    # Resolution-generic: the mask is sized from the input, so the same weights run on
    # finer grids (super-resolution OOD test, Section 8).
    mask = jnp.sin(jnp.pi * jnp.linspace(0, 1, T_curr.shape[0]))
    return T_curr + DT * (T_dot * mask)

def init_params(key):
    keys = jax.random.split(key, 12)
    def norm_init(k, s): return jax.random.normal(k, s) * jnp.sqrt(1.0 / s[-1])
    p = [norm_init(keys[0], (3, WIDTH)), jnp.zeros(WIDTH)]
    for i in range(4):
        p.append(jax.random.normal(keys[i+1], (MODES, WIDTH, WIDTH), dtype=jnp.complex64) * 0.02)
        p.append(norm_init(keys[i+5], (WIDTH, WIDTH)))
    p.extend([norm_init(keys[10], (WIDTH, WIDTH)), jnp.zeros(WIDTH),
              norm_init(keys[11], (WIDTH, 1)), jnp.zeros(1)])
    return p
