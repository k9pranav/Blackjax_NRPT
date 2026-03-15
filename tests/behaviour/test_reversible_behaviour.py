from __future__ import annotations

import jax
import jax.numpy as jnp

from NRPT_Blackjax import NRPTKernel, NRPTConfig, GaussianRMHLocalKernel


def easy_target_logdensity(x):
    z = x[0]
    return -0.5 * z**2


def easy_reference_logdensity(x):
    z = x[0]
    return -0.5 * z**2 / 10.0


def hard_target_logdensity(x):
    z = x[0]
    logp1 = -0.5 * ((z + 5.0) ** 2)
    logp2 = -0.5 * ((z - 5.0) ** 2)
    return jnp.logaddexp(logp1, logp2) - jnp.log(2.0)


def hard_reference_logdensity(x):
    z = x[0]
    return -0.5 * z**2 / 25.0


def make_kernel(target_logdensity_fn, reference_logdensity_fn, betas):
    config = NRPTConfig(
        n_replicas=len(betas),
        betas=jnp.asarray(betas, dtype=jnp.float32),
        initial_direction=+1,
        lift_mode="reversible",
        swap_scheme="adjacent",
        adapt_ladder=False,
    )

    local_kernel = GaussianRMHLocalKernel(sigma=jnp.array([0.8], dtype=jnp.float32))

    return NRPTKernel(
        local_kernel=local_kernel,
        target_logdensity_fn=target_logdensity_fn,
        reference_logdensity_fn=reference_logdensity_fn,
        config=config,
    )


def run_steps(kernel, init_positions, n_steps=12, seed=0):
    key = jax.random.key(seed)
    state = kernel.init(init_positions)

    parity_history = []
    direction_history = []
    blocked_history = []

    for _ in range(n_steps):
        key, subkey = jax.random.split(key)
        state, info = kernel.step(subkey, state)
        parity_history.append(int(info.swap_info.sweep_parity))
        direction_history.append(int(state.direction))
        blocked_history.append(bool(info.swap_info.blocked))

    return state, parity_history, direction_history, blocked_history


def test_reversible_easy_regime_alternates_parity_and_keeps_direction():
    betas = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    kernel = make_kernel(easy_target_logdensity, easy_reference_logdensity, betas)
    init_positions = jnp.array([[-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0]], dtype=jnp.float32)

    _, parity_history, direction_history, _ = run_steps(kernel, init_positions, n_steps=10, seed=11)

    expected = [i % 2 for i in range(10)]
    assert parity_history == expected
    assert set(direction_history) == {1}


def test_reversible_hard_regime_alternates_parity_even_if_blocked_occurs():
    betas = [1.0, 0.5, 0.2, 0.0]
    kernel = make_kernel(hard_target_logdensity, hard_reference_logdensity, betas)
    init_positions = jnp.array([[-5.0], [-5.0], [5.0], [5.0]], dtype=jnp.float32)

    _, parity_history, direction_history, _ = run_steps(kernel, init_positions, n_steps=10, seed=22)

    expected = [i % 2 for i in range(10)]
    assert parity_history == expected
    assert set(direction_history) == {1}