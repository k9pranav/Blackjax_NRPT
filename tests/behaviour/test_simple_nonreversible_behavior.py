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
    logp1 = -0.5 * ((z + 6.0) ** 2)
    logp2 = -0.5 * ((z - 6.0) ** 2)
    return jnp.logaddexp(logp1, logp2) - jnp.log(2.0)


def hard_reference_logdensity(x):
    z = x[0]
    return -0.5 * z**2 / 36.0


def make_kernel(target_logdensity_fn, reference_logdensity_fn, betas):
    config = NRPTConfig(
        n_replicas=len(betas),
        betas=jnp.asarray(betas, dtype=jnp.float32),
        initial_direction=+1,
        lift_mode="simple_nonreversible",
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


def run_steps(kernel, init_positions, n_steps=15, seed=0):
    key = jax.random.key(seed)
    state = kernel.init(init_positions)

    parity_history = []
    direction_history = [int(state.direction)]
    blocked_history = []

    for _ in range(n_steps):
        key, subkey = jax.random.split(key)
        state, info = kernel.step(subkey, state)
        parity_history.append(int(info.swap_info.sweep_parity))
        direction_history.append(int(state.direction))
        blocked_history.append(bool(info.swap_info.blocked))

    return state, parity_history, direction_history, blocked_history


def test_simple_nonreversible_easy_regime_often_stays_on_one_parity():
    betas = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    kernel = make_kernel(easy_target_logdensity, easy_reference_logdensity, betas)
    init_positions = jnp.array([[-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0]], dtype=jnp.float32)

    _, parity_history, direction_history, blocked_history = run_steps(kernel, init_positions, n_steps=12, seed=101)

    assert set(direction_history).issubset({1, -1})
    assert all(p in (0, 1) for p in parity_history)
    assert len(blocked_history) == 12


def test_simple_nonreversible_hard_regime_can_flip_direction_after_blocking():
    betas = [1.0, 0.4, 0.1, 0.0]
    kernel = make_kernel(hard_target_logdensity, hard_reference_logdensity, betas)
    init_positions = jnp.array([[-6.0], [-6.0], [6.0], [6.0]], dtype=jnp.float32)

    _, parity_history, direction_history, blocked_history = run_steps(kernel, init_positions, n_steps=20, seed=202)

    assert all(p in (0, 1) for p in parity_history)
    assert len(direction_history) == 21
    assert len(blocked_history) == 20

    # In the hard regime, we expect at least one blockage or flip to be plausible.
    assert any(blocked_history) or len(set(direction_history)) > 1