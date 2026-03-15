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
        lift_mode="persistent_sweep",
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


def run_steps(kernel, init_positions, n_steps=16, seed=0):
    key = jax.random.key(seed)
    state = kernel.init(init_positions)

    schedule_parities = []
    next_sweep_parities = [int(state.sweep_parity)]
    direction_history = [int(state.direction)]
    blocked_history = []

    for _ in range(n_steps):
        key, subkey = jax.random.split(key)
        state, info = kernel.step(subkey, state)
        schedule_parities.append(int(info.swap_info.sweep_parity))
        next_sweep_parities.append(int(state.sweep_parity))
        direction_history.append(int(state.direction))
        blocked_history.append(bool(info.swap_info.blocked))

    return state, schedule_parities, next_sweep_parities, direction_history, blocked_history


def test_persistent_sweep_easy_regime_alternates_sweep_parity_when_unblocked():
    betas = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    kernel = make_kernel(easy_target_logdensity, easy_reference_logdensity, betas)
    init_positions = jnp.array([[-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0]], dtype=jnp.float32)

    _, schedule_parities, next_sweep_parities, direction_history, blocked_history = run_steps(
        kernel, init_positions, n_steps=10, seed=303
    )

    assert all(p in (0, 1) for p in schedule_parities)
    assert all(p in (0, 1) for p in next_sweep_parities)
    assert set(direction_history).issubset({1, -1})

    # When unblocked on consecutive steps, schedule parity should alternate often.
    alternating_pairs = sum(
        schedule_parities[i] != schedule_parities[i - 1] for i in range(1, len(schedule_parities))
    )
    assert alternating_pairs >= 6


def test_persistent_sweep_hard_regime_can_block_and_flip():
    betas = [1.0, 0.4, 0.1, 0.0]
    kernel = make_kernel(hard_target_logdensity, hard_reference_logdensity, betas)
    init_positions = jnp.array([[-6.0], [-6.0], [6.0], [6.0]], dtype=jnp.float32)

    _, schedule_parities, next_sweep_parities, direction_history, blocked_history = run_steps(
        kernel, init_positions, n_steps=20, seed=404
    )

    assert all(p in (0, 1) for p in schedule_parities)
    assert all(p in (0, 1) for p in next_sweep_parities)
    assert len(direction_history) == 21
    assert len(blocked_history) == 20

    assert any(blocked_history) or len(set(direction_history)) > 1