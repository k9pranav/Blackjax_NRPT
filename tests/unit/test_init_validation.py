from __future__ import annotations

import pytest
import jax.numpy as jnp

from NRPT_Blackjax import NRPTKernel, NRPTConfig, GaussianRMHLocalKernel


def target_logdensity(x):
    return -0.5 * jnp.sum(x**2)


def reference_logdensity(x):
    return -0.5 * jnp.sum(x**2) / 10.0


def make_local_kernel():
    return GaussianRMHLocalKernel(sigma=jnp.array([0.8]))


def test_init_rejects_non_1d_betas():
    config = NRPTConfig(
        n_replicas=6,
        betas=jnp.ones((2, 3)),
        initial_direction=+1,
        lift_mode="reversible",
    )

    kernel = NRPTKernel(
        local_kernel=make_local_kernel(),
        target_logdensity_fn=target_logdensity,
        reference_logdensity_fn=reference_logdensity,
        config=config,
    )

    init_positions = jnp.ones((6, 1))
    with pytest.raises(ValueError, match="config.betas must be a 1D array"):
        kernel.init(init_positions)


def test_init_rejects_length_mismatch_between_betas_and_replicas():
    config = NRPTConfig(
        n_replicas=6,
        betas=jnp.array([1.0, 0.8, 0.6]),
        initial_direction=+1,
        lift_mode="reversible",
    )

    kernel = NRPTKernel(
        local_kernel=make_local_kernel(),
        target_logdensity_fn=target_logdensity,
        reference_logdensity_fn=reference_logdensity,
        config=config,
    )

    init_positions = jnp.ones((6, 1))
    with pytest.raises(ValueError, match="len\\(betas\\) must equal n_replicas"):
        kernel.init(init_positions)


def test_init_rejects_bad_initial_direction():
    config = NRPTConfig(
        n_replicas=6,
        betas=jnp.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0]),
        initial_direction=0,
        lift_mode="reversible",
    )

    kernel = NRPTKernel(
        local_kernel=make_local_kernel(),
        target_logdensity_fn=target_logdensity,
        reference_logdensity_fn=reference_logdensity,
        config=config,
    )

    init_positions = jnp.ones((6, 1))
    with pytest.raises(ValueError, match="initial_direction must be either \\+1 or -1"):
        kernel.init(init_positions)


def test_init_rejects_bad_lift_mode():
    config = NRPTConfig(
        n_replicas=6,
        betas=jnp.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0]),
        initial_direction=+1,
        lift_mode="not_a_mode",
    )

    kernel = NRPTKernel(
        local_kernel=make_local_kernel(),
        target_logdensity_fn=target_logdensity,
        reference_logdensity_fn=reference_logdensity,
        config=config,
    )

    init_positions = jnp.ones((6, 1))
    with pytest.raises(ValueError, match="lift_mode must be one of"):
        kernel.init(init_positions)


def test_init_sets_persistent_sweep_parity_from_direction():
    config = NRPTConfig(
        n_replicas=6,
        betas=jnp.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0]),
        initial_direction=-1,
        lift_mode="persistent_sweep",
    )

    kernel = NRPTKernel(
        local_kernel=make_local_kernel(),
        target_logdensity_fn=target_logdensity,
        reference_logdensity_fn=reference_logdensity,
        config=config,
    )

    init_positions = jnp.ones((6, 1))
    state = kernel.init(init_positions)

    assert int(state.direction) == -1
    assert state.sweep_parity is not None
    assert int(state.sweep_parity) == 1


def test_init_non_persistent_modes_leave_sweep_parity_none():
    for lift_mode in ("reversible", "simple_nonreversible"):
        config = NRPTConfig(
            n_replicas=6,
            betas=jnp.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0]),
            initial_direction=+1,
            lift_mode=lift_mode,
        )

        kernel = NRPTKernel(
            local_kernel=make_local_kernel(),
            target_logdensity_fn=target_logdensity,
            reference_logdensity_fn=reference_logdensity,
            config=config,
        )

        init_positions = jnp.ones((6, 1))
        state = kernel.init(init_positions)

        assert state.sweep_parity is None