from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp

from NRPT_Blackjax import (
    NRPTKernel,
    NRPTConfig,
    GaussianRMHLocalKernel,
    MALALocalKernel,
    HMCLocalKernel,
)


def target_logdensity(x):
    z = x[0]
    logp1 = -0.5 * ((z + 4.0) ** 2) - 0.5 * jnp.log(2.0 * jnp.pi)
    logp2 = -0.5 * ((z - 4.0) ** 2) - 0.5 * jnp.log(2.0 * jnp.pi)
    return jnp.logaddexp(logp1, logp2) - jnp.log(2.0)


def reference_logdensity(x):
    z = x[0]
    sigma2 = 25.0
    return -0.5 * z**2 / sigma2 - 0.5 * jnp.log(2.0 * jnp.pi * sigma2)


def initial_positions():
    return jnp.array(
        [[-4.00], [-3.80], [-4.20], [-3.90], [-4.10], [-4.05]],
        dtype=jnp.float32,
    )


def make_config(lift_mode: str):
    return NRPTConfig(
        n_replicas=6,
        betas=jnp.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0], dtype=jnp.float32),
        initial_direction=+1,
        lift_mode=lift_mode,
        swap_scheme="adjacent",
        adapt_ladder=False,
    )


def assert_state_info_invariants(state, info, n_replicas=6):
    assert state.betas.shape == (n_replicas,)
    assert state.replica_order is None or state.replica_order.shape == (n_replicas,)
    assert int(state.direction) in (-1, 1)

    assert hasattr(state.replica_states, "position")
    assert state.replica_states.position.shape[0] == n_replicas
    assert jnp.all(jnp.isfinite(state.replica_states.position))

    s = info.swap_info
    assert s.proposed_pairs.ndim == 2
    assert s.proposed_pairs.shape[1] == 2
    assert s.log_accept_ratio.shape[0] == s.proposed_pairs.shape[0]
    assert s.acceptance_prob.shape[0] == s.proposed_pairs.shape[0]
    assert s.accepted.shape[0] == s.proposed_pairs.shape[0]
    assert jnp.all(jnp.isfinite(s.acceptance_prob))
    assert jnp.all(s.acceptance_prob >= 0.0)
    assert jnp.all(s.acceptance_prob <= 1.0)
    assert int(s.direction) in (-1, 1)


@pytest.mark.parametrize(
    "local_kernel",
    [
        GaussianRMHLocalKernel(sigma=jnp.array([0.8], dtype=jnp.float32)),
        MALALocalKernel(step_size=0.15),
        HMCLocalKernel(
            step_size=0.10,
            inverse_mass_matrix=jnp.array([1.0], dtype=jnp.float32),
            num_integration_steps=5,
        ),
    ],
)
@pytest.mark.parametrize(
    "lift_mode",
    ["reversible", "simple_nonreversible", "persistent_sweep"],
)
def test_nrpt_single_step_smoke(local_kernel, lift_mode):
    kernel = NRPTKernel(
        local_kernel=local_kernel,
        target_logdensity_fn=target_logdensity,
        reference_logdensity_fn=reference_logdensity,
        config=make_config(lift_mode),
    )

    state = kernel.init(initial_positions())
    key = jax.random.key(0)
    new_state, info = kernel.step(key, state)

    assert new_state.iteration == 1
    assert_state_info_invariants(new_state, info)

    if lift_mode == "persistent_sweep":
        assert new_state.sweep_parity is not None
    else:
        assert new_state.sweep_parity is None


@pytest.mark.parametrize(
    "local_kernel",
    [
        GaussianRMHLocalKernel(sigma=jnp.array([0.8], dtype=jnp.float32)),
        MALALocalKernel(step_size=0.15),
        HMCLocalKernel(
            step_size=0.10,
            inverse_mass_matrix=jnp.array([1.0], dtype=jnp.float32),
            num_integration_steps=5,
        ),
    ],
)
@pytest.mark.parametrize(
    "lift_mode",
    ["reversible", "simple_nonreversible", "persistent_sweep"],
)
def test_nrpt_multiple_steps_smoke(local_kernel, lift_mode):
    kernel = NRPTKernel(
        local_kernel=local_kernel,
        target_logdensity_fn=target_logdensity,
        reference_logdensity_fn=reference_logdensity,
        config=make_config(lift_mode),
    )

    state = kernel.init(initial_positions())
    key = jax.random.key(123)

    n_steps = 10
    for t in range(n_steps):
        key, subkey = jax.random.split(key)
        state, info = kernel.step(subkey, state)
        assert state.iteration == t + 1
        assert_state_info_invariants(state, info)

    if lift_mode == "persistent_sweep":
        assert state.sweep_parity is not None
    else:
        assert state.sweep_parity is None


@pytest.mark.parametrize(
    "local_kernel",
    [
        GaussianRMHLocalKernel(sigma=jnp.array([0.8], dtype=jnp.float32)),
        MALALocalKernel(step_size=0.15),
        HMCLocalKernel(
            step_size=0.10,
            inverse_mass_matrix=jnp.array([1.0], dtype=jnp.float32),
            num_integration_steps=5,
        ),
    ],
)
def test_replica_order_is_a_permutation_after_steps(local_kernel):
    kernel = NRPTKernel(
        local_kernel=local_kernel,
        target_logdensity_fn=target_logdensity,
        reference_logdensity_fn=reference_logdensity,
        config=make_config("reversible"),
    )

    state = kernel.init(initial_positions())
    key = jax.random.key(999)

    for _ in range(10):
        key, subkey = jax.random.split(key)
        state, _ = kernel.step(subkey, state)

    order = state.replica_order
    assert order is not None
    assert set(map(int, order.tolist())) == set(range(6))