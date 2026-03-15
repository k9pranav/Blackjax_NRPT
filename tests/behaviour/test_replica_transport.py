from __future__ import annotations

import jax
import jax.numpy as jnp

from NRPT_Blackjax import NRPTKernel, NRPTConfig, GaussianRMHLocalKernel


def hard_target_logdensity(x):
    z = x[0]
    logp1 = -0.5 * ((z + 6.0) ** 2)
    logp2 = -0.5 * ((z - 6.0) ** 2)
    return jnp.logaddexp(logp1, logp2) - jnp.log(2.0)


def hard_reference_logdensity(x):
    z = x[0]
    return -0.5 * z**2 / 36.0


def make_kernel(lift_mode: str):
    config = NRPTConfig(
        n_replicas=6,
        betas=jnp.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0], dtype=jnp.float32),
        initial_direction=+1,
        lift_mode=lift_mode,
        swap_scheme="adjacent",
        adapt_ladder=False,
    )

    local_kernel = GaussianRMHLocalKernel(sigma=jnp.array([0.8], dtype=jnp.float32))

    return NRPTKernel(
        local_kernel=local_kernel,
        target_logdensity_fn=hard_target_logdensity,
        reference_logdensity_fn=hard_reference_logdensity,
        config=config,
    )


def test_replica_order_remains_permutation_over_time():
    kernel = make_kernel("reversible")
    init_positions = jnp.array([[-6.0], [-6.0], [-6.0], [6.0], [6.0], [6.0]], dtype=jnp.float32)

    key = jax.random.key(505)
    state = kernel.init(init_positions)

    for _ in range(25):
        key, subkey = jax.random.split(key)
        state, _ = kernel.step(subkey, state)
        assert set(map(int, state.replica_order.tolist())) == set(range(6))


def test_transport_occurs_under_at_least_one_lift_mode():
    init_positions = jnp.array([[-6.0], [-6.0], [-6.0], [6.0], [6.0], [6.0]], dtype=jnp.float32)

    moved = False
    for lift_mode in ("reversible", "simple_nonreversible", "persistent_sweep"):
        kernel = make_kernel(lift_mode)
        key = jax.random.key(606)
        state = kernel.init(init_positions)
        initial_order = state.replica_order

        for _ in range(20):
            key, subkey = jax.random.split(key)
            state, _ = kernel.step(subkey, state)

        if not jnp.array_equal(initial_order, state.replica_order):
            moved = True

    assert moved