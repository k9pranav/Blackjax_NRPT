from __future__ import annotations

import jax.numpy as jnp

from NRPT_Blackjax import NRPTKernel, NRPTConfig, GaussianRMHLocalKernel


def target_logdensity(x):
    z = x[0]
    logp1 = -0.5 * ((z + 4.0) ** 2)
    logp2 = -0.5 * ((z - 4.0) ** 2)
    return jnp.logaddexp(logp1, logp2)


def reference_logdensity(x):
    z = x[0]
    return -0.5 * z**2 / 25.0


def make_kernel():
    config = NRPTConfig(
        n_replicas=6,
        betas=jnp.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0]),
        initial_direction=+1,
        lift_mode="reversible",
        swap_scheme="adjacent",
        adapt_ladder=False,
    )
    local_kernel = GaussianRMHLocalKernel(sigma=jnp.array([0.8]))
    return NRPTKernel(
        local_kernel=local_kernel,
        target_logdensity_fn=target_logdensity,
        reference_logdensity_fn=reference_logdensity,
        config=config,
    )


def test_proposed_adjacent_pairs_even():
    pairs = NRPTKernel._proposed_adjacent_pairs(6, jnp.array(0, dtype=jnp.int32))
    expected = jnp.array([[0, 1], [2, 3], [4, 5]], dtype=jnp.int32)
    assert jnp.array_equal(pairs, expected)


def test_proposed_adjacent_pairs_odd():
    pairs = NRPTKernel._proposed_adjacent_pairs(6, jnp.array(1, dtype=jnp.int32))
    expected = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
    assert jnp.array_equal(pairs, expected)


def test_proposed_adjacent_pairs_small_replica_count():
    pairs = NRPTKernel._proposed_adjacent_pairs(1, jnp.array(0, dtype=jnp.int32))
    assert pairs.shape == (0, 2)


def test_swap_array_pairs_1d():
    kernel = make_kernel()
    x = jnp.array([10, 20, 30, 40, 50, 60])
    left_idx = jnp.array([0, 2, 4], dtype=jnp.int32)
    right_idx = jnp.array([1, 3, 5], dtype=jnp.int32)
    accepted = jnp.array([True, False, True])

    y = kernel._swap_array_pairs(x, left_idx, right_idx, accepted)
    expected = jnp.array([20, 10, 30, 40, 60, 50])
    assert jnp.array_equal(y, expected)


def test_swap_array_pairs_2d():
    kernel = make_kernel()
    x = jnp.array([[10.0], [20.0], [30.0], [40.0], [50.0], [60.0]])
    left_idx = jnp.array([0, 2, 4], dtype=jnp.int32)
    right_idx = jnp.array([1, 3, 5], dtype=jnp.int32)
    accepted = jnp.array([True, False, True])

    y = kernel._swap_array_pairs(x, left_idx, right_idx, accepted)
    expected = jnp.array([[20.0], [10.0], [30.0], [40.0], [60.0], [50.0]])
    assert jnp.array_equal(y, expected)


def test_apply_pair_swaps_only_replica_major_leaves():
    kernel = make_kernel()

    pytree = {
        "position": jnp.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]),
        "logdensity": jnp.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
        "scalar_meta": jnp.array(999.0),
        "wrong_leading_dim": jnp.array([7.0, 8.0, 9.0]),
    }

    left_idx = jnp.array([0, 2, 4], dtype=jnp.int32)
    right_idx = jnp.array([1, 3, 5], dtype=jnp.int32)
    accepted = jnp.array([True, False, True])

    out = kernel._apply_pair_swaps(pytree, left_idx, right_idx, accepted)

    assert jnp.array_equal(
        out["position"],
        jnp.array([[2.0], [1.0], [3.0], [4.0], [6.0], [5.0]])
    )
    assert jnp.array_equal(
        out["logdensity"],
        jnp.array([20.0, 10.0, 30.0, 40.0, 60.0, 50.0])
    )
    assert out["scalar_meta"] == pytree["scalar_meta"]
    assert jnp.array_equal(out["wrong_leading_dim"], pytree["wrong_leading_dim"])