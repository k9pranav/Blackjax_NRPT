from __future__ import annotations

import jax.numpy as jnp

from NRPT_Blackjax import (
    NRPTKernel,
    NRPTConfig,
    GaussianRMHLocalKernel,
    TemperedState,
)


def target_logdensity(x):
    return -0.5 * jnp.sum(x**2)


def reference_logdensity(x):
    return -0.5 * jnp.sum(x**2) / 10.0


def make_kernel(lift_mode: str):
    config = NRPTConfig(
        n_replicas=6,
        betas=jnp.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0]),
        initial_direction=+1,
        lift_mode=lift_mode,
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


def make_state(lift_mode: str, direction: int = +1, sweep_parity=None, iteration: int = 0):
    kernel = make_kernel(lift_mode)
    init_positions = jnp.array([[-4.0], [-3.8], [-4.2], [-3.9], [-4.1], [-4.05]])
    state = kernel.init(init_positions)
    return TemperedState(
        replica_states=state.replica_states,
        betas=state.betas,
        direction=jnp.array(direction, dtype=jnp.int32),
        replica_order=state.replica_order,
        adaptation_state=state.adaptation_state,
        iteration=iteration,
        sweep_parity=sweep_parity,
    )


def test_select_swap_pairs_reversible_uses_iteration_parity():
    kernel = make_kernel("reversible")

    state0 = make_state("reversible", iteration=0)
    pairs0, parity0 = kernel._select_swap_pairs(state0)
    assert int(parity0) == 0
    assert jnp.array_equal(pairs0, jnp.array([[0, 1], [2, 3], [4, 5]], dtype=jnp.int32))

    state1 = make_state("reversible", iteration=1)
    pairs1, parity1 = kernel._select_swap_pairs(state1)
    assert int(parity1) == 1
    assert jnp.array_equal(pairs1, jnp.array([[1, 2], [3, 4]], dtype=jnp.int32))


def test_select_swap_pairs_simple_nonreversible_uses_direction():
    kernel = make_kernel("simple_nonreversible")

    state_plus = make_state("simple_nonreversible", direction=+1)
    pairs_plus, parity_plus = kernel._select_swap_pairs(state_plus)
    assert int(parity_plus) == 0
    assert jnp.array_equal(pairs_plus, jnp.array([[0, 1], [2, 3], [4, 5]], dtype=jnp.int32))

    state_minus = make_state("simple_nonreversible", direction=-1)
    pairs_minus, parity_minus = kernel._select_swap_pairs(state_minus)
    assert int(parity_minus) == 1
    assert jnp.array_equal(pairs_minus, jnp.array([[1, 2], [3, 4]], dtype=jnp.int32))


def test_select_swap_pairs_persistent_sweep_uses_state_parity():
    kernel = make_kernel("persistent_sweep")

    state0 = make_state("persistent_sweep", direction=+1, sweep_parity=jnp.array(0, dtype=jnp.int32))
    pairs0, parity0 = kernel._select_swap_pairs(state0)
    assert int(parity0) == 0
    assert jnp.array_equal(pairs0, jnp.array([[0, 1], [2, 3], [4, 5]], dtype=jnp.int32))

    state1 = make_state("persistent_sweep", direction=+1, sweep_parity=jnp.array(1, dtype=jnp.int32))
    pairs1, parity1 = kernel._select_swap_pairs(state1)
    assert int(parity1) == 1
    assert jnp.array_equal(pairs1, jnp.array([[1, 2], [3, 4]], dtype=jnp.int32))


def test_compute_blocked_reversible():
    kernel = make_kernel("reversible")
    state = make_state("reversible")
    pairs = jnp.array([[0, 1], [2, 3], [4, 5]], dtype=jnp.int32)

    blocked_false = kernel._compute_blocked(state, pairs, jnp.array([True, False, False]), jnp.array(0))
    blocked_true = kernel._compute_blocked(state, pairs, jnp.array([False, False, False]), jnp.array(0))

    assert bool(blocked_false) is False
    assert bool(blocked_true) is True


def test_compute_blocked_simple_nonreversible():
    kernel = make_kernel("simple_nonreversible")
    state = make_state("simple_nonreversible")
    pairs = jnp.array([[0, 1], [2, 3], [4, 5]], dtype=jnp.int32)

    blocked_false = kernel._compute_blocked(state, pairs, jnp.array([False, True, False]), jnp.array(0))
    blocked_true = kernel._compute_blocked(state, pairs, jnp.array([False, False, False]), jnp.array(0))

    assert bool(blocked_false) is False
    assert bool(blocked_true) is True


def test_compute_blocked_persistent_sweep_uses_frontier_plus_direction():
    kernel = make_kernel("persistent_sweep")
    pairs = jnp.array([[0, 1], [2, 3], [4, 5]], dtype=jnp.int32)

    state_plus = make_state("persistent_sweep", direction=+1, sweep_parity=jnp.array(0, dtype=jnp.int32))
    # frontier is last pair when direction > 0
    blocked_false = kernel._compute_blocked(state_plus, pairs, jnp.array([False, False, True]), jnp.array(0))
    blocked_true = kernel._compute_blocked(state_plus, pairs, jnp.array([True, True, False]), jnp.array(0))
    assert bool(blocked_false) is False
    assert bool(blocked_true) is True

    state_minus = make_state("persistent_sweep", direction=-1, sweep_parity=jnp.array(1, dtype=jnp.int32))
    # frontier is first pair when direction < 0
    blocked_false = kernel._compute_blocked(state_minus, pairs, jnp.array([True, False, False]), jnp.array(1))
    blocked_true = kernel._compute_blocked(state_minus, pairs, jnp.array([False, True, True]), jnp.array(1))
    assert bool(blocked_false) is False
    assert bool(blocked_true) is True


def test_update_lift_state_reversible_never_flips():
    kernel = make_kernel("reversible")
    state = make_state("reversible", direction=+1)

    new_direction, new_sweep_parity, flipped = kernel._update_lift_state(
        state=state,
        blocked=jnp.array(True),
        schedule_parity=jnp.array(0),
    )

    assert int(new_direction) == 1
    assert new_sweep_parity is None
    assert bool(flipped) is False


def test_update_lift_state_simple_nonreversible_flip_on_blocked():
    kernel = make_kernel("simple_nonreversible")
    state = make_state("simple_nonreversible", direction=+1)

    new_direction, new_sweep_parity, flipped = kernel._update_lift_state(
        state=state,
        blocked=jnp.array(True),
        schedule_parity=jnp.array(0),
    )

    assert int(new_direction) == -1
    assert new_sweep_parity is None
    assert bool(flipped) is True


def test_update_lift_state_simple_nonreversible_keep_direction_when_unblocked():
    kernel = make_kernel("simple_nonreversible")
    state = make_state("simple_nonreversible", direction=+1)

    new_direction, new_sweep_parity, flipped = kernel._update_lift_state(
        state=state,
        blocked=jnp.array(False),
        schedule_parity=jnp.array(0),
    )

    assert int(new_direction) == 1
    assert new_sweep_parity is None
    assert bool(flipped) is False


def test_update_lift_state_persistent_sweep_toggle_when_unblocked():
    kernel = make_kernel("persistent_sweep")
    state = make_state(
        "persistent_sweep",
        direction=+1,
        sweep_parity=jnp.array(0, dtype=jnp.int32),
    )

    new_direction, new_sweep_parity, flipped = kernel._update_lift_state(
        state=state,
        blocked=jnp.array(False),
        schedule_parity=jnp.array(0, dtype=jnp.int32),
    )

    assert int(new_direction) == 1
    assert int(new_sweep_parity) == 1
    assert bool(flipped) is False


def test_update_lift_state_persistent_sweep_flip_and_restart_parity():
    kernel = make_kernel("persistent_sweep")
    state = make_state(
        "persistent_sweep",
        direction=+1,
        sweep_parity=jnp.array(0, dtype=jnp.int32),
    )

    new_direction, new_sweep_parity, flipped = kernel._update_lift_state(
        state=state,
        blocked=jnp.array(True),
        schedule_parity=jnp.array(0, dtype=jnp.int32),
    )

    assert int(new_direction) == -1
    assert int(new_sweep_parity) == 1  # restart parity for direction -1
    assert bool(flipped) is True