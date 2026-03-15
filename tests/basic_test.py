from __future__ import annotations

import jax
import jax.numpy as jnp

# CHANGE THIS IMPORT

from NRPT_Blackjax import NRPTKernel, NRPTConfig, GaussianRMHLocalKernel


# -----------------------------
# Toy target/reference densities
# -----------------------------
def target_logdensity(x):
    """
    1D bimodal target:
        0.5 N(-4, 1) + 0.5 N(+4, 1)
    x expected shape: (1,)
    """
    z = x[0]
    logp1 = -0.5 * ((z + 4.0) ** 2) - 0.5 * jnp.log(2.0 * jnp.pi)
    logp2 = -0.5 * ((z - 4.0) ** 2) - 0.5 * jnp.log(2.0 * jnp.pi)
    return jnp.logaddexp(logp1, logp2) - jnp.log(2.0)


def reference_logdensity(x):
    """
    Broad Gaussian reference:
        N(0, 25)
    """
    z = x[0]
    sigma2 = 25.0
    return -0.5 * z**2 / sigma2 - 0.5 * jnp.log(2.0 * jnp.pi * sigma2)


# -----------------------------
# Helpers
# -----------------------------
def make_kernel(lift_mode: str, n_replicas: int = 6):
    betas = jnp.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0], dtype=jnp.float32)

    config = NRPTConfig(
        n_replicas=n_replicas,
        betas=betas,
        initial_direction=+1,
        lift_mode=lift_mode,
        swap_scheme="adjacent",
        adapt_ladder=False,
    )

    local_kernel = GaussianRMHLocalKernel(
        sigma=jnp.array([0.8], dtype=jnp.float32)
    )

    return NRPTKernel(
        local_kernel=local_kernel,
        target_logdensity_fn=target_logdensity,
        reference_logdensity_fn=reference_logdensity,
        config=config,
    )


def initial_positions(n_replicas: int = 6):
    return jnp.array(
        [[-4.00], [-3.80], [-4.20], [-3.90], [-4.10], [-4.05]],
        dtype=jnp.float32,
    )


def assert_basic_state_invariants(state, n_replicas: int):
    assert state.betas.shape == (n_replicas,)
    assert state.replica_order is None or state.replica_order.shape == (n_replicas,)
    assert state.direction.shape == ()
    assert int(state.direction) in (-1, 1)

    # assumes BlackJAX state has .position
    assert hasattr(state.replica_states, "position")
    pos = state.replica_states.position
    assert pos.shape[0] == n_replicas
    assert jnp.all(jnp.isfinite(pos))


def assert_basic_swapinfo_invariants(info):
    swap_info = info.swap_info

    assert swap_info.proposed_pairs.ndim == 2
    assert swap_info.proposed_pairs.shape[1] == 2

    n_pairs = swap_info.proposed_pairs.shape[0]
    assert swap_info.log_accept_ratio.shape == (n_pairs,)
    assert swap_info.acceptance_prob.shape == (n_pairs,)
    assert swap_info.accepted.shape == (n_pairs,)

    assert jnp.all(jnp.isfinite(swap_info.log_accept_ratio))
    assert jnp.all(jnp.isfinite(swap_info.acceptance_prob))
    assert jnp.all(swap_info.acceptance_prob >= 0.0)
    assert jnp.all(swap_info.acceptance_prob <= 1.0)

    assert swap_info.direction.shape == ()
    assert int(swap_info.direction) in (-1, 1)

    assert isinstance(swap_info.lift_mode, str)
    assert swap_info.lift_mode in ("reversible", "simple_nonreversible", "persistent_sweep")

    assert swap_info.blocked.shape == ()
    assert swap_info.blocked.dtype == jnp.bool_


def print_step_summary(step_idx, state, info):
    print(f"\n=== step {step_idx} ===")
    print("iteration          :", state.iteration)
    print("direction          :", int(state.direction))
    print("sweep_parity       :", None if state.sweep_parity is None else int(state.sweep_parity))
    print("replica_order      :", state.replica_order)
    print("positions          :", state.replica_states.position[:, 0])

    s = info.swap_info
    print("lift_mode          :", s.lift_mode)
    print("schedule_parity    :", None if s.sweep_parity is None else int(s.sweep_parity))
    print("proposed_pairs     :", s.proposed_pairs)
    print("log_accept_ratio   :", s.log_accept_ratio)
    print("acceptance_prob    :", s.acceptance_prob)
    print("accepted           :", s.accepted)
    print("blocked            :", bool(s.blocked))
    print("flipped_direction  :", bool(s.flipped_direction))


# -----------------------------
# Tests
# -----------------------------
def test_init_reversible():
    kernel = make_kernel("reversible")
    state = kernel.init(initial_positions())

    assert_basic_state_invariants(state, 6)
    assert state.iteration == 0
    assert int(state.direction) == 1
    assert state.sweep_parity is None
    assert jnp.all(state.replica_order == jnp.arange(6))

    print("test_init_reversible passed")


def test_init_simple_nonreversible():
    kernel = make_kernel("simple_nonreversible")
    state = kernel.init(initial_positions())

    assert_basic_state_invariants(state, 6)
    assert state.iteration == 0
    assert int(state.direction) == 1
    assert state.sweep_parity is None

    print("test_init_simple_nonreversible passed")


def test_init_persistent_sweep():
    kernel = make_kernel("persistent_sweep")
    state = kernel.init(initial_positions())

    assert_basic_state_invariants(state, 6)
    assert state.iteration == 0
    assert int(state.direction) == 1
    assert state.sweep_parity is not None
    assert int(state.sweep_parity) == 0  # because initial_direction = +1

    print("test_init_persistent_sweep passed")


def test_single_step(lift_mode: str):
    kernel = make_kernel(lift_mode)
    state = kernel.init(initial_positions())

    key = jax.random.key(0)
    state2, info = kernel.step(key, state)

    assert_basic_state_invariants(state2, 6)
    assert_basic_swapinfo_invariants(info)

    assert state2.iteration == 1

    # Pair schedule checks by mode
    if lift_mode == "reversible":
        # iteration starts at 0, so step uses parity 0
        assert info.swap_info.sweep_parity is not None
        assert int(info.swap_info.sweep_parity) == 0

    if lift_mode == "simple_nonreversible":
        assert info.swap_info.sweep_parity is not None
        assert int(info.swap_info.sweep_parity) == 0  # direction initially +1

    if lift_mode == "persistent_sweep":
        assert info.swap_info.sweep_parity is not None
        assert int(info.swap_info.sweep_parity) == 0
        assert state2.sweep_parity is not None

    print_step_summary(1, state2, info)
    print(f"test_single_step({lift_mode}) passed")


def test_multiple_steps(lift_mode: str, n_steps: int = 20):
    kernel = make_kernel(lift_mode)
    state = kernel.init(initial_positions())

    key = jax.random.key(123)

    direction_history = [int(state.direction)]
    sweep_history = [None if state.sweep_parity is None else int(state.sweep_parity)]
    blocked_history = []
    accept_rate_history = []

    for t in range(n_steps):
        key, subkey = jax.random.split(key)
        state, info = kernel.step(subkey, state)

        assert_basic_state_invariants(state, 6)
        assert_basic_swapinfo_invariants(info)
        assert state.iteration == t + 1

        direction_history.append(int(state.direction))
        sweep_history.append(None if state.sweep_parity is None else int(state.sweep_parity))
        blocked_history.append(bool(info.swap_info.blocked))

        if info.swap_info.accepted.shape[0] > 0:
            accept_rate_history.append(float(jnp.mean(info.swap_info.accepted.astype(jnp.float32))))
        else:
            accept_rate_history.append(0.0)

        if t < 4 or t == n_steps - 1:
            print_step_summary(t + 1, state, info)

    print(f"\n=== aggregate: {lift_mode} ===")
    print("direction history  :", direction_history)
    print("sweep history      :", sweep_history)
    print("blocked history    :", blocked_history)
    print("mean accept rate   :", sum(accept_rate_history) / len(accept_rate_history))
    print("final positions    :", state.replica_states.position[:, 0])

    # Mode-specific checks
    if lift_mode == "reversible":
        # state.sweep_parity should remain None in your design
        assert state.sweep_parity is None

    if lift_mode == "simple_nonreversible":
        # still no persistent sweep parity in state
        assert state.sweep_parity is None

    if lift_mode == "persistent_sweep":
        assert state.sweep_parity is not None

    print(f"test_multiple_steps({lift_mode}) passed")


def test_pair_selector():
    kernel = make_kernel("reversible")

    pairs_even = kernel._proposed_adjacent_pairs(6, jnp.array(0, dtype=jnp.int32))
    pairs_odd = kernel._proposed_adjacent_pairs(6, jnp.array(1, dtype=jnp.int32))

    assert jnp.array_equal(pairs_even, jnp.array([[0, 1], [2, 3], [4, 5]], dtype=jnp.int32))
    assert jnp.array_equal(pairs_odd, jnp.array([[1, 2], [3, 4]], dtype=jnp.int32))

    print("test_pair_selector passed")


def test_swap_array_pairs():
    kernel = make_kernel("reversible")

    x = jnp.array([[10.0], [20.0], [30.0], [40.0], [50.0], [60.0]])
    left_idx = jnp.array([0, 2, 4], dtype=jnp.int32)
    right_idx = jnp.array([1, 3, 5], dtype=jnp.int32)
    accepted = jnp.array([True, False, True])

    y = kernel._swap_array_pairs(x, left_idx, right_idx, accepted)

    expected = jnp.array([[20.0], [10.0], [30.0], [40.0], [60.0], [50.0]])
    assert jnp.array_equal(y, expected)

    print("test_swap_array_pairs passed")


def run_all_tests():
    print("\nRunning tests...\n")

    test_pair_selector()
    test_swap_array_pairs()

    test_init_reversible()
    test_init_simple_nonreversible()
    test_init_persistent_sweep()

    test_single_step("reversible")
    test_single_step("simple_nonreversible")
    test_single_step("persistent_sweep")

    test_multiple_steps("reversible", n_steps=20)
    test_multiple_steps("simple_nonreversible", n_steps=20)
    test_multiple_steps("persistent_sweep", n_steps=20)

    print("\nAll tests passed.")


if __name__ == "__main__":
    run_all_tests()