from __future__ import annotations

import jax
import jax.numpy as jnp

from NRPT_Blackjax import NRPTKernel, NRPTConfig, GaussianRMHLocalKernel


def target_logdensity(x):
    z = x[0]
    logp1 = -0.5 * ((z + 6.0) ** 2)
    logp2 = -0.5 * ((z - 6.0) ** 2)
    return jnp.logaddexp(logp1, logp2) - jnp.log(2.0)


def reference_logdensity(x):
    z = x[0]
    return -0.5 * z**2 / 36.0


def summarize(xs, accept_rates, direction_history, blocked_history):
    xs = xs[:, 0]
    return {
        "mean": float(jnp.mean(xs)),
        "std": float(jnp.std(xs)),
        "frac_right_mode": float(jnp.mean(xs > 0)),
        "mode_switches": int(jnp.sum((xs[1:] > 0) != (xs[:-1] > 0))),
        "mean_swap_accept": float(jnp.mean(jnp.array(accept_rates))),
        "direction_values": sorted(set(direction_history)),
        "n_blocked": int(sum(blocked_history)),
    }


def run_mode(lift_mode: str, n_steps=2000, seed=0):
    config = NRPTConfig(
        n_replicas=6,
        betas=jnp.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0], dtype=jnp.float32),
        initial_direction=+1,
        lift_mode=lift_mode,
        swap_scheme="adjacent",
        adapt_ladder=False,
    )

    kernel = NRPTKernel(
        local_kernel=GaussianRMHLocalKernel(sigma=jnp.array([0.8], dtype=jnp.float32)),
        target_logdensity_fn=target_logdensity,
        reference_logdensity_fn=reference_logdensity,
        config=config,
    )

    key = jax.random.key(seed)
    init_positions = jnp.array([[-6.0], [-6.0], [-6.0], [-6.0], [-6.0], [-6.0]], dtype=jnp.float32)
    state = kernel.init(init_positions)

    cold_samples = []
    accept_rates = []
    direction_history = [int(state.direction)]
    blocked_history = []

    for _ in range(n_steps):
        key, subkey = jax.random.split(key)
        state, info = kernel.step(subkey, state)
        cold_samples.append(state.replica_states.position[0])

        if info.swap_info.accepted.shape[0] > 0:
            accept_rates.append(float(jnp.mean(info.swap_info.accepted.astype(jnp.float32))))
        else:
            accept_rates.append(0.0)

        direction_history.append(int(state.direction))
        blocked_history.append(bool(info.swap_info.blocked))

    return summarize(jnp.stack(cold_samples), accept_rates, direction_history, blocked_history)


def main():
    print("\nComparing lift modes\n")
    for lift_mode in ("reversible", "simple_nonreversible", "persistent_sweep"):
        summary = run_mode(lift_mode)
        print(lift_mode, summary)


if __name__ == "__main__":
    main()