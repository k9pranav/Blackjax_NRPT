from __future__ import annotations

import jax
import jax.numpy as jnp
import blackjax

from NRPT_Blackjax import NRPTKernel, NRPTConfig, GaussianRMHLocalKernel, MALALocalKernel, HMCLocalKernel


def target_logdensity(x):
    z = x[0]
    logp1 = -0.5 * ((z + 6.0) ** 2)
    logp2 = -0.5 * ((z - 6.0) ** 2)
    return jnp.logaddexp(logp1, logp2) - jnp.log(2.0)


def reference_logdensity(x):
    z = x[0]
    return -0.5 * z**2 / 36.0


def mode_indicator(x):
    return (x[0] > 0).astype(jnp.float32)


def summarize_chain(samples):
    xs = samples[:, 0]
    mode_jumps = int(jnp.sum((xs[1:] > 0) != (xs[:-1] > 0)))
    mean_x = float(jnp.mean(xs))
    std_x = float(jnp.std(xs))
    frac_right = float(jnp.mean(xs > 0))
    return {
        "mean": mean_x,
        "std": std_x,
        "frac_right_mode": frac_right,
        "mode_switches": mode_jumps,
    }


def run_single_chain_rmh(n_steps=2000, seed=0):
    key = jax.random.key(seed)
    init_x = jnp.array([-6.0], dtype=jnp.float32)

    alg = blackjax.mcmc.random_walk.normal_random_walk(target_logdensity, jnp.array([0.8], dtype=jnp.float32))
    state = alg.init(init_x)

    samples = []
    for _ in range(n_steps):
        key, subkey = jax.random.split(key)
        state, _ = alg.step(subkey, state)
        samples.append(state.position)

    return jnp.stack(samples)


def run_single_chain_mala(n_steps=2000, seed=0):
    key = jax.random.key(seed)
    init_x = jnp.array([-6.0], dtype=jnp.float32)

    alg = blackjax.mala(target_logdensity, 0.15)
    state = alg.init(init_x)

    samples = []
    for _ in range(n_steps):
        key, subkey = jax.random.split(key)
        state, _ = alg.step(subkey, state)
        samples.append(state.position)

    return jnp.stack(samples)


def run_single_chain_hmc(n_steps=2000, seed=0):
    key = jax.random.key(seed)
    init_x = jnp.array([-6.0], dtype=jnp.float32)

    alg = blackjax.hmc(target_logdensity, 0.10, jnp.array([1.0], dtype=jnp.float32), 5)
    state = alg.init(init_x)

    samples = []
    for _ in range(n_steps):
        key, subkey = jax.random.split(key)
        state, _ = alg.step(subkey, state)
        samples.append(state.position)

    return jnp.stack(samples)


def run_pt(local_kernel, n_steps=2000, seed=0):
    config = NRPTConfig(
        n_replicas=6,
        betas=jnp.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0], dtype=jnp.float32),
        initial_direction=+1,
        lift_mode="reversible",
        swap_scheme="adjacent",
        adapt_ladder=False,
    )

    kernel = NRPTKernel(
        local_kernel=local_kernel,
        target_logdensity_fn=target_logdensity,
        reference_logdensity_fn=reference_logdensity,
        config=config,
    )

    key = jax.random.key(seed)
    init_positions = jnp.array([[-6.0], [-6.0], [-6.0], [-6.0], [-6.0], [-6.0]], dtype=jnp.float32)
    state = kernel.init(init_positions)

    cold_samples = []
    for _ in range(n_steps):
        key, subkey = jax.random.split(key)
        state, _ = kernel.step(subkey, state)
        cold_samples.append(state.replica_states.position[0])

    return jnp.stack(cold_samples)


def main():
    runs = {
        "single_rmh": summarize_chain(run_single_chain_rmh()),
        "single_mala": summarize_chain(run_single_chain_mala()),
        "single_hmc": summarize_chain(run_single_chain_hmc()),
        "pt_rmh": summarize_chain(run_pt(GaussianRMHLocalKernel(sigma=jnp.array([0.8], dtype=jnp.float32)))),
        "pt_mala": summarize_chain(run_pt(MALALocalKernel(step_size=0.15))),
        "pt_hmc": summarize_chain(run_pt(HMCLocalKernel(step_size=0.10, inverse_mass_matrix=jnp.array([1.0], dtype=jnp.float32), num_integration_steps=5))),
    }

    print("\nPT vs single-chain comparison\n")
    for name, summary in runs.items():
        print(name, summary)


if __name__ == "__main__":
    main()