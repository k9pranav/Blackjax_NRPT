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


def run_ladder(betas, n_steps=1000, seed=0):
    config = NRPTConfig(
        n_replicas=len(betas),
        betas=jnp.asarray(betas, dtype=jnp.float32),
        initial_direction=+1,
        lift_mode="reversible",
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
    init_positions = jnp.repeat(jnp.array([[-6.0]], dtype=jnp.float32), repeats=len(betas), axis=0)
    state = kernel.init(init_positions)

    accept_rates = []
    blocked = 0

    for _ in range(n_steps):
        key, subkey = jax.random.split(key)
        state, info = kernel.step(subkey, state)

        if info.swap_info.accepted.shape[0] > 0:
            accept_rates.append(float(jnp.mean(info.swap_info.accepted.astype(jnp.float32))))
        else:
            accept_rates.append(0.0)

        blocked += int(bool(info.swap_info.blocked))

    return {
        "betas": list(map(float, betas)),
        "mean_swap_accept": float(jnp.mean(jnp.array(accept_rates))),
        "blocked_steps": blocked,
    }


def main():
    ladders = {
        "dense_6": [1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
        "medium_4": [1.0, 0.5, 0.2, 0.0],
        "coarse_3": [1.0, 0.3, 0.0],
    }

    print("\nLadder stress tests\n")
    for name, betas in ladders.items():
        summary = run_ladder(betas)
        print(name, summary)


if __name__ == "__main__":
    main()