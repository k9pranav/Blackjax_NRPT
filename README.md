# NRPT for BlackJAX

A **Non-Reversible Parallel Tempering (NRPT)** addon for **BlackJAX** implemented as a pure orchestration layer around existing BlackJAX kernels.

This project was inspired by **BlackJAX open issue #740**, which raises the broader need for Parallel Tempering / tempered MCMC support in a way that fits naturally with the existing BlackJAX ecosystem.

The core idea here is:

- **do not modify BlackJAX internals**
- let the user choose **one local sampler family**
- run that same sampler across all replicas under different **tempered targets**
- add **replica swaps**, **lifted non-reversible scheduling**, and **PT diagnostics** on top

This project is intentionally being developed in **addon style**. Even though not all of the technical challenges have been fully solved yet, the goal is to address **as many of them as possible** while preserving compatibility with the current BlackJAX implementation, rather than rewriting or altering existing samplers and infrastructure.

This project currently provides a working prototype with:

- homogeneous vmapped local updates
- adjacent replica swaps
- three swap scheduling modes:
  - `reversible`
  - `simple_nonreversible`
  - `persistent_sweep`
- structured PT diagnostics
- tested support for:
  - Gaussian RMH
  - MALA
  - HMC

Current experiments already show that the PT layer dramatically improves exploration on a hard multimodal target compared with corresponding single-chain samplers.

---

## Motivation

Parallel Tempering is useful when ordinary local MCMC samplers get trapped in one region of the target distribution, especially on **multimodal** energy landscapes.

This project was built around a specific design question:

> Can we add Non-Reversible Parallel Tempering to BlackJAX **without rewriting its samplers or changing core internals**?

The aim is not necessarily to solve every systems and algorithmic challenge immediately, but to push that design as far as possible in a compositional way. In other words, the project tries to capture most of the practical benefits of PT and NRPT while keeping BlackJAX itself unchanged and treating existing kernels as reusable local samplers.

Instead of changing BlackJAX kernels, this project wraps them with:

- tempered log densities
- stacked replica state management
- swap proposals and accept/reject logic
- lifted swap scheduling
- replica transport bookkeeping
- future hooks for ladder adaptation and sharding

---

## Design philosophy

This project follows four main principles.

### 1. BlackJAX remains untouched
All local samplers are reused through their existing `init` and `step` interfaces.

### 2. Homogeneous sampler-family agnostic design
The user may choose the local sampler family, for example:

- Gaussian random walk Metropolis
- MALA
- HMC

but **all replicas use the same family**.

This is intentional. It keeps the system compatible with:

- `jax.vmap`
- uniform state structure across replicas
- future sharding across devices

This project does **not** attempt to support different sampler families at different temperature levels.

### 3. Tempering is applied by modifying the target, not the sampler
Each replica runs the same base sampler on a tempered log density:

\[
\log \pi_\beta(x) = \beta \log p_{\text{target}}(x) + (1-\beta)\log p_{\text{reference}}(x)
\]

so the sampler itself does not need to know anything about PT.

### 4. PT logic lives in an orchestration layer
NRPT manages:

- replica-local updates
- swap pair selection
- lifted state (`direction`, `sweep_parity`)
- replica identity tracking
- PT diagnostics
- future ladder adaptation and sharding hooks

---

## Features implemented

### Local kernel abstraction
A `LocalKernel` protocol provides a common interface:

- `init(position, logdensity_fn)`
- `step(rng_key, state, logdensity_fn)`

Concrete wrappers are implemented for:

- `RMHLocalKernel`
- `GaussianRMHLocalKernel`
- `MALALocalKernel`
- `HMCLocalKernel`

### Tempered targets
A helper constructs tempered log densities from:

- a target log density
- a reference log density
- an inverse temperature `beta`

### Top-level PT state
`TemperedState` stores:

- stacked replica states
- beta ladder
- lifted direction
- replica order
- optional adaptation state
- iteration count
- optional sweep parity

### PT diagnostics
`TemperedInfo` stores:

- `local_info` from the base BlackJAX kernel
- `swap_info` for PT-specific diagnostics

`SwapInfo` includes:

- proposed swap pairs
- log acceptance ratios
- acceptance probabilities
- accepted indicators
- direction used this step
- whether direction flipped
- lift mode
- schedule parity
- blockage flag

### Vmappable local updates
Replica-local updates are applied homogeneously across replicas using `jax.vmap`.

### Adjacent swaps
Adjacent temperature swaps are implemented with proper PT acceptance logic and consistent state exchange across stacked PyTrees.

### Lift modes
Three scheduling modes are supported.

#### `reversible`
Standard alternating even/odd adjacent swaps.

#### `simple_nonreversible`
Parity is determined by a lifted direction variable `+1 / -1`. Direction flips when the current step is blocked under the chosen blockage rule.

#### `persistent_sweep`
Maintains both direction and sweep parity. While unblocked, parity toggles so the scheduler sweeps across the ladder. On blockage at the current frontier, direction reverses and the sweep restarts accordingly.

---

## What is not yet implemented

Two major planned extensions are still pending.

### 1. Robbins-Monro ladder adaptation
There is placeholder bookkeeping for adaptation state, but the actual Robbins-Monro temperature update is not yet implemented.

What remains to be added:

- per-edge swap acceptance tracking during warmup
- Robbins-Monro ladder updates
- ladder freezing after warmup

### 2. Device-mesh sharding
The code is currently local and `vmap`-based. Replica sharding across devices is not yet implemented.

Planned future extension:

- distribute replicas across accelerators
- preserve local updates in parallel
- handle nearest-neighbor swap communication across shards

### Other current limitations

- adjacent swaps only
- current swap stage assumes sampler states expose `.position`
- persistent-sweep blockage semantics are implemented but may still be refined
- current experiments are informative but not yet fully compute-budget matched

---

## Current status

This project is already beyond the “initial prototype that merely runs” stage.

At this point it has:

- passing **unit tests**
- passing **integration tests**
- working **behavioral tests**
- experiment scripts comparing:
  - PT vs non-PT
  - reversible vs non-reversible lift modes
  - dense vs coarse ladders

So the implementation is best described as:

> a working and useful PT prototype with solid mechanics, meaningful diagnostics, and clear experimental signal, with ladder adaptation and sharding still left to implement

---

## Experimental results so far

## 1. PT vs single-chain sampling

A hard symmetric bimodal target was used to compare:

- single-chain RMH
- single-chain MALA
- single-chain HMC

against:

- PT-RMH
- PT-MALA
- PT-HMC

### What happened

The single-chain samplers all got stuck in one mode:

- no mode switches
- all samples remained in the left well
- mean stayed near the initial mode

In contrast, all PT variants crossed modes repeatedly and explored both wells.

### Interpretation

This is the main success result of the project:

> on a hard multimodal target, the PT layer enables barrier crossing where ordinary single-chain local samplers fail

Among the tested PT variants, PT-HMC gave the most balanced occupancy in that run, but all PT versions were dramatically better than their non-PT baselines.

---

## 2. Reversible vs non-reversible lift modes

The three implemented lift modes were compared on the same target.

### Reversible
- strong mode switching
- but still somewhat biased toward the initial side in the tested run

### Simple non-reversible
- fewer mode switches in one run
- but more balanced occupancy across the two modes

### Persistent sweep
- highest switching activity
- many blockage-triggered reversals
- clearly active lifted state dynamics

### Interpretation

The lift mode matters.

The results show that:

- reversible PT works well
- simple non-reversible can produce more balanced occupation
- persistent sweep creates the most visibly non-reversible transport dynamics

So the non-reversible scheduling is not cosmetic. It changes the sampler’s behavior in meaningful ways.

---

## 3. Ladder stress tests

Three ladders were compared:

- dense ladder
- medium ladder
- coarse ladder

### Observed pattern

As the ladder became coarser:

- mean swap acceptance decreased
- blocked steps increased sharply

### Interpretation

This matches the expected PT tradeoff:

- denser ladders are smoother and easier to swap across
- coarser ladders use fewer replicas but transport becomes much worse

This also strongly motivates the remaining Robbins-Monro ladder adaptation work.

---

## Is it effective?

Yes — based on current tests and experiments, the implementation is already effective in the most important practical sense:

- it composes cleanly with BlackJAX samplers
- it runs end-to-end without changing BlackJAX internals
- it significantly improves multimodal exploration relative to non-PT local samplers
- it produces meaningful differences between reversible and non-reversible swap scheduling

So while the project is not feature-complete yet, it is already a **working and useful PT library prototype**.

---

## Installation

This project currently assumes a Python environment with JAX and BlackJAX installed.

### Recommended
Use a Python 3.10+ virtual environment.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install jax jaxlib blackjax pytest