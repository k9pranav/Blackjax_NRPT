from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, NamedTuple, Optional, Protocol, TypeVar

import jax
import jax.numpy as jnp
import blackjax

Array = jax.Array
PyTree = Any #Ensuring Any is PyTree JAX compatible

StateT = TypeVar("StateT") #Abstraction of State for different samplers
InfoT = TypeVar("InfoT") #Abstraction of Info for different samplers

class LocalKernel(Protocol[StateT, InfoT]):
    def init(self, position:PyTree, logdensity_fn:Callable[[PyTree], Array]) -> StateT: ...
    
    def step(self, rng_key: Array, state: StateT, logdensity_fn: Callable[[PyTree], Array]) -> tuple[StateT, InfoT]: ...

#Implementations of concrete samplers 

# rmh sampler
@dataclass(frozen=True)
class RMHLocalKernel:
    proposal_generator: Callable[[Array, PyTree], PyTree]
    proposal_logdensity_fn: Callable[[PyTree], Array] | None = None

    def init(
        self,
        position: PyTree,
        logdensity_fn: Callable[[PyTree], Array],
    ):
        alg = blackjax.rmh(
            logdensity_fn,
            self.proposal_generator,
            self.proposal_logdensity_fn,
        )
        return alg.init(position)

    def step(
        self,
        rng_key: Array,
        state,
        logdensity_fn: Callable[[PyTree], Array],
    ):
        alg = blackjax.rmh(
            logdensity_fn,
            self.proposal_generator,
            self.proposal_logdensity_fn,
        )
        return alg.step(rng_key, state)


# Gaussian random walk RMH
@dataclass(frozen=True)
class GaussianRMHLocalKernel:
    sigma: Array

    def init(
        self,
        position: PyTree,
        logdensity_fn: Callable[[PyTree], Array],
    ):
        alg = blackjax.mcmc.random_walk.normal_random_walk(logdensity_fn, self.sigma)
        return alg.init(position)

    def step(
        self,
        rng_key: Array,
        state,
        logdensity_fn: Callable[[PyTree], Array],
    ):
        alg = blackjax.mcmc.random_walk.normal_random_walk(logdensity_fn, self.sigma)
        return alg.step(rng_key, state)
    
#MALA
@dataclass(frozen=True)
class MALALocalKernel:
    step_size: float

    def init(
        self,
        position: PyTree,
        logdensity_fn: Callable[[PyTree], Array],
    ):
        alg = blackjax.mala(logdensity_fn, self.step_size)
        return alg.init(position)

    def step(
        self,
        rng_key: Array,
        state,
        logdensity_fn: Callable[[PyTree], Array],
    ):
        alg = blackjax.mala(logdensity_fn, self.step_size)
        return alg.step(rng_key, state)


#HMC
@dataclass(frozen=True)
class HMCLocalKernel:
    step_size: float
    inverse_mass_matrix: Array
    num_integration_steps: int

    def init(
        self,
        position: PyTree,
        logdensity_fn: Callable[[PyTree], Array],
    ):
        alg = blackjax.hmc(
            logdensity_fn,
            self.step_size,
            self.inverse_mass_matrix,
            self.num_integration_steps, 
        )
        return alg.init(position)

    def step(
        self,
        rng_key: Array,
        state,
        logdensity_fn: Callable[[PyTree], Array],
    ):
        alg = blackjax.hmc(
            logdensity_fn,
            self.step_size,
            self.inverse_mass_matrix,
            self.num_integration_steps,
        )
        return alg.step(rng_key, state)


#Tempered Helpers
def make_tempered_logdensity(target_logdensity_fn: Callable[[PyTree], Array], reference_logdensity_fn: Callable[[PyTree], Array], beta: Array) -> Callable[[PyTree], Array]:

    def logdensity_fn(x: PyTree) -> Array:
        return beta * target_logdensity_fn(x) + (1.0 - beta)*reference_logdensity_fn(x)
    
    return logdensity_fn

#Swap diagnostics, record keeping of every swap proposed
class SwapInfo(NamedTuple):
    proposed_pairs: Array #2D array -> [n_pairs, 2]. Contains indices of replicas that tried to swap
    log_accept_ratio: Array #Log-accepted ratio
    acceptance_prob: Array #Actual probability
    accepted: Array 
    direction: Array #+1 or -1 representing the current flow of the scheme
    flipped_direction: Array #boolean flag to indicate if the direction was reversed during the step
    lift_mode: str #Lift statergy selected by the user
    sweep_parity: Array | None
    blocked: Array

#Transition Diagnostics
class TemperedInfo(NamedTuple, Generic[InfoT]):
    local_info: InfoT #Diagnostic data from the underlying sampler
    swap_info: SwapInfo #Instance of swapInfo

#Warmup adatptaion State; used for Robbins-Monro adaptation
@dataclass
class LadderAdaptationState:
    enabled: bool
    step: int #Number of adaptation steps taken
    target_accept: float #goal swap rate
    step_size: float #How aggresively the ladder should change when it misses the target
    running_mean_accept: Optional[Array] = None #Average acceptance over time

#NRPT Runtime State -> Everything contained to resume the simulation
@dataclass
class TemperedState(Generic[StateT]): 
    replica_states: StateT #Current internal states of all N samples
    betas: Array #Temp ladder
    direction: Array #Current lifted direction +1 or -1 for non-reversible swaps
    replica_order: Optional[Array] #Tracks which replica is at what state
    adaptation_state: Optional[LadderAdaptationState] = None 
    iteration: int = 0
    sweep_parity: Array| None = None

#User-facing Config
@dataclass(frozen=True)
class NRPTConfig:
    n_replicas: int #Number of Chains
    betas: Array #Betas of those chains
    initial_direction: int = +1

    #swap_scheduling
    # use_nonreversible_swaps: bool = True
    # swap_scheme: str = "adjacent" #For neighbour only swaps
    # schedule: str = "directional" #even_odd/directional

    lift_mode: str = "simple_nonreversible" #Simple non-reversible strat. Options are (i)reversible, (ii)simple_nonreversible, (iii)persistent_sweep

    swap_scheme: str = "adjacent" #For neighbour only swaps. Right now, only focusing on adjacent; might change it later to allow for more robust comparisons

    #ladder adaptation
    adapt_ladder: bool = False #Boolean to enable/disable automatic temperature tuning
    target_swap_accept: float = 0.234
    adaptation_step_size: float = 0.01

#Main orchestration object
@dataclass(frozen=True)
class NRPTKernel(Generic[StateT, InfoT]):
    local_kernel: LocalKernel[StateT, InfoT] #Sampler being used
    target_logdensity_fn: Callable[[PyTree], Array] #Cold Dist (Beta = 1.0)
    reference_logdensity_fn: Callable[[PyTree], Array] #Hot distribution (Beta = 0.0)
    config: NRPTConfig

    #Initialization
    def init(self, initial_positions:PyTree) -> TemperedState[StateT]:
        """
        Initialize all replicas using the SAME sampler, but each has its own tempered log density
        """

        betas = jax.numpy.asarray(self.config.betas)

        #Basic Config Checks
        if betas.ndim != 1:
            raise ValueError("config.betas must be a 1D array.")
        if betas.shape[0] != self.config.n_replicas:
            raise ValueError("len(betas) must equal n_replicas.")
        if self.config.initial_direction not in (-1, +1):
            raise ValueError("initial_direction must be either +1 or -1.")
        if self.config.lift_mode not in ("reversible", "simple_nonreversible", "persistent_sweep"):
            raise ValueError("lift_mode must be one of: 'reversible', 'simple_nonreversible', 'persistent_sweep'.")

        def init_one(position_k:PyTree, beta_k:Array) -> StateT:
            logdensity_k = make_tempered_logdensity(
                self.target_logdensity_fn,
                self.reference_logdensity_fn,
                beta_k
            )

            return self.local_kernel.init(position_k, logdensity_k)
        
        replica_states = jax.vmap(init_one)(initial_positions, betas)

        adaptation_state = None

        if self.config.adapt_ladder:
            adaptation_state = LadderAdaptationState(
                enabled=True,
                step=0,
                target_accept=self.config.target_swap_accept,
                step_size=self.config.adaptation_step_size,
                running_mean_accept=None
            )
        
        initial_direction = jnp.asarray(self.config.initial_direction, dtype=jnp.int32)

        if self.config.lift_mode == "persistent_sweep":
            initial_sweep_parity = jnp.where(initial_direction>0, 0, 1).astype(jnp.int32)
        else:
            initial_sweep_parity = None
        
        return TemperedState(
            replica_states=replica_states,
            betas=betas,
            direction = jax.numpy.asarray(self.config.initial_direction, dtype=jax.numpy.int32),
            replica_order = jax.numpy.arange(self.config.n_replicas),
            iteration=0,
            adaptation_state=adaptation_state,
            sweep_parity=initial_sweep_parity
        )
    
    #One full transition
    def step(self, rng_key:Array, state: TemperedState[StateT]) -> tuple[TemperedState[StateT], TemperedInfo[InfoT]]:
        """
        One NRPT Step:
            1) homogeneous vmapped local updates
            2) adjacent swap proposals
            3) optional lifted direction logic
            4) optional ladder adaptation
        """

        key_local, key_swap = jax.random.split(rng_key, 2)

        #Local Updates
        new_replica_states, local_info = self._local_step(key_local, state)

        intermediate_state = TemperedState(
            replica_states=new_replica_states,
            betas=state.betas,
            direction=state.direction,
            replica_order=state.replica_order,
            adaptation_state=state.adaptation_state,
            iteration=state.iteration,
            sweep_parity=state.sweep_parity
        )

        # Swap Stage
        swapped_state, swap_info = self._swap_step(key_swap, intermediate_state)

        #Optional Laddder Adaptation
        adapted_state = self._maybe_adapt_ladder(swapped_state, swap_info)

        final_state = TemperedState(
            replica_states=adapted_state.replica_states,
            betas=adapted_state.betas,
            direction=adapted_state.direction,
            replica_order=adapted_state.replica_order,
            adaptation_state=adapted_state.adaptation_state,
            iteration=state.iteration + 1,
            sweep_parity=adapted_state.sweep_parity
        )

        info = TemperedInfo(
            local_info=local_info, swap_info=swap_info
        )

        return final_state, info
    
    #Local Step
    def _local_step(self, rng_key: Array, state:TemperedState[StateT]) -> tuple[StateT, InfoT]:
        """Applying the same local sampler family accross all replicas, each with its own tempered log density"""
        n = self.config.n_replicas
        keys = jax.random.split(rng_key, n)

        def step_one(key_k: Array, state_k: StateT, beta_k: Array) -> tuple[StateT, InfoT]:
            logdensity_k = make_tempered_logdensity(
                self.target_logdensity_fn,
                self.reference_logdensity_fn,
                beta_k,
            )
            return self.local_kernel.step(key_k, state_k, logdensity_k)

        return jax.vmap(step_one)(keys, state.replica_states, state.betas)
    
    def _swap_step(self, rng_key:Array, state: TemperedState[StateT]) -> tuple[TemperedState[StateT], SwapInfo]:
        """
        Proposing Neighbour Swap
        Assumes all chains use the same sampler
        if at least one proposed swap is accepted, keep direction, else flip direction
        """

        proposed_pairs, schedule_parity = self._select_swap_pairs(state)

        n_pairs = proposed_pairs.shape[0] #The pairs to check

        
        
        if n_pairs == 0:
            accepted = jnp.zeros((0,), dtype = bool)

            blocked = self._compute_blocked(state=state, proposed_pairs=proposed_pairs, accepted=accepted, schedule_parity=schedule_parity)
            
            new_direction, new_sweep_parity, flipped_direction = self._update_lift_state(
                state=state,
                blocked=blocked,
                schedule_parity=schedule_parity,
            )

            new_state = TemperedState(
                replica_states=state.replica_states,
                betas=state.betas,
                direction=new_direction,
                replica_order=state.replica_order,
                adaptation_state=state.adaptation_state,
                iteration=state.iteration,
                sweep_parity=new_sweep_parity
            )

            swap_info = SwapInfo(
                proposed_pairs=proposed_pairs,
                log_accept_ratio=jnp.zeros((0,)),
                acceptance_prob=jnp.zeros((0,)),
                accepted=jnp.zeros((0,), dtype=bool),
                direction=state.direction,
                flipped_direction=flipped_direction,
                lift_mode=self.config.lift_mode,
                sweep_parity=schedule_parity,
                blocked=blocked
            )

            return new_state, swap_info

        left_idx = proposed_pairs[:, 0] #First set of pairs
        right_idx = proposed_pairs[:, 1] #Second set of pairs

        positions = self._get_positions(state.replica_states) #Getting position on the posterior

        x_left = jax.tree_util.tree_map(lambda x: x[left_idx], positions) #Colder Beta
        x_right = jax.tree_util.tree_map(lambda x: x[right_idx], positions) #Warmer Beta

        beta_left = state.betas[left_idx] #Extracting Beta left side
        beta_right = state.betas[right_idx] #Extracting Right left side

        target_left = jax.vmap(self.target_logdensity_fn)(x_left) #Applying log density left-side
        target_right = jax.vmap(self.target_logdensity_fn)(x_right) #Appling log density right-side

        reference_left = jax.vmap(self.reference_logdensity_fn)(x_left)
        reference_right = jax.vmap(self.reference_logdensity_fn)(x_right)

        #Current total tempered log density of the pair
        current_pair_logprob = (beta_left * target_left
            + (1.0 - beta_left) * reference_left
            + beta_right * target_right
            + (1.0 - beta_right) * reference_right
        )

        # Proposed total tempered log density after swapping positions
        swapped_pair_logprob = (
            beta_left * target_right
            + (1.0 - beta_left) * reference_right
            + beta_right * target_left
            + (1.0 - beta_right) * reference_left
        )

        log_accept_ratio = swapped_pair_logprob - current_pair_logprob
        acceptance_prob = jnp.minimum(1.0, jnp.exp(log_accept_ratio))


        key_accept = rng_key
        log_u = jnp.log(jax.random.uniform(key_accept, shape=(n_pairs,)))
        accepted = log_u < jnp.minimum(log_accept_ratio, 0.0)

        new_replica_states = self._apply_pair_swaps(
            state.replica_states,
            left_idx,
            right_idx,
            accepted,
        )

        new_replica_order = state.replica_order
        if new_replica_order is not None:
            new_replica_order = self._apply_pair_swaps(
                state.replica_order,
                left_idx,
                right_idx,
                accepted,
            )

        blocked = self._compute_blocked(
            state = state, 
            proposed_pairs=proposed_pairs, 
            accepted=accepted,
            schedule_parity=schedule_parity,
        )
        
        #The NR part in NRPT
        new_direction, new_sweep_parity, flipped_direction = self._update_lift_state(
            state=state,
            blocked=blocked,
            schedule_parity=schedule_parity,
        )
        
        new_state = TemperedState(
            replica_states=new_replica_states,
            betas=state.betas,
            direction=new_direction,
            replica_order=new_replica_order,
            adaptation_state=state.adaptation_state,
            iteration=state.iteration,
            sweep_parity=new_sweep_parity
        )

        swap_info = SwapInfo(
            proposed_pairs=proposed_pairs,
            log_accept_ratio=log_accept_ratio,
            acceptance_prob=jnp.minimum(1.0, jnp.exp(log_accept_ratio)),
            accepted=accepted,
            direction=state.direction,
            flipped_direction=flipped_direction,
            lift_mode=self.config.lift_mode,
            sweep_parity=schedule_parity,
            blocked=blocked

        )

        return new_state, swap_info
    
    def _select_swap_pairs(self, state:TemperedState[StateT]) -> tuple[Array, Array|None]:
        """
        Returns: 
            proposed_pairs
            schedule_parity used this step
        """

        mode = self.config.lift_mode

        if mode == "reversible":
            parity = jnp.asarray(state.iteration%2, dtype=jnp.int32)
            return self._proposed_adjacent_pairs(self.config.n_replicas, parity), parity
        
        if mode == "simple_nonreversible":
            parity = jnp.where(state.direction > 0, 0, 1).astype(jnp.int32)
            return self._proposed_adjacent_pairs(self.config.n_replicas, parity), parity
        
        if mode == "persistent_sweep":
            if state.sweep_parity is None:
                raise ValueError("persistent_sweep requires sweep_parity in TemperedState.")
            parity = state.sweep_parity.astype(jnp.int32)
            return self._proposed_adjacent_pairs(self.config.n_replicas, parity), parity
        
        raise ValueError(f"Unkown lift_mode: {mode}")
    
    def _compute_blocked(self, state: TemperedState[StateT], proposed_pairs: Array, accepted: Array, schedule_parity:Array|None):
        mode = self.config.lift_mode

        n_pairs = proposed_pairs.shape[0]
        if n_pairs == 0:
            return jnp.asarray(True)
        
        if mode == "reversible":
            return jnp.logical_not(jnp.any(accepted))
        
        if mode == "simple_nonreversible":
            return jnp.logical_not(jnp.any(accepted))
        
        if mode == "persistent_sweep":
            frontier_idx = jnp.where(state.direction > 0, n_pairs -1, 0).astype(jnp.int32)
            frontier_accepted = accepted[frontier_idx]
            return jnp.logical_not(frontier_accepted)
        
        raise ValueError(f"Unknown lift_mode: {mode}")
    
    def _update_lift_state(self, state:TemperedState[StateT], blocked:Array, schedule_parity:Array|None) -> tuple[Array, Array|None, Array]:
        """
        Returns: 
            new_direction, 
            new_sweep_parity,
            flipped_direction,
        """

        mode = self.config.lift_mode
       

        if mode == "reversible":
            return state.direction, state.sweep_parity, jnp.asarray(False)

        if mode == "simple_nonreversible":
            flipped_direction = blocked
            new_direction = jnp.where(flipped_direction, -state.direction, state.direction)
            return new_direction, None, flipped_direction

        if mode == "persistent_sweep":
            flipped_direction = blocked
            new_direction = jnp.where(flipped_direction, -state.direction, state.direction)

            if schedule_parity is None:
                raise ValueError("persistent_sweep requires schedule_parity.")
            
            restart_parity = jnp.where(new_direction > 0, 0, 1).astype(jnp.int32)
            toggled_parity = (1 - schedule_parity).astype(jnp.int32)

            new_sweep_parity = jnp.where(
                flipped_direction,
                restart_parity,
                toggled_parity,
            ).astype(jnp.int32)

            return new_direction, new_sweep_parity, flipped_direction
        
        raise ValueError(f"Unknown lift_mode: {mode}")


    
    def _get_positions(self, replica_states: StateT) -> PyTree:
        """
        Extract stacked position PyTree from stacked replica states

        Assumes existence of `.position` field
        """
        if not hasattr(replica_states, "position"):
            raise AttributeError(
                "NRPT swap step assumes the base sampler state has a `.position` field."
            )
        return replica_states.position
    
    def _swap_array_pairs(self, x: Array, left_idx:Array, right_idx:Array, accepted:Array) -> Array:
        """
            Swap x[left_idx[k]] <-> x[right_idx[k]] for every accepted[k] = True.
            Assumes pairs are disjoint.

            Works for arbitary-rank replica major leaves
        """

        left_vals = x[left_idx]
        right_vals = x[right_idx]

        make_shape = (accepted.shape[0], ) + (1,)*(left_vals.ndim -1)
        accepted_mask = accepted.reshape(make_shape)

        new_left_vals = jnp.where(accepted_mask, right_vals, left_vals)
        new_right_vals = jnp.where(accepted_mask, left_vals, right_vals)

        x = x.at[left_idx].set(new_left_vals)
        x = x.at[right_idx].set(new_right_vals)
        return x
    
    def _apply_pair_swaps(self, pytree: PyTree, left_idx: Array, right_idx: Array, accepted: Array) -> PyTree:
        """
            Apply the same set of accepted pair swaps to every leaf of a stacked PyTree.
            Each leaf must be indexed by replica along axis 0.
        """
        def swap_leaf(leaf):
            if not hasattr(leaf, "ndim") or leaf.ndim < 1 :
                return leaf

            if leaf.shape[0] != self.config.n_replicas:
                return leaf

            return self._swap_array_pairs(leaf, left_idx, right_idx, accepted)

        return jax.tree_util.tree_map(swap_leaf, pytree)


    def _maybe_adapt_ladder(
        self,
        state: TemperedState[StateT],
        swap_info: SwapInfo,
    ) -> TemperedState[StateT]:
        """
        Placeholder Robbins-Monro ladder update.
        Adaptation is kept outside the core transition logic conceptually.
        """
        adaptation_state = state.adaptation_state
        if adaptation_state is None or not adaptation_state.enabled:
            return state

        # Placeholder: no ladder movement yet, only step bookkeeping.
        new_adaptation_state = LadderAdaptationState(
            enabled=adaptation_state.enabled,
            step=adaptation_state.step + 1,
            target_accept=adaptation_state.target_accept,
            step_size=adaptation_state.step_size,
            running_mean_accept=adaptation_state.running_mean_accept,
        )

        return TemperedState(
            replica_states=state.replica_states,
            betas=state.betas,
            direction=state.direction,
            replica_order=state.replica_order,
            adaptation_state=new_adaptation_state,
            iteration=state.iteration,
            sweep_parity=state.sweep_parity
        )
    
    @staticmethod
    def _proposed_adjacent_pairs(n_replicas: int, parity:Array) -> Array:
        """
            parity = 0 -> even pairs: (0,1), (2,3), ...
            parity = 1 -> odd pairs:  (1,2), (3,4), ...
        """
        if n_replicas < 2:
            return jnp.zeros((0, 2), dtype=jnp.int32)

        start = parity.astype(jnp.int32)
        left = jnp.arange(start, n_replicas - 1, 2, dtype=jnp.int32)
        right = left + 1
        return jnp.stack([left, right], axis = 1)
        


















    




