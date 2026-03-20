# Good Ideas from Legacy Env (deleted 2026-03-14)

Patterns and optimizations worth preserving from `crypto_trader/envs_legacy/` before deletion.

## Vectorization Patterns

### Shared Data Optimization
When all envs use the same data, store 1 copy with `(1, steps, features)` layout. Index with `min(i, data.shape[0]-1)`. Massive memory savings: 2yr data = 692MB single copy vs N copies.

### Pre-allocated Output Buffers
Reuse `obs_buf`, `rewards_buf`, `term_buf` every step. Zero allocation in inner loop.

### Partial Reset with Masks
`reset(env_mask=...)` resets only done envs. Crucial for continuous rollout loops.

## Numba JIT

### Multi-kernel Architecture
Three separate kernels (step, build_obs, reset) instead of one monolith. Each independently optimized, faster JIT.

### Auto-dispatch Serial vs Parallel
```python
PARALLEL_THRESHOLD = 1024
fn = step_kernel_parallel if num_envs >= PARALLEL_THRESHOLD else step_kernel_serial
```
Prange overhead only worth it at 1024+ envs. Below that, serial Numba is faster.

### Inline Always
`@numba.njit(cache=True, fastmath=True, inline='always')` for the single-env step function, called by both serial and parallel kernels.

### Contiguous Memory
`np.ascontiguousarray(data, dtype=np.float32)` before passing to Numba. Significant perf difference.

## Branchless GPU Code

### Masks as Float Multipliers
```python
close_mask = (changing & (self.positions != 0.0)).float()
cash = self.cash + (self.holdings * prices - close_fee) * close_mask
```
No `.if_()`, no `.any()`, no GPU-CPU syncs. Compatible with torch.compile and CUDAGraphs.

### Immutable Semantics (No In-Place Mutations)
```python
# NO: self.cash -= x
# YES: cash = self.cash - x * close_mask; self.cash = cash
```
Required for torch.compile tracing.

### Reward Masking
```python
rewards = (new_equity - prev_equity) * (done | active).float()
```
Zero out rewards for done envs in one branchless op.

## Three-Phase Position Transitions
1. Close existing position (if changing)
2. Open long (if target=long)
3. Open short (if target=short)

Handles all transitions cleanly including flips (long->short = close + open).

## Benchmarking Results (RTX 4090, 2026-02-15)
- Numba: ~10M steps/sec single-threaded up to 1024 envs
- Crossover: Numba wins <256 envs, GPU wins >=1024 envs
- torch.compile: +50-70% on GPU env
- VRAM per row: ~88 bytes (2yr 7.8M rows = 692MB)
- GTX 1050 Ti: too tight for GPU env, use Numba+256 envs
- RTX 4090: GPU+4096 envs optimal

## Testing Patterns
- Kernel-level unit tests (test logic before wrapping in classes)
- Reference implementation validation (run same actions through single-env and vectorized, compare)
- Device-agnostic tests (`'cuda' if available else 'cpu'`)

## Config Worth Preserving
- `max_episode_steps` — soft truncation limit (separate from data end termination)
- `fee_rate` — exposed as param
- Per-env `n_steps_per_env` — handles heterogeneous episode lengths
