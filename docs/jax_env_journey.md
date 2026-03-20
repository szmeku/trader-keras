# JAX Env: Journey to Deterministic Overfitting

How we got from "build a fast trading env" to a policy that memorizes 1440 bars of BTCUSD with $4.93M deterministic PnL. Each section covers a decision point where we changed direction based on evidence.

## Phase 1-5: JAX Env Implementation (2026-03-19)

Built `crypto_trader/jax_env/` — pure functional trading sim in JAX, mirrors `icmarkets_env/TradingSim` exactly. Branchless (`jnp.where`), JIT-able, vmap-ready.

**Validated**: 2100-bar equivalence test vs Python env. 50 tests passing.

No surprises here — straightforward port following the [[jax_env_plan]].

## Phase 6: Benchmarks — JAX Scan vs Python Loop

First real decision point. Benchmarked different approaches:

| Backend | Steps/sec |
|---------|-----------|
| Numba (Python loop) | 330k |
| JAX scan CPU | 99k |
| JAX scan GPU | 101k |
| JAX Python loop CPU | 1.1k |
| JAX Python loop GPU | 541 |

**Key finding**: JAX with Python loop is **300x slower** than with `lax.scan`. The Python overhead dominates completely. For single-env training (overfitting), `lax.scan` is essential.

**Decision**: All JAX env interaction must go through `lax.scan`, not Python loops.

## Phase 7a: First Training Attempt — Python Loop + Old Obs

Started with a Python training loop: extract PyTorch weights → JAX arrays, run env step-by-step in Python, compute REINFORCE gradient.

**Problem**: 352 steps/sec. Training 1440 bars took 4+ seconds per epoch. Way too slow for iterating.

**Decision**: Move the entire rollout (obs → normalize → forward → sample → decode → step) into `lax.scan`.

## Phase 7b: lax.scan Rollout — 36x Speedup

Created `jax_env/rollout.py`: pure JAX MLP forward pass (no PyTorch during rollout), Gumbel-max for discrete actions, Normal sampling for continuous params. Full episode in one compiled XLA call.

Result: **12,839 steps/sec** — 36x faster than Python loop.

**What we learned building this**:
- `extract_mlp_params(policy)`: converts PyTorch weights → JAX arrays each epoch
- `_mlp_forward`: just `h @ w.T + b` with `jax.nn.relu` — no need for Flax/Equinox
- `_decode_action`: branchless parametric decoder (same logic as Python `parametric_decode_single`)
- `lax.dynamic_slice` for lookback window indexing inside the scan

## Phase 7c: Observation Design — Raw Prices vs Log Returns

Initial obs used the standard 21-dim vector from `jax_env/obs.py` (balance, bid, ask, entry_price, etc.). Raw prices leaked asset identity — the NN could distinguish BTC ($74k) from EURUSD ($1.08) trivially.

**Problem**: Overfitting worked but only in stochastic mode. Deterministic (argmax) produced only 2-3 trades — the policy hadn't learned generalizable patterns, just noise.

**Decision**: Switch to asset-agnostic log-return observations (from [[jax_env_goal]] design doc).

Later redesigned further (2026-03-20): all OHLCV as `log(x_t/x_{t-1})`, time features removed entirely to keep patterns asset/timezone-agnostic:
- `N_BAR_FEATURES = 6`: `log(o_t/o_{t-1})`, `log(h_t/h_{t-1})`, `log(l_t/l_{t-1})`, `log(c_t/c_{t-1})`, `log(vol_t/vol_{t-1})`, `rel_spread`
- `N_STATE_FEATURES = 6`: `equity/initial_balance - 1`, position info (no time)
- Lookback window of 10 bars → obs_dim = 10×6 + 6 = 66

**Result**: The NN now builds its own internal representations from relative price changes. No raw prices, no time, no hand-crafted indicators anywhere.

## Phase 7d: Reward Signal — Equity vs Realized PnL

Tried two reward signals:

1. **Equity change** (dense): reward every step = equity(t+1) - equity(t). Most steps have non-zero reward when holding a position.
2. **Realized PnL** (sparse): reward only when a trade closes. Most steps have reward=0.

**First attempt used realized PnL**: PnL jumped from $780 (type-only log_prob) to $72k avg (full log_prob including continuous params). But the signal was too sparse for REINFORCE — only ~200 out of 1440 steps had non-zero reward.

**Key discovery**: Adding continuous params to the REINFORCE log_prob was critical. With type-only log_prob, the policy learned WHICH action to take but not HOW (volume, price). Adding `Normal(mu, std).log_prob(raw_params)` let it optimize volume and price simultaneously.

## Phase 7e: REINFORCE Collapse and Fixes

After ~270 epochs of REINFORCE training, the policy collapsed: converged to STOP_SELL only → 0 trades → loss=0 → stuck.

**Root cause**: No exploration pressure. Once the policy found a local optimum (always SELL), REINFORCE couldn't escape because zero trades = zero gradient.

**Attempted fixes** (all applied together):
1. Entropy bonus (0.05) — prevent action distribution collapse
2. Value baseline (`policy.value_head`) — reduce gradient variance
3. Discounted returns (gamma=0.999) — propagate sparse rewards
4. Lower lr (3e-3 → 1e-3)
5. Gradient clipping (1.0)

**Result**: Prevented collapse but created new problem — entropy bonus was too strong. With 6 action types, max entropy = ln(6) ≈ 1.79, and the policy sat at 1.75 (basically uniform random). PnL oscillated wildly between -$10k and +$137k across epochs.

## Phase 7f: Multiple Rollouts — Variance Reduction

Single-trajectory REINFORCE has extremely high variance. Each epoch runs one stochastic rollout, gets one noisy gradient. The policy can never converge because each rollout explores different actions.

**Fix**: 8 rollouts per gradient update, equity rewards (dense signal instead of sparse close_pnl), entropy annealing (0.03 → 0.001).

**Result**: Much better trend — avg PnL grew from ~$0 to ~$66k, best=$89k. But still too slow and noisy. Entropy remained high (1.65/1.79). After 500 epochs, PnL still fluctuating ±$30k between epochs.

**Diagnosis**: REINFORCE is fundamentally wrong for memorization. The gradient estimator `∇log π(a|s) · A(s,a)` has variance proportional to policy entropy. With near-uniform action distribution (required for exploration), gradients point in random directions. More rollouts reduce variance linearly but convergence is still O(1/sqrt(N)).

## Phase 7g: CEM — The Breakthrough

Switched from REINFORCE to **Cross-Entropy Method (CEM)**:

1. Run 16 stochastic rollouts with current policy
2. Rank by total equity reward
3. Select top-4 (elite trajectories)
4. Train policy supervised on elite data: cross-entropy for action types + Normal NLL for continuous params
5. Repeat

**Why CEM works where REINFORCE failed**:
- **No gradient estimation**: supervised loss gives clean, unambiguous gradients
- **No credit assignment problem**: trajectory-level selection handles it implicitly
- **Natural exploration decay**: as policy improves, stochastic rollouts cluster tighter → elite trajectories become more consistent → supervised targets stabilize
- **log_std shrinks naturally**: Normal NLL loss pushes `params_log_std` negative → policy becomes more deterministic over time

**Results** (500 epochs from scratch):

| Epoch | Avg PnL | Elite Equity | log_std |
|-------|---------|-------------|---------|
| 1 | $4k | $21k | 0.00 |
| 100 | $1.0M | $1.1M | -0.03 |
| 200 | $3.4M | $3.4M | -0.18 |
| 300 | $4.3M | $4.4M | -0.35 |
| 500 | $4.7M | $4.7M | -0.84 |

**Monotonic improvement** — no oscillation, no collapse, no hyperparameter sensitivity.

Continued for 500 more epochs (lr=3e-4):

| Epoch | Avg PnL | log_std |
|-------|---------|---------|
| 600 | $4.8M | -0.87 |
| 800 | $4.9M | -0.97 |
| 1000 | $4.9M | -1.06 |

## Final Results

**Deterministic rollout** (temperature=0, argmax types, mean params):
- PnL: **$4.93M** from $10k starting balance
- Trades: 706
- Win rate: 100%
- Actions: BUY_STOP (720) + SELL_STOP (719) exclusively
- Deterministic **outperforms** stochastic ($4.93M vs $4.7M avg)

**Trade pattern**: Alternates BUY_STOP ↔ SELL_STOP every bar (nets position each time). When price trends strongly, repeats same-direction orders for 2-7 bars (effectively holding). Never uses HOLD action — uses same-direction STOP orders as de facto hold (see [[issues]] for this sim discrepancy).

## Key Takeaways

1. **lax.scan is mandatory** for single-env JAX training. Python loop overhead kills performance (300x slower).
2. **Log-return obs > raw prices** for overfitting. Asset-agnostic features force the NN to learn relative patterns.
3. **REINFORCE is wrong for memorization**. High entropy (needed for exploration) makes gradients too noisy. Multiple rollouts help linearly but don't fix the fundamental issue.
4. **CEM is ideal for overfitting**: trajectory-level selection + supervised loss = clean gradients + natural exploration decay.
5. **Equity reward > realized PnL** for REINFORCE (dense signal vs sparse). For CEM, we select on total equity anyway.
6. **Continuous params matter**: type-only policy plateaus at $780 PnL. Adding volume/price optimization via `Normal(mu, std).log_prob` → $72k+ immediately.

## Discovered Issues

- **Same-direction STOP orders are no-ops**: env silently ignores same-side fills instead of increasing position. MT5 would increase position size. See [[issues#2026-03-20]].
- **Commission = 0.0**: trade register shows zero commission everywhere. Needs investigation.
- **cuDNN conflict**: JAX needs cudnn>=9.8, torch GRU kernel breaks with cudnn 9.20. Pre-existing, documented.

## Files

| File | Purpose |
|------|---------|
| `crypto_trader/jax_env/rollout.py` | lax.scan rollout, MLP forward, action decoder, obs construction |
| `tools/train_validation_policy.py` | CEM training loop, trade register generation |
| `validation_output/overfit_policy.pt` | Trained checkpoint (policy, obs stats, config) |
| `validation_output/trades_jax_btcusd.csv` | 706-trade register from deterministic rollout |
