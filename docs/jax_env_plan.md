# POC Plan: JAX/GPU-Native Trading Environment

## Objective

Build a JAX/GPU-native trading simulator that produces **identical results** to the existing Numba sim validated against MT5 Strategy Tester (1138 trades, zero tolerance). One clean interface — same `step`/`reset` for production and testing, no hacks.

## Scope

**In:**
- Pure-functional JAX sim with full trading mechanics (4-tick model, pending orders, netting, margin, stop-out, commission, swap, rollover)
- gymnax-style `Environment` interface (explicit state, jit-able, vmap-able)
- MT5 parity validation via the standard env interface
- vmap batching + benchmarking

**Out (post-POC):**
- Normalized "fraction of balance" observation/action space (goal doc design)
- Lookback window observations
- Data augmentation, WFO windowing, multi-asset batching
- RL framework integration
- `delay_bars` (unless needed for parity)

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  Environment  (env.py)                               │
│                                                      │
│  reset(key, params) → (obs, state)                   │
│  step(state, action, params) → (obs, state, reward,  │
│                                  done, close_info)   │
│                                                      │
│  Delegates to:                                       │
│    sim.py  — pure functional trading mechanics       │
│    obs.py  — observation construction                │
└──────────────────────────────────────────────────────┘
                    ↑ vmap over batches
```

No `key` argument in `step` — the sim is fully deterministic (no stochastic elements). `key` in `reset` is only for gymnax interface compatibility.

### Data flow per step

```
action (type, volume, price)
  │
  ▼
step_bar(state, spec, bar, action)
  │
  ├─ 1. Check OLD pending at open tick
  ├─ 2. Check stop-out at open
  ├─ 3. Place NEW order (if not stopped out, not in rollover)
  ├─ 4. Process 4 ticks (check NEW pending + stop-out at each)
  ├─ 5. Charge swap (if hour ≥ 23, once per day)
  ├─ 6. Compute reward = equity_now − prev_equity
  └─ 7. Advance step_idx
  │
  ▼
(obs, new_state, reward, done, close_info)
```

---

## Types

All types are `NamedTuple`s registered as JAX pytrees. `float32` throughout — MT5 reference data will be regenerated to match.

### EnvParams (static per episode)

| Field | Type | Description |
|---|---|---|
| `open` | `(n_bars,) f32` | Bar open prices |
| `high` | `(n_bars,) f32` | Bar high prices |
| `low` | `(n_bars,) f32` | Bar low prices |
| `close` | `(n_bars,) f32` | Bar close prices |
| `spread` | `(n_bars,) f32` | Spread in points |
| `hour` | `(n_bars,) i32` | Server hour |
| `minute` | `(n_bars,) i32` | Server minute |
| `spec` | `InstrumentSpec` | Instrument parameters |
| `initial_balance` | `f32` | Starting balance |

### InstrumentSpec

| Field | Example (BTCUSD) |
|---|---|
| `point` | 0.01 |
| `contract_size` | 1.0 |
| `commission_per_lot_side` | 0.0 |
| `volume_min` | 0.01 |
| `volume_max` | 10.0 |
| `leverage` | 25.0 |
| `swap_rate` | 0.50 |

### EnvState (mutable per step)

| Field | Type | Description |
|---|---|---|
| `balance` | `f32` | Liquid cash |
| `has_position` | `bool` | Position flag |
| `pos_side` | `i32` | 0=BUY, 1=SELL |
| `pos_volume` | `f32` | Lots |
| `pos_entry_price` | `f32` | Entry price |
| `has_pending` | `bool` | Pending order flag |
| `pend_order_type` | `i32` | 0=LIMIT, 1=STOP |
| `pend_side` | `i32` | 0=BUY, 1=SELL |
| `pend_volume` | `f32` | Lots |
| `pend_price` | `f32` | Trigger price |
| `prev_equity` | `f32` | Previous step equity (for reward) |
| `swap_charged_today` | `bool` | Daily swap flag |
| `step_idx` | `i32` | Current bar index |

### Action

| Field | Values |
|---|---|
| `type` | 0=HOLD, 1=LIMIT_BUY, 2=LIMIT_SELL, 3=STOP_BUY, 4=STOP_SELL, 5=CANCEL |
| `volume` | Lots (e.g. 0.01) |
| `price` | Order trigger price |

### CloseInfo (always output; `has_close` indicates validity)

`has_close, side, volume, entry_price, exit_price, pnl, commission`

---

## Sim Mechanics — Reference

Matching the existing Numba sim (`../trader/crypto_trader/icmarkets_env/numba_sim.py`) exactly.

### Price model
- All OHLC data are **bid** prices
- `ask = bid + spread × point`
- BUY fills at ask, SELL fills at bid

### 4-tick sequence
- Bullish bar (`close ≥ open`): O → L → H → C
- Bearish bar (`close < open`): O → H → L → C

### Pending order triggers
| Order | Condition |
|---|---|
| LIMIT BUY | `ask ≤ order_price` |
| LIMIT SELL | `bid ≥ order_price` |
| STOP BUY | `ask ≥ order_price` |
| STOP SELL | `bid ≤ order_price` |

### Netting (close-and-reverse)
When pending fills against an existing opposite position:
1. Close existing position → balance += pnl − commission
2. Remainder = `order_volume − old_volume`
3. If `remainder ≥ vol_min` AND margin check passes → open new position, balance −= commission
4. Otherwise → just close, no new position

### Stop-out
At any tick with a position: `margin_level = (equity / margin) × 100`. If ≤ 50% → force close, clear pending.

### Rollover window
`(hour == 23 AND minute ≥ 58) OR (hour == 0 AND minute < 5)` → reject new orders.

### Daily swap
If `hour ≥ 23` and not yet charged today: `balance −= swap_rate × margin` (margin at close price). Check stop-out after. Reset flag when `hour < 23`.

### Commission
`commission_per_lot_side × volume`, charged on both open and close, deducted from balance.

---

## Observations (21-dim)

Matching existing layout for parity. Evolve to goal doc's design post-POC.

```
 0  balance
 1  equity
 2  free_margin (equity − margin)
 3  margin_level_pct (equity / margin × 100, 0 if no position)
 4  bid
 5  ask
 6  spread_price (spread × point)
 7  has_position (0/1)
 8  pos_side (0=BUY, 1=SELL)
 9  pos_volume
10  pos_entry_price
11  pos_unrealized_pnl
12  has_pending (0/1)
13  pend_order_type
14  pend_side
15  pend_volume
16  pend_price
17  sin(hour)
18  cos(hour)
19  sin(dow) — placeholder 0
20  cos(dow) — placeholder 0
```

---

## Validation Strategy

### Unit Tests (per phase)

Each sim function and env method is tested with known inputs → expected outputs. Tests cover:
- Sim primitives: P&L, margin, equity calculations
- Pending order triggers: all LIMIT/STOP conditions
- Netting: close-and-reverse with remainder, margin check
- Stop-out: threshold behavior
- Bar-level step: full step scenarios (open, close, swap, rollover, stop-out during ticks)
- Env interface: multi-bar episodes, obs correctness, episode termination

### MT5 Parity Test (TBD)

Format and reference data to be determined. User will regenerate MT5 reference trades with float32-compatible precision. The test will use the standard `env.step()` interface — no backdoors.

---

## Implementation Phases

### Phase 1: Project Setup & Types

- `pyproject.toml` with deps (jax, jaxlib, pytest, pandas, numpy)
- `types.py`: all NamedTuples above + pytree registration
- Verify: `jax.jit` over a trivial function using these types compiles
- **Tests:** type creation, pytree flatten/unflatten roundtrip

### Phase 2: Sim Primitives

Pure functions in `sim.py`:
- `compute_ask`, `compute_unrealized_pnl`, `compute_equity`, `compute_margin`, `compute_margin_level`
- `close_position(state, spec, bid, ask) → (state, close_info)`
- `open_position(state, spec, side, volume, price, bid, ask) → state`
- **Tests:** each function with known inputs → expected outputs, including edge cases (no position, zero spread)

### Phase 3: Pending Orders & Stop-Out

- `check_pending_trigger(pend_type, pend_side, pend_price, bid, ask) → triggered`
- `process_pending_fill(state, spec, bid, ask) → (state, close_info)` — handles netting logic
- `check_stop_out(state, spec, bid, ask) → should_stop`
- `execute_stop_out(state, spec, bid, ask) → (state, close_info)`
- **Tests:** LIMIT/STOP trigger conditions, netting with remainder, margin check on remainder, stop-out threshold

### Phase 4: Bar-Level Step

- `generate_ticks(o, h, l, c) → (4,) tick array`
- `step_bar(state, spec, o, h, l, c, spread, hour, minute, action) → (state, reward, close_info)`
  - Full logic: old pending → stop-out → place → 4-tick loop → swap → reward
  - 4-tick loop: unrolled with `stopped_out` carry flag (no early exit in JAX, but remaining ticks become no-ops)
  - Rollover check before placing new order
- **Tests:** single-bar scenarios — open position, close on next bar, stop-out during ticks, swap charge, rollover rejection, close-and-reverse

### Phase 5: Environment Interface

- `env.py`: `TradingEnv` with:
  - `reset(key, params) → (obs, state)`
  - `step(state, action, params) → (obs, state, reward, done, close_info)`
- `obs.py`: `make_obs(state, spec, bid, ask, spread, hour) → (21,) array`
- Data loading helper: parquet → EnvParams
- `done` when `step_idx ≥ n_bars`
- **Tests:** multi-bar episodes, obs values correct, episode termination

### Phase 6: vmap & Benchmarking

- `vmap` step over batch dimension (batched state + batched params)
- Benchmark configurations:
  - Single env: baseline steps/sec
  - 100, 500, 1000, 1200 parallel envs
  - Compare to existing Numba sim speed
- Profile: identify bottlenecks (compilation time, tick processing, obs construction)

---

## File Structure

```
trader-rl-env/
├── src/
│   └── trading_env/
│       ├── __init__.py
│       ├── types.py          # EnvState, EnvParams, InstrumentSpec, Action, CloseInfo
│       ├── sim.py            # Pure functional sim mechanics
│       ├── env.py            # Environment interface (reset/step)
│       └── obs.py            # Observation construction
├── tests/
│   ├── conftest.py           # Shared fixtures (spec, params, bar data loading)
│   ├── test_types.py         # Phase 1
│   ├── test_sim.py           # Phase 2-3
│   ├── test_step.py          # Phase 4
│   └── test_env.py           # Phase 5
├── pyproject.toml
├── CLAUDE.md
├── the-goal.md
└── poc-plan.md
```

---

## Key Technical Challenges

### 1. Conditionals in JIT
The sim is branch-heavy. Strategy:
- `jnp.where` for simple value selection (same shape, no side effects)
- `jax.lax.cond` for branches with different state mutations
- Both branches must return same-shaped outputs (pad with dummy CloseInfo where needed)

### 2. 4-tick loop with early exit
Stop-out at any tick should skip remaining ticks. JAX has no break statement.
**Solution:** unroll all 4 ticks. Each tick checks a `stopped_out` flag — if True, it's a no-op returning state unchanged. Slightly wasteful but deterministic and jit-friendly.

### 3. Close-and-reverse atomicity
Netting requires sequential: close → compute remainder → margin check → open. These depend on each other (post-close balance needed for margin check). Must be expressed as chained state transformations, not parallel ops.

### 4. Multiple close events per bar
A bar can theoretically produce two closes (old pending fills at open → opens remainder → new pending fills during 4-tick loop → closes remainder). The existing Numba sim handles this by **overwriting** CloseInfo on each close — only the **last close per bar** is recorded. We match this behavior: single CloseInfo, last close wins.

---

## Data Dependencies

| What | Where |
|---|---|
| BTCUSD 1m bars | `../trader/tests/test_data/icmarkets_btcusd_*.parquet` |
| Instrument specs | `~/projects/data/instrument_specs.json` |
| MT5 reference trades | TBD — user will regenerate with float32 precision |

---

## Success Criteria

1. **Correctness:** all unit tests pass — sim mechanics match existing Numba sim behavior
2. **Clean interface:** all tests use `reset`/`step` only — zero test-only methods or internal state access
3. **JIT-able:** `step` and `reset` compile under `jax.jit` without errors
4. **vmap-able:** `step` works over batched `(state, params)` with `jax.vmap`
5. **Speed:** >100k steps/sec single env on GPU (baseline); near-linear scaling with batch count
6. **MT5 parity:** TBD — once reference data is regenerated
