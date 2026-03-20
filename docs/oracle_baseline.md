# Oracle Baseline — Theoretical Profit Ceiling

## What it is

A backward dynamic programming solver that computes the **maximum possible profit** given perfect future knowledge of OHLC prices. It runs through our actual `TradingSim` (commissions, spread, swap, margin), so the result is a realistic ceiling — not a fantasy number.

```bash
python tools/validate.py oracle --symbol XAUUSD --from 2025-12-01 --to 2025-12-15 --lot 0.01 --levels 3
python tools/validate.py oracle --symbol XAUUSD --from 2025-12-01 --to 2025-12-15 --lot 0.01 --levels 3 --wandb
```

## Lot and Volume Levels

### What is a "lot"?

A **lot** is MT5's standard unit of trade size. It's NOT "how much money you invest" — it's a fixed multiplier:

| Instrument | 1 lot = | 0.01 lot = |
|---|---|---|
| XAUUSD | 100 oz gold | 1 oz gold |
| Forex (EURUSD) | 100,000 units | 1,000 units |
| BTCUSD | 1 BTC | 0.01 BTC |

The `--lot` flag sets the **base position size in lots** per trade. With XAUUSD at $4238 and 0.01 lot, one trade controls $4,238 worth of gold. With 25× leverage, the margin (collateral) is only $169.

Volume **can vary trade to trade** — each order specifies its own volume. Our SMA strategy uses a fixed lot, but the oracle picks from multiple sizes.

### Volume levels (`--levels`)

The `--levels N` parameter is **our oracle-specific setting** (not an MT5 concept). It controls how many position sizes the DP can choose from. With `--lot 0.01 --levels 3`:

| Level | Volume | = lot × level |
|---|---|---|
| 1 | 0.01 | base bet |
| 2 | 0.02 | 2× base |
| 3 | 0.03 | 3× base |

At each bar the DP picks the optimal state from `{flat, long@0.01, long@0.02, long@0.03, short@0.01, short@0.02, short@0.03}` — that's `1 + 2×N` states total (7 for N=3).

With perfect foresight, the DP almost always picks the maximum volume (99%+ of trades at 0.03 in testing). More levels = higher ceiling, but also more states in the DP (still trivially fast with Numba — 525K bars in 0.14s).

### Capital model

**Fixed capital, no reinvestment.** The DP picks from the same volume set (e.g., 0.01/0.02/0.03) at every bar regardless of current balance. Profits accumulate in the account but never increase position sizes. This is a flat-rate model, not compounding.

Default balance is **$10,000** with **25× leverage** (set via `--balance` / `--leverage`). These are sim/account settings, not strategy settings — passed through `validate.py` CLI to `TradingSim`.

With 0.03 lot XAUUSD at $4238, the position controls $12,714 worth of gold. **Margin** is the collateral the broker locks while the position is open: `$12,714 / 25 = $508`. That's only 5% of the $10K balance, so the oracle is never margin-constrained. The large PnL numbers come from trading ~17,000 times in 2 weeks, each making a few dollars.

## W&B Logging

Runs log to W&B by default (project: `strategy-validation`, `--no-wandb` to skip):

- **Equity curve** — cumulative PnL over time (step = trade index)
- **Trade profit** — per-trade P&L
- **Volume** — position size per trade
- **Cumulative commission** — running total of costs
- **Trades table** — full trade log (browsable in W&B UI)
- **Summary metrics** — total PnL, win rate, profit factor, max drawdown, return %

Works with any strategy, not just oracle.

## Simp Oracle vs DP Oracle

### The numbers (XAUUSD, 2 weeks, 20160 bars)

| Metric | Simp Oracle | DP (0.01 lot) | DP (3 levels) |
|---|---|---|---|
| **Net PnL** | $17,143 | $32,399 | $96,678 |
| **Commission** | $0 | $1,210 | $3,609 |
| **Trades** | ~10,260 | 17,294 | 17,289 |

DP(1 level) is **1.89×** simp — despite paying $1,210 in commission that simp ignores entirely.

### Why they differ

**1. Close-to-close vs intra-bar range (the big one)**

Simp computes `sum(|log(close[i] / close[i-1])|)` — only the net directional move between bar closes. The DP oracle buys at `ask@low` and sells at `bid@high`, capturing the full **high-low range** within each bar. Since `(high - low)` is almost always larger than `|close_change|`, the DP extracts ~2× more gross profit from the same data.

This is why DP beats simp even after paying real costs.

**2. Simp ignores all costs; DP pays them and still wins**

Simp's $17K assumes zero commission, zero spread. If simp actually traded its 10,260 direction changes at 0.01 lot, it would pay ~$718 commission + ~$513 spread ≈ $1,231, netting ~$15,912. DP still beats this adjusted number by 2×.

**3. Simp flips every bar; DP knows when to hold**

Simp implicitly flips position every time the return sign changes (50.9% of bars). DP makes more total trades (17,294) because it profitably exploits bars where simp would hold — but it also skips flips where intra-bar range doesn't cover commission + spread.

**4. Volume: DP picks max bet 99% of the time**

With 3 levels and perfect foresight: 17,137 of 17,289 trades used 0.03 (max). Perfect knowledge = max confidence = max bet. DP(3 levels) / DP(1 level) = 2.98×, almost exactly 3×.

### Bottom line

| Use case | Which oracle |
|---|---|
| Quick directional edge metric during training | `simp_baseline_both` |
| Absolute ceiling for strategy comparison | DP oracle via `tools/validate.py oracle` |

Simp is useful as a fast, cost-free training signal. DP oracle is the ground truth for "how much money could we possibly make on this data with our broker's cost structure."

## Implementation

- `tools/strategies/oracle.py` — Numba-JIT backward DP + `OracleStrategy` class (Strategy protocol) + `export_states()` for MQL5
- `tools/strategies/OracleStrategy.mq5` — MQL5 EA that reads precomputed states from CSV and replays them with limit orders
- Core DP runs only in Python (Numba). MQL5 EA is a thin replay layer — no DP duplication.
- 17 unit tests in `tests/test_oracle.py`

### Architecture: Export & Replay

The MQL5 EA no longer contains a DP solver. Instead:

1. **Python side**: `oracle.py` runs the DP, then `export_states()` writes a CSV to MT5's `Common/Files/` directory
2. **MQL5 side**: `OracleStrategy.mq5` reads that CSV in `OnInit()`, populates `g_states[]` + `g_barTimes[]`
3. **OnTick**: Same as before — replays the optimal states with limit orders at bar extremes

The states CSV format is simple: `time,state` per row, where time is `YYYY.MM.DD HH:MM:SS` and state is the integer DP state. Filename encodes symbol and date range for uniqueness: `oracle_states_BTCUSD_20260120_20260227.csv`.

### MQL5 Oracle EA

The `OracleStrategy.mq5` EA replays precomputed optimal states in MT5 Strategy Tester:

1. **`OnInit()`**: Reads precomputed states CSV (`StatesFile` input), loads bar data via `CopyRates()` for limit order prices
2. **`OnTick()`**: On each new bar, looks up precomputed optimal state, places pending limit orders at bar extremes (buy at low+spread, sell at high) — matches Python's fill prices exactly
3. **`OnDeinit()`**: Exports `trade_log_mt5.csv` + `trade_log_mt5_config.csv` for comparison

```bash
# Full validation: Python sim + MT5 comparison
# (validate.py automatically exports states before launching MT5)
python tools/validate.py oracle --symbol XAUUSD --from 2025-12-01 --to 2025-12-15 --lot 0.01 --levels 3 --auto
```

The EA uses pending limit orders (not market orders) so fills occur at the same prices as the Python sim: `BUY_LIMIT` at bar low + spread, `SELL_LIMIT` at bar high. This matches the DP's cost model exactly.

`validate.py` calls `strategy.export_states()` automatically before launching MT5, and passes the `StatesFile` input to the EA via `ea_inputs()`.

MQL_TESTER guard prevents running on live charts.
