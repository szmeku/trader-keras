# Trading Sim & Env Architecture

**See also:** [[rl-oracle-ideas]], [[good-ideas-from-legacy-env]]

---

## Layer Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        CONSUMERS                                │
│                                                                 │
│   RL Agent (PPO/DQN)    Oracle Solver    Strategy (SMA/etc)     │
│         │                    │                  │                │
│         │    ┌───────────────┘                  │                │
│         │    │  (needs set_state               │                │
│         │    │   for backward                  │                │
│         │    │   iteration)                    │                │
│         ▼    ▼                                 ▼                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                     TradingEnv (env.py)                          │
│                     ══════════════════                           │
│                     Gymnasium interface                          │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │   reset()    │  │   step()     │  │  from_dataframe()     │  │
│  │ → obs, info  │  │ → obs, r,   │  │  (factory from        │  │
│  │              │  │   term,trunc │  │   parquet DataFrame)  │  │
│  │              │  │   info       │  │                       │  │
│  └──────────────┘  └──────┬───────┘  └───────────────────────┘  │
│                           │                                     │
│  Responsibilities:        │                                     │
│  • Owns OHLC bar data     │                                     │
│  • Tracks bar index       │                                     │
│  • Decodes RL actions     │                                     │
│    to sim Actions         │                                     │
│  • Episode boundaries     │                                     │
│    (truncation at end)    │                                     │
│                           │                                     │
├───────────────────────────┼─────────────────────────────────────┤
│                           ▼                                     │
│                    TradingSim (core.py)                          │
│                    ═══════════════════                           │
│                    MT5-fidelity engine                           │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  step(bar: Bar, action: Action) → (reward, done, info)  │   │
│  │                                                          │   │
│  │  Per-bar processing (4-tick MT5 OHLC model):             │   │
│  │                                                          │   │
│  │  1. Check old pending order at Open tick                 │   │
│  │  2. Check stop-out at Open tick                          │   │
│  │  3. Place agent's new order (if not rollover)            │   │
│  │  4. For each of 4 ticks:                                 │   │
│  │     • Check pending order fill                           │   │
│  │     • Check stop-out (50% margin level)                  │   │
│  │  5. Check swap (daily charge at hour ≥ 23)               │   │
│  │  6. reward = equity_now - equity_prev                    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  State:                    Constraints:                          │
│  • balance (float)         • Rollover: 23:58-00:05 rejects      │
│  • position (Position?)      new orders                         │
│  • pending (Order?)        • Stop-out: liquidate at 50%         │
│  • prev_equity             • Margin: reject if insufficient     │
│  • swap_charged_today      • Volume: clamp to [min, max]        │
│  • last_close_info         • Swap: $50/lot/day at hour ≥ 23     │
│                            • Commission: $/lot/side on          │
│                              open AND close                     │
│                                                                 │
│  Price model:              Tick sequence:                        │
│  • Data = BID prices       • Bullish: O → L → H → C            │
│  • ask = bid + spread*pt   • Bearish: O → H → L → C            │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                   InstrumentSpec (instruments.py)                │
│                   ══════════════════════════════                 │
│                   Frozen dataclass, loaded from JSON             │
│                                                                 │
│  symbol ─── point ─── contract_size ─── tick_value              │
│  volume_min ─── volume_max ─── volume_step                      │
│  commission_per_lot_side ─── digits                             │
│                                                                 │
│  Source: ~/projects/data/instrument_specs.json                  │
│  Exported from MT5 via: python tools/mt5_collect.py --specs     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Types

```
Action                          Bar
├── type: ActionType            ├── timestamp: float
├── volume: float               ├── open, high, low, close: float
└── price: float                ├── spread: float (in points)
                                ├── hour: int
ActionType (IntEnum)            └── minute: int
├── HOLD = 0                   — do nothing
├── LIMIT_BUY = 1              — enter LONG when price dips to target (buy the pullback)
├── LIMIT_SELL = 2             — enter SHORT when price rises to target (sell the rally)
├── STOP_BUY = 3               — enter LONG when price breaks above target (buy the breakout)
├── STOP_SELL = 4              — enter SHORT when price breaks below target (sell the breakdown)
└── CANCEL = 5                 — cancel pending order

                                Position
                                ├── side: Side (BUY=0, SELL=1)
                                ├── volume: float
                                └── entry_price: float
                                Order
                                ├── order_type: OrderType (LIMIT=0, STOP=1)
                                ├── side: Side
                                ├── volume: float
                                └── price: float
```

### Order types — market semantics

All 4 order actions (LIMIT_BUY, LIMIT_SELL, STOP_BUY, STOP_SELL) are **entry** orders — they open new positions. None of them close positions directly.

| Action | Direction | Trigger | Trading style |
|---|---|---|---|
| LIMIT_BUY | Long | price ≤ target | Mean-reversion (buy the dip) |
| LIMIT_SELL | Short | price ≥ target | Mean-reversion (sell the rally) |
| STOP_BUY | Long | price ≥ target | Momentum (buy the breakout) |
| STOP_SELL | Short | price ≤ target | Momentum (sell the breakdown) |

**Params** `Box(2)` = `[volume, price]` — the lot size and trigger price for the order. There is no built-in TP/SL mechanism.

**Exits** happen in two ways only:
1. **Netting** — place an opposite-side order while holding a position (closes existing, opens remainder if volume exceeds)
2. **Stop-out** — broker force-closes at 50% margin level

The agent must actively manage exits by placing opposite-side orders. "Stop loss" (the risk-management concept) is **not** a built-in feature — the agent must learn to exit losing positions on its own.

**Netting/reversal**: placing an opposite entry while a position is open will close the existing position and (if volume > existing) open a new one in the opposite direction (see [env_specs](env_specs.md)).

---

## Observation Vector (21 floats)

```
Index  Field              Source
─────  ─────              ──────
 0     balance            sim.balance
 1     equity             balance + unrealized_pnl
 2     free_margin        equity - used_margin
 3     margin_level       (equity / margin * 100) or 0

 4     bid                bar.close (data = bid prices)
 5     ask                bid + spread * point
 6     spread_dollars     spread * point

 7     has_position       1.0 or 0.0
 8     pos_side           0=BUY, 1=SELL
 9     pos_volume         lot size
10     pos_entry_price    fill price
11     pos_pnl            unrealized P&L in $

12     has_pending        1.0 or 0.0
13     pend_order_type    0=LIMIT, 1=STOP
14     pend_side          0=BUY, 1=SELL
15     pend_volume        lot size
16     pend_price         order price

17     hour_sin           sin(2π * hour / 24)
18     hour_cos           cos(2π * hour / 24)
19     dow_sin            0.0 (placeholder)
20     dow_cos            0.0 (placeholder)
```

---

## TradingEnv Action Space

```
action = {
    "type":   int 0-5  (ActionType enum),
    "params": [vol_frac, price_offset_pct]
}

vol_frac ∈ [0, 1]:
    0.0 → volume_min
    1.0 → max affordable volume (95% of balance * leverage / notional)
    snapped to volume_step

price_offset_pct ∈ [-1, 1]:
    0.0 → exact bid (sell) or ask (buy)
    ±1  → ±1% offset from reference price
```

---

## Info Dict (from step)

```python
{
    "balance":      float,  # cash after realized P&L
    "equity":       float,  # balance + unrealized P&L
    "margin":       float,  # used margin for open position
    "free_margin":  float,  # equity - margin
    "margin_level": float,  # (equity/margin)*100 or inf
    "stopped_out":  bool,   # liquidated this bar
}
```

---

## Key Formulas

```
bid = close                          (data is bid prices)
ask = close + spread * point

unrealized_pnl:
  LONG:   (bid - entry_price) * volume * contract_size
  SHORT:  (entry_price - ask) * volume * contract_size

equity = balance + unrealized_pnl
margin = (price * volume * contract_size) / leverage
margin_level = (equity / margin) * 100

commission = commission_per_lot_side * volume    (charged on open AND close)
swap = swap_cost * volume                        (once/day at hour ≥ 23)

reward = equity_now - equity_prev                (per bar)
```

---

## Order Execution Flow (per bar)

```
                    ┌──── Bar arrives ────┐
                    │                     │
                    ▼                     │
            Generate 4 ticks             │
         (O,L,H,C or O,H,L,C)          │
                    │                     │
                    ▼                     │
         ┌── Tick[0] (Open) ──┐          │
         │  Check old pending │          │
         │  Check stop-out    │          │
         └────────┬───────────┘          │
                  │                      │
                  ▼                      │
         ┌── Rollover? ──┐              │
         │ 23:58-00:05   │──YES──► skip │
         └──────┬────────┘       order  │
                │ NO                    │
                ▼                       │
         Place new order                │
                │                       │
                ▼                       │
         ┌── All 4 ticks ──┐           │
         │  Check pending   │           │
         │  Check stop-out  │           │
         │  (break on       │           │
         │   stop-out)      │           │
         └────────┬─────────┘           │
                  │                     │
                  ▼                     │
           Check swap                   │
           (hour ≥ 23)                  │
                  │                     │
                  ▼                     │
         reward = equity - prev_equity  │
         prev_equity = equity           │
                  │                     │
                  ▼                     │
         Return (reward, done, info)    │
                    │                     │
                    └─────────────────────┘
```

---

## What Mirrors MT5

| Python | MT5 |
|--------|-----|
| TradingSim | Strategy Tester engine |
| TradingEnv | EA runtime environment |
| Agent/Strategy | Expert Advisor (.ex5) |
| InstrumentSpec | SymbolInfoDouble/Integer |
| Bar | MqlRates |
| Action | OrderSend request |
| obs() | AccountInfo + PositionGet + OrderGet |

The sim is validated against MT5 with **zero tolerance** — exact price/P&L match on 1138+ SMA trades (see `tests/test_sim_vs_mt5.py`).
