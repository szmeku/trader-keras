---

**Trading Simulation Environment — Full Requirements**

**Broker: IC Markets Raw Spread Account**

---

**1. Fees**

Commission: $3.50/lot/side MT4/MT5, $3.00/lot/side cTrader. Spreads from 0.0 pips, variable, EUR/USD avg ~0.1 pips. Spreads widen during low liquidity and news events, tighten during London/NY overlap. No deposit, withdrawal, or inactivity fees.

**2. Swaps**

Applied on positions held past 5PM EST daily. Direction-specific and instrument-specific rates. Triple swap on Wednesday. Configurable per instrument.

**3. Leverage by Asset Class**

Forex Majors 1:30, Forex Minors 1:20, Forex Exotics 1:20, Commodities 1:10, Gold 1:20, Metals 1:10, Energies 1:10, Cryptos 1:2, Futures 1:10, Bonds 1:5, Shares 1:5, Indices Majors 1:20, Indices Minors 1:10. Leverage determined per instrument class, not account-wide. Configurable for different regulatory regimes.

**4. Margin**

Required margin = (price × volume) / instrument_leverage. Free margin = equity − used margin. Margin level = (equity / used margin) × 100. Reject orders if required margin > free margin at fill time.

**5. Margin Call & Stop-Out**

Margin call at 100% margin level. Stop-out at 50% — auto-close most unprofitable position first, loop until margin level > 50%. Negative balance allowed in gap/slippage scenarios.

**6. Execution**

1000ms base latency on all orders. Fill price = price after delay, not at submission. Slippage = price movement during delay. Optional jitter 800–1200ms.

**7. Price Feed**

Dual bid/ask always. Variable spread — wider at session boundaries, tighter London/NY overlap. Support tick or OHLC granularity. Historical replay and live modes.

**8. Sessions**

No trading Sat/Sun except crypto. Sydney/Tokyo/London/NY session awareness for spread modeling. Weekend gap on Monday open optional. Rollover window detection for swap application.

**9. Account State**

Track balance, equity, used margin, free margin, margin level. Update every tick/bar.

**10. Instruments**

Forex pairs minimum, extensible to indices/commodities/crypto/CFDs. Configurable pip value, contract size (1 lot = 100k units), tick size per instrument.

**11. Logging**

Every order: timestamps submitted and filled, requested price, fill price, slippage, commission, spread. Every position: open/close time, direction, volume, entry/exit price, accumulated swap, gross and net P&L. Equity curve at configurable intervals.

**12. Risk Controls**

Configurable max open positions, max lot size, max drawdown auto-halt.

---

**Agent Action Space**

`OPEN_LONG(instrument, volume)` — market buy. `OPEN_SHORT(instrument, volume)` — market sell. `CLOSE_POSITION(position_id)` — full close. `CLOSE_PARTIAL(position_id, volume)` — partial close. `PLACE_LIMIT_BUY(instrument, volume, price)` — limit buy order. `PLACE_LIMIT_SELL(instrument, volume, price)` — limit sell order. `PLACE_STOP_BUY(instrument, volume, price)` — stop buy order. `PLACE_STOP_SELL(instrument, volume, price)` — stop sell order. `SET_SL(position_id, price)` — set/modify stop-loss. `SET_TP(position_id, price)` — set/modify take-profit. `SET_TRAILING_STOP(position_id, distance_pips)` — set/modify trailing stop. `CANCEL_ORDER(order_id)` — cancel pending order. `HOLD` — do nothing.

**Continuous Parameters:** volume (0.01–max_lot, step 0.01), price (for limit/stop), distance_pips (trailing stop).

---

**Environment Auto-Actions (not agent-controlled)**

`MARGIN_CALL` — warning at 100% margin level. `STOP_OUT` — forced close at 50% margin level. `SWAP_CHARGE` — applied at 5PM EST rollover. `COMMISSION_DEDUCT` — applied at order fill. `SPREAD_APPLY` — applied at fill via bid/ask difference. `SLIPPAGE_APPLY` — price shift during 1s execution delay. `ORDER_FILL` — pending order triggered by price. `SL_TRIGGER` / `TP_TRIGGER` / `TRAILING_TRIGGER` — auto-close on price hit.

---

**Observation Space**

Current bid/ask per instrument. Spread. Account: balance, equity, free margin, margin level. Open positions: instrument, direction, volume, entry price, unrealized P&L, swap accumulated, current SL/TP. Pending orders: type, instrument, volume, target price. Time: hour, day of week, session. Price history: last N candles or ticks.

---

**Reward Signal Candidates**

Realized P&L net of commission, swap, spread. Change in equity step-to-step. Risk-adjusted return (Sharpe per episode). Penalty for stop-out. Penalty for excessive drawdown.

