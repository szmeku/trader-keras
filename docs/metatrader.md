# MetaTrader 5 Data Bridge

Export market data from MetaTrader 5 (IC Markets) to `~/projects/data/` for training.

## Architecture

```
Python (host)                    MQ5 EA (MT5 in Bottles/Wine)
─────────────                    ────────────────────────────
tools/mt5_collect.py             mql5/DataExporter.mq5
  │                                │
  ├─ writes trigger file ──────>   ├─ polls every 500ms
  │   (symbols, bars/date range)   ├─ reads symbols + params from trigger
  │                                ├─ exports CSV per symbol to Files/export/
  ├─ waits for done file  <──────  ├─ writes done file
  │                                │
  └─ converts CSVs to parquet → ~/projects/data/icmarkets_*.parquet
```

**File-based IPC** through the shared filesystem (MQL5/Files/ is on the host via Bottles).

Output parquets conform to `OHLCVSchema` (defined in `crypto_trader/constants.py`) — the same schema validated by the data pipeline loader.

## Setup

| Item | Detail |
|---|---|
| MT5 | MetaTrader 5 via **Bottles** (Flatpak) |
| Broker | IC Markets |
| Bottles bottle name | `MT5` |
| MT5 install path (host) | `~/.var/app/com.usebottles.bottles/data/bottles/bottles/MT5/drive_c/Program Files/MetaTrader 5/` |
| MQL5 Files dir (host) | `...MetaTrader 5/MQL5/Files/` |
| MT5 mode | Portable (data stored in install dir, not AppData) |
| Python in MT5 Wine | **Does not work** — use host Python |

### One-time MT5 setup

1. Copy `mql5/DataExporter.mq5` to `...MQL5/Experts/Advisors/`
2. Open MetaEditor, compile it (F7)
3. In MT5, drag `DataExporter` EA onto any chart
4. Enable **Algo Trading** button in toolbar (green = on)
5. Never touch it again — Python controls everything

## Usage

```bash
# List all available symbols from MT5
python tools/mt5_collect.py --list

# Search symbols by name (case-insensitive)
python tools/mt5_collect.py --search gold

# Export specific symbols (default 180 bars = 3h M1)
python tools/mt5_collect.py EURUSD XAUUSD

# Custom bar count
python tools/mt5_collect.py --bars 60 EURUSD GBPJPY BTCUSD

# Last 7 days of M1 data (7 * 1440 = 10080 bars)
python tools/mt5_collect.py --days 7 EURUSD XAUUSD

# Date range: from a specific date to now
python tools/mt5_collect.py --from 2026-03-04 EURUSD XAUUSD

# Date range: exact start and end
python tools/mt5_collect.py --from 2026-03-04 --to 2026-03-07 EURUSD

# With timeout (useful for large exports)
python tools/mt5_collect.py --timeout 300 --days 30 EURUSD XAUUSD

# Export instrument specs (contract size, point, tick value, etc.)
python tools/mt5_collect.py --specs XAUUSD EURUSD
```

## Output

Parquet files at `~/projects/data/icmarkets_<symbol>_<start>_to_<end>_1m.parquet` with columns:

`timestamp, open, close, high, low, volume, spread, real_volume`

These are directly loadable by the training pipeline via `data.pattern: "icmarkets_*.parquet"`.

## Key paths

- **EA source (repo)**: `mql5/DataExporter.mq5`
- **Trigger file**: `...MQL5/Files/export_trigger.txt`
- **Done file**: `...MQL5/Files/export_done.txt`
- **Export CSVs**: `...MQL5/Files/export/*.csv`
- **Collected data**: `~/projects/data/icmarkets_*.parquet`
- **Instrument specs**: `~/projects/data/instrument_specs.json`

## Strategy Tester CLI Automation

MT5 Strategy Tester can be launched from code via `terminal64.exe /config:<ini_path>`:

```bash
# Full auto validation: Python sim + MT5 tester + comparison
python tools/validate.py sma --symbol BTCUSD --auto
```

**How it works** (`tools/mt5_tester.py`):
1. Generates UTF-16LE `.ini` config with `[Tester]` section (dates as unix timestamps, `ShutdownTerminal=1`)
2. Writes `.set` file for EA inputs (SMA_Fast, SMA_Slow, etc.) to `MQL5/Profiles/Tester/`
3. Launches via `flatpak run com.usebottles.bottles -b MT5 -e terminal64.exe -a "/config:C:\..."`
4. MT5 starts, runs the backtest, exports results, and shuts down automatically
5. Python reads exported CSVs and runs comparison

**Key paths (tester):**
- **Tester config**: `...Config/test_auto.ini` (generated)
- **EA inputs**: `...MQL5/Profiles/Tester/SMAStrategy.set` (generated)
- **Trade export (Common)**: `...AppData/Roaming/MetaQuotes/Terminal/Common/Files/export/trade_log_mt5.csv`
- **Trade export (Tester agent)**: `...Tester/Agent-127.0.0.1-3000/MQL5/Files/export/trade_log_mt5.csv`

**Strategy files** are co-located: `tools/strategies/sma_crossover.py` (Python) + `tools/strategies/SMAStrategy.mq5` (MQL5). `mql5/deploy.sh` copies `.mq5` files from both `mql5/` and `tools/strategies/` to MT5.

## ONNX Oracle Tester

Run a trained Stage 2 RL oracle policy in MT5 Strategy Tester via ONNX inference.

### Pipeline

```
1. Train oracle model          → python run.py train oracle_config.yml
2. Export to ONNX               → python tools/export_stage2_onnx.py <run_or_path>
3. Export bar data for oracle   → python tools/export_bars_csv.py <pattern>
4. Deploy to MT5                → bash mql5/deploy.sh
5. Compile in MetaEditor (F7)
6. Run in Strategy Tester       → BTCUSD, M1, date range within CSV coverage
```

### ONNX Export

```bash
# From W&B run name
python tools/export_stage2_onnx.py young-violet-1244

# From local checkpoint
python tools/export_stage2_onnx.py artifacts/models-stage2:v39/stage2_oracle_20260317_165654.pt

# Custom output path
python tools/export_stage2_onnx.py young-violet-1244 --out mql5/Files/my_model.onnx
```

The export **bakes obs normalization** (running mean/var from training) into the ONNX graph — MT5 feeds raw observations, no preprocessing needed.

**Outputs:** `type_logits (1, 6)` + `params_mu (1, 2)`.

### Bar Data Export (for oracle future-leak)

The oracle model needs to peek at future bars. Since MT5 Strategy Tester can't look ahead, we pre-export all bars to a CSV that the EA loads at init. The EA matches bars by **unix timestamp** (not by order), so you can run any date sub-range.

```bash
# Full dataset (same as training)
python tools/export_bars_csv.py "icmarkets_btcusd*1m.parquet" --bar-seconds 60 --limit 100000

# Specific date range (export a superset — include n_future=50 extra bars beyond test end)
python tools/export_bars_csv.py "icmarkets_btcusd*1m.parquet" --start 2024-08-01 --end 2024-08-15
```

Output: `mql5/Files/bars_oracle.csv` — columns: `timestamp, open, high, low, close, spread`.

### young-violet-1244 Model Details

- **Asset:** BTCUSD (IC Markets)
- **Source file:** `icmarkets_btcusd_20240714_1846_to_20260313_1643_1m.parquet`
- **Training data:** first 100k bars (`load_limit: 100000`) → **2024-07-14 18:46 → 2024-09-22 21:12**
- **Bar period:** M1 (60s)
- **Oracle:** n_future=50, hidden=256, layers=3
- **Config:** `oracle_config.yml`

Strategy Tester date range must stay within **2024-07-14 – 2024-09-22** (minus 50 bars at the end for oracle lookahead).

### MT5 Strategy Tester Setup

1. `bash mql5/deploy.sh` — copies `OracleTrader.mq5` + `stage2_oracle.onnx` + `bars_oracle.csv` to MT5
2. Compile `OracleTrader.mq5` in MetaEditor (F7)
3. Strategy Tester settings:
   - **Expert:** OracleTrader
   - **Symbol:** BTCUSD
   - **Period:** M1 (must match `bar_seconds` from training)
   - **Date range:** within CSV coverage (check export output for time range)
   - **Deposit:** 10000, Leverage: 1:20 (must match training config)

### EA Inputs

| Input | Default | Description |
|---|---|---|
| `InpBarsCSV` | `bars_oracle.csv` | CSV filename in MQL5/Files/ |
| `InpBalance` | 10000 | Must match training balance |
| `InpLeverage` | 20 | Must match training leverage |
| `InpNFuture` | 50 | Future bars for oracle (must match `n_future` in config) |
| `InpPriceScale` | 0.01 | Price offset scale for parametric decoder |

### How the EA Works

1. **On init:** loads ONNX model (embedded resource) + CSV bars into memory
2. **On each new bar:**
   - Finds current bar's timestamp in CSV (advancing cursor, O(1))
   - Builds 21-dim base obs from MT5 account state (balance, equity, position, pending orders, time encoding)
   - Reads next 50 bars from CSV → normalizes OHLC relative to current close → 250 oracle dims
   - Feeds 271-dim obs to ONNX → `type_logits (6)` + `params_mu (2)`
   - Deterministic action: `argmax(type_logits)` → action type (HOLD/LIMIT_BUY/LIMIT_SELL/STOP_BUY/STOP_SELL/CANCEL)
   - Decodes params: `volume_frac → lots`, `price_offset → limit/stop price`
   - Executes via `CTrade`

### Observation Vector (271 dims)

**Base 21 dims** (from MT5 account state):

| Index | Field | Source |
|---|---|---|
| 0 | Balance | `ACCOUNT_BALANCE` |
| 1 | Equity | `ACCOUNT_EQUITY` |
| 2 | Free Margin | `ACCOUNT_MARGIN_FREE` |
| 3 | Margin Level % | `equity / margin * 100` |
| 4 | Bid | `SYMBOL_BID` |
| 5 | Ask | `SYMBOL_ASK` |
| 6 | Spread (price) | `Ask - Bid` |
| 7-11 | Position | has, side, volume, entry, PnL |
| 12-16 | Pending order | has, type, side, volume, price |
| 17-18 | Time | `sin/cos(2π·hour/24)` |
| 19-20 | Unused | 0.0 |

**Oracle 250 dims** (from CSV, indices 21-270): for each of 50 future bars, 5 values:
- `(future_open - current_close) / current_close`
- `(future_high - current_close) / current_close`
- `(future_low - current_close) / current_close`
- `(future_close - current_close) / current_close`
- `future_spread` (raw, in points)

### Key Files

| File | Purpose |
|---|---|
| `mql5/OracleTrader.mq5` | MT5 EA — ONNX inference + oracle data loading |
| `mql5/Files/stage2_oracle.onnx` | Exported model (with baked-in normalization) |
| `mql5/Files/bars_oracle.csv` | Bar data for oracle future-leak |
| `tools/export_stage2_onnx.py` | Python: checkpoint → ONNX export |
| `tools/export_bars_csv.py` | Python: parquet → CSV for MT5 |
| `oracle_config.yml` | Training config for oracle model |
