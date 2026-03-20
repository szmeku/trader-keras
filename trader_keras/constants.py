"""Single source of truth for feature columns and defaults."""

# Default active features (matches original trader project convention)
FEATURE_COLS: list[str] = [
    "high", "low", "volume",
    "log_returns", "volatility", "volume_ma", "volume_ratio",
    "log_volume", "log_volatility", "log_volume_ma",
    "vol_regime", "price_accel", "time_in_day_counter",
]

# Optional — included when present in data
OPTIONAL_FEATURE_COLS: list[str] = ["buy_ratio", "log_spread"]

DATA_DIR = "~/projects/data"
ARTIFACTS_DIR = "artifacts"
