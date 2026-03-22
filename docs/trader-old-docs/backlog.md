# Codebase Reduction Backlog

Deferred tasks from 2026-03-19 reduction sprint. Each has a cost/benefit note.

---

## High Priority

### H1 — Delete unused ABC base classes
**Files:** `crypto_trader/rl/policies.py` (`BasePolicy`), potentially others
**Action:** `BasePolicy` (ABC + `nn.Module`) is only used by `MLPPolicy` and `TransformerPolicy`, which are not used by stage2 (stage2 uses `MixedPolicy` directly). If `MLPPolicy`/`TransformerPolicy` are truly unused, delete all three + `POLICY_REGISTRY` entries.
**Benefit:** ~50 lines removed
**Risk:** Check if any tests or tools reference these before deleting.

### H2 — Unify forecaster and policy registries
**Files:** `crypto_trader/registry.py`, `crypto_trader/models/forecaster.py`, `crypto_trader/rl/policies.py`
**Action:** `forecaster.py` uses `make_registry()` (dynamic, decorator-based). Policy/loss/reward use `registry_factory()` (static dict). Unify to one pattern — either all `make_registry` (better for extensibility) or all static dicts.
**Decision needed:** Static dict is simpler if we don't need runtime registration.

### H3 — Merge `targets.py` into `features.py`
**Files:** `crypto_trader/data/targets.py`, `crypto_trader/data/features.py`
**Action:** `targets.py` is ~50 lines of simple log-return computation. The data pipeline calls both sequentially. Merging into one `features.py` or a `dataset.py` reduces the file count without violating size limits.
**Benefit:** Simpler import graph.

### H4 — Remove unused re-exports
**Files:** `crypto_trader/rl/__init__.py` and similar
**Action:** Audit `__init__.py` files for re-exports that nothing outside the package imports. Remove dead exports.
**Tool:** `grep -r "from crypto_trader.rl import" .` to find actual usages.

---

## Medium Priority

### M8 — ActionDecoder Protocol → Callable
**Files:** `crypto_trader/rl/action_decoder.py`, `crypto_trader/eval/rollout_eval.py`
**Action:** `ActionDecoder` is a Protocol with one method `__call__`. Replace with `Callable[[...], ...]` type alias — no Protocol needed.
**Benefit:** 10 lines removed, simpler types.

### M9 — Replace `_compute_features_numba` with pandas
**Files:** `crypto_trader/data/features.py`
**Action:** If any remaining numba functions exist (check with `grep -n "@nb.njit\|@numba" features.py`), replace with pandas equivalents. We already removed the rolling Numba loop in the 2026-03-19 sprint.
**Note:** Verify with `grep numba crypto_trader/data/features.py` first — may already be done.

### M11 — Device inference helper
**Files:** `crypto_trader/trainer/stage2.py`, `crypto_trader/eval/rollout_eval.py`, others
**Action:** `device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")` appears in several places. Extract to `crypto_trader/constants.py` as `def get_device(device_str: str = "") -> str`.
**Benefit:** DRY, single place to change device selection logic.

---

## Low Priority / Notes

### H6 — Pre-compute percentile thresholds explanation
**Context:** `by_percentile()` in `percentile.py` recomputes `np.quantile(abs_s, q)` in a loop. For large arrays called many times, pre-computing all thresholds once would be faster.
**Action:** Add a `by_percentile_fast(selector, metric_fn, quantiles)` that computes all thresholds in one `np.quantile(abs_s, quantiles)` call, then iterates.
**Note:** Only matters if profiling shows this is a bottleneck.

### Task 12 — empyrical / quantstats replacement
**Status:** Abandoned — empyrical broken on Python 3.12 (`SafeConfigParser` removed). `quantstats` requires pandas Series input (would need adapters). Existing `risk_metrics.py` (pure numpy, 83 lines) is clean and fit-for-purpose. Keep as-is.

### Task 13 — omegaconf
**Status:** Done 2026-03-19. `load_config` uses `OmegaConf.load` + `OmegaConf.to_container`. Supports `${variable}` interpolation in YAML for free.
