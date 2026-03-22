# ICMarkets Asset Similarity Analysis

**Generated:** 2026-03-05
**Assets:** 10 largest ICMarkets 1-minute bar files (86,400 rows each)
**Method:** Euclidean distance on z-scored statistical features + Ward's hierarchical clustering

---

## Summary Statistics

| Asset | Type | Price | LR Std | LR Skew | LR Kurt | Spread Rel | Vol Std | % Zero |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BTCUSD | crypto | 79,121.80 | 0.000836 | 1.46 | 78 | 0.014 | 0.30 | 0.3% |
| AAPL.NAS | us_equity | 238.24 | 0.000968 | -1.33 | 595 | 0.030 | 0.24 | 3.9% |
| XEC.NYSE | us_equity | 58.11 | 0.002237 | 0.68 | 123 | 1.780 | 1.08 | 4.5% |
| LB.NYSE | us_equity | 51.32 | 0.001919 | -13.30 | 2,763 | 0.396 | 0.68 | 7.0% |
| MTS.MAD | eu_equity | 35.75 | 0.001691 | 0.92 | 73 | 7.001 | 0.76 | 9.8% |
| TKA.ETR | eu_equity | 10.11 | 0.002154 | -12.91 | 4,851 | 8.555 | 0.83 | 9.3% |
| SZG.ETR | eu_equity | 29.98 | 0.003193 | 0.08 | 1,088 | 2.774 | 0.91 | 11.4% |
| WAF.ETR | eu_equity | 43.78 | 0.003284 | -0.07 | 3,375 | 2.610 | 0.91 | 12.7% |
| TRE.MAD | eu_equity | 23.10 | 0.001881 | -4.10 | 1,197 | 21.118 | 0.89 | 16.1% |
| ITRK.LSE | uk_equity | 47.09 | 0.000789 | -38.94 | 4,401 | 5.859 | 0.85 | 16.2% |

> **LR** = log_return_close | **Spread Rel** = spread_mean / price_mean | **Vol Std** = std of log_return_volume | **% Zero** = fraction of zero log returns (liquidity proxy)

---

## Clusters

Three natural groups emerge from Ward's hierarchical clustering:

### Cluster 1 -- High Liquidity (BTCUSD, AAPL)

- Lowest volatility per bar (LR Std < 0.001)
- Tightest relative spreads (< 0.03)
- Fewest zero returns (< 4%) -- finest tick resolution
- Stable volume (Vol Std ~ 0.27)

### Cluster 2 -- Illiquid Equities (TRE, WAF, SZG, LB, MTS, ITRK, TKA)

- 2-4x higher volatility than Cluster 1
- Wide relative spreads (0.4 -- 21.1)
- 7-16% zero returns -- tick-size constrained
- Extreme kurtosis (73 -- 4,851) from low liquidity tail events
- Heavy negative skew in ITRK (-38.9), TKA (-12.9), LB (-13.3)

### Cluster 3 -- Outlier (XEC.NYSE)

- Extreme volume skewness (24.2) -- very bursty volume
- Highest volume variability (1.08)
- Moderate return stats but unique microstructure

---

## Most Similar Pairs

| Rank | Asset 1 | Asset 2 | Distance | Notes |
|---:|---|---|---:|---|
| 1 | WAF.ETR | SZG.ETR | **1.42** | Both German small-caps, near-identical profiles |
| 2 | SZG.ETR | MTS.MAD | 2.30 | Similar spread/volume characteristics |
| 3 | WAF.ETR | TKA.ETR | 2.35 | German ETR exchange cluster |
| 4 | LB.NYSE | MTS.MAD | 2.59 | Cross-exchange match |
| 5 | BTCUSD | AAPL.NAS | 2.75 | Both high-liquidity, low zero-returns |

## Most Dissimilar Pairs

| Rank | Asset 1 | Asset 2 | Distance |
|---:|---|---|---:|
| 1 | XEC.NYSE | ITRK.LSE | **6.37** |
| 2 | XEC.NYSE | AAPL.NAS | 6.28 |
| 3 | BTCUSD | ITRK.LSE | 5.78 |

---

## Uniqueness Ranking

Average distance to all other assets (higher = more unique):

| Asset | Avg Distance | |
|---|---:|---|
| XEC.NYSE | 5.09 | most unique |
| ITRK.LSE | 4.79 | |
| AAPL.NAS | 4.59 | |
| BTCUSD | 4.42 | |
| TRE.MAD | 4.35 | |
| WAF.ETR | 3.61 | |
| TKA.ETR | 3.60 | |
| SZG.ETR | 3.47 | |
| LB.NYSE | 3.43 | |
| MTS.MAD | 3.24 | most typical |

---

## Key Findings

1. **BTCUSD behaves like AAPL** at 1-min resolution -- lower per-bar volatility than most equities, tight spreads, near-zero stale bars. The 24/7 trading spreads volatility across more bars.

2. **German small-caps cluster tightly** -- WAF, SZG, TKA share similar spread/return/volume profiles. Good candidates for cross-asset transfer learning.

3. **Crash-prone distributions** -- ITRK (skew=-38.9) and TKA (skew=-12.9) have extreme left tails. Models trained on these need robust tail-risk handling.

4. **TRE.MAD is most expensive to trade** -- relative spread 21x price, making it impractical for high-frequency strategies.

5. **US equities show bimodal liquidity** -- XEC, LB, AAPL have ~45-48% missing spread values from off-hours data, creating two distinct trading regimes.

6. **XEC.NYSE is a true outlier** -- its extreme volume burstiness (skew=24.2) makes it statistically distinct from everything else in this dataset.
