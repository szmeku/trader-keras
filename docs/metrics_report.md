# Trading Model Evaluation Metrics

## Key Invariants

The oracle strategy always picks the correct sign on every bar:

$$\text{logret}_\text{oracle} = \sum_{i=1}^{N} |r_i| > 0$$

where $r_i = \log(p_i / p_{i-1})$ is the log return (additive in log-price space).

Any model with imperfect predictions can only do equal or worse:

$$\text{logret}_\text{model} \leq \text{logret}_\text{oracle} \quad \Rightarrow \quad \text{ratio} \in (-\infty,\ 1]$$

---

## Metric 1 — Logret Ratio (Ceiling Proximity)

$$\text{logret\_ratio} = \frac{\text{logret}_\text{model}}{\text{logret}_\text{oracle}}$$

**Scale invariance.** If all returns are scaled by a constant $k$, both numerator and denominator scale by $k$:

$$\text{ratio} = \frac{\sum_i \text{sign}(\hat{r}_i) \cdot r_i}{\sum_i |r_i|}$$

Empirically verified over 500 seeds across oracle wealth ranging from $2\times$ to $130\times$: expected ratio shifts by $< 0.005$ across regimes.

**Why it flatters.** The logret ratio lives in log-return space. The corresponding wealth ratio is:

$$\text{wealth\_ratio} = \frac{e^{\text{logret}_\text{model}}}{e^{\text{logret}_\text{oracle}}} = e^{(\text{ratio} - 1) \cdot \text{logret}_\text{oracle}}$$

For $\text{ratio} = 0.7$ and $\text{logret}_\text{oracle} = 2.8$ (roughly a 1-year strong oracle):

$$\text{wealth\_ratio} = e^{-0.3 \times 2.8} \approx 0.43$$

A logret ratio of $0.70$ corresponds to capturing only $43\%$ of oracle's wealth gain.

**Honest alternative.** Simple-return ratio preserves sign and is in wealth space:

$$\text{simple\_ratio} = \frac{W_\text{model} - 1}{W_\text{oracle} - 1}$$

where $W = e^{\text{logret}}$. Always $\leq 1$, crosses zero at break-even, negative for losing models.

---

## Metric 2 — Sortino Ratio (Floor Constraint)

$$\text{Sortino} = \frac{\mathbb{E}[r_\text{model} - r_\text{benchmark}]}{\sigma_\text{down}}, \qquad \sigma_\text{down} = \sqrt{\mathbb{E}\left[\max(0,\ r_\text{benchmark} - r_\text{model})^2\right]}$$

**Benchmark choice:**
- Asset appreciated over the period → benchmark = buy-and-hold long
- Asset depreciated over the period → benchmark = buy-and-hold short

This ensures the floor is always the passive strategy a non-predictive investor would use.

---

## Two-Metric Framework

| Metric | Role | Answers |
|---|---|---|
| Logret ratio | Ceiling proximity | How close to theoretical maximum? |
| Sortino vs benchmark | Floor constraint | Am I beating passive hold, risk-adjusted? |

A good strategy satisfies both. The gap between them is informative:

- **Wide gap** (good Sortino, low logret ratio) — safely above floor but far from ceiling; better predictions would directly translate to gains
- **Narrow gap** (both high) — close to squeezing out all available alpha
- **High Sortino, low logret ratio in bear/choppy market** — Sortino looks good because benchmark is weak; logret ratio stays honest

---

## Combined Loss for Neural Network Training

### Normalized Efficiency Metric

Place the model on a scale between floor and ceiling:

$$M = \frac{\text{logret}_\text{model} - \text{logret}_\text{benchmark}}{\text{logret}_\text{oracle} - \text{logret}_\text{benchmark}}$$

| Value | Interpretation |
|---|---|
| $M = 1$ | Model equals oracle (ceiling) |
| $M = 0$ | Model equals benchmark (floor) |
| $M < 0$ | Model worse than passive hold |
| $M > 1$ | Impossible by construction |

The denominator $\text{logret}_\text{oracle} - \text{logret}_\text{benchmark} > 0$ always (oracle always beats passive), so no division-by-zero risk.

### Apex Loss (implemented in `crypto_trader/models/losses.py`)

With benchmark = 0 (break-even), both terms become dimensionless and scale-invariant:

$$\mathcal{L} = \underbrace{1 - \frac{\mathbb{E}[\tanh(\hat{r}) \cdot r]}{\mathbb{E}[|r|]}}_{\text{capture term}} + \lambda \cdot \underbrace{\frac{\mathbb{E}[\max(0,\ -\tanh(\hat{r}) \cdot r)^2]}{\mathbb{E}[|r|]^2}}_{\text{downside term}}$$

where $r = r_i$ is the actual log return and $\tanh(\hat{r})$ is the predicted position (smooth sign relaxation).

- **Capture term**: $1 - M$ with benchmark=0 — ranges $[0, 2]$, reaches $0$ at oracle, $1$ at random/init
- **Downside term**: quadratic penalty on bars where the strategy loses; divided by $\mathbb{E}[|r|]^2$ for true dimensionlessness ($r^2/r^2$)

> **Note:** Dividing the downside term by $\mathbb{E}[|r|]$ alone would leave units of $r$. Using $\mathbb{E}[|r|]^2$ makes both terms truly scale-invariant.

- First term pulls toward oracle (maximize ceiling proximity)
- Second term penalizes every bar the strategy loses money (Sortino downside curvature)

**Implicit magnitude weighting.** Because the capture term multiplies `tanh(pred)` by the actual return `r` (not just `sign(r)`), large moves contribute proportionally more to the loss. Being correct on a 2% bar matters ~20× more than being correct on a 0.1% bar. The downside term has the same property quadratically — a wrong call on a big move is penalised as the square of the missed return. No extra configuration needed.

### Interpreting Loss Values

| Loss value | Meaning |
|---|---|
| **0.0** | Theoretical oracle — perfect sign, full tanh confidence |
| **0.0 to 0.3** | Excellent — high oracle capture, low downside |
| **0.3 to 0.7** | Good |
| **~1.0** | Random predictions / fresh initialisation baseline |
| **> 1.0** | Net losing — predictions are destructive |

The two terms trade off: a model can reach e.g. -0.8 either via 80% capture + zero downside, or 90% capture + 10% downside penalty.

### Stability Notes

**Sign function.** Hard $\text{sign}(\hat{r})$ has zero gradient almost everywhere. Use a continuous relaxation for training:

$$\text{position}_i = \tanh(\hat{r}_i) \quad \text{or} \quad \text{position}_i = \hat{r}_i \text{ (raw output)}$$

**Short episodes.** With small $N$, the denominator $\text{logret}_\text{oracle} - \text{logret}_\text{benchmark}$ can be small and noisy. Compute over rolling windows or normalize per epoch.

**$\lambda$ tuning.** Both terms are dimensionless. Start with $\lambda = 1$ and tune based on whether the model prioritizes return capture or downside protection. Config key: `apex_lam`.

**logret\_oracle is a constant** during training (precomputed from realized returns), so it introduces no gradient issues — it is purely a scaling factor.
