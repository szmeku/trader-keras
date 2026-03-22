# Ideas

Future work and architecture options. See [[architecture]] for current implementation.

---

## Architecture Options

### Option A: Two-Stage Training (Current) ✓

Pre-train GRU on return prediction, then freeze and use for RL.

**Pros:** Stable gradients, easier debugging, RL focuses on decisions, works with limited data.
**Cons:** Slightly less powerful than end-to-end.

```python
# Stage 1: Supervised pre-training
GRU(features) -> embedding -> MLP -> predicted_return
Loss: MSE(predicted_return, actual_return)

# Stage 2: RL with frozen GRU
embedding = GRU(features)  # frozen, pre-trained
action_probs = Policy(embedding)  # trainable
Reward: PnL after fees
```

### Option B: Joint Training (Not Implemented)

Train GRU + Policy end-to-end with RL reward only.

**Pros:** Maximum flexibility, learns representations optimized for trading.
**Cons:** Unstable gradients, hard to debug, needs lots of data, credit assignment problem.

```python
embedding = GRU(features)  # trainable
action_probs = Policy(embedding)
Reward: PnL after fees (backprops through everything)
```

### Option C: Distillation (Not Implemented)

Train big GRU on prediction, distill to small GRU + RL.

**Pros:** Compact model for inference, fast Stage 2 training, good performance/compute tradeoff.
**Cons:** More complex pipeline, distillation loss adds hyperparameters.

```python
# Stage 1: Train big teacher GRU
Teacher_GRU -> predictions (MSE)

# Stage 2: Distill to student + RL
Student_GRU(features) -> embedding (smaller)
Policy(embedding) -> actions
Loss: α * KL(Teacher, Student) + β * RL_loss
```

---

## Modular RL Policies & Losses ✓ DONE

Implemented on `modular-rl-policies` branch, merged to main.

Swappable policies and losses via registries:

```bash
# Standard baseline
python run.py train --policy standard --loss standard

# Opportunistic - only act when confident
python run.py train --policy opportunistic --confidence-threshold 0.7

# Uncertainty-aware - hold when uncertain
python run.py train --policy uncertainty --uncertainty-threshold 0.5

# Conservative - avoid false positives
python run.py train --policy standard --loss conservative
```

**Philosophy:** "Better to do nothing when uncertain than to act wrongly."

---

## Next Steps (Priority Order)

### 1. Switch to PPO ✓ DONE
Implemented PPO with clipped surrogate, value baseline, multiple epochs per batch. See [[architecture#PPO Loss Function]].

### 2. Improve Stage 1 First ← ACTIVE
See [[stage1_proposal]] for full experiment plan — focuses on the 117KB 1-hour AAVE file.
Key changes: fee-adjusted 3-class classification target, directional accuracy metric, overfit sanity check, Stage 1→Stage 2 paired comparison.

### 3. Use Larger Data + Longer Bars
- Load full 2-year dataset instead of 1-hour/1-day samples
- Try `bar_seconds: 60` (1-min bars) — price moves ~0.05% per bar, exceeding fee threshold
- More diverse market conditions = better generalization

### 4. Tune Trade Incentives
- Increase `flat_penalty_pct` from 0.5 to 2-5
- Add explicit per-trade reward (small bonus for each round-trip)
- Curriculum: start with high flat penalty, decay as agent learns

### 5. Multi-Asset Training
Train on BTC, ETH, SOL for robustness.

### 6. Continuous Position Sizing (DDPG/SAC)
Replace discrete long/short/flat with continuous allocation.

### 7. Walk-Forward Optimization
Rolling train/val windows to avoid overfitting to one regime.

### 8. Live Paper Trading
Forward testing against real market data.

---

## Performance Ideas (If More Speed Needed)

### EnvPool / Isaac Gym
C++/GPU-native environments. 10-50x speedup. HIGH effort (1-2 weeks).

### Full JAX/Flax Rewrite (2026-03-20)
Rewrite entire pipeline (TradingSim + GRU policy + PPO) in JAX to run everything on GPU. Eliminates CPU-GPU transfers and enables `vmap` over 1000+ parallel envs.
- **Speedup**: 100-4000x (experiments taking hours → seconds/minutes)
- **Key constraint**: JAX requires pure functions, no mutation, no Python control flow in jitted code — TradingSim (order matching, netting, spread logic) is the hard part
- **GRU specifically**: ~parity with PyTorch (cuDNN fused kernels vs JAX scan — both hit the sequential dependency wall). Gains come from env vectorization, not model speed.
- **GTX 1050 Ti concern**: JAX preallocates 75% VRAM by default (3GB/4GB), needs tuning via `XLA_PYTHON_CLIENT_MEM_FRACTION`
- **Precedent**: JAX-LOB, JaxMARL-HFT prove it's viable for trading envs
- **Effort**: HIGH (2-4 weeks). Only worth it if env throughput is the dominant bottleneck.
- **Alternative**: switching only the policy to Flax gives ~0 speedup — the env is the bottleneck

### Profile Stage 1 GRU
Now takes 38% of pipeline time, could be next optimization target.

### DataLoader Optimization (2026-03-03)
`StridedLoader` is already near-optimal (zero-copy GPU-resident windows). For the `DataLoader` fallback path:
- **Quick win:** add `num_workers=2-4` + `pin_memory=True` to overlap CPU→GPU transfer with compute
- **Pre-normalize on GPU:** fold `NormalizedSequenceDataset` normalization into `StridedLoader` (subtract mean / divide std on the full tensor once, instead of per-sample in `__getitem__`)
- **No Rust DataLoader needed** — data is already in-memory numpy/GPU tensors, not disk-bound. The bottleneck is model forward/backward, not data loading.
