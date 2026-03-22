# Status Report — 2026-03-22

## Source files (trader_keras/)

| File | Lines | Role |
|---|---|---|
| `steps/rl.py` | **191** | PPO training loop (largest, at budget ceiling) |
| `steps/train.py` | 96 | Supervised training + save |
| `steps/data.py` | 84 | load, featurize, window |
| `config.py` | 75 | Dataclass configs (Backbone, Data, Train, RL, Env, Wandb) |
| `steps/model.py` | 65 | Model build + checkpoint loading |
| `models/gru.py` | 57 | GRU model builder |
| `steps/ppo_loss.py` | 49 | PPO loss computation |
| `pipeline.py` | 44 | Ctx type + pipe() runner |
| `models/policy.py` | 36 | Actor-critic MLP for RL |
| `data/features.py` | 35 | Feature engineering (6 log-ratio cols) |
| `steps/env.py` | 24 | Thin wrapper for icmarkets_env |
| **Total source** | **757** | All files under 200-line budget |

## Tests

| File | Lines | Tests | Covers |
|---|---|---|---|
| `test_steps_rl.py` | 204 | 11 | PPO rollout, GAE, action sampling, fit_rl |
| `test_steps_data.py` | 172 | 12 | load, featurize, window |
| `test_steps_model.py` | 159 | 12 | model build, checkpoint, freeze |
| `test_steps_train.py` | 126 | 6 | fit_supervised, save, pipe integration |
| `test_config.py` | 91 | 14 | Hydra config groups, overrides, skip |
| `test_steps_env.py` | 89 | 6 | env step (skip if no data) |
| `test_policy_model.py` | 85 | 6 | Actor-critic shapes, gradients |
| `test_data_pipeline.py` | 30 | 1 | create_targets direct |
| `test_model.py` | 25 | 2 | Loss function direct |
| **Total tests** | **981** | **70** | **All passing** |

Test-to-source ratio: **1.3:1** (981 test lines / 757 source lines).

## Implementation status

| Feature | Status |
|---|---|
| Composable pipeline (`ctx→ctx`, `pipe()`) | Done |
| Predict pipeline (load→featurize→window→model→checkpoint→fit→save) | Done |
| RL pipeline (load→featurize→env→model→checkpoint→fit_rl→save) | Done |
| Hydra config groups (predict, rl, bench) | Done |
| Config split (backbone, data, train, rl, env) | Done |
| PPO (rollout, GAE, clipped surrogate, actor-critic) | Done |
| `pipeline.skip` for step skipping | Done |
| Streaming GRU research (lax.scan, stateful, batch-as-isolation) | Documented |
| `predict_then_rl` composed pipeline | Not started |
| Backbone registry (GRU/LSTM/Transformer swap) | Not started |
| Streaming GRU implementation | Not started |

## Complexity notes

- `steps/rl.py` (191 lines) — 6 functions. PPO is inherently complex. Could split `_collect_rollout` and `_train_on_rollout` into separate files if it grows.
- `config.py` (75 lines) — 7 dataclasses, will grow with more pipelines. Fine for now.
- Everything else is straightforward single-responsibility.

## Active docs (docs/)

| File | Lines | Purpose |
|---|---|---|
| `open_lookback_windowing.md` | 196 | Streaming GRU research: lax.scan, stateful, VRAM estimates, batch-as-isolation |
| `predictor_vs_rl_pipeline.md` | 137 | Predict vs RL pipeline design rationale |
| `proposal_pipeline_design.md` | 63 | Pipeline architecture (source of truth) |
| `proposal_registries.md` | 95 | Registry pattern design |
| `issues.md` | 120 | Active issues |
| `plan_backbone_embedding.md` | 41 | Backbone slicing plan (use Keras model surgery) |
| `plan_registry.md` | 29 | Registry extraction plan |

Legacy docs from `../trader` moved to `docs/trader-old-docs/` (42 files, 7,800+ lines). Reference only.

## Config structure

```
conf/
  config.yaml              # shared: backbone, data, wandb (25 lines)
  pipeline/
    predict.yaml           # train params (15 lines)
    rl.yaml                # data overrides + env + rl params (27 lines)
    bench.yaml             # benchmark config (19 lines)
```

## Entry point

`run.py` (49 lines) — Hydra main, PIPELINES dict, skip logic.
