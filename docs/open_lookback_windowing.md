# Open Question: Engineered Lookback vs Learned Context

## The question

Should we provide a fixed lookback window of bars to the model, or let the architecture decide how much context it needs?

## Current approach: fixed lookback

We slice `lookback` bars into a window `(batch, lookback, features)` and feed it to the model. The model sees exactly `lookback` bars — no more, no less. This applies to both:
- **Supervised**: `window` step creates `(lookback, features)` sequences
- **RL**: env provides `lookback` bars as part of its flat observation

## Arguments for fixed lookback (current)

- Simple, predictable memory usage — `O(lookback)` per sample
- Easy to reason about what the model sees
- Works with any architecture (MLP, GRU, Transformer)
- Consistent between supervised and RL pipelines
- Batching is trivial — all sequences same length

## Arguments for learned/variable context

- A recurrent model (GRU, LSTM) naturally handles variable-length sequences — forcing a fixed window throws away its strength
- A seq2seq / encoder-decoder architecture could process the full available history and learn which parts matter
- Fixed lookback is an arbitrary hyperparameter — the "right" lookback depends on the regime, asset, and time scale
- Attention-based models can attend to variable context with masking

## Architecture-specific considerations

**GRU/LSTM (current)**:
- Can process variable-length sequences natively via `mask` or packing
- In theory, hidden state carries information beyond the lookback window
- But in practice: training on `lookback=60` means the model never learns to use longer context
- Alternative: feed the full episode as one sequence, let the recurrent state accumulate. The GRU hidden state IS the learned embedding — no windowing needed

**Transformer**:
- Fixed context window is natural (positional encodings, attention matrix)
- But context can be much larger (thousands of bars)
- Lookback maps directly to attention span

**Streaming / online**:
- For live trading, we get one bar at a time
- A recurrent model can maintain state across bars — no lookback needed
- A windowed model needs to re-process the last `lookback` bars each step (wasteful)

## Possible directions

1. **Keep fixed lookback** but make it large enough and let the model learn what to attend to
2. **Episode-level sequences** for recurrent models: feed all bars as one sequence, backprop through time (memory-expensive but lets the model learn its own context length)
3. **Variable lookback** with padding/masking: train on different lookback lengths so the model is robust
4. **Streaming GRU**: in RL, step the GRU one bar at a time, carry hidden state across steps. The hidden state replaces the lookback window entirely
5. **Hybrid**: use lookback for the feature window but also pass hidden state from previous steps (streaming recurrence + local context)

## Streaming GRU — most interesting direction

Option 4 is appealing for both supervised and RL:
- The env already steps one bar at a time
- Instead of flattening `lookback` bars into the observation, pass one bar `(1, features)` to the GRU each step
- The GRU hidden state carries forward automatically — it IS the learned context
- Observation size shrinks from `lookback * 6 + 6` to `6 + hidden_size + 6` (current bar + hidden state + account state)
- The model learns how far back to "remember"
- Matches how it would run in production (live streaming)

The tricky part for RL: PPO collects rollouts and replays them — but hidden states are sequential. Solutions exist (store hidden states in rollout buffer, or re-roll hidden states from episode start before each PPO epoch).

## Simplified model: lookback as single knob

No separate `stateful` flag needed. Just `lookback`:
- `lookback=60` → reset hidden state every 60 steps. Equivalent to current windowed approach but streaming.
- `lookback=None` → never reset. GRU accumulates state indefinitely, learns its own context length.

The training loop is always streaming (one bar at a time). The only difference is whether/when `reset_states()` is called:
```
for step, bar in enumerate(bars):
    out = model(bar)  # state carried automatically
    if lookback and step % lookback == 0:
        model.reset_states()
```
No nested loops, no windowing logic. One unified approach.

## Keras JAX backend: `lax.scan` confirmed

Verified in Keras source (`keras/src/backend/jax/rnn.py`, line 198): when `unroll=False` (default), GRU uses `jax.lax.scan` internally:
```python
new_states, outputs = lax.scan(
    f=_step,
    init=initial_states,
    xs=scan_xs,
    reverse=go_backwards,
)
```
This compiles the sequential loop into a fused XLA kernel. Both windowed and streaming paths use the same efficient primitive.

## Keras `stateful=True` — built-in streaming support

Keras GRU natively supports stateful mode:
- `stateful=True` → hidden state carried between `__call__` invocations automatically
- Requires **fixed batch size**: use `batch_shape=(B, 1, n_features)` instead of `input_shape`
- `model.reset_states()` clears all hidden states
- `return_state=True` available for manual state management if needed
- `shuffle=False` required during training (state carries across batches, shuffling breaks temporal order)
- `stateless_call()` API available for JAX functional transforms (jax.grad, jit)

**Note:** `reset_states()` resets ALL batch slots. If different slots need different reset schedules (e.g., different assets with different fold boundaries), use manual state management with `return_state=True` instead.

## Batch dimension as natural stream isolation

**Key insight:** if we never mix streams in the first place, we don't need to track segment boundaries.

The `../trader` codebase uses `segment_id` to un-mix concatenated dataframes — detecting boundaries, enforcing no cross-segment windows, filtering in backtest. That complexity exists because everything is concatenated into one big dataframe first.

**Simpler approach:** stack data as `(n_streams, n_bars, features)` upfront. Each batch slot = one independent stream. No concatenation, no segment_id, no boundary detection.

```python
# Data prep: just stack sequences into a matrix
# shape: (n_streams, n_bars_per_fold, 6)
streams[0]  = EURUSD fold 1 clean
streams[1]  = EURUSD fold 1 noised
streams[2]  = EURUSD fold 2 clean
streams[3]  = XAUUSD fold 1 clean
...

# Training: feed one timestep at a time
model(streams[:, t, :])  # shape (n_streams, 1, 6)
# Each slot has its own hidden state. Nothing ever mixes.
```

The batch dimension **is** the separation. No tracking needed. Data prep = "stack your sequences into a matrix."

**Why this works for streaming GRU:**
- Each slot maintains its own hidden state independently
- `reset_states()` resets all slots (fine if all folds are same length)
- For variable-length folds: pad shorter ones or use `return_state=True` for manual per-slot reset
- All streams processed in parallel on GPU with one `lax.scan` call

## Capacity estimates

### Stream counts
- 200 assets × 20 folds (5yr / 3-month) = 4,000 base streams
- With noise augmentation (5x): ~20,000 streams

### VRAM for hidden states (float32, 2-layer GRU)

| hidden_size | weights (shared) | state/stream | 20K streams | fits in 8GB? |
|---|---|---|---|---|
| 64 | ~155 KB | 512 B | 10 MB | trivially |
| 128 | ~605 KB | 1 KB | 20 MB | trivially |
| 256 | ~2.4 MB | 2 KB | 40 MB | trivially |
| 512 | ~9.5 MB | 4 KB | 80 MB | trivially |

During training (optimizer states + gradients + activations), multiply weight memory by ~4x. Still under 200 MB total for H=512 with 20K streams.

**Hidden state memory is never the bottleneck.** GPU compute saturation (diminishing throughput from more streams) hits first, typically at tens of thousands of streams. For our 20K case — completely fine.

### Input tensor size concern

Full sequence in memory: `(20K, 90K, 6)` float32 = ~43 GB — doesn't fit in VRAM.

Solutions:
1. Process 2-4K streams at a time (5-10 passes)
2. Use shorter sub-sequences with state carry-over
3. Preprocess features to parquet, load chunks at training time

### Training time estimates (H=256, 2-layer, ~600K params)

Data: 200 assets × 5 years × ~250 days × 1440 min ≈ 360M bars total.

Per-fold: ~90K bars (3 months forex). With 20K parallel streams, one GRU step per bar:

| GPU | per step | per epoch (90K steps) | overfit (10-30 epochs) |
|---|---|---|---|
| 4090 | ~1 ms | ~90 sec | **15-45 min** |
| 3090 | ~2 ms | ~180 sec | **30-90 min** |

Features are preprocessed and stored as parquet — training is pure matmuls, no CPU-side feature pipeline overhead.

**600K params vs 300M+ samples means the model will generalize more than overfit unless trained hard.**

## GRU internals: weights vs hidden state

Two kinds of parameters, both controlled by `hidden_size`:

| | **Weights** | **Hidden state** |
|---|---|---|
| What | Gate matrices (W_z, W_r, W_h) | Recurrent activation vector |
| Shape | `(input_dim, hidden_size)` + `(hidden_size, hidden_size)` per gate | `(batch, hidden_size)` per layer |
| Changes | Per optimizer step (epoch/batch) | Per forward step (every bar) |
| Learned via | Backprop (gradient descent) | Forward pass (GRU update equations) |
| Shared across | All timesteps and all streams | Nothing — each stream has its own |
| Persisted | Saved in model weights file | Transient, discarded after sequence |

`hidden_size` controls both the weight matrix dimensions AND the hidden state vector size. Larger `hidden_size` = more expressive short-term memory AND more learnable parameters. `num_layers` stacks GRUs — each layer has its own weights and hidden state.

## Decision

Not decided yet. Current fixed lookback works and is simple. The streaming approach is confirmed viable on Keras + JAX (lax.scan, stateful=True). Revisit when we have baseline results to compare against.
