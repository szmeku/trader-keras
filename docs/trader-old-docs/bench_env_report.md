# Env Backend Benchmark: Numba vs GPU

**GPU:** NVIDIA GeForce RTX 4090  
**Date:** 2026-02-15 10:43  
**PyTorch:** 2.7.1+cu128, CUDA 12.8  
**RAM:** 504 GB  
**Warmup:** 50 steps, **Bench:** 500 steps

## Raw Results

| Data | Rows | num_envs | Backend | Steps/sec | Time (s) | RAM +MB | GPU +MB | GPU peak MB |
|------|------|----------|---------|-----------|----------|---------|---------|-------------|
| 1h | 835 | 16 | numba | 5,226,661 | 0.00 | 9 | 0 | — |
| 1h | 835 | 16 | gpu | 25,181 | 0.32 | 401 | 0 | 2 |
| 1h | 835 | 64 | numba | 17,961,303 | 0.00 | 0 | 0 | — |
| 1h | 835 | 64 | gpu | 100,873 | 0.32 | 0 | 0 | 2 |
| 1h | 835 | 256 | numba | 43,761,434 | 0.00 | 0 | 0 | — |
| 1h | 835 | 256 | gpu | 399,816 | 0.32 | 0 | 0 | 2 |
| 1h | 835 | 1024 | numba | 28,927,037 | 0.02 | 10 | 0 | — |
| 1h | 835 | 1024 | gpu | 1,591,556 | 0.32 | 0 | 0 | 2 |
| 1d | 17,659 | 16 | numba | 5,202,552 | 0.00 | 0 | 0 | — |
| 1d | 17,659 | 16 | gpu | 25,156 | 0.32 | 0 | 2 | 22 |
| 1d | 17,659 | 64 | numba | 17,881,150 | 0.00 | 0 | 0 | — |
| 1d | 17,659 | 64 | gpu | 100,776 | 0.32 | 0 | 2 | 22 |
| 1d | 17,659 | 256 | numba | 44,601,229 | 0.00 | 0 | 0 | — |
| 1d | 17,659 | 256 | gpu | 398,567 | 0.32 | 0 | 2 | 22 |
| 1d | 17,659 | 1024 | numba | 31,820,016 | 0.02 | 0 | 0 | — |
| 1d | 17,659 | 1024 | gpu | 1,589,018 | 0.32 | 0 | 2 | 22 |
| 19d | 294,404 | 16 | numba | 5,198,589 | 0.00 | 26 | 0 | — |
| 19d | 294,404 | 16 | gpu | 23,468 | 0.34 | 0 | 26 | 28 |
| 19d | 294,404 | 64 | numba | 15,967,594 | 0.00 | 26 | 0 | — |
| 19d | 294,404 | 64 | gpu | 100,825 | 0.32 | 0 | 26 | 28 |
| 19d | 294,404 | 256 | numba | 22,345,120 | 0.01 | 0 | 0 | — |
| 19d | 294,404 | 256 | gpu | 395,091 | 0.32 | 0 | 26 | 28 |
| 19d | 294,404 | 1024 | numba | 28,048,997 | 0.02 | 0 | 0 | — |
| 19d | 294,404 | 1024 | gpu | 1,588,550 | 0.32 | 0 | 26 | 28 |
| 3mo | 2,192,421 | 16 | numba | 5,093,432 | 0.00 | 192 | 0 | — |
| 3mo | 2,192,421 | 16 | gpu | 25,039 | 0.32 | 0 | 192 | 196 |
| 3mo | 2,192,421 | 64 | numba | 12,740,685 | 0.00 | 192 | 0 | — |
| 3mo | 2,192,421 | 64 | gpu | 99,980 | 0.32 | 0 | 192 | 196 |
| 3mo | 2,192,421 | 256 | numba | 21,530,296 | 0.01 | 192 | 0 | — |
| 3mo | 2,192,421 | 256 | gpu | 390,336 | 0.33 | 0 | 192 | 196 |
| 3mo | 2,192,421 | 1024 | numba | 27,014,787 | 0.02 | 192 | 0 | — |
| 3mo | 2,192,421 | 1024 | gpu | 1,581,074 | 0.32 | 0 | 193 | 196 |
| 2yr | 7,843,125 | 16 | numba | 5,107,771 | 0.00 | 688 | 0 | — |
| 2yr | 7,843,125 | 16 | gpu | 25,000 | 0.32 | 0 | 688 | 692 |
| 2yr | 7,843,125 | 64 | numba | 12,042,053 | 0.00 | 688 | 0 | — |
| 2yr | 7,843,125 | 64 | gpu | 99,922 | 0.32 | 0 | 688 | 692 |
| 2yr | 7,843,125 | 256 | numba | 20,356,108 | 0.01 | 688 | 0 | — |
| 2yr | 7,843,125 | 256 | gpu | 396,759 | 0.32 | 0 | 688 | 692 |
| 2yr | 7,843,125 | 1024 | numba | 28,967,506 | 0.02 | 688 | 0 | — |
| 2yr | 7,843,125 | 1024 | gpu | 1,580,789 | 0.32 | 0 | 688 | 692 |

## GPU Speedup over Numba

| Data | num_envs | Numba steps/s | GPU steps/s | Speedup |
|------|----------|---------------|-------------|---------|
| 19d | 16 | 5,198,589 | 23,468 | **0.00x** |
| 19d | 64 | 15,967,594 | 100,825 | **0.01x** |
| 19d | 256 | 22,345,120 | 395,091 | **0.02x** |
| 19d | 1024 | 28,048,997 | 1,588,550 | **0.06x** |
| 1d | 16 | 5,202,552 | 25,156 | **0.00x** |
| 1d | 64 | 17,881,150 | 100,776 | **0.01x** |
| 1d | 256 | 44,601,229 | 398,567 | **0.01x** |
| 1d | 1024 | 31,820,016 | 1,589,018 | **0.05x** |
| 1h | 16 | 5,226,661 | 25,181 | **0.00x** |
| 1h | 64 | 17,961,303 | 100,873 | **0.01x** |
| 1h | 256 | 43,761,434 | 399,816 | **0.01x** |
| 1h | 1024 | 28,927,037 | 1,591,556 | **0.06x** |
| 2yr | 16 | 5,107,771 | 25,000 | **0.00x** |
| 2yr | 64 | 12,042,053 | 99,922 | **0.01x** |
| 2yr | 256 | 20,356,108 | 396,759 | **0.02x** |
| 2yr | 1024 | 28,967,506 | 1,580,789 | **0.05x** |
| 3mo | 16 | 5,093,432 | 25,039 | **0.00x** |
| 3mo | 64 | 12,740,685 | 99,980 | **0.01x** |
| 3mo | 256 | 21,530,296 | 390,336 | **0.02x** |
| 3mo | 1024 | 27,014,787 | 1,581,074 | **0.06x** |

## Analysis

### Key Finding: Numba env.step() is 17-200x faster than GPU env.step()

This benchmark measures **raw env stepping throughput only** (no policy inference).
The GPU env is slower at stepping because:

1. **Python loop overhead dominates**: Each of 500 `env.step()` calls goes through Python → CUDA kernel launch → synchronize. The kernels themselves are tiny (simple arithmetic on small tensors).
2. **Numba JIT compiles to native code**: The step kernel runs as a tight C loop with no Python overhead, and parallelizes across CPU cores for large num_envs.
3. **GPU kernels need more data to be efficient**: With only `num_envs` elements (16-1024), the GPU is vastly underutilized. GPUs shine at 100K+ parallel operations.

### What this benchmark does NOT measure

The GPU env's real value is **eliminating CPU↔GPU data transfers during training**:

| | Numba env | GPU env |
|---|---|---|
| env.step() | CPU (fast) | GPU (slow) |
| Obs → policy inference | CPU→GPU transfer needed | Already on GPU (zero-copy) |
| Actions ← policy | GPU→CPU transfer needed | Already on GPU (zero-copy) |

In the training loop, each rollout step requires:
- Numba: `env.step()` (CPU) → copy obs to GPU → policy forward → copy actions to CPU
- GPU: `env.step()` (GPU) → policy forward (same device, zero-copy)

The transfer overhead becomes significant with large num_envs and frequent steps.

### Memory Scaling

| Data size | Rows | GPU VRAM (MB) | Ratio |
|-----------|------|---------------|-------|
| 1h | 835 | 2 | baseline |
| 1d | 17,659 | 22 | ~11x rows, 11x VRAM |
| 19d | 294,404 | 28 | ~17x rows, 14x VRAM |
| 3mo | 2,192,421 | 196 | ~2.6K rows, 98x VRAM |
| 2yr | 7,843,125 | 692 | ~9.4K rows, 346x VRAM |

VRAM scales linearly with data rows (~88 bytes/row with 23 float32 features).
The 2yr dataset uses 692MB — well within RTX 4090's 24GB.
On GTX 1050 Ti (4GB), the 2yr dataset would leave ~3.3GB for model + rollout buffer.

### Recommendations

1. **For env stepping benchmarks**: Numba wins decisively. The Numba JIT kernels are extremely well-optimized.
2. **For end-to-end training**: Need to benchmark the full rollout loop (env + policy + transfer) to see if GPU env's zero-copy advantage outweighs its slower stepping.
3. **For GTX 1050 Ti**: GPU env saves VRAM vs having separate CPU and GPU copies, but the 2yr dataset (692MB) leaves limited room. Consider `max_rows` or `max_episode_steps` to limit data on GPU.
