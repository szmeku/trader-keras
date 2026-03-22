# Full Rollout Benchmark: Numba vs GPU (with policy inference)

**GPU:** NVIDIA GeForce RTX 4090  
**Date:** 2026-02-15 10:55  
**PyTorch:** 2.7.1+cu128, CUDA 12.8  
**RAM:** 504 GB  
**max_episode_steps:** 65536  
**Policy:** MLPPolicy (26→128→128→3)  
**Benchmark:** median of 5 rollouts (2 warmup)

## Results

| Data | Rows      | Approach        | num_envs | rollout_steps | Steps/sec | Time (s) | GPU peak MB |
| ---- | --------- | --------------- | -------- | ------------- | --------- | -------- | ----------- |
| 1d   | 17,659    | numba           | 64       | 2048          | 95,136    | 0.022    | 22          |
| 1d   | 17,659    | gpu             | 64       | 2048          | 51,001    | 0.040    | 22          |
| 1d   | 17,659    | gpu+compile     | 64       | 2048          | 57,410    | 0.036    | 118         |
| 1d   | 17,659    | gpu+compile_env | 64       | 2048          | 104,754   | 0.020    | 118         |
| 1d   | 17,659    | numba           | 256      | 2048          | 407,748   | 0.005    | 118         |
| 1d   | 17,659    | gpu             | 256      | 2048          | 215,320   | 0.010    | 118         |
| 1d   | 17,659    | gpu+compile     | 256      | 2048          | 214,418   | 0.010    | 118         |
| 1d   | 17,659    | gpu+compile_env | 256      | 2048          | 355,022   | 0.006    | 118         |
| 1d   | 17,659    | numba           | 1024     | 2048          | 160,344   | 0.013    | 120         |
| 1d   | 17,659    | gpu             | 1024     | 2048          | 695,488   | 0.003    | 120         |
| 1d   | 17,659    | gpu+compile     | 1024     | 2048          | 734,233   | 0.003    | 120         |
| 1d   | 17,659    | gpu+compile_env | 1024     | 2048          | 1,138,175 | 0.002    | 120         |
| 1d   | 17,659    | numba           | 2048     | 4096          | 331,232   | 0.012    | 124         |
| 1d   | 17,659    | gpu             | 2048     | 4096          | 1,341,938 | 0.003    | 124         |
| 1d   | 17,659    | gpu+compile     | 2048     | 4096          | 1,474,643 | 0.003    | 120         |
| 1d   | 17,659    | gpu+compile_env | 2048     | 4096          | 2,286,101 | 0.002    | 120         |
| 1d   | 17,659    | numba           | 4096     | 4096          | 467,934   | 0.009    | 122         |
| 1d   | 17,659    | gpu             | 4096     | 4096          | 2,671,173 | 0.002    | 122         |
| 1d   | 17,659    | gpu+compile     | 4096     | 4096          | 2,594,137 | 0.002    | 122         |
| 1d   | 17,659    | gpu+compile_env | 4096     | 4096          | 3,839,783 | 0.001    | 122         |
| 1d   | 17,659    | numba           | 8192     | 8192          | 979,775   | 0.008    | 124         |
| 1d   | 17,659    | gpu             | 8192     | 8192          | 4,610,902 | 0.002    | 144         |
| 1d   | 17,659    | gpu+compile     | 8192     | 8192          | 5,100,281 | 0.002    | 124         |
| 1d   | 17,659    | gpu+compile_env | 8192     | 8192          | 7,603,511 | 0.001    | 124         |
| 3mo  | 2,192,421 | numba           | 64       | 2048          | 116,404   | 0.018    | 120         |
| 3mo  | 2,192,421 | gpu             | 64       | 2048          | 58,435    | 0.035    | 314         |
| 3mo  | 2,192,421 | gpu+compile     | 64       | 2048          | 58,452    | 0.035    | 294         |
| 3mo  | 2,192,421 | gpu+compile_env | 64       | 2048          | 97,386    | 0.021    | 294         |
| 3mo  | 2,192,421 | numba           | 256      | 2048          | 414,617   | 0.005    | 120         |
| 3mo  | 2,192,421 | gpu             | 256      | 2048          | 222,962   | 0.009    | 314         |
| 3mo  | 2,192,421 | gpu+compile     | 256      | 2048          | 219,021   | 0.009    | 314         |
| 3mo  | 2,192,421 | gpu+compile_env | 256      | 2048          | 350,381   | 0.006    | 314         |
| 3mo  | 2,192,421 | numba           | 1024     | 2048          | 160,680   | 0.013    | 122         |
| 3mo  | 2,192,421 | gpu             | 1024     | 2048          | 716,846   | 0.003    | 316         |
| 3mo  | 2,192,421 | gpu+compile     | 1024     | 2048          | 736,798   | 0.003    | 314         |
| 3mo  | 2,192,421 | gpu+compile_env | 1024     | 2048          | 1,115,127 | 0.002    | 314         |
| 3mo  | 2,192,421 | numba           | 2048     | 4096          | 255,884   | 0.016    | 124         |
| 3mo  | 2,192,421 | gpu             | 2048     | 4096          | 1,388,094 | 0.003    | 318         |
| 3mo  | 2,192,421 | gpu+compile     | 2048     | 4096          | 1,464,663 | 0.003    | 314         |
| 3mo  | 2,192,421 | gpu+compile_env | 2048     | 4096          | 2,138,590 | 0.002    | 314         |
| 3mo  | 2,192,421 | numba           | 4096     | 4096          | 481,970   | 0.008    | 122         |
| 3mo  | 2,192,421 | gpu             | 4096     | 4096          | 2,726,976 | 0.002    | 316         |
| 3mo  | 2,192,421 | gpu+compile     | 4096     | 4096          | 2,639,390 | 0.002    | 316         |
| 3mo  | 2,192,421 | gpu+compile_env | 4096     | 4096          | 3,782,552 | 0.001    | 316         |
| 3mo  | 2,192,421 | numba           | 8192     | 8192          | 1,063,598 | 0.008    | 124         |
| 3mo  | 2,192,421 | gpu             | 8192     | 8192          | 5,471,178 | 0.001    | 318         |
| 3mo  | 2,192,421 | gpu+compile     | 8192     | 8192          | 5,223,104 | 0.002    | 318         |
| 3mo  | 2,192,421 | gpu+compile_env | 8192     | 8192          | 7,537,205 | 0.001    | 318         |
| 2yr  | 7,843,125 | numba           | 64       | 2048          | 115,812   | 0.018    | 120         |
| 2yr  | 7,843,125 | gpu             | 64       | 2048          | 58,186    | 0.035    | 810         |
| 2yr  | 7,843,125 | gpu+compile     | 64       | 2048          | 57,920    | 0.035    | 810         |
| 2yr  | 7,843,125 | gpu+compile_env | 64       | 2048          | 97,021    | 0.021    | 810         |
| 2yr  | 7,843,125 | numba           | 256      | 2048          | 414,753   | 0.005    | 120         |
| 2yr  | 7,843,125 | gpu             | 256      | 2048          | 221,452   | 0.009    | 810         |
| 2yr  | 7,843,125 | gpu+compile     | 256      | 2048          | 215,302   | 0.010    | 810         |
| 2yr  | 7,843,125 | gpu+compile_env | 256      | 2048          | 346,037   | 0.006    | 810         |
| 2yr  | 7,843,125 | numba           | 1024     | 2048          | 164,675   | 0.012    | 122         |
| 2yr  | 7,843,125 | gpu             | 1024     | 2048          | 702,621   | 0.003    | 812         |
| 2yr  | 7,843,125 | gpu+compile     | 1024     | 2048          | 716,702   | 0.003    | 810         |
| 2yr  | 7,843,125 | gpu+compile_env | 1024     | 2048          | 1,055,731 | 0.002    | 810         |
| 2yr  | 7,843,125 | numba           | 2048     | 4096          | 256,676   | 0.016    | 124         |
| 2yr  | 7,843,125 | gpu             | 2048     | 4096          | 1,346,948 | 0.003    | 814         |
| 2yr  | 7,843,125 | gpu+compile     | 2048     | 4096          | 1,448,360 | 0.003    | 810         |
| 2yr  | 7,843,125 | gpu+compile_env | 2048     | 4096          | 2,128,346 | 0.002    | 810         |
| 2yr  | 7,843,125 | numba           | 4096     | 4096          | 502,071   | 0.008    | 122         |
| 2yr  | 7,843,125 | gpu             | 4096     | 4096          | 2,653,126 | 0.002    | 812         |
| 2yr  | 7,843,125 | gpu+compile     | 4096     | 4096          | 2,530,236 | 0.002    | 812         |
| 2yr  | 7,843,125 | gpu+compile_env | 4096     | 4096          | 3,556,955 | 0.001    | 812         |
| 2yr  | 7,843,125 | numba           | 8192     | 8192          | 981,431   | 0.008    | 124         |
| 2yr  | 7,843,125 | gpu             | 8192     | 8192          | 5,359,748 | 0.002    | 814         |
| 2yr  | 7,843,125 | gpu+compile     | 8192     | 8192          | 5,081,419 | 0.002    | 814         |
| 2yr  | 7,843,125 | gpu+compile_env | 8192     | 8192          | 7,076,356 | 0.001    | 814         |

## Speedup vs Numba baseline

| Data | num_envs | Numba             | GPU               | GPU+compile       | GPU+compile_env   | Best speedup |
| ---- | -------- | ----------------- | ----------------- | ----------------- | ----------------- | ------------ |
| 1d   | 64       | 95,136 (1.00x)    | 51,001 (0.54x)    | 57,410 (0.60x)    | 104,754 (1.10x)   | **1.10x**    |
| 1d   | 256      | 407,748 (1.00x)   | 215,320 (0.53x)   | 214,418 (0.53x)   | 355,022 (0.87x)   | **1.00x**    |
| 1d   | 1024     | 160,344 (1.00x)   | 695,488 (4.34x)   | 734,233 (4.58x)   | 1,138,175 (7.10x) | **7.10x**    |
| 1d   | 2048     | 331,232 (1.00x)   | 1,341,938 (4.05x) | 1,474,643 (4.45x) | 2,286,101 (6.90x) | **6.90x**    |
| 1d   | 4096     | 467,934 (1.00x)   | 2,671,173 (5.71x) | 2,594,137 (5.54x) | 3,839,783 (8.21x) | **8.21x**    |
| 1d   | 8192     | 979,775 (1.00x)   | 4,610,902 (4.71x) | 5,100,281 (5.21x) | 7,603,511 (7.76x) | **7.76x**    |
| 2yr  | 64       | 115,812 (1.00x)   | 58,186 (0.50x)    | 57,920 (0.50x)    | 97,021 (0.84x)    | **1.00x**    |
| 2yr  | 256      | 414,753 (1.00x)   | 221,452 (0.53x)   | 215,302 (0.52x)   | 346,037 (0.83x)   | **1.00x**    |
| 2yr  | 1024     | 164,675 (1.00x)   | 702,621 (4.27x)   | 716,702 (4.35x)   | 1,055,731 (6.41x) | **6.41x**    |
| 2yr  | 2048     | 256,676 (1.00x)   | 1,346,948 (5.25x) | 1,448,360 (5.64x) | 2,128,346 (8.29x) | **8.29x**    |
| 2yr  | 4096     | 502,071 (1.00x)   | 2,653,126 (5.28x) | 2,530,236 (5.04x) | 3,556,955 (7.08x) | **7.08x**    |
| 2yr  | 8192     | 981,431 (1.00x)   | 5,359,748 (5.46x) | 5,081,419 (5.18x) | 7,076,356 (7.21x) | **7.21x**    |
| 3mo  | 64       | 116,404 (1.00x)   | 58,435 (0.50x)    | 58,452 (0.50x)    | 97,386 (0.84x)    | **1.00x**    |
| 3mo  | 256      | 414,617 (1.00x)   | 222,962 (0.54x)   | 219,021 (0.53x)   | 350,381 (0.85x)   | **1.00x**    |
| 3mo  | 1024     | 160,680 (1.00x)   | 716,846 (4.46x)   | 736,798 (4.59x)   | 1,115,127 (6.94x) | **6.94x**    |
| 3mo  | 2048     | 255,884 (1.00x)   | 1,388,094 (5.42x) | 1,464,663 (5.72x) | 2,138,590 (8.36x) | **8.36x**    |
| 3mo  | 4096     | 481,970 (1.00x)   | 2,726,976 (5.66x) | 2,639,390 (5.48x) | 3,782,552 (7.85x) | **7.85x**    |
| 3mo  | 8192     | 1,063,598 (1.00x) | 5,471,178 (5.14x) | 5,223,104 (4.91x) | 7,537,205 (7.09x) | **7.09x**    |

## Analysis

### The crossover point: num_envs >= 1024

| num_envs | Winner | Speedup range |
|----------|--------|---------------|
| 64 | Numba (2x) | GPU is 0.5x (half as fast) |
| 256 | Numba (1.2-1.9x) | GPU catches up but still loses |
| **1024** | **GPU (4.3-7.1x)** | Crossover — GPU starts winning |
| 2048 | GPU (5.2-8.4x) | Clear GPU advantage |
| 4096 | GPU (5.3-8.2x) | Continues scaling |
| 8192 | GPU (5.1-7.8x) | Peak throughput: 7.6M steps/s |

### Why this differs from the raw env.step() benchmark

The raw benchmark (see [[bench_env_report]]) showed Numba winning 17-200x. With policy inference included:
- **Numba bottleneck shifts** from env stepping to CPU↔GPU data transfer for each policy forward pass
- **GPU env eliminates transfers** — obs and actions stay on GPU, policy inference is zero-copy
- At low num_envs, Python loop overhead dominates both paths; at high num_envs, the transfer cost becomes the bottleneck for Numba

### torch.compile effects

| Approach | Effect |
|----------|--------|
| `gpu+compile` (policy only) | **Marginal** (+5-10%), not worth the 96MB extra VRAM for compilation cache |
| `gpu+compile_env` (env.step compiled) | **Significant** (+50-70% over base GPU), env.step ops fused into fewer kernels |

Note: `torch.compile` on env.step falls back from CUDAGraphs due to in-place mutations (`self.step_idx += active_long`). Even without CUDAGraphs, the op fusion still helps. Eliminating the mutations could unlock CUDAGraph capture for even more speedup.

### Memory (2yr data, 7.8M rows)

| Approach | GPU peak MB |
|----------|-------------|
| Numba | 120-124 (model + rollout buffer only) |
| GPU | 810-814 (data on GPU + model + buffer) |
| GPU+compile | 810 (same + compile cache) |

Data VRAM: ~690MB for 7.8M rows × 23 features × 4 bytes.
On GTX 1050 Ti (4GB): 690MB data + 120MB model/buffer = 810MB — fits, but leaves only ~3.2GB for compile cache and other overhead.

### Data size independence

Throughput is nearly identical across data sizes (1d, 3mo, 2yr) for the same num_envs. This makes sense:
- With `max_episode_steps=65536`, all envs see the same window size regardless of total data
- The only difference is VRAM usage for storing the data tensor

### Recommendations for training config

| Hardware | Recommended config |
|----------|-------------------|
| **RTX 4090 (24GB)** | `env: gpu`, `num_envs: 4096-8192`, consider `torch.compile` on env.step for +60% |
| **GTX 1050 Ti (4GB)** | `env: numba`, `num_envs: 256` — GPU env eats too much VRAM for the 2yr dataset |
| **Any GPU, small data** | `env: numba`, `num_envs: 256` — Numba is fastest at low env counts |
| **VastAI training** | `env: gpu`, `num_envs: 4096`, `rollout_steps: 8192` — maximize GPU utilization |
