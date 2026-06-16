# Why MI300X leads on the mixed-family fleet — surface-d5 BP profile

Root-cause for the v4 "MI300X serves 2× the H200 on mixed-family" result, with the
confounds stated. surface-d5 min-sum BP `decode_batch`, latency vs batch size,
20-call median, `torch.cuda.synchronize()` around each. (`profile_bp.py`.)

| batch | H200 (cuda / triton 3.0.0) | MI300X-VF (rocm7 / triton 3.4.0) |
|---|---|---|
| 1 | **1.08 ms** · 924 syn/s | 2.19 ms · 456 syn/s |
| 64 | **1.07 ms** · 60.0k | 2.12 ms · 30.2k |
| 256 | **1.06 ms** · 241k | 2.02 ms · 127k |
| 1024 | **1.76 ms** · 583k | 2.33 ms · 439k |
| 4096 | 6.20 ms · 661k | **4.67 ms · 878k** |
| 8192 | 12.0 ms · 682k (1.47 µs/syn) | **9.48 ms · 864k (1.16 µs/syn)** |

## Decomposition
- **H200 wins small batch:** ~2× lower per-call/launch overhead (1.08 vs 2.19 ms at
  B=1; flat to B≈256). Lower fixed cost per kernel launch.
- **MI300X wins large batch:** ~1.27× higher throughput at B=8192 (1.16 vs 1.47
  µs/syn). The crossover is ~B=1024–4096.
- **The fleet/serving regime is large-batch** (continuous batching fills to the
  bucket), so the MI300X throughput edge dominates → it leads on mixed-family.

## Confounds (this is a STACK comparison, not a hardware verdict)
1. **Different triton versions:** 3.4+rocm7 (MI300X) vs 3.0+cu124 (H200). Newer
   triton codegen could account for the large-batch edge by itself.
2. **MI300X is a VF** (virtualized partition on DigitalOcean), not bare-metal.
3. Per-stack kernel configs (`_GPU_TUNED`) differ.
A clean hardware attribution would need matched triton versions on both. The honest
claim: *on rocm7/triton-3.4 vs cuda/triton-3.0, MI300X's large-batch surface-d5 BP
throughput is ~1.3× higher; the version gap is a confound.*

## Why this is trustworthy regardless
Accuracy is **bit-identical** across the two (deterministic decoders + same x86_64
stim shots — see `benchmark_accuracy_parity.png`), so the capacity difference is
pure kernel/stack work, not a measurement artifact.
