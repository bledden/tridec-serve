# Known limit: tridec BP triton kernel fails at surface d≥15 (issue draft)

**Status:** **FIXED on tridec master** — `46c63a4` "lift surface d>=15 BP
ceiling (1-D grid)" (2026-06-17) — but in **no PyPI release yet** (latest:
0.2.1, which has the ceiling). This receipt documents the 0.2.1 behavior the
benchmark measured; the d≥15 re-measure is queued for the next release.

## Summary
`RelayBpDecoder(algorithm="bp").decode_batch(...)` raises a Triton launch error at
surface code distance **d=15** (rotated_memory_z, rounds=15; **3360 detectors**),
on **both** NVIDIA/CUDA and AMD/ROCm. d=14 (2729 detectors) is the last working
point. It is a **kernel-launch limit, not OOM** (3360 detectors is tiny; both GPUs
have ≥140 GB). matching (PyMatching, CPU) has no such ceiling.

## Failing call
`tridec/backends/bp_triton.py:356` → `_bit_update_kernel[grid_bit](...)`
The grid for the bit-update kernel at this problem size exceeds a launch limit
(`invalid argument` on kernel launch — a grid/block dimension out of range).

## Minimal repro
```python
import numpy as np, stim, tridec
sc = stim.Circuit.generated("surface_code:rotated_memory_z", distance=15, rounds=15,
        after_clifford_depolarization=0.003, after_reset_flip_probability=0.003,
        before_measure_flip_probability=0.003, before_round_data_depolarization=0.003)
det,_ = sc.compile_detector_sampler(seed=0).sample(16, separate_observables=True)
dec = tridec.from_dem(sc.detector_error_model(decompose_errors=False), algorithm="bp")
dec.decode_batch(np.ascontiguousarray(np.asarray(det, bool)))   # -> Triton Error: invalid argument
# d=14 (rounds=14, 2729 detectors) works; d=15 (3360 detectors) fails.
```

## Reproduced on (env)
- **NVIDIA H200** (RunPod): driver 570.211.01, torch 2.4.1+cu124, **triton 3.0.0** →
  `RuntimeError: Triton Error [CUDA]: invalid argument`
- **AMD MI300X VF** (DigitalOcean): torch 2.9.0.dev+rocm7.0, **triton 3.4.0+rocm7.0** →
  `RuntimeError: Triton Error [HIP]: Code: 1, Message: invalid argument`

Same failure, same distance, two independent triton backends → it's a tridec
kernel-side grid-sizing assumption, not a vendor/driver bug.

## Likely fix direction
Cap/tile the `_bit_update_kernel` grid (the bit-update grid scales with detectors ×
batch; at d=15 it overflows the max grid dimension) — chunk the grid or 2D-tile it
so the launch stays within limits. Until then: **surface d≤14 is the supported
range for the GPU BP lane on tridec ≤0.2.1** (matching/CPU is unbounded).
Upstream fixed it exactly this way (1-D grid tiling, `46c63a4`).
