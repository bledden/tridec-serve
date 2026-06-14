# QEC decode-serving benchmark (v0)

A **vendor-portable, decoder-agnostic** benchmark for quantum error-correction
decoders that adds the axis existing benchmarks miss.

## The gap it fills

QEC decoder benchmarks today (sinter, decoder zoos, the CUDA-Q QEC numbers)
report **accuracy** — logical error rate at a fixed shot count. None reports
**serving capacity**: how many logical-qubit syndrome streams one accelerator
can decode in real time at a latency SLA. As machines scale to many logical
qubits, *that* is the operational question — decoding becomes a serving problem
(see `../README.md`). This benchmark measures **both axes** per
(decoder × code × platform):

- **accuracy** — LER ± 95% Wilson CI (the standard axis), and
- **serving** — max sustained logical-qubits/GPU (bounded backlog) + capacity at
  p99 SLA budgets {100, 250, 500 ms} + peak throughput (the **new** axis).

See `benchmark_pareto_metal.png` — the accuracy-vs-serving-capacity tradeoff.

## Decoder-agnostic by construction

Any decoder plugs in via a ~3-line adapter exposing `decode_batch(dets) ->
preds`. Reference entries: tridec Relay-BP (accurate) and min-sum BP (fast).
External entry shipped to prove it: the `relay_bp` Rust oracle (CPU). Obvious
next adapters: PyMatching/MWPM, `ldpc` BP-OSD, NVIDIA CUDA-Q QEC decoders, and
neural decoders (AlphaQubit-style).

## First results — `bb72_r6_p0.003_Z`, Metal (M4 Max)

| decoder | LER | max sustained q/GPU | @p99 250 ms | peak syn/s |
|---|---|---|---|---|
| tridec **Relay-BP** (accurate) | 1.95% | ~8 | 8 | ~11k |
| tridec **min-sum BP** (fast) | 8.40% | ~32 | 16 | ~32k |
| `relay_bp` Rust oracle (CPU, accurate) | 1.75% | **0** | 0 | ~0.9k |

Reads: accurate decoders cost serving capacity; the **CPU oracle can't sustain
even one logical qubit at a 1 ms round** (~0.9k syn/s < 1k needed) → real-time
serving *requires* the GPU. **Vendor-portable:** the same serving harness on
H200 / MI300X measured min-sum BP at **~1024 logical qubits/GPU at single-digit-
ms p99** (`../decode_serving_knee.png`) — so the capacity column scales ~30× on
datacenter GPUs, portably.

## Why this is *more* relevant given NVIDIA's CUDA-Q QEC rollout

NVIDIA shipping GPU QEC decoding (CUDA-Q QEC, real-time NVQLink) **validates the
thesis** — but their numbers are CUDA-only, their protocol, their decoders.
There is no **vendor-neutral** serving benchmark. That's the MLPerf role: the
neutral standard exists *because* one vendor dominates — it's the counterweight
everyone else uses to show competitiveness. The **AMD angle is the strongest
case**: AMD needs a way to show MI300X/ROCm is competitive for QEC decoding in a
domain NVIDIA is claiming CUDA-only, and NVIDIA structurally won't build that. A
benchmark where **MI300X is first-class** (it already is here — relay/BP run on
ROCm; we measured the 1024-qubit knee on MI300X) is exactly the artifact to
bring to AMD.

**Honest scope:** "the standard nobody's defined" is aspirational — a benchmark
becomes a *standard* only if adopted. The realistic value is a clean, novel,
vendor-neutral serving benchmark = a portfolio artifact + a natural
AMD-collaboration vehicle + a direct demonstration of GPU × serving × QEC.

## Run

```bash
python qecserve_bench.py     # accuracy + serving for tridec relay/bp + relay_bp -> results_<plat>.json
python make_pareto.py        # accuracy-vs-capacity figure -> benchmark_pareto_metal.png
```
(`pip install "tridec[torch,decoders]"` + a GPU, or the Metal env.)

## Limitations / next (v0)

- **Prototype scheduler** (`../serve.py`, Python-threaded) → the serving numbers
  are a pessimistic lower bound; a production server (out-of-process worker,
  CUDA-graph bucket shapes) would do better.
- **Slow-decoder knee is coarse.** A high-latency decoder produces few batches
  per window, so its exact sustained-K is noisy ±1 grid point (4 s window used);
  report it as a range / with CIs for precision. The *relative* picture and the
  datacenter capacities are robust.
- **Single code so far** (`bb72`). Next: surface-code distances (with PyMatching/
  MWPM), mixed code-distance fleets, more decoders, and running this unified
  harness cross-vendor on H200/MI300X (one pod session — the serving half is
  already cross-vendor-validated).
