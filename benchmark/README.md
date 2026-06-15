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
**Four decoder families shipped** across two codes — tridec Relay-BP & min-sum
BP, plus externals: `relay_bp` Rust oracle, `ldpc` BP-OSD, and **PyMatching/MWPM**
(a *matching* decoder — a different family entirely). Next: NVIDIA CUDA-Q QEC
decoders, neural (AlphaQubit-style).

## Results — code × decoder matrix, Metal (M4 Max) — `benchmark_pareto.png`

| code | decoder | LER | max q/GPU | peak syn/s |
|---|---|---|---|---|
| **BB [[72,12,6]] qLDPC** | tridec Relay-BP | 1.95% | ~8 | ~11k |
| | tridec min-sum BP | 8.40% | ~32 | ~34k |
| | `ldpc` BP-OSD (CPU) | **1.55%** | 0 | ~0.8k |
| | `relay_bp` oracle (CPU) | 1.75% | 0 | ~0.6k |
| **surface d=5** | tridec Relay-BP | 1.30% | 0 | ~0.7k |
| | tridec min-sum BP | 7.65% | ~16 | ~16k |
| | **PyMatching MWPM** | **0.10%** | **~1536** | **~1.0M** |

The matrix shows the **right tool per code**, on both axes:
- **Surface code → matching dominates.** PyMatching wins *both* accuracy (0.10%,
  ~13–80× better than BP-family) *and* serving (~1536 q/GPU at ~1M syn/s). On
  surface codes the specialized decoder beats the general BP decoders outright.
- **BB qLDPC → matching N/A; it's a BP-family + capacity story.** BP-OSD is most
  accurate (1.55%) but **CPU-bound — can't sustain even one logical qubit at a
  1 ms round** (~0.8k syn/s < 1k needed). The tridec **GPU** decoders are what
  provide real-time serving capacity (relay 8 q / BP 32 q). Accurate-CPU vs
  serving-capable-GPU is the qLDPC tradeoff.
- **Vendor-portable:** the same serving harness on H200 / MI300X measured min-sum
  BP at **~1024 logical qubits/GPU at single-digit-ms p99**
  (`../decode_serving_knee.png`) — the capacity column scales ~30× on datacenter
  GPUs, portably.

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
python qecserve_bench.py     # code x decoder matrix: accuracy + serving -> results_<plat>.json
python make_pareto.py        # accuracy-vs-capacity figure -> benchmark_pareto.png
```
(`pip install "tridec[torch,decoders]"` + a GPU/Metal; `pymatching` for the
surface-code matching entry.)

## Limitations / next

- **Prototype scheduler** (`../serve.py`, Python-threaded) → serving numbers are
  a pessimistic lower bound; a production server (out-of-process worker,
  CUDA-graph bucket shapes) would do better.
- **Slow-decoder knee is coarse.** A high-latency decoder produces few batches
  per window, so its exact sustained-K is noisy ±1 grid point (4 s window used);
  CIs / longer windows for precision. The *relative* picture is robust.
- ✅ **Two codes + 4 decoder families** (BB qLDPC, surface d=5) — done.
- **Next:** NVIDIA **CUDA-Q QEC** decoder adapters (the head-to-head that makes
  the vendor-neutral point land), neural decoders, more surface distances +
  mixed code-distance fleets, CIs on the knee, and running this unified harness
  **cross-vendor on H200/MI300X** (one pod session — the serving half is already
  cross-vendor-validated).
