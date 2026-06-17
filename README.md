# tridec-serve — a decode-serving benchmark for QEC (LLM-serving × QEC)

**Status:** research benchmark. Built on
[tridec](https://github.com/bledden/tridec). This top-level README is the
original first-order model + Metal spike; the full **cross-vendor benchmark**
(Apple Metal / AMD MI300X / NVIDIA H200, six decoder families, incl. a
head-to-head vs NVIDIA's CUDA-Q `nv-qldpc`) lives in
[`benchmark/`](benchmark/README.md).

## Thesis

At scale, QEC decoding is a **serving** problem: a fault-tolerant machine has
many logical qubits, each emitting a syndrome stream at the QEC clock, and one
accelerator must decode all of them within a per-stream **reaction-latency
SLA** (so the correction is ready before backlog compounds). That's the problem
the LLM-serving stack solved — **continuous batching**: multiplex many streams
onto the device, batch arrivals to maximize throughput, bound per-request
latency. So the question isn't "how fast is one decode" — it's **"how many
logical qubits can one GPU serve at a given SLA."**

## Model (first-order, pipelined continuous batching)

- Each logical qubit emits 1 syndrome per `T_round` (the QEC clock).
- The scheduler accumulates arrivals into a batch `B` and decodes; while batch
  *i* decodes, batch *i+1* fills → steady-state **reaction latency ≈ 2·L(B)**,
  where `L(B)` is tridec's measured decode latency at batch `B`.
- Within an SLA, the usable batch is the largest `B` with `2·L(B) ≤ SLA`, and
  one GPU sustains **`K = (B/L(B)) · T_round`** logical qubits (throughput ×
  round period).

`L(B)` is **measured empirically** from tridec (`measure_latency.py`), so the
model is grounded in the real decoder, not a guess.

## Result (Apple M4 Max / Metal — see `decode_serving_metal.png`)

Peak throughput on this box (canonical BB `[[72,12,6]]` cell):
- **Relay-BP** (accurate, the v0.2.1 default): **13.8k syndromes/s**
- **min-sum BP** (fast, less accurate): **46.2k syndromes/s**

→ logical qubits served per GPU:

| QEC clock | Relay-BP | min-sum BP |
|---|---|---|
| **superconducting** (1 µs/round) | **0.014** | **0.046** | 
| **neutral-atom / trapped-ion** (1 ms/round) | **~14** | **~46** |

**The honest, quantified takeaway** (now measured):
- At a **1 µs superconducting round**, even max throughput serves **<1 qubit/GPU
  in real time** → the sub-µs inner loop is **FPGA/ASIC territory**, not GPU.
  GPU decode-serving is *not* the play there.
- At **ms-scale clocks** (high-rate qLDPC on neutral-atom / trapped-ion — where
  *many* logical qubits are encoded), **one GPU serves 10s of logical qubits**,
  and the count rises with the SLA as continuous batching unlocks bigger batches
  (the step-up in the figure). **This is the GPU decode-serving regime** — and
  it's exactly where the many-logical-qubit hardware is heading.

## Live scheduler — measured (`serve.py` → `serve_latency_vs_load.png`)

The above is the analytical model. `serve.py` is the **real system**: a running
continuous-batching scheduler (worker thread doing actual tridec decodes against
a live arrival process), so we measure the **reaction-latency tail** (p50/p99/
p99.9) and whether the GPU keeps up (**bounded backlog**), not a formula.
`bench_serve.py` drives increasing offered load and finds the knee.

Measured on Metal (M4 Max, 1 ms round) — max **sustained** logical qubits/GPU
(backlog stays bounded), and at a p99 reaction-latency SLA:

| decoder | max sustained | p99 ≤ 250 ms | p99 ≤ 500 ms |
|---|---|---|---|
| **Relay-BP** (accurate, default) | **~12** | 8 | 8 |
| **min-sum BP** (fast) | **~32** | 32 | 32 |

Measured (12 / 32) sits a bit under the analytical model (14 / 46) — the honest
real-system overhead (Python-threaded worker, bucket-padding waste, queueing).
The shape matches: throughput-bound, loose SLA, 10s of qubits.

**A finding the live run surfaced (and a fix that's itself an LLM-serving
technique):** naive continuous batching produces a *new batch size almost every
cycle*, and Triton **recompiles per new shape** — those compile stalls dominated
the latency tail (p99 jumped to ~1.7 s). The fix is **batch-size bucketing**
(pad each batch up to a fixed set of shapes → compile once), exactly what LLM
servers do (bucketing / CUDA graphs). With bucketing the tail collapses to the
steady-state decode latency. (`DecodeServer(buckets=...)`.)

## Vendor-portable — measured on 3 platforms (`decode_serving_xvendor.png`)

Same code (the v0.2.1 default decoders), measured on Metal (M4 Max), NVIDIA H200
(CUDA), and AMD MI300X (ROCm). Logical qubits served per GPU at a 1 ms round
(throughput model, with the live scheduler's max-sustained in parens):

| platform | Relay-BP (accurate) | min-sum BP (fast) |
|---|---|---|
| Metal (M4 Max) | **14** (live 12) | 46 (live 32) |
| **NVIDIA H200** (CUDA) | **18** (live 16) | **1008** (live ≥256\*) |
| **AMD MI300X** (ROCm) | **11** (live 8) | **1177** (live ≥256\*) |

\* the live BP runs are capped at the sweep's max offered load (K=256), not BP's
knee — the Python load generator, not the decoder, is the limit there.

Two honest findings:
- **Relay-BP (the accurate default) is compute-bound and portable** — ~11–18
  qubits/GPU everywhere. The relay schedule itself is the cost; the GPU is
  almost incidental. That's the realistic accurate-decoding capacity per GPU.
- **min-sum BP is launch-bound on Metal (~46k syndromes/s) but ~1.0–1.2M/s on
  the datacenter GPUs** (their kernel launches are ~free), so a single H200/
  MI300X serves **hundreds of logical qubits** with the fast decoder at
  **single-digit-ms p99** — but BP trades accuracy for that. The
  accuracy-vs-capacity choice is the real knob.

Vendor-portable throughout (NVIDIA's CUDA-Q QEC stack is CUDA-locked; tridec is
not). Per-platform L(B) curves in `latency_{metal,h200,mi300x}.json`; live runs
in `serve_measured_{h200,mi300x}.json`.

## Scaling to ~1000 qubits live (`decode_serving_knee.png`)

With a **round-batched load generator** (the producer enqueues one array per
round, not one syndrome at a time, so it's O(rounds/s) not O(syndromes/s)) the
live scheduler reaches min-sum BP's real knee on datacenter GPUs:

| platform | min-sum BP (fast) | Relay-BP (accurate) |
|---|---|---|
| **H200** | **1024 qubits @ p99 13 ms** (~995k syndromes/s) | 16 @ p99 339 ms |
| **MI300X** | **1024 qubits @ p99 7 ms** (~977k syndromes/s) | 8 @ p99 565 ms |

So **one datacenter GPU serves ~1024 logical qubits live at single-digit-ms
p99** with the fast decoder (overloads at K=1536). Relay-BP, compute-bound,
holds ~8–16 at 100s-of-ms. (`bench_hetero.py`.)

**Heterogeneous load — honest negative.** Making 90% of qubits idle
(error-free) barely helped: **none for min-sum BP** (fixed 30 iterations — idle
costs the same), and **small/inconsistent for Relay-BP** (MI300X 8→16, H200
16→16). Relay's per-shot early-exit only saves the relay-leg *tail*; the fixed
`pre_iter=80` first leg runs over every shot regardless, so it dominates and
idle shots aren't much cheaper. A serving win from heterogeneity would need a
decoder whose *bulk* cost is per-shot-adaptive, not just its tail.

## Run

```bash
python measure_latency.py     # measure L(B) on the local accelerator -> latency_<plat>.json
python make_figure.py         # analytical model + figure -> decode_serving_metal.png
python bench_serve.py         # LIVE scheduler load sweep -> serve_latency_vs_load.png
python make_xvendor.py        # combined cross-vendor figure -> decode_serving_xvendor.png
python bench_hetero.py        # round-batched load gen, high-K + heterogeneous -> serve_hetero.json
python make_knee.py           # the ~1024-qubit knee figure -> decode_serving_knee.png
```
(needs tridec installed for the local backend: `pip install "tridec[torch]"` +
a GPU, or the experimental Metal env.)

## Caveats / next steps

- **Python-threaded prototype.** `serve.py`'s worker is a Python thread; a
  production server would run the decode worker out-of-process / in C++ with
  CUDA-graph-captured bucket shapes. The measured tail is therefore a (pessimistic)
  lower bound on what's achievable.
- **Single-cell `L(B)`.** Uses the BB cell; real serving mixes code distances /
  families — `L(B)` is per-decoder-config.
- ✅ **Faster (round-batched) load generator — done** (`serve.py`); it reaches
  BP's ~1024-qubit knee live on datacenter GPUs (`decode_serving_knee.png`).
- ✅ **Heterogeneous idle/active mix — done, honest negative** (above): little
  benefit at this config (BP fixed-iter; relay's `pre_iter` floor). Documented,
  not pursued further unless a per-shot-adaptive-bulk decoder motivates it.
- **Next:** (a) ✅ cross-vendor curves + live sweep on H200/MI300X — *done*
  (`decode_serving_xvendor.png`); (b) ✅ faster load gen + heterogeneous —
  *done*; (c) mixed code-distance fleets (the real heterogeneity lever); (d) the
  QEC decode-serving benchmark — the standard nobody's defined.
