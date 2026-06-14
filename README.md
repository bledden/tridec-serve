# tridec-serve — decode-serving prototype (LLM-serving × QEC, lane #3 spike)

**Status:** research spike / proof-of-concept (1 figure). Built on
[tridec](https://github.com/bledden/tridec). Companion to the scoping doc
`lane3_llm_serving_x_qec_scoping_2026-06-14.md`.

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

**The honest, quantified takeaway** (the scoping doc's caveat, now measured):
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

## Vendor-portable (the tridec edge)

Same code, higher throughput on datacenter GPUs (relay, fp64, this session's
spot-bench): **H200 ≈ 20.7k syndromes/s**, **MI300X ≈ 14.2k/s** vs Metal's
13.8k/s — so the qubits-per-GPU curve shifts up proportionally on CUDA/ROCm,
**portably** (NVIDIA's CUDA-Q QEC stack is CUDA-locked; tridec is not). Full
cross-vendor curves are the obvious next measurement.

## Run

```bash
python measure_latency.py     # measure L(B) on the local accelerator -> latency_metal.json
python make_figure.py         # analytical model + figure -> decode_serving_metal.png
python bench_serve.py         # LIVE scheduler load sweep -> serve_latency_vs_load.png
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
- **Heterogeneous load not yet modeled.** The biggest expected win is multiplexing
  *idle vs active* patches (continuous batching's bread and butter); the sweep
  here is a uniform stream.
- **Next:** (a) cross-vendor curves + live sweep on H200/MI300X (one pod session,
  same code); (b) heterogeneous idle/active arrival mix; (c) the QEC
  decode-serving benchmark (scoping idea #4 — the standard nobody's defined).
