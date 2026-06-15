# QEC decode-serving benchmark (v1)

A **vendor-portable, decoder-agnostic** benchmark for quantum error-correction
decoders that adds the axis existing benchmarks miss — now measured across
**three GPU vendors** (Apple Metal, AMD MI300X/ROCm, NVIDIA H200/CUDA) and
**five decoder families**, including a head-to-head against **NVIDIA's own
CUDA-Q QEC GPU decoder**.

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

See `benchmark_pareto_{metal,mi300x,h200}.png` — the accuracy-vs-serving-capacity
tradeoff per platform.

## Decoder-agnostic by construction

Any decoder plugs in via a ~3-line adapter exposing `decode_batch(dets) ->
preds`. **Five decoder families shipped** across two codes and three platforms:
tridec Relay-BP (accurate) & min-sum BP (fast), plus externals `relay_bp` Rust
oracle and `ldpc` BP-OSD, **PyMatching/MWPM** (a *matching* decoder — a different
family entirely), and **NVIDIA CUDA-Q `nv-qldpc`** (the vendor's own GPU LDPC
decoder, CUDA-only). Next: neural (AlphaQubit-style), CUDA-Q tensor-network.

## Results — code × decoder × **platform** matrix

Per-platform Pareto: `benchmark_pareto_{metal,mi300x,h200}.png`. `max q` = max
sustained logical qubits/GPU at a 1 ms round (bounded backlog); `peak` = peak
throughput (syn/s).

| code | decoder | Metal (M4 Max) | MI300X (ROCm) | H200 (CUDA) |
|---|---|---|---|---|
| **BB qLDPC** | tridec Relay-BP | 1.95% · 8q | 1.95% · 8q | 1.95% · **16q** |
| | tridec min-sum BP | 8.40% · 32q | 7.80% · **1024q** | 7.80% · **1024q** |
| | `ldpc` BP-OSD (CPU) | 1.55% · 0q | **1.15%** · 1q | 1.15% · 0q |
| | `relay_bp` oracle (CPU) | 1.75% · 0q | 1.40% · 32q | 1.40% · 0q |
| | **NVIDIA CUDA-Q `nv-qldpc`+OSD** | — *(CUDA-only)* | — *(CUDA-only)* | **1.45% · 1q** |
| **surface d=5** | tridec Relay-BP | 1.30% · 0q | 1.60% · 0q | 1.60% · 1q |
| | tridec min-sum BP | 7.65% · 16q | 7.90% · 1024q | 7.90% · 512q |
| | **PyMatching MWPM** | **0.10% · 1536q** | **0.15% · 1536q** | **0.15% · 1024q** |
| | **NVIDIA CUDA-Q `nv-qldpc`+OSD** | — *(CUDA-only)* | — *(CUDA-only)* | **0.55% · 2q** |

(LER % · sustained qubits/GPU. 2000 shots, p=0.003.)

Three findings:
- **Right tool per code.** On **surface**, matching dominates *both* axes
  (0.10–0.15%, ~1024–1536 q/GPU) — the specialized decoder beats every BP-family
  decoder outright. On **BB qLDPC** (matching N/A), it's a BP + capacity story:
  the accurate decoders (BP-OSD, relay) sit at ~1–2% LER, the fast GPU min-sum BP
  trades accuracy (~8%) for **1024 q/GPU**.
- **Vendor-portable, and datacenter GPUs scale the capacity ~30×.** The *same*
  tridec code runs on Metal, ROCm, *and* CUDA; the GPU BP family jumps from
  Metal's 16–32 q to **1024 q** on MI300X *and* H200. AMD MI300X is first-class —
  it hits the same 1024/1536-qubit knee as the H200.
- **Head-to-head vs NVIDIA's own decoder — and the vendor-lock made concrete.**
  CUDA-Q `nv-qldpc`+OSD is **accurate** (1.45% BB, 0.55% surface — competitive
  with relay/BP-OSD) but **GPU-throughput-bound at ~1.5–2.7k syn/s → 1–2 logical
  qubits/GPU** (measured: the BP iteration *is* the cost — 78% of batch time is
  the GPU decode, not marshalling). On NVIDIA's **own H200**, vendor-neutral
  tridec Relay-BP serves **16 q at comparable 1.95% accuracy** (≈16×), and tridec
  min-sum BP serves **1024 q**. And critically: **`nv-qldpc` is CUDA-locked — it
  cannot run on the MI300X at all.** tridec runs on both.

## NVIDIA Ising pre-decoder (documented — not yet runnable here)

NVIDIA's **Ising decoder** is an AI *pre-decoder* for the surface code: it
sparsifies syndrome data block-wise *before* the main decoder, reaching ~1 µs/
round end-to-end on **GB300** and improving LER out to distance 13 (QuEra /
Infleqtion collaboration). It targets the **sub-µs superconducting regime** —
exactly the regime this project's serving model flags as FPGA/ASIC territory for
a *full* GPU decode (`../README.md`). It's **not in cudaq-qec 0.6.0** (only
`nv-qldpc` is registered), so it stays a *documented* comparison, not a runnable
entry. In this benchmark it would slot in as an optional **pre-decode stage**
ahead of any `decode_batch` decoder — a natural v2 addition (and another
CUDA-locked direction the vendor-neutral framing covers).

## Why this is *more* relevant given NVIDIA's CUDA-Q QEC rollout

NVIDIA shipping GPU QEC decoding (CUDA-Q QEC, the Ising pre-decoder, real-time
NVQLink) **validates the thesis** — but their stack is CUDA-only, their protocol,
their decoders. There is no **vendor-neutral** serving benchmark. That's the
MLPerf role: the neutral standard exists *because* one vendor dominates — the
counterweight everyone else uses to show competitiveness. This benchmark now
makes the case concretely: it **runs NVIDIA's own `nv-qldpc` head-to-head**, and
shows a vendor-neutral decoder is **Pareto-competitive-to-better on serving even
on NVIDIA's own H200** — *and* the only one that also runs on AMD. The **AMD
angle is the strongest case**: MI300X is first-class here (same 1024-qubit knee
as H200), in a domain NVIDIA is claiming CUDA-only and structurally won't build a
neutral benchmark for. This is exactly the artifact to bring to AMD.

**Honest scope:** "the standard nobody's defined" is aspirational — a benchmark
becomes a *standard* only if adopted. The realistic value is a clean, novel,
vendor-neutral serving benchmark = a portfolio artifact + a natural
AMD-collaboration vehicle + a direct demonstration of GPU × serving × QEC.

## Run

```bash
python qecserve_bench.py        # code x decoder matrix: accuracy + serving -> results_<plat>.json
python make_pareto.py h200      # accuracy-vs-capacity figure -> benchmark_pareto_h200.png
```
(`pip install "tridec[torch,decoders]"` + a GPU/Metal; `pymatching` for the
surface matching entry; `cudaq-qec` on CUDA for the NVIDIA `nv-qldpc` entry —
auto-skipped on AMD/Metal, which is the point.)

## Limitations / next

- **Prototype scheduler** (`../serve.py`, Python-threaded) → serving numbers are
  a pessimistic lower bound; a production server (out-of-process worker,
  CUDA-graph bucket shapes) would do better.
- **Slow-decoder knee is coarse.** A high-latency decoder produces few batches
  per window, so its exact sustained-K is noisy ±1 grid point (4 s window used);
  CIs / longer windows for precision. The *relative* picture is robust.
- ✅ **Two codes, 5 decoder families, 3 vendors** (Metal / MI300X / H200) — done.
- ✅ **NVIDIA CUDA-Q `nv-qldpc` head-to-head** (H200) — done, with a fair OSD
  config (default is BP-only → ~5% LER; OSD → ~1.5%).
- **Next:** the NVIDIA **Ising pre-decoder** as a pre-decode stage (above);
  neural decoders (AlphaQubit-style); CUDA-Q **tensor-network** decoder; more
  surface distances + mixed code-distance fleets; CIs on the slow-decoder knee.
