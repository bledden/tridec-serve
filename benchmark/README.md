# QEC decode-serving benchmark (v2)

A **vendor-portable, decoder-agnostic** benchmark for quantum error-correction
decoders that adds the axis existing benchmarks miss — measured across
**three GPU vendors** (Apple Metal, AMD MI300X/ROCm, NVIDIA H200/CUDA) and
**six decoder families**, including a head-to-head against **NVIDIA's own
CUDA-Q QEC GPU decoders** (`nv-qldpc` + tensor-network). v2 adds **multi-seed
error bars** on the serving knee and a **pre-decode-stage** architecture (the
slot NVIDIA's Ising AI pre-decoder would fill).

## The gap it fills

QEC decoder benchmarks today (sinter, decoder zoos, the CUDA-Q QEC numbers)
report **accuracy** — logical error rate at a fixed shot count. None reports
**serving capacity**: how many logical-qubit syndrome streams one accelerator
can decode in real time at a latency SLA. As machines scale to many logical
qubits, *that* is the operational question — decoding becomes a serving problem
(see `../README.md`). This benchmark measures **both axes** per
(decoder × code × platform):

- **accuracy** — LER ± 95% Wilson CI (the standard axis), and
- **serving** — max sustained logical-qubits/GPU (bounded backlog, **median over
  3 seeds with [min–max] band**) + capacity at p99 SLA budgets {100, 250, 500 ms}
  + peak throughput (the **new** axis).

See `benchmark_pareto_{metal,mi300x,h200}.png` — accuracy-vs-serving-capacity per
platform, now with error bars (x = multi-seed knee band, y = Wilson LER CI).

## Decoder-agnostic by construction

Any decoder plugs in via a ~3-line adapter exposing `decode_batch(dets) ->
preds`. **Six decoder families** across two codes and three platforms: tridec
Relay-BP (accurate) & min-sum BP (fast), externals `relay_bp` Rust oracle and
`ldpc` BP-OSD, **PyMatching/MWPM** (a *matching* decoder — a different family),
and **NVIDIA CUDA-Q** `nv-qldpc` (GPU LDPC) + `tensor-network` (both CUDA-only).
Plus a **pre-decode-stage** composition (cheap pre-pass → accurate cleanup on the
residual) — the architectural slot for an AI pre-decoder. Next: neural
(AlphaQubit-style).

## Results — code × decoder × **platform** matrix

Per-platform Pareto: `benchmark_pareto_{metal,mi300x,h200}.png`. `max q` = max
sustained logical qubits/GPU at a 1 ms round (bounded backlog); `peak` = peak
throughput (syn/s).

LER % (95% Wilson CI) · sustained qubits/GPU (median of 3 seeds [min–max]).
2000 shots, p=0.003.

| code | decoder | Metal (M4 Max) | MI300X (ROCm) | H200 (CUDA) |
|---|---|---|---|---|
| **BB qLDPC** | tridec Relay-BP | 1.95% · 8q | 1.95% · 12q [8-12] | 1.95% · **16q [12-16]** |
| | tridec min-sum BP | 8.40% · 32q | 7.80% · **1024q** | 7.80% · **1024q** |
| | `ldpc` BP-OSD (CPU) | 1.55% · 0q | **1.15%** · 0q [0-1] | 1.15% · 0q |
| | `relay_bp` oracle (CPU) | 1.75% · 0q | 1.40% · 8q [1-8] | 1.40% · 0q |
| | **NVIDIA `nv-qldpc`+OSD** | — *(CUDA-only)* | — *(CUDA-only)* | **1.45% · 12q [8-12]** |
| | **NVIDIA pre-decode cascade** | — *(CUDA-only)* | — *(CUDA-only)* | **1.25% · 2q [2-3]** |
| **surface d=5** | tridec Relay-BP | 1.30% · 0q | 1.60% · 0q | 1.60% · 1q [0-1] |
| | tridec min-sum BP | 7.65% · 16q | 7.90% · 1024q | 7.90% · 512q |
| | **PyMatching MWPM** | **0.10% · 1536q** | **0.15% · 1536q** | **0.15% · 1024q** |
| | **NVIDIA `nv-qldpc`+OSD** | — *(CUDA-only)* | — *(CUDA-only)* | **0.55% · 12q [8-12]** |
| | **NVIDIA tensor-network** | — | — | *intractable at d=5* † |

† TN decoder is exact-ish but **intractable at surface d=5/5-rounds** (TN
treewidth blows up; >90 s/decode). Measured at small scale: surface d=3/3-rounds
≈ 5% LER at ~4 decodes/s (→ 0 q served). It's an accuracy decoder that can't
serve and doesn't scale — a clean negative.

Findings:
- **Right tool per code.** On **surface**, matching dominates *both* axes
  (0.10–0.15%, ~1024–1536 q/GPU) — the specialized decoder beats every BP-family
  decoder outright. On **BB qLDPC** (matching N/A), it's a BP + capacity story:
  the accurate decoders (BP-OSD, relay, `nv-qldpc`) sit at ~1–2% LER, the fast GPU
  min-sum BP trades accuracy (~8%) for **1024 q/GPU**.
- **Vendor-portable, and datacenter GPUs scale the capacity ~30×.** The *same*
  tridec code runs on Metal, ROCm, *and* CUDA; the GPU BP family jumps from
  Metal's 16–32 q to **1024 q** on MI300X *and* H200. AMD MI300X is first-class —
  it hits the same 1024/1536-qubit knee as the H200.
- **Head-to-head vs NVIDIA's own decoder — Pareto-close, and the vendor-lock made
  concrete.** CUDA-Q `nv-qldpc`+OSD is **accurate** (1.45% BB, 0.55% surface —
  competitive with relay/BP-OSD) and **sustains ~12 q/GPU** at a 1 ms round. At
  the accurate tier it's neck-and-neck with tridec Relay-BP (1.95% · 16 q): the
  CIs overlap on accuracy and the knee bands ([8-12] vs [12-16]) nearly touch —
  genuinely Pareto-close, neither dominates. (tridec's *fast* min-sum BP serves
  1024 q at ~8% LER — a different point on the curve.) The serving cap is
  `nv-qldpc`'s **~18 ms per-batch latency** (heavy BP iterations): its *raw*
  large-batch throughput is ~54k syn/s (≈50 q ceiling under a production
  scheduler), but the
  per-batch latency bounds sustained-K at a tight reaction-latency SLA — the same
  prototype-scheduler caveat applies to every decoder here. **Two non-default
  knobs are required to be fair to it:** `use_osd=True` (default BP-only ≈ 5% LER)
  and `bp_batch_size` (default is tiny → ~serial ~4k syn/s; set to the batch width
  → true GPU batching, 12× at the same LER). And the durable point: **`nv-qldpc`
  is CUDA-locked — it cannot run on the MI300X at all.** tridec runs on both.
- **Pre-decode stage (v2) — architecture works; cheap pre is the missing piece.**
  The cascade (a fast BP-only pre-pass → accurate +OSD on the *residual*
  syndrome, results combined) is **more accurate** (1.25% vs 1.45% for `nv-qldpc`
  alone) but serves **less** (2 q vs 12 q): the BP-proxy pre-pass is a *second*
  decode, so it costs throughput. A serving win needs a genuinely *cheap* pre —
  exactly what NVIDIA's Ising AI pre-decoder is. The plumbing is correct and
  ready; the cheap-pre slot is where Ising would drop in.

## NVIDIA Ising pre-decoder (documented — the pre-decode slot is now built)

NVIDIA's **Ising decoder** is an AI *pre-decoder* for the surface code: it
sparsifies syndrome data block-wise *before* the main decoder, reaching ~1 µs/
round end-to-end on **GB300** and improving LER out to distance 13 (QuEra /
Infleqtion collaboration). It targets the **sub-µs superconducting regime** —
exactly the regime this project's serving model flags as FPGA/ASIC territory for
a *full* GPU decode (`../README.md`). It's **not in cudaq-qec 0.6.0** (only
`nv-qldpc` and `tensor_network_decoder` are registered), so it stays a
*documented* comparison, not a runnable entry. **v2 builds the slot it would fill:**
`cudaq_predecode_entry` is the pre-decode-stage composition (cheap pre-pass →
accurate cleanup on the residual). Today the "pre" is a BP proxy that costs a
second decode; Ising would be the cheap pre that makes it a serving win. (Another
CUDA-locked direction the vendor-neutral framing covers.)

## Why this is *more* relevant given NVIDIA's CUDA-Q QEC rollout

NVIDIA shipping GPU QEC decoding (CUDA-Q QEC, the Ising pre-decoder, real-time
NVQLink) **validates the thesis** — but their stack is CUDA-only, their protocol,
their decoders. There is no **vendor-neutral** serving benchmark. That's the
MLPerf role: the neutral standard exists *because* one vendor dominates — the
counterweight everyone else uses to show competitiveness. This benchmark now
makes the case concretely: it **runs NVIDIA's own `nv-qldpc` head-to-head**, and
shows a vendor-neutral decoder is **Pareto-competitive on NVIDIA's own H200**
(neck-and-neck with `nv-qldpc` at the accurate tier) — *and* the only one that
also runs on AMD. The **AMD angle is the strongest case**: MI300X is first-class
here (same 1024-qubit knee as H200), in a domain NVIDIA is claiming CUDA-only and
structurally won't build a neutral benchmark for. This is exactly the artifact to
bring to AMD.

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
- ✅ **Multi-seed knee bands (v2).** The serving knee is now median-of-3-seeds
  with a [min–max] band + a finer K grid in the contested 0–64 zone, so the
  accurate-tier head-to-head (`nv-qldpc` [8-12] vs Relay-BP [12-16]) is
  statistically legible, not a single coarse point.
- ✅ **Two codes, 6 decoder families, 3 vendors** (Metal / MI300X / H200) — done.
- ✅ **NVIDIA CUDA-Q `nv-qldpc` head-to-head** (H200) — done, fair config
  (`use_osd=True` + `bp_batch_size`; defaults badly under-report it).
- ✅ **CUDA-Q tensor-network decoder** — measured; intractable at d=5 (small-scale
  negative, above).
- ✅ **Pre-decode-stage architecture** — built (`cudaq_predecode_entry`); needs a
  cheap pre (Ising) for a serving win.
- **Next:** drop in NVIDIA's **Ising** model if/when public; neural decoders
  (AlphaQubit-style); more surface distances + mixed code-distance fleets;
  out-of-process / CUDA-graph scheduler to lift the prototype's pessimistic floor.
