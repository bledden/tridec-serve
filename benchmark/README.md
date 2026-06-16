# QEC decode-serving benchmark (v4)

A **vendor-portable, decoder-agnostic** benchmark for quantum error-correction
decoders that adds the axis existing benchmarks miss — measured across
**three GPU vendors** (Apple Metal, AMD MI300X/ROCm, NVIDIA H200/CUDA) and
**six decoder families**, including a head-to-head against **NVIDIA's own
CUDA-Q QEC GPU decoders** (`nv-qldpc` + tensor-network). v2 added **multi-seed
error bars** on the serving knee and a **pre-decode-stage** architecture (the
slot NVIDIA's Ising AI pre-decoder would fill). v3 added a surface-code
**distance sweep (d=3/5/7)** — how the accuracy↔capacity tradeoff scales with
distance. **v4 adds mixed-distance / mixed-family FLEETS** — one GPU serving a
*heterogeneous* fleet of logical qubits, each routed to its own decoder instance
(the real heterogeneity lever, and the vendor-neutrality move a single
CUDA-locked decoder structurally can't make).

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

> **A correction we lead with (why you can trust these numbers).** An earlier
> cut of this benchmark reported NVIDIA's `nv-qldpc` at 1–2 q/GPU and claimed
> tridec serves *~16× more on NVIDIA's own H200*. That was **wrong** — an artifact
> of NVIDIA's *default* `bp_batch_size` (which runs the decode ~serially), not
> their decoder. A verify-before-claiming pass (a timing breakdown + a config
> sweep of *their* decoder) caught it before anything went out; with its proper
> config (+ OSD) `nv-qldpc` is **1.45% LER · ~12 q — Pareto-close** with tridec
> Relay-BP (1.95% · 16 q), neither dominating. Corrected in commit `8ba1722`. The
> durable claim survives and is stronger: **`nv-qldpc` is CUDA-locked — it cannot
> run on the MI300X at all; tridec runs on both.** Every comparative number below
> was run through the same discipline (sweep the competitor's obvious knobs first).

Per-platform Pareto: `benchmark_pareto_{metal,mi300x,h200}.png`. `max q` = max
sustained logical qubits/GPU at a 1 ms round (bounded backlog); `peak` = peak
throughput (syn/s).

LER % (95% Wilson CI) · sustained qubits/GPU (median of 3 seeds [min–max]).
2000 shots, p=0.003.

| code | decoder | Metal (M4 Max) | MI300X (ROCm7)‡ | H200 (CUDA) |
|---|---|---|---|---|
| **BB qLDPC** | tridec Relay-BP | 1.95% · 8q | 1.80% · 16q [6-16] | 1.95% · **16q [12-16]** |
| | tridec min-sum BP | 8.40% · 32q | 7.80% · **1024q** | 7.80% · **1024q** |
| | `ldpc` BP-OSD (CPU) | 1.55% · 0q | **1.15%** · 0q | 1.15% · 0q |
| | `relay_bp` oracle (CPU) | 1.75% · 0q | 1.40% · 0q | 1.40% · 0q |
| | **NVIDIA `nv-qldpc`+OSD** | — *(CUDA-only)* | — *(CUDA-only)* | **1.45% · 12q [8-12]** |
| | **NVIDIA pre-decode cascade** | — *(CUDA-only)* | — *(CUDA-only)* | **1.25% · 2q [2-3]** |
| **surface d=5** | tridec Relay-BP | 1.30% · 0q | 1.60% · 0q | 1.60% · 1q [0-1] |
| | tridec min-sum BP | 7.65% · 16q | 7.90% · 768q | 7.90% · 512q |
| | **PyMatching MWPM** | **0.10% · 1536q** | **0.15% · 1024q** | **0.15% · 1024q** |
| | **NVIDIA `nv-qldpc`+OSD** | — *(CUDA-only)* | — *(CUDA-only)* | **0.55% · 12q [8-12]** |
| | **NVIDIA tensor-network** | — | — | *intractable at d=5* † |

‡ MI300X = a DigitalOcean MI300X droplet on **ROCm 7 / triton 3.4**. (RunPod's
MI300X — rocm6.2/triton3.1 — became unusable mid-session: a host `memlock` cap of
8 MB broke ROCm/HSA after a stop/resume, on both the resumed and a freshly
deployed pod; DigitalOcean's ROCm7 image has proper ulimits.) The relay megakernel
block/warps were **re-tuned for the rocm7 stack** to (1024,16) — the shipped
(512,8), tuned on rocm6.2/triton3.1, gave only 4q for BB relay here; (1024,16)
recovers 16q (parity with the H200). min-sum BP & matching needed no re-tune
(already at parity: 1024q / 1024-1536q).

† TN decoder is exact-ish but **intractable at surface d=5/5-rounds** (TN
treewidth blows up; >90 s/decode). Measured at small scale: surface d=3/3-rounds
≈ 5% LER at ~4 decodes/s (→ 0 q served). It's an accuracy decoder that can't
serve and doesn't scale — a clean negative.

## Surface-code distance sweep (v3) — `benchmark_distance_{h200,mi300x}.png`

The serving question that only this benchmark asks of code distance: **stronger
protection costs serving capacity.** Sweeping surface d=3/5/7 on both the H200 and
the AMD MI300X (ROCm7), and pushing the sweep to **d=9/11/13** on the MI300X (the
box that was available — d=13 fits well within either GPU's memory; H200 coverage
of d≥9 is a TODO, not a capability limit):

| d | PyMatching LER · cap | min-sum BP LER · cap | nv-qldpc+OSD (H200) |
|---|---|---|---|
| **3** | 0.55% · 1536q | 4.25% · 1536q | 0.90% · 96q |
| **5** | 0.15% · 1024q | 7.90% · 512–768q | 0.55% · 12q |
| **7** | 0.35%† · 256q | 15.2% · 192–256q | 0.60% · 3q |
| **9** *(MI300X)* | 0.10% · 96q | 22.2% · 96q | — |
| **11** *(MI300X)* | 0.05% · 64q | 30.95% · 48q | — |
| **13** *(MI300X)* | **0.00%** (0/2000) · 32q | **33.75%** · 32q | — |

(LER and capacity per platform; matching/BP capacities for d≤7 agree closely
across H200/MI300X. d=9–13 are MI300X-only **coverage** (matching + min-sum BP);
d=13 is Gu et al.'s max. †d=7 matching LER is noise-limited at 2000 shots
— 7/2000 — but the d=9–13 points confirm the monotone suppression: 0.10% → 0.05%
→ **0/2000**. At d=13 matching is a *perfect* decoder on these 2000 shots while BP
fails a third of them.)

Two clean findings, **identical across both vendors** (it's decoder physics):
- **Capacity drops monotonically with distance** — bigger code = more
  detectors/error-mechanisms per shot = fewer logical qubits/GPU. PyMatching
  1536→1024→256; min-sum BP 1536→~600→~220; nv-qldpc 96→12→3. This is the
  quantified *serving cost of stronger protection*.
- **BP-family LER *rises* monotonically with distance on the surface code** (min-sum
  BP 4.25%→7.90%→15.2%→22.2%→**30.95%** out to d=11; relay 0.75%→1.60%→6.05%),
  while **matching keeps suppressing** (0.55%→…→**0.05%** at d=11) and `nv-qldpc`+OSD
  stays low — the textbook weakness of plain BP on topological codes
  (loops/degeneracy), now quantified out to **d=11**: matching gives ~600× lower
  LER than BP there. "Matching is the right tool for surface; BP-family is for
  qLDPC" — and it only sharpens with distance.

Findings (BB qLDPC + d=5 slice):
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

## Mixed-distance / mixed-family FLEETS (v4) — `benchmark_fleet_{h200,mi300x}.png`

v1–v3 decode one homogeneous code at a time. A real fault-tolerant machine isn't
homogeneous: it has logical qubits at **different code distances** (different
protection levels) and **different code families** at once. v4 serves that on
**one GPU** — a `FleetServer` keeps a per-lane queue + its own decoder instance
(different codes can't share a batch) and **time-shares the accelerator** across
lanes (`../fleet_serve.py`). Metric: max sustained **total** fleet, with each
lane routed to the right decoder. Measured **cross-vendor** (1 ms round, tridec BP
per lane, median of 3 seeds):

| fleet composition (one box) | H200 (CUDA) | MI300X (ROCm7) |
|---|---|---|
| **mixed-distance** — surface d3 + d5 + d7 (equal), BP all | **384 q** | **384 q** |
| **mixed-family** — surface d5 + BB qLDPC, BP all | **512 q** | **1024 q** |
| **best-tool** — surface d3,d5 → *matching* + BB qLDPC → *BP* | — | **1536 q** |
| **mixed-distance + d9** — surface d3+d5+d7+d9, BP all | — | **256 q** |

Findings:
- **One box serves a heterogeneous fleet of hundreds of logical qubits**, routing
  each lane to its own decoder instance. The knee is set by the **bottleneck lane**
  — the biggest/slowest code — exactly as the per-code v3 capacities predict once
  the accelerator is split across lanes (adding a heavier d=9 lane drops the
  equal-split total to 256 q).
- **Best-tool routing wins on both axes.** Routing surface→matching + qLDPC→BP
  (the right tool per code) sustains **1536 q** on the MI300X — 1.5–3× the all-BP
  fleets — *and* is far more accurate on the surface lanes (matching ~0.1% vs BP
  ~8%). The accuracy-optimal fleet is also the highest-capacity one. (Adding a
  *slower* lane re-confirms the bottleneck principle: a 4-lane best-tool fleet that
  also carries d7-surface-matching drops to 512 q, gated by that slowest lane.)
- **Cross-vendor, and AMD holds its own.** Mixed-distance is at **parity** (384 q
  on both). On mixed-family the **MI300X serves 2× the H200** (1024 vs 512 q) —
  its surface-d5 min-sum BP is faster on the rocm7/triton-3.4 stack. AMD isn't just
  "supported" here; on this workload it leads.
- **This is the vendor-neutrality lever made operational.** A heterogeneous fleet
  needs *multiple* decoder instances (one per code/distance) — and the right one
  per code (matching for surface, BP for qLDPC). A single CUDA-locked `nv-qldpc`
  is one BP decoder bound to one parity-check matrix; it cannot *be* a mixed
  fleet's router, and cannot run on AMD at all. The portable multi-decoder stack
  can — on NVIDIA *and* AMD, with AMD ahead on the mixed-family fleet.

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

## Run (reproducible from a clean checkout)

Paths are relative + env-overridable (`TRIDEC_PLAT`, `TRIDEC_FIX`); fixtures are
bundled in `fixtures/`; deps pinned in `requirements.txt`; artifact provenance in
`receipts/MANIFEST.md`. No per-machine source edits.

```bash
pip install -r requirements.txt                  # + "tridec[torch]" for GPU lanes; cudaq-qec on CUDA
TRIDEC_PLAT=h200 python qecserve_bench.py         # accuracy + serving matrix -> results_h200.json
python make_pareto.py h200                        # accuracy-vs-capacity Pareto
python make_distance.py h200                      # surface distance sweep d=3/5/7
TRIDEC_PLAT=h200 python fleetbench.py             # v4 fleets -> results_fleet_h200.json
python make_fleet.py h200                         # v4 fleet knee figure
```
The `cudaq-qec` (NVIDIA `nv-qldpc` + TN) lanes are CUDA-only and **auto-skip** on
AMD/Metal — that's the point. On AMD/ROCm7 the relay megakernel wants re-tuning
(`relay_retune.py`). A CPU-only checkout still runs the routing + PyMatching lanes.

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
- ✅ **Surface-code distance sweep d=3/5/7 (v3)** — cross-vendor (H200 + AMD
  MI300X/ROCm7); capacity-vs-distance + the BP-LER-rises finding, above.
- ✅ **AMD MI300X re-validated on ROCm7** (DigitalOcean) after RunPod's MI300X
  was lost to a host memlock cap; relay megakernel re-tuned for triton-3.4.
- ✅ **Mixed-distance / mixed-family / best-tool fleets (v4), cross-vendor** — one
  box serves a heterogeneous fleet, routing each lane to its own decoder:
  mixed-distance 384 q (H200 = MI300X); mixed-family 512 q (H200) / **1024 q
  (MI300X)**; best-tool (matching+BP) **1536 q (MI300X)**. The vendor-neutrality
  lever, above.
- ✅ **Distance sweep pushed to d=9/11/13 on the MI300X** (coverage; fits either GPU) — past the H200's
  d=7; matching suppresses to 0.05% LER while BP climbs to ~31%.
- **Next:** drop in NVIDIA's **Ising** if public; neural decoders; out-of-process /
  CUDA-graph scheduler to lift the prototype's pessimistic serving floor; then the
  arXiv writeup (repro gate is cleared — see `receipts/`).
