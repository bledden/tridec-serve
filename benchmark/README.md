# QEC decode-serving benchmark (v4+)

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
2000 shots, p=0.003, **tridec 0.2.1** — except the MI300X/H200 surface-d5
min-sum-BP and matching LERs, which are the better-powered **50k-shot** rerun
(the same raw rows as the distance table below; the original 2000-shot rows
were superseded in-place by that rerun).

| code | decoder | Metal (M4 Max) | MI300X (ROCm7)‡ | H200 (CUDA) |
|---|---|---|---|---|
| **BB qLDPC** | tridec Relay-BP | 1.95% · 8q | 1.80% · 16q [6-16] | 1.95% · **16q [8-16]** |
| | tridec min-sum BP | 8.40% · 32q | 7.80% · **1024q** | 7.80% · **1024q** |
| | `ldpc` BP-OSD (CPU) | 1.55% · 0q | **1.15%** · 0q | 1.15% · 0q |
| | `relay_bp` oracle (CPU) | 1.75% · 0q | 1.40% · 0q | 1.40% · 0q |
| | **NVIDIA `nv-qldpc`+OSD** | — *(CUDA-only)* | — *(CUDA-only)* | **1.45% · 12q [8-12]** |
| | **NVIDIA pre-decode cascade** | — *(CUDA-only)* | — *(CUDA-only)* | **1.25% · 3q [2-3]** |
| **surface d=5** | tridec Relay-BP | 1.30% · 0q | 1.60% · 0q | 1.60% · 1q [1-1] |
| | tridec min-sum BP | 7.65% · 16q | 8.66% · 768q | 8.66% · 512q |
| | **PyMatching MWPM** | **0.10% · 1536q** | **0.29% · 1024q** | **0.29% · 1024q** |
| | **NVIDIA `nv-qldpc`+OSD** | — *(CUDA-only)* | — *(CUDA-only)* | **0.55% · 8q [8-8]** |
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
≈ 5% LER at ~4 decodes/s (→ 0 q served; measured on-pod, stdout not archived). It's an accuracy decoder that can't
serve and doesn't scale — a clean negative.

## Surface-code distance sweep (v3, extended to d=13 in v4+) — `benchmark_distance_{h200,mi300x}.png`

The serving question that only this benchmark asks of code distance: **stronger
protection costs serving capacity.** Full sweep **d=3..13 on BOTH the H200 and the
AMD MI300X** (ROCm7), accuracy at **50k shots** (Wilson CI):

| d | PyMatching LER | min-sum BP LER | capacity (q/GPU) |
|---|---|---|---|
| **3** | 0.632% [0.57–0.71] | 4.01% | 1536 |
| **5** | 0.286% [0.24–0.34] | 8.66% | ~512–1024 |
| **7** | 0.140% [0.11–0.18] | 15.31% | ~192–256 |
| **9** | 0.068% [0.05–0.10] | 22.18% | ~64–128 |
| **11** | 0.018% [0.01–0.03] | 29.15% | ~48–128 |
| **13** | **0.010%** [0.004–0.02] | **34.65%** | ~24–32 |
| **15** ‖ | <0.05% (0/2000) | **39.30%** [37.2–41.5] | — |
| **17** ‖ | <0.05% (0/2000) | **44.25%** [42.1–46.4] | — |

‖ **d=15/17 added on tridec 0.2.2** (the d≥15 BP-kernel-ceiling lift, `46c63a4`),
H200, **2000 shots** — min-sum BP is ~200 ms/shot at these sizes, so the ladder's
50k methodology is ~3 h/point here; BP's ~40% LER is well-resolved at 2000, and
matching stays below the 2000-shot floor (`0/2000`, <0.05%). A tight matching LER
at d≥15 is a cheap CPU-only follow-on (no GPU). LER is hardware-independent, so
the single-GPU H200 row is canonical.

**Accuracy (LER) is hardware-independent** — bit-identical across H200 and MI300X
(`benchmark_accuracy_parity.png` shows the two LER columns side by side, identical
to the last digit): both are x86_64 Linux, so stim's sampler yields the same shots
and the decoders are deterministic, so the LER ladder reproduces to the digit on
either GPU. (This
also defuses any "you tuned for one vendor" objection: accuracy is *deterministic*
and identical, so the capacity differences below are pure stack/kernel work, not a
measurement artifact.) Capacity, in contrast, is stack-dependent (shown as the
H200↔MI300X range). The 50k shots dissolved the 2000-shot noise — matching ~halves
each step (clean threshold; the old d=7 "0.35%" wiggle was noise, now 0.140%), and
at d=13 matching gives **0.010% (5/50000) — ~3500× lower LER than BP**.

**The d=13 "wall" was a decoder-version limit — now lifted and measured past.**
On **tridec 0.2.1** the two-kernel BP path launched on a 2-D grid whose unit
dimension (bit/check count) capped at 65535, so surface BP failed identically at
**d=15** on both CUDA and HIP (`invalid argument` on kernel launch) — a kernel
limit, *not* memory or vendor (d=13 is only 2184 detectors; both GPUs have
≥140 GB, and the MI300X's 192 GB bought no extra reach). **tridec 0.2.2 lifts it**
(`46c63a4`, 1-D flattened grid, bit-identical), and the ladder now runs past the
old wall — measured live on the H200: **d=15 (3360 detectors) → BP 39.30%, d=17
(4896) → BP 44.25%**, matching below the floor throughout (table above). So the
192-vs-141 GB memory story was never the ceiling, and the kernel ceiling itself
is now gone.

Two clean findings, **identical across both vendors** (it's decoder physics):
- **Capacity drops monotonically with distance** — bigger code = more
  detectors/error-mechanisms per shot = fewer logical qubits/GPU. PyMatching
  1536→1024→256→128→…→32; min-sum BP 1536→~512→~192→…→~24. This is the quantified
  *serving cost of stronger protection*.
- **BP-family LER *rises* monotonically; matching keeps suppressing** (50k shots):
  min-sum BP **4.01→8.66→15.31→22.18→29.15→34.65%** across d=3..13, while PyMatching
  **0.632→0.286→0.140→0.068→0.018→0.010%** (each ~halving — clean threshold). At
  d=13 matching gives **~3500× lower LER than BP**. The textbook weakness of plain
  BP on topological codes (loops/degeneracy), now quantified to d=13: "matching is
  the right tool for surface; BP-family is for qLDPC" — and it only sharpens with
  distance.

Findings (BB qLDPC + d=5 slice):
- **Right tool per code.** On **surface**, matching dominates *both* axes
  — the specialized decoder beats every BP-family
  decoder outright (0.10–0.29%, ~1024–1536 q/GPU). On **BB qLDPC** (matching N/A), it's a BP + capacity story:
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
  CIs overlap on accuracy and the knee bands ([8-12] vs [8-16]) overlap —
  genuinely Pareto-close, neither dominates. (tridec's *fast* min-sum BP serves
  1024 q at ~8% LER — a different point on the curve.) The serving cap is
  `nv-qldpc`'s **~18 ms per-batch latency** (heavy BP iterations): its *raw*
  large-batch throughput is ~54k syn/s (≈50 q ceiling under a production
  scheduler; the 18 ms / 54k figures are on-pod measurements whose stdout was
  not archived in receipts — indicative, not receipt-backed), but the
  per-batch latency bounds sustained-K at a tight reaction-latency SLA — the same
  prototype-scheduler caveat applies to every decoder here. **Two non-default
  knobs are required to be fair to it:** `use_osd=True` (default BP-only ≈ 5% LER)
  and `bp_batch_size` (default is tiny → ~serial ~4k syn/s; set to the batch width
  → true GPU batching, 12× at the same LER). And the durable point: **`nv-qldpc`
  is CUDA-locked — it cannot run on the MI300X at all.** tridec runs on both.
- **Pre-decode stage (v2) — architecture works; cheap pre is the missing piece.**
  The cascade (a fast BP-only pre-pass → accurate +OSD on the *residual*
  syndrome, results combined) is **more accurate** (1.25% vs 1.45% for `nv-qldpc`
  alone) but serves **less** (3 q vs 12 q): the BP-proxy pre-pass is a *second*
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
| **best-tool** — surface d3,d5 → *matching* + BB qLDPC → *BP* | **768 q** | **1536 q** |
| **mixed-distance + d9** — surface d3+d5+d7+d9, BP all | **128 q** | **256 q** |

(best-tool/d9-fleet now cross-vendor too. Note best-tool routes the surface lanes
to *CPU* matching, so its total is partly host-CPU-bound — the H200↔MI300X gap
there reflects the whole box, not just the GPU; median-of-5 shows the MI300X
best-tool occasionally dips to 768q (band [768–1536]) from CPU jitter, while
mixed-family is rock-solid. **SLA caveat (torture sweep, `benchmark_torture_mi300x.png`):**
"max sustained" is the *backlog* knee; the cliff is a gradual p99 ramp
(26→273→574 ms across 1152→1536q) then hard overload at 1728q — so the **p99<500 ms
SLA-bound** capacity is ~1344q, below the 1536q backlog knee. Both metrics reported;
neither hidden.)

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
  on both). On mixed-family the **MI300X serves 2× the H200** (1024 vs 512 q),
  because the fleet runs at large batches where the MI300X's surface-d5 BP throughput
  is higher. **Root-caused, with the confounds stated** (a profile of surface-d5 BP
  across batch sizes): the H200 has ~2× *lower* per-call/launch overhead (1.08 vs
  2.19 ms at B=1), the MI300X has ~1.3× *higher* large-batch throughput (1.16 vs
  1.47 µs/syn at B=8192); the serving regime is large-batch, so MI300X wins there.
  **But this is a *stack* comparison, not a hardware verdict:** the two triton
  versions differ (**3.4+rocm7 vs 3.0+cu124**), and the DO part is an **MI300X VF**
  (virtualized partition), so the gap reflects codegen + config + partition, not an
  architectural claim. A same-triton-version test would be needed to attribute it to
  hardware. (Accuracy, by contrast, *is* hardware-independent — next section — so the
  capacity gap is pure stack/kernel, not a measurement artifact.)
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
python make_distance.py h200                      # surface distance sweep d=3..13
TRIDEC_PLAT=h200 python ../fleetbench.py          # v4 fleets (script at repo root) -> results_fleet_h200.json here
python make_fleet.py h200                         # v4 fleet knee figure
```
The `cudaq-qec` (NVIDIA `nv-qldpc` + TN) lanes are CUDA-only and **auto-skip** on
AMD/Metal — that's the point. On AMD/ROCm7 the relay megakernel wants re-tuning
(`relay_retune.py`). A CPU-only checkout still runs the routing + PyMatching lanes.

## Limitations / next

- **Prototype scheduler** (`../serve.py`, Python-threaded) → serving numbers are
  a pessimistic lower bound; a production server (out-of-process worker,
  CUDA-graph bucket shapes) would do better. All serving numbers were measured
  with **tridec 0.2.1**; tridec master has since made a CUDA-graph fast path
  default-on (`46c63a4`, 1.59× at batch-1), which shifts exactly the small-batch
  accurate-tier knees (relay, the `nv-qldpc` head-to-head) — re-measure on the
  next tridec release.
- ✅ **Multi-seed knee bands (v2).** The serving knee is now median-of-3-seeds
  with a [min–max] band + a finer K grid in the contested 0–64 zone, so the
  accurate-tier head-to-head (`nv-qldpc` [8-12] vs Relay-BP [8-16]) is
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
- ✅ **Distance sweep pushed to d=9/11/13 on BOTH GPUs** (50k-shot LER; H200
  parity commit `1a52616`) — matching suppresses to 0.010% LER while BP climbs
  to 34.65% at d=13 (table above).
- ✅ **d≥15 frontier — done** (tridec **0.2.2** lifted the ceiling; measured live
  on the H200: d=15 BP 39.30%, d=17 BP 44.25%, matching below floor — the distance
  table above). `d15_extend.py` → `results_dist_d15d17_h200.json`.
- **Next:** drop in NVIDIA's **Ising** if public; neural decoders; a tight
  matching LER at d≥15 (cheap CPU-only, millions of shots, no GPU); out-of-process
  / CUDA-graph scheduler to lift the prototype's pessimistic serving floor; then
  the arXiv writeup (repro gate is cleared — see `receipts/`).
