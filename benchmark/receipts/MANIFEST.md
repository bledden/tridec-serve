# Receipts — provenance for the benchmark result artifacts

The result JSONs + figures live in `../` (next to the `make_*.py` that read them).
This manifest pins where each came from, so a reader can tell which platform /
vintage / config produced each number. All runs: p=0.003, depolarizing+readout,
stim seed=0 sampling. "Pin artifacts, not generators" — the `.dem`/`.stim` inputs
are bundled in `../fixtures/`.

## How to reproduce (clean checkout — run from `benchmark/`)
```bash
pip install -r requirements.txt             # + tridec[torch] for GPU lanes; cudaq-qec on CUDA
TRIDEC_PLAT=<plat> python qecserve_bench.py # accuracy + serving matrix -> results_<plat>.json
python make_pareto.py <plat>                # Pareto figure
python make_distance.py <plat>              # distance sweep figure
TRIDEC_PLAT=<plat> python ../fleetbench.py  # v4 fleets (script at repo root) -> results_fleet_<plat>.json here
python make_fleet.py <plat>                 # fleet figure
```
Paths are relative / env-overridable (`TRIDEC_FIX`, `TRIDEC_PLAT`); no source edits
needed per machine. CPU-only checkouts run the routing + PyMatching lanes; the
tridec GPU lanes need a triton backend; the `nv-qldpc`/TN lanes need CUDA.

## Artifacts

| file | platform / stack | shots · seeds | scope (harness vintage) |
|---|---|---|---|
| `results_h200.json` | NVIDIA H200, CUDA (torch 2.4.1+cu124 / triton 3.0) | 2000 · 3; LER 50k | v3+ — BB qLDPC + surface **d=3..13** (matching + min-sum BP; 50k-shot LER) + 6 decoder families at d≤7 incl. NVIDIA `nv-qldpc`+OSD & pre-decode cascade |
| `results_maxd_{h200,mi300x}.json` | both | 4000 | device-max probe: **both wall at d=15** (tridec BP triton kernel `invalid argument`) — a kernel limit, not memory/vendor. Vintage: tridec **0.2.1**; lifted on tridec master `46c63a4` (unreleased, unmeasured here) |
| `results_mi300x.json` | AMD MI300X, ROCm7 (DigitalOcean; torch 2.9+rocm7 / triton 3.4) | 2000 · 3 | v3 — BB + surface d=3/5/7 (all decoders) **+ d=9..13 (matching + min-sum BP)**; relay re-tuned (1024,16) for rocm7 |
| `results_fleet_extra_mi300x.json` | AMD MI300X, ROCm7 | per-lane · 3 | v4 extras — best-tool fleet (matching+BP, 1536q) + d3+d5+d7+d9 fleet (256q) |
| `results_dist_d9d11_mi300x.json` | AMD MI300X, ROCm7 | 2000 · 3 | raw d=9/d=11 distance rows (merged into results_mi300x.json) |
| `results_metal.json` | Apple M4 Max, Metal (triton-msl) | 2000 · 1 | v1 — BB + surface d=5 (single-seed, pre-CI vintage; not re-run with the multi-seed harness) |
| `results_fleet_h200.json` | NVIDIA H200, CUDA | per-lane · 3 | v4 — mixed-distance (384q) + mixed-family (512q) fleets, tridec BP per lane |
| `results_fleet_mi300x.json` | AMD MI300X, ROCm7 (DigitalOcean) | per-lane · 3 | v4 — mixed-distance (384q) + mixed-family (1024q) fleets, tridec BP per lane |
| `results_fleet_extra_h200.json` | NVIDIA H200, CUDA | per-lane · 3 | v4 extras — best-tool fleet (768q) + d3+d5+d7+d9 fleet (128q) |
| `results_dist_d9d11_h200.json` | NVIDIA H200, CUDA | 2000 · 3 | raw d=9/d=11 distance rows (merged into results_h200.json) |
| `results_dist_d13_{h200,mi300x}.json` | both | 2000 · 3 | raw d=13 rows (capacity; the d=13 LER in the README is the 50k rerun) |
| `results_hishot_dist_{h200,mi300x}.json` | both | **50k** | 50k-shot LER ladder d=3..13 — the accuracy numbers in the README tables (bit-identical across platforms) |
| `results_bigfleet_{h200,mi300x}.json` | both | per-lane · 3 | 4-lane best-tool fleet incl. a d7-matching lane (512q both — the bottleneck-lane check) |
| `results_seeds5_{h200,mi300x}.json` | both | per-lane · 5 | median-of-5 stability: MI300X best-tool band [768–1536] (CPU jitter), mixed-family solid |
| `results_torture_mi300x.json` | AMD MI300X, ROCm7 | per-lane | fine-grained overload cliff (p99 26→273→574 ms across 1152→1536q; SLA-bound ~1344q) |
| `benchmark_pareto_{h200,mi300x,metal}.png` | — | — | accuracy-vs-capacity Pareto per platform (v3; error bars where multi-seed) |
| `benchmark_distance_{h200,mi300x}.png` | — | — | surface distance sweep d=3..13 (v3 + v4+ extension) |
| `benchmark_fleet_{h200,mi300x}.png` | — | — | v4 fleet p99-vs-size knee (cross-vendor) |
| `benchmark_torture_mi300x.png` | — | — | the overload-cliff figure (p99 vs fleet size, SLA line) |
| `benchmark_accuracy_parity.png` | — | — | H200 vs MI300X 50k LER columns side-by-side (identical to the digit) |

## Known non-reproducible-by-construction items (documented, not bugs)
- stim's circuit→DEM is platform-dependent at the ulp level, and its seeded
  sampler is platform-dependent → exact cross-platform count reproduction is
  impossible. The bundled `.dem`/`.stim` bytes are the canonical inputs; the
  statistical (Wilson-CI) tier binds cross-platform.
- The NVIDIA `nv-qldpc` numbers require its non-default config (`use_osd=True`,
  `bp_batch_size`) — defaults badly under-report it (see the README correction /
  commit `8ba1722`). The relay megakernel block/warps are per-stack tuned.
- RunPod MI300X is unusable (host `memlock` cap → ROCm/HSA fails); the MI300X
  receipts are from a DigitalOcean ROCm7 box.

## Status
- ✅ v4 fleet **cross-vendor done** (H200 + DO ROCm7 MI300X): mixed-distance at
  parity (384q both); mixed-family MI300X 1024q vs H200 512q (AMD leads).
  Reproduce on a DO ROCm7 box: `TRIDEC_PLAT=mi300x python ../fleetbench.py`.
