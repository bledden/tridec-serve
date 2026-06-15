"""QEC decode-serving benchmark (vendor-portable, decoder-agnostic).

The standard QEC benchmarks (sinter, decoder zoos) measure ACCURACY (logical
error rate at fixed shots). This one adds the missing axis: SERVING CAPACITY —
how many logical-qubit syndrome streams one accelerator sustains at a latency
SLA. Any decoder plugs in via a 3-line adapter; results are reported per
(decoder x code x platform) as (LER ± Wilson CI, sustained qubits/GPU @ p99 SLA,
peak throughput). Vendor-portable by construction: Metal / CUDA / ROCm, with
AMD MI300X a first-class citizen (NVIDIA's CUDA-Q QEC stack is CUDA-locked).
"""
import sys, json, time, numpy as np, stim
sys.path.insert(0, "/Users/bledden/Documents/tridec-serve")
from serve import run_load
import tridec
from tridec.validation import wilson_ci

FIX = "/Users/bledden/Documents/tridec/tests/fixtures/bb72/"


# --- the decoder-agnostic adapter: any entry just needs decode_batch + flags ---
class Entry:
    """name, accurate(bool), and a tridec-style decoder with .decode_batch(dets)."""
    def __init__(self, name, accurate, decoder):
        self.name, self.accurate, self.decoder = name, accurate, decoder


def tridec_entry(dem, algorithm, name=None, **opts):
    dec = tridec.from_dem(dem, algorithm=algorithm, **opts)
    acc = (algorithm == "relay")
    return Entry(name or f"tridec-{algorithm}[{dec.backend}]", acc, dec)


def relaybp_entry(dem, name="relay_bp Rust oracle (CPU, accurate)"):
    """External decoder, ~3-line adapter — proves the harness is decoder-agnostic
    (anything with a decode_batch(dets)->preds plugs in). CPU baseline."""
    import relay_bp
    from relay_bp.stim import CheckMatrices
    cm = CheckMatrices.from_dem(dem)
    orc = relay_bp.RelayDecoderF64(cm.check_matrix, error_priors=cm.error_priors,
            gamma0=0.1, pre_iter=80, num_sets=60, set_max_iter=60,
            gamma_dist_interval=(-0.24, 0.66), stop_nconv=5, stopping_criterion="nconv")
    runner = relay_bp.ObservableDecoderRunner(orc, cm.observables_matrix, include_decode_result=False)

    class _Wrap:
        backend = "cpu"
        def decode_batch(self, dets, device=None):
            p = np.asarray(runner.decode_observables_batch(np.asarray(dets).astype(np.uint8))) % 2
            return p.astype(bool).reshape(len(dets), -1)
    return Entry(name, True, _Wrap())


def pymatching_entry(dem, name="PyMatching MWPM (accurate)"):
    """External, DIFFERENT decoder family (matching, not BP). Surface codes /
    matchable DEMs only. Reference decoder everyone compares to."""
    import pymatching
    m = pymatching.Matching.from_detector_error_model(dem)

    class _W:
        backend = "cpu"
        def decode_batch(self, dets, device=None):
            d = np.asarray(dets).astype(np.uint8)
            preds = np.asarray(m.decode_batch(d))
            return preds.astype(bool).reshape(len(d), -1)
    return Entry(name, True, _W())


def bposd_entry(dem, name="ldpc BP-OSD (accurate, CPU)"):
    """External, accurate qLDPC decoder (BP + ordered-statistics post-proc)."""
    import ldpc
    from tridec.dem import extract
    ex = extract(dem)
    H = ex["H"]; pri = list(np.clip(ex["priors"], 1e-6, 1 - 1e-6))
    Lo = ex["Lo"].toarray().astype(np.uint8)
    dec = ldpc.BpOsdDecoder(H, error_channel=pri, max_iter=30, bp_method="minimum_sum",
                            ms_scaling_factor=0.625, osd_method="osd_cs", osd_order=10)

    class _W:
        backend = "cpu"
        def decode_batch(self, dets, device=None):
            d = np.asarray(dets).astype(np.uint8)
            out = np.zeros((len(d), Lo.shape[0]), dtype=bool)
            for i in range(len(d)):
                e = np.asarray(dec.decode(d[i]), dtype=np.uint8)
                out[i] = (Lo @ e) & 1
            return out
    return Entry(name, True, _W())


def cudaq_entry(dem, name="NVIDIA CUDA-Q nv-qldpc+OSD (GPU)"):
    """NVIDIA's CUDA-Q QEC GPU LDPC decoder (cudaq-qec). CUDA-only -> runs on the
    H200, auto-skipped on AMD/Metal -- exactly the vendor-lock point the benchmark
    makes concrete (the head-to-head we CAN'T port to MI300X). GPU sparse batch
    decoding (use_sparsity=True). TWO non-default knobs matter for a FAIR
    comparison:
      * use_osd=True   -- default is BP-only (~5% LER on this BB cell); OSD brings
                          it to ~1.5% and is nearly free (BP dominates runtime).
      * bp_batch_size  -- THE throughput knob. Default is tiny, so 'batch' decode
                          runs ~serially (~4k syn/s); setting it to the batch
                          width gives true GPU batching (~54k syn/s, 12x, SAME
                          LER). Leaving it default badly under-reports NVIDIA's
                          decoder -- so we set it to the max serving bucket.
    The NVIDIA *Ising* pre-decoder is a research artifact, not in cudaq-qec 0.6.0,
    so it stays a documented number (README), not a runnable entry here."""
    import cudaq_qec as qec
    from tridec.dem import extract
    ex = extract(dem)
    Hd = ex["H"].toarray().astype(np.uint8)
    pri = list(np.clip(np.asarray(ex["priors"]), 1e-6, 1 - 1e-6))
    Lo = ex["Lo"].toarray().astype(np.uint8)
    dec = qec.get_decoder("nv-qldpc-decoder", Hd, error_rate_vec=pri,
                          use_sparsity=True, use_osd=True, bp_batch_size=4096)
    # pick the fastest accepted batch-input form once (ndarray if allowed, else
    # list-of-lists) so the serving number isn't penalised by needless conversion
    try:
        dec.decode_batch(np.zeros((2, Hd.shape[0]), dtype=np.uint8)); as_list = False
    except Exception:
        as_list = True

    class _W:
        backend = "cuda"
        def decode_batch(self, dets, device=None):
            d = np.asarray(dets).astype(np.uint8)
            rb = dec.decode_batch([row.tolist() for row in d] if as_list else d)
            E = (np.array([np.asarray(x.result) for x in rb]) > 0.5).astype(np.uint8)
            return ((E @ Lo.T) & 1).astype(bool)
    return Entry(name, True, _W())


def cudaq_predecode_entry(dem, name="CUDA-Q nv-qldpc PRE->+OSD cascade (GPU)"):
    """v2b -- the PRE-DECODE STAGE architecture (the slot NVIDIA's Ising AI
    pre-decoder would fill). A cheap BP-only pre-pass produces a partial error
    estimate e_pre; the RESIDUAL syndrome s ^ (H @ e_pre) goes to the accurate
    +OSD 'main'; the estimates combine (e = e_pre ^ e_main). Correct by
    construction for ANY pre. Here the 'pre' is a fast BP-only nv-qldpc proxy --
    NVIDIA's Ising model would be a far cheaper, smarter pre. This measures
    whether a cheap pre-pass + accurate cleanup helps; CUDA-only."""
    import cudaq_qec as qec
    from tridec.dem import extract
    ex = extract(dem)
    Hd = ex["H"].toarray().astype(np.uint8)
    pri = list(np.clip(np.asarray(ex["priors"]), 1e-6, 1 - 1e-6))
    Lo = ex["Lo"].toarray().astype(np.uint8)
    pre = qec.get_decoder("nv-qldpc-decoder", Hd, error_rate_vec=pri,
                          use_sparsity=True, use_osd=False, max_iterations=10,
                          bp_batch_size=4096)
    main = qec.get_decoder("nv-qldpc-decoder", Hd, error_rate_vec=pri,
                           use_sparsity=True, use_osd=True, bp_batch_size=4096)

    def _e(dec, d):
        rb = dec.decode_batch(d)
        return (np.array([np.asarray(x.result) for x in rb]) > 0.5).astype(np.uint8)

    class _W:
        backend = "cuda"
        def decode_batch(self, dets, device=None):
            d = np.asarray(dets).astype(np.uint8)
            e_pre = _e(pre, d)                              # (n, E) partial estimate
            s_res = (d ^ (e_pre @ Hd.T)) & 1               # residual syndrome (n, D)
            e = e_pre ^ _e(main, s_res.astype(np.uint8))   # combine: e_pre ^ e_main
            return ((e @ Lo.T) & 1).astype(bool)
    return Entry(name, True, _W())


def cudaq_tn_entry(dem, name="NVIDIA CUDA-Q tensor-network (GPU)"):
    """v2b -- NVIDIA CUDA-Q QEC tensor-network decoder (cudaq-qec[tensor-network-
    decoder]): contracts the Tanner-graph TN (cuTensorNet) to compute P(logical
    flip | syndrome) directly. Exact-ish; surface-code scale only (TN treewidth
    blows up on qLDPC). Float syndromes; result is the logical-flip prob, so no Lo
    multiply. CUDA-only."""
    import cudaq_qec as qec
    from tridec.dem import extract
    ex = extract(dem)
    Hd = ex["H"].toarray().astype(np.uint8)
    pri = list(np.clip(np.asarray(ex["priors"]), 1e-6, 1 - 1e-6))
    Lo = ex["Lo"].toarray().astype(np.uint8)
    nL = Lo.shape[0]
    dec = qec.get_decoder("tensor_network_decoder", Hd, logical_obs=Lo, noise_model=pri)
    try:
        dec.decode_batch(np.zeros((2, Hd.shape[0]), np.float32)); batch = True
    except Exception:
        batch = False

    def _res(x):
        return np.asarray(x.result if hasattr(x, "result") else x).ravel()

    class _W:
        backend = "cuda"
        def decode_batch(self, dets, device=None):
            d = np.asarray(dets).astype(np.float32)
            rb = dec.decode_batch(d) if batch else [dec.decode(r.tolist()) for r in d]
            P = np.array([_res(x) for x in rb]).reshape(len(d), nL)
            return P > 0.5
    return Entry(name, True, _W())


# --- accuracy tier (LER + Wilson CI), the standard axis ---
def accuracy(entry, dets, obs):
    pred = entry.decoder.decode_batch(np.ascontiguousarray(dets))
    fails = int(np.any(pred != obs, axis=1).sum())
    lo, hi = wilson_ci(fails, len(dets))
    return {"fails": fails, "shots": len(dets), "ler": fails/len(dets),
            "ler_ci95": [lo, hi]}


# --- serving tier (the NEW axis): max sustained qubits/GPU + capacity at a few
#     SLA budgets (the latency-capacity tradeoff; one fixed SLA is unfair across
#     decoders whose base latency differs by 100x). "sustained" = bounded backlog.
# v2a: FINER K grid in the contested 0-64 zone (where the accurate decoders'
#      knee lives) + MULTI-SEED so the knee gets an error bar, not a single point.
KS_GRID = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384,
           512, 768, 1024, 1536)
SURFACE_DISTANCES = (3, 5, 7)   # v3 distance sweep


def _sweep(decoder, pool, Ks, t_round, duration, seed):
    rows = []
    for K in Ks:
        r = run_load(decoder, pool, K=K, t_round=t_round, duration=duration, seed=seed)
        rows.append(r)
        if r["overloaded"]:
            break
    ok = [r for r in rows if not r["overloaded"] and r["p99_ms"] is not None]
    return rows, (ok[-1]["K"] if ok else 0), ok


def serving(entry, pool, slas_ms=(100, 250, 500), t_round=1e-3,
            Ks=KS_GRID, duration=3.0, seeds=(0, 1, 2)):
    per_seed, peak, last_rows, last_ok = [], 0.0, [], []
    for sd in seeds:
        rows, mx, ok = _sweep(entry.decoder, pool, Ks, t_round, duration, sd)
        per_seed.append(mx)
        peak = max(peak, max((r["throughput_per_s"] for r in rows), default=0.0))
        last_rows, last_ok = rows, ok
    srt = sorted(per_seed)
    median = srt[len(srt) // 2]
    return {
        "max_sustained_qubits": median,            # median over seeds
        "max_sustained_seeds": per_seed,
        "max_sustained_lo": min(per_seed),
        "max_sustained_hi": max(per_seed),
        "p99_at_max_ms": last_ok[-1]["p99_ms"] if last_ok else None,
        "peak_throughput_per_s": peak,
        "sustained_at_sla": {f"{s}ms": max((r["K"] for r in last_ok if r["p99_ms"] <= s),
                                           default=0) for s in slas_ms},
        "sweep": last_rows,
    }


def run(entries, dets, obs, pool):
    rows = []
    for e in entries:
        a = accuracy(e, dets, obs)
        s = serving(e, pool)
        rows.append({"decoder": e.name, "accurate": e.accurate, **a, "serving": s})
        ci = a["ler_ci95"]
        print(f"  {e.name:30s} LER {a['ler']*100:5.2f}% [{ci[0]*100:4.2f}-{ci[1]*100:4.2f}]  "
              f"max {s['max_sustained_qubits']:>4}q [{s['max_sustained_lo']}-{s['max_sustained_hi']}] "
              f"(p99 {s['p99_at_max_ms'] or 0:.0f}ms) | peak {s['peak_throughput_per_s']:>9.0f} syn/s")
    return rows


def _opt(label, fn):
    try:
        return [fn()]
    except Exception as ex:
        print(f"  (skipped {label}: {type(ex).__name__}: {str(ex)[:60]})")
        return []


def build_codes():
    """The code x decoder matrix. Matching decoders (PyMatching) need a
    decomposed DEM; BP/relay use the plain DEM. Same shots for both."""
    codes = []
    # --- BB [[72,12,6]] qLDPC (matching does NOT apply; BP-family + OSD) ---
    bb = stim.DetectorErrorModel.from_file(FIX+"bb72_r6_p0.003_Z.dem")
    bc = stim.Circuit.from_file(FIX+"bb72_r6_p0.003_Z.stim")
    bd, bo = bc.compile_detector_sampler(seed=0).sample(2000, separate_observables=True)
    be = [tridec_entry(bb, "relay", "tridec Relay-BP"),
          tridec_entry(bb, "bp", "tridec min-sum BP")]
    be += _opt("ldpc BP-OSD", lambda: bposd_entry(bb))
    be += _opt("relay_bp oracle", lambda: relaybp_entry(bb))
    be += _opt("cudaq nv-qldpc", lambda: cudaq_entry(bb))
    be += _opt("cudaq pre-decode cascade", lambda: cudaq_predecode_entry(bb))  # v2b
    codes.append(("BB [[72,12,6]] qLDPC (p=0.003)", np.asarray(bd, bool), np.asarray(bo, bool), be))
    # --- surface code DISTANCE SWEEP (v3): d=3,5,7 -- accuracy improves with d,
    #     serving capacity drops (bigger code = more work/syndrome = fewer q/GPU).
    #     matching IS the reference here; nv-qldpc on H200 (CUDA-only). TN omitted
    #     (intractable at d>=5/multi-round; see tn_small.py).
    for d in SURFACE_DISTANCES:
        sc = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d,
                after_clifford_depolarization=0.003, after_reset_flip_probability=0.003,
                before_measure_flip_probability=0.003, before_round_data_depolarization=0.003)
        sd, so = sc.compile_detector_sampler(seed=0).sample(2000, separate_observables=True)
        plain = sc.detector_error_model(decompose_errors=False)
        se = [tridec_entry(plain, "relay", "tridec Relay-BP"),
              tridec_entry(plain, "bp", "tridec min-sum BP")]
        se += _opt("PyMatching MWPM", lambda dem=sc.detector_error_model(decompose_errors=True): pymatching_entry(dem))
        se += _opt("cudaq nv-qldpc", lambda p=plain: cudaq_entry(p))
        codes.append((f"surface d={d} rotated_memory_z (p=0.003)",
                      np.asarray(sd, bool), np.asarray(so, bool), se))
    return codes


if __name__ == "__main__":
    backend = tridec.resolve_backend("auto")
    print(f"QEC decode-serving benchmark | backend={backend}")
    all_rows = []
    for label, dets, obs, entries in build_codes():
        print(f"\n== {label} ==")
        rows = run(entries, dets, obs, np.ascontiguousarray(dets))
        for r in rows:
            r["code"] = label
        all_rows += rows
    out = {"backend": backend, "slas_ms": [100, 250, 500], "rows": all_rows}
    json.dump(out, open("/Users/bledden/Documents/tridec-serve/benchmark/results_metal.json", "w"), indent=2)
    print("\nsaved benchmark/results_metal.json")
