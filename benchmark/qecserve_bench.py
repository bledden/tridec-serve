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
def serving(entry, pool, slas_ms=(100, 250, 500), t_round=1e-3,
            Ks=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536), duration=4.0):
    rows = []
    for K in Ks:
        r = run_load(entry.decoder, pool, K=K, t_round=t_round, duration=duration)
        rows.append(r)
        if r["overloaded"]:
            break
    ok = [r for r in rows if not r["overloaded"] and r["p99_ms"] is not None]
    max_sus = ok[-1]["K"] if ok else 0
    return {
        "max_sustained_qubits": max_sus,
        "p99_at_max_ms": ok[-1]["p99_ms"] if ok else None,
        "peak_throughput_per_s": max((r["throughput_per_s"] for r in rows), default=0.0),
        "sustained_at_sla": {f"{s}ms": max((r["K"] for r in ok if r["p99_ms"] <= s),
                                           default=0) for s in slas_ms},
        "sweep": rows,
    }


def run(entries, dets, obs, pool):
    rows = []
    for e in entries:
        a = accuracy(e, dets, obs)
        s = serving(e, pool)
        rows.append({"decoder": e.name, "accurate": e.accurate, **a, "serving": s})
        sla = s["sustained_at_sla"]
        print(f"  {e.name:28s} LER {a['ler']*100:5.2f}%  max {s['max_sustained_qubits']:>4}q "
              f"(p99 {s['p99_at_max_ms'] or 0:.0f}ms) | @SLA "
              f"100ms:{sla['100ms']:>4} 250ms:{sla['250ms']:>4} 500ms:{sla['500ms']:>4} | "
              f"peak {s['peak_throughput_per_s']:>9.0f} syn/s")
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
    codes.append(("BB [[72,12,6]] qLDPC (p=0.003)", np.asarray(bd, bool), np.asarray(bo, bool), be))
    # --- surface d=5 (the canonical code; matching IS the reference here) ---
    sc = stim.Circuit.generated("surface_code:rotated_memory_z", distance=5, rounds=5,
            after_clifford_depolarization=0.003, after_reset_flip_probability=0.003,
            before_measure_flip_probability=0.003, before_round_data_depolarization=0.003)
    sd, so = sc.compile_detector_sampler(seed=0).sample(2000, separate_observables=True)
    plain = sc.detector_error_model(decompose_errors=False)
    se = [tridec_entry(plain, "relay", "tridec Relay-BP"),
          tridec_entry(plain, "bp", "tridec min-sum BP")]
    se += _opt("PyMatching MWPM", lambda: pymatching_entry(sc.detector_error_model(decompose_errors=True)))
    codes.append(("surface d=5 rotated_memory_z (p=0.003)", np.asarray(sd, bool), np.asarray(so, bool), se))
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
