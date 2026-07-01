"""v4 fleet benchmark: one GPU serving a HETEROGENEOUS fleet — logical qubits at
mixed code distances (surface d=3/5/7) and mixed families (surface + BB qLDPC),
each routed to its own decoder instance on the shared accelerator.

The questions v4 answers that v1-v3 couldn't:
  - max sustained TOTAL fleet on one GPU when the qubits are NOT homogeneous,
  - how that compares to the homogeneous per-code baselines (the routing cost),
  - the vendor-neutrality lever: a mixed fleet needs MULTIPLE decoder instances
    (one per code/distance) on one accelerator — a single CUDA-locked decoder
    can't be the whole fleet's decoder, and can't run on AMD at all.

Decoder factory picks tridec (GPU, when available) else PyMatching (CPU) so the
routing logic is testable locally; the headline numbers come from a GPU run.
"""
import sys, os, json, numpy as np, stim
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fleet_serve import run_fleet_load, max_sustained_fleet

# bundled fixtures by default; override with TRIDEC_FIX for a pod's repo path.
FIX = os.environ.get("TRIDEC_FIX", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "benchmark", "fixtures") + os.sep)
P = 0.003
SHOTS = 1500


def _surface(d):
    sc = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d,
            after_clifford_depolarization=P, after_reset_flip_probability=P,
            before_measure_flip_probability=P, before_round_data_depolarization=P)
    det, _ = sc.compile_detector_sampler(seed=0).sample(SHOTS, separate_observables=True)
    return sc, np.asarray(det, bool)


def _pymatching(dem):
    import pymatching
    m = pymatching.Matching.from_detector_error_model(dem)

    class _W:
        backend = "cpu"
        def decode_batch(self, dets, device=None):
            return np.asarray(m.decode_batch(np.asarray(dets).astype(np.uint8))).astype(bool)
    return _W()


def _tridec(dem, algo):
    import tridec
    return tridec.from_dem(dem, algorithm=algo)


def surface_lane(d, K, kind):
    sc, pool = _surface(d)
    if kind == "pymatching":
        dec = _pymatching(sc.detector_error_model(decompose_errors=True))
    else:
        dec = _tridec(sc.detector_error_model(decompose_errors=False), kind.split("-")[1])
    return {"name": f"surf-d{d}", "decoder": dec, "pool": pool, "K": K}


def bb_lane(K, kind):  # BB qLDPC: matching N/A -> always a BP-family decoder
    dem = stim.DetectorErrorModel.from_file(FIX + "bb72_r6_p0.003_Z.dem")
    c = stim.Circuit.from_file(FIX + "bb72_r6_p0.003_Z.stim")
    det, _ = c.compile_detector_sampler(seed=0).sample(SHOTS, separate_observables=True)
    algo = kind.split("-")[1] if kind.startswith("tridec") else "bp"
    return {"name": "bb-qldpc", "decoder": _tridec(dem, algo), "pool": np.asarray(det, bool), "K": K}


def main():
    import tridec
    backend = tridec.resolve_backend("auto")
    gpu = backend in ("triton",) or backend == "metal"
    surf_kind = "tridec-bp" if gpu else "pymatching"
    have_bb = gpu and os.path.exists(FIX + "bb72_r6_p0.003_Z.dem")
    print(f"fleet benchmark | backend={backend} | surface decoder={surf_kind} | BB lane={have_bb}")

    SCALES = [4, 8, 16, 32, 64, 128, 256, 512]

    # --- mixed-DISTANCE surface fleet: equal split d=3/d=5/d=7, all on one GPU ---
    def mixed_distance(scale):
        return [surface_lane(3, scale, surf_kind),
                surface_lane(5, scale, surf_kind),
                surface_lane(7, scale, surf_kind)]
    md = max_sustained_fleet(mixed_distance, SCALES, duration=3.0)
    print(f"\nMIXED-DISTANCE surface fleet (d3+d5+d7, equal): max sustained TOTAL "
          f"{md['max_fleet']}q [{md['lo']}-{md['hi']}]  (per-lane ~{md['max_fleet']//3})")

    out = {"backend": backend, "surface_decoder": surf_kind, "mixed_distance": md}

    # --- mixed-FAMILY fleet (GPU only): surface d=5 + BB qLDPC on one GPU ---
    if have_bb:
        def mixed_family(scale):
            return [surface_lane(5, scale, surf_kind), bb_lane(scale, "tridec-bp")]
        mf = max_sustained_fleet(mixed_family, SCALES, duration=3.0)
        print(f"MIXED-FAMILY fleet (surface-d5 + BB-qLDPC, one GPU): max sustained TOTAL "
              f"{mf['max_fleet']}q [{mf['lo']}-{mf['hi']}]")
        out["mixed_family"] = mf

    plat = os.environ.get("TRIDEC_PLAT", backend)
    json.dump(out, open(f"results_fleet_{plat}.json", "w"), indent=2)
    print(f"\nsaved results_fleet_{plat}.json")


if __name__ == "__main__":
    main()
