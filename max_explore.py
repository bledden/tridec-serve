"""Device-max: push surface distance upward until the box hits a wall (GPU OOM on
the BP lane, or impractical wall-time), to find each accelerator's ceiling and
where the MI300X's 192 GB pulls ahead of the H200's 141 GB. matching=CPU reference;
BP=GPU (chunked) is the memory-bound lane that decides the GPU ceiling."""
import os, json, time, traceback, numpy as np, stim
os.environ.setdefault("TRIDEC_FIX", "/workspace/bench/fixtures/")
from qecserve_bench import tridec_entry, pymatching_entry

PLAT = os.environ.get("TRIDEC_PLAT", "gpu")
SHOTS, P, CHUNK, TIME_CAP = 4000, 0.003, 2048, 240
rows = []
for d in range(13, 36, 2):                       # 13,15,...,35
    t0 = time.perf_counter()
    try:
        sc = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d,
                after_clifford_depolarization=P, after_reset_flip_probability=P,
                before_measure_flip_probability=P, before_round_data_depolarization=P)
        det, obs = sc.compile_detector_sampler(seed=0).sample(SHOTS, separate_observables=True)
        det = np.asarray(det, bool); obs = np.asarray(obs, bool)
        plain = sc.detector_error_model(decompose_errors=False)
        row = {"d": d, "detectors": int(det.shape[1])}
        m = pymatching_entry(sc.detector_error_model(decompose_errors=True))
        pm = np.asarray(m.decoder.decode_batch(np.ascontiguousarray(det)))
        row["matching_ler"] = float(np.any(pm != obs, axis=1).mean())
        bp = tridec_entry(plain, "bp"); fails = 0
        for i in range(0, len(det), CHUNK):
            p = np.asarray(bp.decoder.decode_batch(np.ascontiguousarray(det[i:i+CHUNK])))
            fails += int(np.any(p != obs[i:i+CHUNK], axis=1).sum())
        row["bp_ler"] = fails / len(det)
        row["sec"] = round(time.perf_counter() - t0, 1)
        print(f"d={d:2d} dets={row['detectors']:>6} matching {row['matching_ler']*100:6.3f}% "
              f"bp {row['bp_ler']*100:6.2f}%  ({row['sec']}s)", flush=True)
        rows.append(row)
        if row["sec"] > TIME_CAP:
            print(f"d={d} exceeded {TIME_CAP}s wall -> impractical, stopping.", flush=True); break
    except Exception as ex:
        msg = f"{type(ex).__name__}: {str(ex)[:120]}"
        print(f"d={d} WALL HIT: {msg}", flush=True)
        rows.append({"d": d, "error": msg}); break

maxd = max((r["d"] for r in rows if "error" not in r), default=None)
json.dump({"plat": PLAT, "max_d_reached": maxd, "rows": rows},
          open(f"/workspace/bench/results_maxd_{PLAT}.json", "w"), indent=2)
print(f"MAX d reached on {PLAT}: {maxd}")
