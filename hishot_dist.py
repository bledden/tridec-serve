"""Publication-grade accuracy: re-run the surface distance ladder (d=3..13) at
50k shots for PyMatching + tridec min-sum BP, to tighten the Wilson CIs that were
single-digit-failure thin at 2000 shots. Chunked decode (8192) so big batches fit.
LER is config-independent, so this is the clean accuracy receipt; capacity numbers
(separate axis) are unaffected."""
import os, json, numpy as np, stim
os.environ.setdefault("TRIDEC_FIX", "/workspace/bench/fixtures/")
from qecserve_bench import tridec_entry, pymatching_entry
from tridec.validation import wilson_ci

SHOTS, P, CHUNK = 50000, 0.003, 8192
PLAT = os.environ.get("TRIDEC_PLAT", "mi300x")

def acc(e, det, obs):
    fails = 0
    for i in range(0, len(det), CHUNK):
        p = np.asarray(e.decoder.decode_batch(np.ascontiguousarray(det[i:i+CHUNK])))
        fails += int(np.any(p != obs[i:i+CHUNK], axis=1).sum())
    lo, hi = wilson_ci(fails, len(det))
    return fails, fails / len(det), [lo, hi]

out = []
for d in [3, 5, 7, 9, 11, 13]:
    sc = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d,
            after_clifford_depolarization=P, after_reset_flip_probability=P,
            before_measure_flip_probability=P, before_round_data_depolarization=P)
    det, obs = sc.compile_detector_sampler(seed=0).sample(SHOTS, separate_observables=True)
    det = np.asarray(det, bool); obs = np.asarray(obs, bool)
    plain = sc.detector_error_model(decompose_errors=False)
    for e in [pymatching_entry(sc.detector_error_model(decompose_errors=True)),
              tridec_entry(plain, "bp", "tridec min-sum BP")]:
        f, ler, ci = acc(e, det, obs)
        print(f"d={d:2d} {e.name:28s} LER {ler*100:6.3f}% [{ci[0]*100:.3f}-{ci[1]*100:.3f}]  ({f}/{SHOTS})", flush=True)
        out.append({"code": f"surface d={d}", "decoder": e.name, "shots": SHOTS,
                    "fails": f, "ler": ler, "ler_ci95": ci})
json.dump(out, open(f"/workspace/bench/results_hishot_dist_{PLAT}.json", "w"), indent=2)
print("saved")
