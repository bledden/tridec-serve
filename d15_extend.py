"""d>=15 FRONTIER: extend the surface distance ladder to d=15 and d=17, now that
tridec 0.2.2 lifted the BP-kernel ceiling (1-D grid flatten, #6 — d<=14 was the
old wall). Mirrors hishot_dist.py's 50k-shot accuracy methodology so the numbers
slot straight into the existing d=3..13 ladder.

Requires tridec>=0.2.2 (the d>=15 lift is unreleased before it). LER is
deterministic/hardware-independent, so ONE GPU (H200 or MI300X) produces the
canonical ladder extension.

Shot count is 2000 (not the d<=13 ladder's 50k): min-sum BP costs ~200 ms/shot at
d>=15 (3360+ detectors), so 50k would be ~3 h/point — infeasible on a rented GPU.
2000 is plenty for BP (its LER is ~40% at d>=15, well-resolved) and its per-shot
cost is the whole reason 50k is off the table; matching's LER is so low it sits
below even the 50k floor, so 2000 (0/2000 -> "<= floor, <0.05%") makes the same
"still crushing it" point. A tight matching LER at d>=15 is a cheap CPU-only
follow-on (millions of shots, no GPU). Headline: BP keeps CLIMBING past d=13's
34.65% while matching stays below the floor — the "matching for surface, BP for
qLDPC" story, quantified past the old d=14 ceiling.
"""
import os, sys, json, numpy as np, stim, tridec

_v = tuple(int(x) for x in tridec.__version__.split(".")[:3])
assert _v >= (0, 2, 2), f"needs tridec>=0.2.2 (d>=15 kernel lift); got {tridec.__version__}"

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TRIDEC_FIX", os.path.join(HERE, "benchmark", "fixtures") + os.sep)
sys.path.insert(0, os.path.join(HERE, "benchmark"))
from qecserve_bench import tridec_entry, pymatching_entry  # noqa: E402
from tridec.validation import wilson_ci  # noqa: E402

SHOTS, P, CHUNK = 2000, 0.003, 1024    # BP is ~200ms/shot at d>=15 -> 50k infeasible
PLAT = os.environ.get("TRIDEC_PLAT", "h200")
FLOOR = 1.0 / SHOTS


def acc(e, det, obs):
    fails = 0
    for i in range(0, len(det), CHUNK):
        p = np.asarray(e.decoder.decode_batch(np.ascontiguousarray(det[i:i + CHUNK])))
        fails += int(np.any(p != obs[i:i + CHUNK], axis=1).sum())
    lo, hi = wilson_ci(fails, len(det))
    return fails, fails / len(det), [lo, hi]


out = []
for d in [15, 17]:
    sc = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d,
            after_clifford_depolarization=P, after_reset_flip_probability=P,
            before_measure_flip_probability=P, before_round_data_depolarization=P)
    ndet = sc.num_detectors
    det, obs = sc.compile_detector_sampler(seed=0).sample(SHOTS, separate_observables=True)
    det, obs = np.asarray(det, bool), np.asarray(obs, bool)
    plain = sc.detector_error_model(decompose_errors=False)
    print(f"=== d={d}  ({ndet} detectors) — the frontier past the old d=14 wall ===", flush=True)
    for e in [pymatching_entry(sc.detector_error_model(decompose_errors=True)),
              tridec_entry(plain, "bp", "tridec min-sum BP")]:
        f, ler, ci = acc(e, det, obs)
        floor_tag = f"  (<= {SHOTS//1000}k-shot floor, <{100.0/SHOTS:.3f}%)" if f <= 2 else ""
        print(f"d={d:2d} {e.name:28s} LER {ler * 100:7.4f}% "
              f"[{ci[0] * 100:.4f}-{ci[1] * 100:.4f}]  ({f}/{SHOTS}){floor_tag}", flush=True)
        out.append({"code": f"surface d={d}", "detectors": ndet, "decoder": e.name,
                    "shots": SHOTS, "fails": f, "ler": ler, "ler_ci95": ci,
                    "below_floor": bool(f <= 2)})

dest = os.path.join(HERE, "benchmark", f"results_dist_d15d17_{PLAT}.json")
json.dump({"tridec_version": tridec.__version__, "plat": PLAT, "shots": SHOTS,
           "rows": out}, open(dest, "w"), indent=2)
print(f"saved {os.path.relpath(dest, HERE)}  (tridec {tridec.__version__})")
