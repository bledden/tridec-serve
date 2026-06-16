"""Last AMD-window items: (A) d=13 distance point (matches Gu et al.'s max,
exploits 192 GB), (B) a big best-tool fleet (surface d3,d5,d7 -> matching + BB
qLDPC -> BP) to find how many TOTAL mixed logical qubits one MI300X sustains."""
import os, json, numpy as np, stim
os.environ.setdefault("TRIDEC_FIX", "/workspace/bench/fixtures/")
from qecserve_bench import tridec_entry, pymatching_entry, accuracy, serving
from fleetbench import surface_lane, bb_lane
from fleet_serve import max_sustained_fleet

# (A) d=13
P = 0.003
sc = stim.Circuit.generated("surface_code:rotated_memory_z", distance=13, rounds=13,
        after_clifford_depolarization=P, after_reset_flip_probability=P,
        before_measure_flip_probability=P, before_round_data_depolarization=P)
det, obs = sc.compile_detector_sampler(seed=0).sample(2000, separate_observables=True)
det = np.asarray(det, bool); obs = np.asarray(obs, bool)
plain = sc.detector_error_model(decompose_errors=False)
rows = []
for e in [tridec_entry(plain, "bp", "tridec min-sum BP"),
          pymatching_entry(sc.detector_error_model(decompose_errors=True))]:
    a = accuracy(e, det, obs); s = serving(e, np.ascontiguousarray(det))
    print(f"d=13 {e.name:28s} LER {a['ler']*100:5.2f}%  cap {s['max_sustained_qubits']}q")
    rows.append({"code": "surface d=13 rotated_memory_z (p=0.003)", "decoder": e.name, **a, "serving": s})
json.dump({"backend": "triton", "rows": rows}, open("/workspace/bench/results_dist_d13_mi300x.json", "w"), indent=2)

# (B) big best-tool fleet: 3 surface distances (matching) + a qLDPC block (BP)
def big(scale):
    return [surface_lane(3, scale, "pymatching"),
            surface_lane(5, scale, "pymatching"),
            surface_lane(7, scale, "pymatching"),
            bb_lane(scale, "tridec-bp")]
bf = max_sustained_fleet(big, [64, 128, 256, 384, 512, 768, 1024], duration=3.0)
print("BIG best-tool fleet (surf d3,d5,d7 matching + BB qLDPC BP):", bf["max_fleet"], bf["seeds"])
for r in bf["last_sweep"]:
    print(f"  total={r['fleet_K']:>5} {'OVERLOAD' if r['overloaded'] else 'ok'} p99={r['p99_ms']}")
json.dump(bf, open("/workspace/bench/results_bigfleet_mi300x.json", "w"), indent=2)
print("saved")
