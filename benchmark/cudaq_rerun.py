"""Surgically recompute just the two cudaq nv-qldpc rows with the fair OSD config
and patch them into results_h200.json (the other rows are unchanged)."""
import json, numpy as np, stim
from qecserve_bench import cudaq_entry, accuracy, serving

RES = "/workspace/tridec-serve/benchmark/results_h200.json"
FIX = "/workspace/tridec/tests/fixtures/bb72/"

def bb():
    dem = stim.DetectorErrorModel.from_file(FIX + "bb72_r6_p0.003_Z.dem")
    c = stim.Circuit.from_file(FIX + "bb72_r6_p0.003_Z.stim")
    d, o = c.compile_detector_sampler(seed=0).sample(2000, separate_observables=True)
    return "BB [[72,12,6]] qLDPC (p=0.003)", cudaq_entry(dem), np.asarray(d, bool), np.asarray(o, bool)

def surf():
    c = stim.Circuit.generated("surface_code:rotated_memory_z", distance=5, rounds=5,
            after_clifford_depolarization=0.003, after_reset_flip_probability=0.003,
            before_measure_flip_probability=0.003, before_round_data_depolarization=0.003)
    d, o = c.compile_detector_sampler(seed=0).sample(2000, separate_observables=True)
    plain = c.detector_error_model(decompose_errors=False)
    return "surface d=5 rotated_memory_z (p=0.003)", cudaq_entry(plain), np.asarray(d, bool), np.asarray(o, bool)

doc = json.load(open(RES))
for code, e, dets, obs in [bb(), surf()]:
    a = accuracy(e, dets, obs)
    s = serving(e, np.ascontiguousarray(dets))
    print(f"{code[:22]:22s} {e.name}: LER {a['ler']*100:.2f}%  max {s['max_sustained_qubits']}q  peak {s['peak_throughput_per_s']:.0f}/s")
    row = {"decoder": e.name, "accurate": e.accurate, **a, "serving": s, "code": code}
    # replace any existing cudaq row for this code, else append
    doc["rows"] = [r for r in doc["rows"] if not (r["code"] == code and "nv-qldpc" in r["decoder"])]
    doc["rows"].append(row)

json.dump(doc, open(RES, "w"), indent=2)
print("patched", RES)
