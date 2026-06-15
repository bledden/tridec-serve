"""Re-measure the tridec Relay-BP rows on the DO MI300X with the rocm7-retuned
megakernel config (1024,16) -- the shipped (512,8) was tuned for rocm6.2/triton3.1.
Patches results_mi300x.json (relay rows only; BP & matching already at parity)."""
import json, numpy as np, stim
import tridec.backends.megakernel as mk
mk._CUDA_TUNED.setdefault("gfx942", {})["relay"] = (1024, 16)   # best rocm7/triton3.4
from qecserve_bench import tridec_entry, accuracy, serving

RES = "/workspace/bench/benchmark/results_mi300x.json"
FIX = "/workspace/bench/fixtures/"

def bb():
    dem = stim.DetectorErrorModel.from_file(FIX + "bb72_r6_p0.003_Z.dem")
    c = stim.Circuit.from_file(FIX + "bb72_r6_p0.003_Z.stim")
    d, o = c.compile_detector_sampler(seed=0).sample(2000, separate_observables=True)
    return "BB [[72,12,6]] qLDPC (p=0.003)", dem, np.asarray(d, bool), np.asarray(o, bool)

def surf(dist):
    sc = stim.Circuit.generated("surface_code:rotated_memory_z", distance=dist, rounds=dist,
            after_clifford_depolarization=0.003, after_reset_flip_probability=0.003,
            before_measure_flip_probability=0.003, before_round_data_depolarization=0.003)
    d, o = sc.compile_detector_sampler(seed=0).sample(2000, separate_observables=True)
    return f"surface d={dist} rotated_memory_z (p=0.003)", sc.detector_error_model(decompose_errors=False), np.asarray(d, bool), np.asarray(o, bool)

doc = json.load(open(RES))
for code, dem, dets, obs in [bb(), surf(3), surf(5), surf(7)]:
    e = tridec_entry(dem, "relay", "tridec Relay-BP")
    a = accuracy(e, dets, obs); s = serving(e, np.ascontiguousarray(dets))
    print(f"{code[:22]:22s} relay (1024,16): LER {a['ler']*100:.2f}%  max {s['max_sustained_qubits']}q [{s['max_sustained_lo']}-{s['max_sustained_hi']}]")
    row = {"decoder": e.name, "accurate": e.accurate, **a, "serving": s, "code": code}
    doc["rows"] = [r for r in doc["rows"] if not (r["code"] == code and r["decoder"] == "tridec Relay-BP")]
    doc["rows"].append(row)
json.dump(doc, open(RES, "w"), indent=2)
print("patched", RES)
