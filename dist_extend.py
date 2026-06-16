"""Extend the surface distance sweep to d=9 and d=11 on the MI300X (192 GB) —
past the H200's d=7. Focused on the two decoders with meaningful high-d capacity
(tridec min-sum BP on GPU + PyMatching); skips the slow CPU decoders. Extends the
v3 capacity-vs-distance + BP-LER-rises findings."""
import os, json, numpy as np, stim
os.environ.setdefault("TRIDEC_FIX", "/workspace/bench/fixtures/")
from qecserve_bench import tridec_entry, pymatching_entry, accuracy, serving

SHOTS, P = 2000, 0.003
out = []
for d in [9, 11]:
    sc = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d,
            after_clifford_depolarization=P, after_reset_flip_probability=P,
            before_measure_flip_probability=P, before_round_data_depolarization=P)
    det, obs = sc.compile_detector_sampler(seed=0).sample(SHOTS, separate_observables=True)
    det = np.asarray(det, bool); obs = np.asarray(obs, bool)
    plain = sc.detector_error_model(decompose_errors=False)
    entries = [tridec_entry(plain, "bp", "tridec min-sum BP")]
    try:
        entries.append(pymatching_entry(sc.detector_error_model(decompose_errors=True)))
    except Exception as e:
        print("pymatching skipped:", str(e)[:60])
    for e in entries:
        a = accuracy(e, det, obs); s = serving(e, np.ascontiguousarray(det))
        print(f"d={d} {e.name:28s} LER {a['ler']*100:5.2f}%  cap {s['max_sustained_qubits']:>5}q [{s['max_sustained_lo']}-{s['max_sustained_hi']}]")
        out.append({"code": f"surface d={d} rotated_memory_z (p=0.003)", "decoder": e.name, **a, "serving": s})

json.dump({"backend": "triton", "rows": out},
          open("/workspace/bench/results_dist_d9d11_mi300x.json", "w"), indent=2)
print("saved results_dist_d9d11_mi300x.json")
