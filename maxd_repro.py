"""Minimal repro for the tridec BP triton-kernel ceiling: find the exact distance /
detector count that first triggers the kernel error on each stack, with full
versions + the failing call. Goes verbatim into the GitHub issue."""
import os, traceback, numpy as np, stim, tridec, torch, triton
PLAT = os.environ.get("TRIDEC_PLAT", "gpu")
P = 0.003
print(f"=== {PLAT} | torch {torch.__version__} | triton {triton.__version__} | "
      f"hip/cuda {getattr(torch.version,'hip',None) or getattr(torch.version,'cuda',None)} ===")
try:
    print("device:", torch.cuda.get_device_name(0))
except Exception:
    pass
for d in [12, 13, 14, 15]:
    dets = None
    try:
        sc = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d,
                after_clifford_depolarization=P, after_reset_flip_probability=P,
                before_measure_flip_probability=P, before_round_data_depolarization=P)
        s, _ = sc.compile_detector_sampler(seed=0).sample(32, separate_observables=True)
        s = np.asarray(s, bool); dets = s.shape[1]
        dec = tridec.from_dem(sc.detector_error_model(decompose_errors=False), algorithm="bp")
        dec.decode_batch(np.ascontiguousarray(s[:16]))
        print(f"d={d:2d} detectors={dets:>6}  OK")
    except Exception as ex:
        print(f"d={d:2d} detectors={dets}  *** FIRST FAILURE *** {type(ex).__name__}: {str(ex)[:200]}")
        traceback.print_exc()
        break
