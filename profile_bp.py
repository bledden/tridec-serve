"""AMD-leads root cause: time surface-d5 min-sum BP decode_batch across batch sizes
on each stack, to locate WHY MI300X (rocm7/triton3.4) is ~2x the H200 (cuda/triton3.0)
on the surface-d5 BP lane. Launch-bound (batch=1 dominates) vs compute-bound
(batch>=1024) is the key discriminator; also reports the tridec path/config used."""
import os, time, numpy as np, stim, tridec, torch, triton
PLAT = os.environ.get("TRIDEC_PLAT", "gpu")
P = 0.003
print(f"PLAT={PLAT} backend={tridec.resolve_backend('auto')} torch={torch.__version__} "
      f"triton={triton.__version__} hipcuda={getattr(torch.version,'hip',None) or getattr(torch.version,'cuda',None)}")
try:
    print("device:", torch.cuda.get_device_name(0), "arch:", getattr(torch.cuda.get_device_properties(0), 'gcnArchName', 'n/a'))
except Exception as e:
    print("dev info:", e)

sc = stim.Circuit.generated("surface_code:rotated_memory_z", distance=5, rounds=5,
        after_clifford_depolarization=P, after_reset_flip_probability=P,
        before_measure_flip_probability=P, before_round_data_depolarization=P)
det, _ = sc.compile_detector_sampler(seed=0).sample(8192, separate_observables=True)
det = np.asarray(det, bool)
dec = tridec.from_dem(sc.detector_error_model(decompose_errors=False), algorithm="bp")
print(f"BP decoder backend={dec.backend}  detectors(ncols)={det.shape[1]}")
sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
for B in [1, 64, 256, 1024, 4096, 8192]:
    batch = np.ascontiguousarray(det[np.arange(B) % len(det)])
    dec.decode_batch(batch); sync()                      # warm + compile
    t = time.perf_counter()
    for _ in range(20):
        dec.decode_batch(batch)
    sync()
    dt = (time.perf_counter() - t) / 20
    print(f"  B={B:5d}  {dt*1e3:9.3f} ms/call  {B/dt:11.0f} syn/s  {dt*1e6/B:8.2f} us/syn")
