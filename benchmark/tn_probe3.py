"""Nail the TN decode interface (float syndromes; result = logical-flip prob) and
check surface d=5 feasibility + timing before wiring into the sweep."""
import time, numpy as np, stim, cudaq_qec as qec
from tridec.dem import extract

sc = stim.Circuit.generated("surface_code:rotated_memory_z", distance=5, rounds=5,
        after_clifford_depolarization=0.003, after_reset_flip_probability=0.003,
        before_measure_flip_probability=0.003, before_round_data_depolarization=0.003)
det, obs = sc.compile_detector_sampler(seed=0).sample(200, separate_observables=True)
det = np.asarray(det, bool); obs = np.asarray(obs, bool)
plain = sc.detector_error_model(decompose_errors=False)
ex = extract(plain); Hd = ex["H"].toarray().astype(np.uint8)
pri = list(np.clip(np.asarray(ex["priors"]), 1e-6, 1 - 1e-6)); Lo = ex["Lo"].toarray().astype(np.uint8)
print("surface d=5: H", Hd.shape, "Lo", Lo.shape)

t0 = time.perf_counter()
dec = qec.get_decoder("tensor_network_decoder", Hd, logical_obs=Lo, noise_model=pri)
print("constructed in", round(time.perf_counter() - t0, 2), "s")

# single decode (float syndrome)
t = time.perf_counter()
r = dec.decode(det[0].astype(np.float32).tolist())
dt1 = time.perf_counter() - t
print("single decode:", round(dt1, 3), "s | type", type(r).__name__,
      "| result", np.asarray(getattr(r, "result", r)))

# batch?
try:
    t = time.perf_counter()
    rb = dec.decode_batch(det[:50].astype(np.float32))
    print("decode_batch(50) ok in", round(time.perf_counter() - t, 2), "s | len", len(rb))
    P = np.array([np.asarray(x.result) for x in rb]).reshape(50, Lo.shape[0])
    has_batch = True
except Exception as e:
    print("decode_batch fail:", str(e)[:80], "-> per-shot")
    has_batch = False

# accuracy over 200 (per-shot if needed), with timing
t = time.perf_counter()
if has_batch:
    rb = dec.decode_batch(det.astype(np.float32))
    P = np.array([np.asarray(x.result) for x in rb]).reshape(len(det), Lo.shape[0])
else:
    P = np.array([np.asarray(dec.decode(row.astype(np.float32).tolist()).result) for row in det]).reshape(len(det), Lo.shape[0])
dt = time.perf_counter() - t
pred = P > 0.5
ler = np.any(pred != obs, axis=1).mean()
print(f"TN LER {ler*100:.2f}%  over {len(det)} shots in {dt:.1f}s -> {len(det)/dt:.0f} syn/s")
