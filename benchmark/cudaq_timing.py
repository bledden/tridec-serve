"""Where does cudaq nv-qldpc's serving throughput actually go? Time (a) GPU
decode_batch only, (b) result-marshalling only, (c) full adapter, for a 1024
batch. Tells us if the low serving number is the GPU or the Python API."""
import time, numpy as np, stim, cudaq_qec as qec
from tridec.dem import extract

fix = "/workspace/tridec/tests/fixtures/bb72/bb72_r6_p0.003_Z"
dem = stim.DetectorErrorModel.from_file(fix + ".dem")
circ = stim.Circuit.from_file(fix + ".stim")
det, _ = circ.compile_detector_sampler(seed=0).sample(2000, separate_observables=True)
det = np.asarray(det, np.uint8)
ex = extract(dem)
Hd = ex["H"].toarray().astype(np.uint8)
pri = list(np.clip(np.asarray(ex["priors"]), 1e-6, 1 - 1e-6))
Lo = ex["Lo"].toarray().astype(np.uint8)
dec = qec.get_decoder("nv-qldpc-decoder", Hd, error_rate_vec=pri, use_sparsity=True)

B = 1024
batch = det[np.arange(B) % len(det)]
try:
    dec.decode_batch(np.zeros((2, Hd.shape[0]), np.uint8)); as_list = False
except Exception:
    as_list = True
print("batch input form:", "list-of-lists" if as_list else "ndarray")
arg = [r.tolist() for r in batch] if as_list else batch

def timeit(fn, n=10):
    fn()  # warm
    t = time.perf_counter()
    for _ in range(n): r = fn()
    dt = (time.perf_counter() - t) / n
    return dt, r

# (a) GPU decode only
dt_gpu, rb = timeit(lambda: dec.decode_batch(arg))
# (b) marshalling only (reuse rb)
dt_marsh, _ = timeit(lambda: ((np.array([np.asarray(x.result) for x in rb]) > 0.5).astype(np.uint8) @ Lo.T) & 1)
# (c) full adapter
def full():
    r = dec.decode_batch(arg)
    return ((np.array([np.asarray(x.result) for x in r]) > 0.5).astype(np.uint8) @ Lo.T) & 1
dt_full, _ = timeit(full)

print(f"GPU decode_batch only : {dt_gpu*1e3:7.1f} ms/batch -> {B/dt_gpu:10.0f} syn/s")
print(f"result-marshalling    : {dt_marsh*1e3:7.1f} ms/batch -> {B/dt_marsh:10.0f} syn/s")
print(f"full adapter          : {dt_full*1e3:7.1f} ms/batch -> {B/dt_full:10.0f} syn/s")
print(f"marshalling is {dt_marsh/dt_full*100:.0f}% of full-adapter time")
