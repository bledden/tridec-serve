"""Bulletproof the head-to-head: sweep nv-qldpc THROUGHPUT knobs (bp_batch_size,
n_threads, max_iterations, bp_method) on a 1024 batch, with LER held, so the
"~16x slower than tridec on NVIDIA's own GPU" claim isn't a bad-config artifact.
Default-config serving was ~1.5-2.7k syn/s -> 1-2 qubits; if a config does far
better we report THAT."""
import time, numpy as np, stim, cudaq_qec as qec
from tridec.dem import extract

fix = "/workspace/tridec/tests/fixtures/bb72/bb72_r6_p0.003_Z"
dem = stim.DetectorErrorModel.from_file(fix + ".dem")
circ = stim.Circuit.from_file(fix + ".stim")
det, obs = circ.compile_detector_sampler(seed=0).sample(2000, separate_observables=True)
det = np.asarray(det, np.uint8); obs = np.asarray(obs, bool)
ex = extract(dem)
Hd = ex["H"].toarray().astype(np.uint8)
pri = list(np.clip(np.asarray(ex["priors"]), 1e-6, 1 - 1e-6))
Lo = ex["Lo"].toarray().astype(np.uint8)

# print defaults
cfg = qec.nv_qldpc_decoder_config
print("config knobs:", [a for a in dir(cfg) if not a.startswith("_") and a not in
      ("from_heterogeneous_map", "to_heterogeneous_map")])

B = 1024
batch = det[np.arange(B) % len(det)]

def run(label, **kw):
    try:
        dec = qec.get_decoder("nv-qldpc-decoder", Hd, error_rate_vec=pri,
                              use_sparsity=True, use_osd=True, **kw)
    except Exception as e:
        print(f"  {label:40s} CTOR FAIL: {str(e)[:70]}"); return
    try:
        rb = dec.decode_batch(det)
        E = (np.array([np.asarray(x.result) for x in rb]) > 0.5).astype(np.uint8)
        ler = np.any(((E @ Lo.T) & 1).astype(bool) != obs, axis=1).mean()
        dec.decode_batch(batch)  # warm
        t = time.perf_counter()
        for _ in range(10): dec.decode_batch(batch)
        dt = (time.perf_counter() - t) / 10
        print(f"  {label:40s} LER {ler*100:5.2f}%  {B/dt:10.0f} syn/s  ({dt*1e3:6.1f} ms/{B})")
    except Exception as e:
        print(f"  {label:40s} RUN FAIL: {str(e)[:70]}")

print("\n-- throughput sweep (1024 batch, OSD on) --")
run("default")
for bs in [256, 1024, 4096]:
    run(f"bp_batch_size={bs}", bp_batch_size=bs)
for nt in [8, 32, 128]:
    run(f"n_threads={nt}", n_threads=nt)
for it in [10, 20]:
    run(f"max_iterations={it}", max_iterations=it)
for m in ["min_sum", "product_sum"]:
    run(f"bp_method={m}", bp_method=m)
# best-guess combined high-throughput config
run("bp_batch_size=4096 + n_threads=128", bp_batch_size=4096, n_threads=128)
run("bs=4096 + nt=128 + max_iter=20", bp_batch_size=4096, n_threads=128, max_iterations=20)
