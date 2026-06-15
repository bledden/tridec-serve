"""Be fair to nv-qldpc: enumerate its config knobs and sweep a few (iterations,
OSD on/off) for accuracy + throughput, so the reported number isn't a bad-default
artifact."""
import time, numpy as np, stim, cudaq_qec as qec
from tridec.dem import extract

cfg = getattr(qec, "nv_qldpc_decoder_config", None)
print("nv_qldpc_decoder_config attrs:", [a for a in dir(cfg) if not a.startswith("_")] if cfg else "NONE")
print("get_decoder doc:", (qec.get_decoder.__doc__ or "")[:200])

fix = "/workspace/tridec/tests/fixtures/bb72/bb72_r6_p0.003_Z"
dem = stim.DetectorErrorModel.from_file(fix + ".dem")
circ = stim.Circuit.from_file(fix + ".stim")
det, obs = circ.compile_detector_sampler(seed=0).sample(2000, separate_observables=True)
det = np.asarray(det, np.uint8); obs = np.asarray(obs, bool)
ex = extract(dem)
Hd = ex["H"].toarray().astype(np.uint8)
pri = list(np.clip(np.asarray(ex["priors"]), 1e-6, 1 - 1e-6))
Lo = ex["Lo"].toarray().astype(np.uint8)
B = 1024
batch = det[np.arange(B) % len(det)]

def run(label, **kw):
    try:
        dec = qec.get_decoder("nv-qldpc-decoder", Hd, error_rate_vec=pri, use_sparsity=True, **kw)
    except Exception as e:
        print(f"  {label:34s} CTOR FAIL: {str(e)[:80]}"); return
    # accuracy over all shots
    rb = dec.decode_batch(det)
    E = (np.array([np.asarray(x.result) for x in rb]) > 0.5).astype(np.uint8)
    pred = ((E @ Lo.T) & 1).astype(bool)
    ler = np.any(pred != obs, axis=1).mean()
    # throughput on a 1024 batch
    dec.decode_batch(batch)
    t = time.perf_counter()
    for _ in range(10): dec.decode_batch(batch)
    dt = (time.perf_counter() - t) / 10
    print(f"  {label:34s} LER {ler*100:5.2f}%  {B/dt:9.0f} syn/s  ({dt*1e3:.1f} ms/{B})")

print("\n-- config sweep --")
run("default")
for it in [10, 20, 50]:
    run(f"max_iterations={it}", max_iterations=it)
for osd in [0, 1]:
    run(f"use_osd={osd}", use_osd=bool(osd))
run("osd_order=0", use_osd=True, osd_order=0)
