import cudaq_qec as qec, numpy as np, stim
from tridec.dem import extract

fix = "/workspace/tridec/tests/fixtures/bb72/bb72_r6_p0.003_Z"
dem = stim.DetectorErrorModel.from_file(fix + ".dem")
circ = stim.Circuit.from_file(fix + ".stim")
det, obs = circ.compile_detector_sampler(seed=0).sample(300, separate_observables=True)
det = np.asarray(det, bool); obs = np.asarray(obs, bool)
ex = extract(dem)
Hd = ex["H"].toarray().astype(np.uint8)
pri = np.clip(np.asarray(ex["priors"]), 1e-6, 1 - 1e-6)
Lo = ex["Lo"].toarray().astype(np.uint8)

dec = qec.get_decoder("nv-qldpc-decoder", Hd, error_rate_vec=list(pri), use_sparsity=True)
r = dec.decode(det[0].astype(np.uint8).tolist())
print("single type:", type(r).__name__, "attrs:", [a for a in dir(r) if not a.startswith("_")])
res = getattr(r, "result", None); conv = getattr(r, "converged", None)
print("converged:", conv, "result len:", (len(res) if res is not None else None))
print("sample result:", np.asarray(res)[:6] if res is not None else None)

rb = dec.decode_batch([d.astype(np.uint8).tolist() for d in det])
print("batch type:", type(rb).__name__, "len", len(rb))
E = np.array([np.asarray(x.result) for x in rb])
print("E shape", E.shape, "dtype", E.dtype, "max", float(E.max()))
Eb = (E > 0.5).astype(np.uint8)
pred = (Eb @ Lo.T) & 1
fails = int(np.any(pred.astype(bool) != obs, axis=1).sum())
print("cudaq nv-qldpc LER = %d/%d = %.2f%%" % (fails, len(det), fails / len(det) * 100))
