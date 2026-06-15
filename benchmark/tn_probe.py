"""After the torch upgrade from the TN-extra install: (1) confirm tridec + nv-qldpc
still decode correctly, (2) find the TN decoder name + construct it on BB."""
import numpy as np, stim, cudaq_qec as qec
from tridec.dem import extract
import tridec

fix = "/workspace/tridec/tests/fixtures/bb72/bb72_r6_p0.003_Z"
dem = stim.DetectorErrorModel.from_file(fix + ".dem")
circ = stim.Circuit.from_file(fix + ".stim")
det, obs = circ.compile_detector_sampler(seed=0).sample(300, separate_observables=True)
det = np.asarray(det, bool); obs = np.asarray(obs, bool)
ex = extract(dem); Hd = ex["H"].toarray().astype(np.uint8)
pri = list(np.clip(np.asarray(ex["priors"]), 1e-6, 1 - 1e-6)); Lo = ex["Lo"].toarray().astype(np.uint8)

# (1) tridec still correct?
d = tridec.from_dem(dem, algorithm="relay")
pred = d.decode_batch(np.ascontiguousarray(det))
print("tridec relay LER:", round(np.any(pred != obs, axis=1).mean() * 100, 2), "% (backend", d.backend + ")")

# (1b) nv-qldpc still correct?
dec = qec.get_decoder("nv-qldpc-decoder", Hd, error_rate_vec=pri, use_sparsity=True, use_osd=True, bp_batch_size=4096)
rb = dec.decode_batch(det.astype(np.uint8))
E = (np.array([np.asarray(x.result) for x in rb]) > 0.5).astype(np.uint8)
print("nv-qldpc+OSD LER:", round(np.any(((E @ Lo.T) & 1).astype(bool) != obs, axis=1).mean() * 100, 2), "%")

# (2) TN decoder name + construct
print("TN candidates:")
for n in ["tensor_network_decoder", "tensor-network-decoder", "tn-decoder", "tensor_network"]:
    try:
        td = qec.get_decoder(n, Hd)
        print("  OK", n, "-> methods:", [m for m in dir(td) if not m.startswith("_")][:8])
    except Exception as e:
        print("  nope", n, "|", str(e)[:70])
