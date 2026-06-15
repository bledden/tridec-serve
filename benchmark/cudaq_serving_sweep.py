"""Find nv-qldpc's best SERVING config: sweep bp_batch_size through the real
serving harness (not raw timing). Big bp_batch_size = great raw throughput but
heavy fixed overhead on the small batches the round-scheduler sends at low K, so
it overloads early. The fair serving number is the best sustained-K over the
sweep."""
import numpy as np, stim, cudaq_qec as qec
from tridec.dem import extract
from qecserve_bench import serving, Entry

def load_bb():
    fix = "/workspace/tridec/tests/fixtures/bb72/bb72_r6_p0.003_Z"
    dem = stim.DetectorErrorModel.from_file(fix + ".dem")
    c = stim.Circuit.from_file(fix + ".stim")
    d, o = c.compile_detector_sampler(seed=0).sample(2000, separate_observables=True)
    return dem, np.asarray(d, bool), np.asarray(o, bool)

dem, dets, obs = load_bb()
ex = extract(dem)
Hd = ex["H"].toarray().astype(np.uint8)
pri = list(np.clip(np.asarray(ex["priors"]), 1e-6, 1 - 1e-6))
Lo = ex["Lo"].toarray().astype(np.uint8)

def mk(bs):
    dec = qec.get_decoder("nv-qldpc-decoder", Hd, error_rate_vec=pri,
                          use_sparsity=True, use_osd=True, bp_batch_size=bs)
    class _W:
        backend = "cuda"
        def decode_batch(self, d, device=None):
            d = np.asarray(d).astype(np.uint8)
            rb = dec.decode_batch(d)
            E = (np.array([np.asarray(x.result) for x in rb]) > 0.5).astype(np.uint8)
            return ((E @ Lo.T) & 1).astype(bool)
    return Entry(f"nv-qldpc bs={bs}", True, _W())

pool = np.ascontiguousarray(dets)
print("BB qLDPC, nv-qldpc+OSD serving vs bp_batch_size:")
for bs in [64, 128, 256, 512, 1024, 2048, 4096]:
    s = serving(mk(bs), pool)
    print(f"  bp_batch_size={bs:5d}  max_sustained={s['max_sustained_qubits']:>5}q  "
          f"peak={s['peak_throughput_per_s']:>9.0f}/s  p99@max={s['p99_at_max_ms']}")
