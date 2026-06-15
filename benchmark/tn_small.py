"""Honest TN data point at the scale where it's tractable. d=5/5-rounds hung
(>300s); pinpoint where TN stops being tractable by timing construct+decode at
d=3 (r=3), d=3 (r=1), d=5 (r=1), each under a hard per-step alarm, with flush."""
import signal, time, sys, numpy as np, stim, cudaq_qec as qec
from tridec.dem import extract

class TO(Exception): pass
def _to(s, f): raise TO()
signal.signal(signal.SIGALRM, _to)

def trial(d, r, nshots=40, budget=90):
    sc = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=r,
            after_clifford_depolarization=0.003, after_reset_flip_probability=0.003,
            before_measure_flip_probability=0.003, before_round_data_depolarization=0.003)
    det, obs = sc.compile_detector_sampler(seed=0).sample(nshots, separate_observables=True)
    det = np.asarray(det, bool); obs = np.asarray(obs, bool)
    plain = sc.detector_error_model(decompose_errors=False)
    ex = extract(plain); Hd = ex["H"].toarray().astype(np.uint8)
    pri = list(np.clip(np.asarray(ex["priors"]), 1e-6, 1 - 1e-6)); Lo = ex["Lo"].toarray().astype(np.uint8)
    tag = f"d={d} r={r} (H {Hd.shape})"
    try:
        signal.alarm(budget); t = time.perf_counter()
        dec = qec.get_decoder("tensor_network_decoder", Hd, logical_obs=Lo, noise_model=pri)
        tc = time.perf_counter() - t
        t = time.perf_counter()
        P = []
        for row in det:
            r0 = dec.decode(row.astype(np.float32).tolist())
            P.append(np.asarray(getattr(r0, "result", r0)).ravel())
        signal.alarm(0)
        td = time.perf_counter() - t
        P = np.array(P).reshape(nshots, Lo.shape[0])
        ler = np.any((P > 0.5) != obs, axis=1).mean()
        print(f"{tag}: construct {tc:.1f}s | {nshots} decodes {td:.1f}s ({nshots/td:.0f}/s) | LER {ler*100:.1f}%", flush=True)
    except TO:
        signal.alarm(0); print(f"{tag}: INTRACTABLE (> {budget}s)", flush=True)
    except Exception as e:
        signal.alarm(0); print(f"{tag}: ERR {str(e)[:80]}", flush=True)

for d, r in [(3, 1), (3, 3), (5, 1), (5, 5)]:
    trial(d, r)
