"""Find the TN decoder's constructor signature and try a quick surface-code decode
with a timeout (TN contraction can blow up on bad treewidth -> we must know if it's
even tractable before wiring it into the serving sweep)."""
import inspect, signal, time, numpy as np, stim, cudaq_qec as qec
from tridec.dem import extract

# locate the class to read its signature
TN = None
for modname in ["cudaq_qec.plugins.decoders.tensor_network_decoder",
                "cudaq_qec.plugins.decoders"]:
    try:
        m = __import__(modname, fromlist=["*"])
        for nm in dir(m):
            o = getattr(m, nm)
            if isinstance(o, type) and "TensorNetwork" in nm:
                TN = o; print("found class", modname + "." + nm)
    except Exception as e:
        pass
if TN:
    try: print("signature:", str(inspect.signature(TN.__init__)))
    except Exception as e: print("sig err", e)

# surface d=3 (small -> tractable) for the API test
sc = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=3,
        after_clifford_depolarization=0.003, after_reset_flip_probability=0.003,
        before_measure_flip_probability=0.003, before_round_data_depolarization=0.003)
det, obs = sc.compile_detector_sampler(seed=0).sample(50, separate_observables=True)
det = np.asarray(det, bool); obs = np.asarray(obs, bool)
plain = sc.detector_error_model(decompose_errors=False)
ex = extract(plain); Hd = ex["H"].toarray().astype(np.uint8)
pri = list(np.clip(np.asarray(ex["priors"]), 1e-6, 1 - 1e-6)); Lo = ex["Lo"].toarray().astype(np.uint8)
print("surface d=3: H", Hd.shape, "Lo", Lo.shape)

class TO(Exception): pass
def to(s, f): raise TO()
for label, args, kw in [
    ("get_decoder(H, Lo, pri)", (Hd, Lo, pri), {}),
    ("get_decoder(H, logical_obs=Lo, noise=pri)", (Hd,), {"logical_obs": Lo, "noise_model": pri}),
    ("get_decoder(H, Lo, noise_model=pri)", (Hd, Lo), {"noise_model": pri}),
]:
    signal.signal(signal.SIGALRM, to); signal.alarm(60)
    try:
        dec = qec.get_decoder("tensor_network_decoder", *args, **kw)
        signal.alarm(0)
        t = time.perf_counter()
        r = dec.decode(det[0].astype(np.uint8).tolist())
        print("  OK", label, "| decode ok in", round(time.perf_counter()-t, 2), "s | result attrs:",
              [a for a in dir(r) if not a.startswith("_")][:4])
        break
    except TO:
        print("  TIMEOUT(>60s)", label)
    except Exception as e:
        signal.alarm(0); print("  fail", label, "|", str(e)[:90])
