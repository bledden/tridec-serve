"""Measure tridec decode latency L(B) vs batch size on the local accelerator.
This is the empirical input to the decode-serving model: throughput B/L(B) and
the latency curve determine how many logical-qubit syndrome streams one GPU can
serve at a given SLA. Replays the canonical BB [[72,12,6]] cell from stim."""
import time, json, sys, numpy as np, stim, tridec
FIX = "/Users/bledden/Documents/tridec/tests/fixtures/bb72/"
dem = stim.DetectorErrorModel.from_file(FIX+"bb72_r6_p0.003_Z.dem")
c   = stim.Circuit.from_file(FIX+"bb72_r6_p0.003_Z.stim")
NMAX = 2048
dets_all, _ = c.compile_detector_sampler(seed=0).sample(NMAX, separate_observables=True)

def measure(dec, batches, warm=2, reps=3):
    out = {}
    for B in batches:
        d = np.ascontiguousarray(dets_all[:B])
        for _ in range(warm): dec.decode_batch(d)
        ts = []
        for _ in range(reps):
            t = time.perf_counter(); dec.decode_batch(d); ts.append(time.perf_counter()-t)
        out[B] = min(ts)
        print(f"  B={B:>5}  L={min(ts)*1e3:8.2f} ms   {min(ts)*1e6/B:8.2f} us/syndrome")
    return out

dev = tridec.resolve_backend("auto")
print(f"device/backend: {dev}")
res = {"backend": dev, "code": "bb72_r6_p0.003_Z", "n_obs": int(dem.num_observables)}
print("RELAY (megakernel, accurate path):")
relay = tridec.from_dem(dem, algorithm="relay")
res["relay_impl"] = type(relay._impl).__name__
res["relay"] = {str(k): v for k, v in measure(relay, [1,4,16,64,256,1024,2048]).items()}
print("BP (fast path):")
bp = tridec.from_dem(dem, algorithm="bp")
res["bp"] = {str(k): v for k, v in measure(bp, [1,2,4,8,16,32,64,128,256,512,1024,2048]).items()}
json.dump(res, open("/Users/bledden/Documents/tridec-serve/latency_metal.json","w"), indent=2)
print("saved latency_metal.json")
