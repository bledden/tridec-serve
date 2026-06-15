"""Re-tune the relay megakernel (block, num_warps) for the DO MI300X's
rocm7/triton-3.4 stack -- the shipped gfx942 config (512,8) was autotuned on
rocm6.2/triton-3.1 and under-performs here. Times BB relay decode_batch(1024)
across candidate configs; the winner is what we use for parity. (min-sum BP is
already at parity, so only relay is swept.)"""
import time, numpy as np, stim, tridec
import tridec.backends.megakernel as mk

dem = stim.DetectorErrorModel.from_file("/workspace/bench/fixtures/bb72_r6_p0.003_Z.dem")
c = stim.Circuit.from_file("/workspace/bench/fixtures/bb72_r6_p0.003_Z.stim")
det, _ = c.compile_detector_sampler(seed=0).sample(2000, separate_observables=True)
det = np.ascontiguousarray(np.asarray(det, bool))
B = 1024
batch = det[np.arange(B) % len(det)]
print("shipped gfx942 relay cfg:", mk._CUDA_TUNED.get("gfx942", {}).get("relay"))

best = None
for cfg in [(512, 8), (256, 8), (1024, 8), (512, 16), (1024, 16), (256, 4), (128, 4), (256, 16)]:
    mk._CUDA_TUNED.setdefault("gfx942", {})["relay"] = cfg
    try:
        dec = tridec.from_dem(dem, algorithm="relay")
        dec.decode_batch(batch)  # warm + compile
        t = time.perf_counter()
        for _ in range(5):
            dec.decode_batch(batch)
        dt = (time.perf_counter() - t) / 5
        sps = B / dt
        print(f"  relay {str(cfg):10s} {dt*1e3:7.1f} ms/{B}  -> {sps:9.0f} syn/s")
        if best is None or sps > best[1]:
            best = (cfg, sps)
    except Exception as e:
        print(f"  relay {str(cfg):10s} FAIL {str(e)[:60]}")
print(f"BEST relay cfg: {best[0]}  ({best[1]:.0f} syn/s)  vs shipped (512,8)")
