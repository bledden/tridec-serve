"""#4 prototype: CUDA-graph-capture the BP decode core for a fixed batch shape,
prove it's bit-identical to decode_batch, and measure the batch-1 latency win.
Only the per-decode-variable (ssign) is copied into a static tensor; the n_iter
kernel loop + Lo projection are captured + replayed (eliminates per-launch overhead)."""
import numpy as np, stim, torch, time, triton, tridec
from tridec.backends.bp_triton import _check_update_kernel, _bit_update_kernel, _INF

P = 0.003
sc = stim.Circuit.generated("surface_code:rotated_memory_z", distance=5, rounds=5,
        after_clifford_depolarization=P, after_reset_flip_probability=P,
        before_measure_flip_probability=P, before_round_data_depolarization=P)
det, _ = sc.compile_detector_sampler(seed=0).sample(2000, separate_observables=True)
det = np.asarray(det, bool)
dec = tridec.from_dem(sc.detector_error_model(decompose_errors=False), algorithm="bp")
impl = dec._impl
dev = torch.device("cuda")
print("backend", dec.backend, "| n_bits", impl.n_bits, "n_checks", impl.n_checks, "n_edges", impl.n_edges)

S = 1
lam, cedge, bedge, edge_bit, mu_init = impl._device_tensors(dev)
Lo = impl._Lo_dev(dev)
E, N, C = impl.n_edges, impl.n_bits, impl.n_checks
BLOCK_S, n_iter, ms = impl.block_s, impl.max_iter, impl.ms
grid_s = triton.cdiv(S, BLOCK_S); grid_chk = (grid_s, C); grid_bit = (grid_s, N)
mu_init_exp = mu_init.unsqueeze(1).expand(E, S).contiguous()
mu = torch.empty((E, S), dtype=torch.float32, device=dev)
nu = torch.empty((E, S), dtype=torch.float32, device=dev)
post = torch.empty((N, S), dtype=torch.float32, device=dev)
ssign_static = torch.empty((C, S), dtype=torch.float32, device=dev)
one = torch.ones((), dtype=torch.float32, device=dev)

def make_ssign(dets_S):
    syn = torch.as_tensor(np.ascontiguousarray(dets_S), device=dev)
    return torch.where((syn.to(torch.int32) & 1).t() == 0, one, -one).contiguous()

def core():
    mu.copy_(mu_init_exp)
    for _ in range(n_iter):
        _check_update_kernel[grid_chk](mu, nu, ssign_static, cedge, S, E, C, ms,
                                       MAXDEG_C=impl.MAXDEG_C, BLOCK_S=BLOCK_S, INF=_INF)
        _bit_update_kernel[grid_bit](nu, mu, post, lam, bedge, S, E, N,
                                     MAXDEG_B=impl.MAXDEG_B, BLOCK_S=BLOCK_S)
    e_hat = (post < 0.0).to(torch.float32)
    return torch.remainder(Lo @ e_hat, 2.0)

# warm (compile kernels) on a side stream, then capture
ssign_static.copy_(make_ssign(det[:S]))
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        pred = core()
torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    pred_static = core()

# correctness: replay vs decode_batch across 50 distinct shots
ok = True
for i in range(50):
    ds = det[i:i+1]
    ref = dec.decode_batch(np.ascontiguousarray(ds))
    ssign_static.copy_(make_ssign(ds)); g.replay(); torch.cuda.synchronize()
    gi = (pred_static > 0.5).t().cpu().numpy()
    ok &= np.array_equal(gi, ref)
print("50-shot graph-replay bit-identical to decode_batch:", ok)

# latency: non-graph decode_batch vs graph replay (batch-1)
for _ in range(20): dec.decode_batch(np.ascontiguousarray(det[:1]))
torch.cuda.synchronize(); t = time.perf_counter()
for _ in range(300): dec.decode_batch(np.ascontiguousarray(det[:1]))
torch.cuda.synchronize(); dt_ng = (time.perf_counter() - t) / 300
t = time.perf_counter()
for _ in range(300):
    ssign_static.copy_(make_ssign(det[:1])); g.replay()
torch.cuda.synchronize(); dt_g = (time.perf_counter() - t) / 300
print(f"batch-1: non-graph {dt_ng*1e3:.3f} ms | graph {dt_g*1e3:.3f} ms | speedup {dt_ng/dt_g:.2f}x")
