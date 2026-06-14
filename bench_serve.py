"""Drive the live DecodeServer at increasing offered load and measure the
reaction-latency tail + whether the GPU keeps up. Produces the measured
'latency vs load' knee and the max sustained logical-qubits-per-GPU at an SLA
— the real-system counterpart to make_figure.py's analytical model."""
import json, numpy as np, stim, tridec
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from serve import run_load

FIX = "/Users/bledden/Documents/tridec/tests/fixtures/bb72/"
dem = stim.DetectorErrorModel.from_file(FIX+"bb72_r6_p0.003_Z.dem")
c   = stim.Circuit.from_file(FIX+"bb72_r6_p0.003_Z.stim")
pool, _ = c.compile_detector_sampler(seed=0).sample(4096, separate_observables=True)
pool = np.ascontiguousarray(pool)

T_ROUND = 1e-3          # 1 ms/round (neutral-atom / trapped-ion regime — the GPU lane)
DURATION = 2.0
SLAS_MS = [100, 250, 500]

configs = [
    ("relay", "Relay-BP (accurate, default)", "#1f77b4", [2, 4, 8, 12, 16, 20]),
    ("bp",    "min-sum BP (fast)",            "#ff7f0e", [8, 16, 32, 48, 64, 80]),
]
results = {}
fig, ax = plt.subplots(figsize=(8.5, 5.2))
for algo, name, col, Ks in configs:
    dec = tridec.from_dem(dem, algorithm=algo)
    for b in (16, 64, 256, 1024, 4096):            # warmup: compile every bucket shape
        dec.decode_batch(pool[:b])
    rows = []
    for K in Ks:
        r = run_load(dec, pool, K=K, t_round=T_ROUND, duration=DURATION)
        rows.append(r)
        flag = " OVERLOADED" if r["overloaded"] else ""
        print(f"  {algo:5s} K={K:>3} | p50 {r['p50_ms']:6.1f} p99 {r['p99_ms']:7.1f} "
              f"p99.9 {r['p999_ms']:7.1f} ms | batch<= {r['max_batch_used']:>4} "
              f"backlog<= {r['max_backlog']:>5} | thr {r['throughput_per_s']:.0f}/s{flag}")
    results[algo] = rows
    Karr = np.array([r["K"] for r in rows])
    p99  = np.array([r["p99_ms"] for r in rows])
    over = np.array([r["overloaded"] for r in rows])
    ax.plot(Karr[~over], p99[~over], "o-", color=col, lw=2, label=name)
    if over.any():
        ax.plot(Karr[over], p99[over], "x", color=col, ms=10, mew=2)  # overloaded
for sla in SLAS_MS:
    ax.axhline(sla, color="0.6", ls=":", lw=1)
    ax.text(0.5, sla*1.05, f"{sla} ms SLA", color="0.5", fontsize=8)
ax.set_yscale("log")
ax.set_xlabel("offered load  (logical qubits, 1 syndrome / 1 ms round)")
ax.set_ylabel("reaction latency p99 (ms)")
ax.set_title("tridec-serve: measured p99 reaction latency vs offered load\n"
             "(live continuous-batching scheduler, Apple M4 Max / Metal; × = couldn't keep up)")
ax.grid(alpha=0.3, which="both"); ax.legend(loc="upper left")
plt.tight_layout(); plt.savefig("/Users/bledden/Documents/tridec-serve/serve_latency_vs_load.png", dpi=150)

# max sustained qubits per GPU at each SLA (largest non-overloaded K with p99<=SLA)
summary = {}
for algo, rows in results.items():
    sustained = [r["K"] for r in rows if not r["overloaded"]]
    summary[algo] = {"max_sustained_K": max(sustained) if sustained else 0}
    for sla in SLAS_MS:
        ok = [r["K"] for r in rows if not r["overloaded"] and r["p99_ms"] is not None and r["p99_ms"] <= sla]
        summary[algo][f"{sla}ms"] = max(ok) if ok else 0
json.dump({"t_round_ms": T_ROUND*1e3, "duration_s": DURATION,
           "results": results, "max_qubits_at_sla": summary},
          open("/Users/bledden/Documents/tridec-serve/serve_measured.json", "w"), indent=2)
print("\nmax sustained logical qubits/GPU at p99 SLA:")
for algo, s in summary.items():
    print(f"  {algo}: " + ", ".join(f"{k}->{v}q" for k, v in s.items()))
print("saved serve_latency_vs_load.png + serve_measured.json")
