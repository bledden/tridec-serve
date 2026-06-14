"""Heterogeneous-load + high-K sweep with the faster (round-batched) load
generator. Two questions:
  1. With the round-batched producer, can we drive the fast decoder to its real
     knee (BP ~1M syndromes/s on datacenter GPUs)?
  2. Does heterogeneous load (most qubits idle / error-free) raise capacity?
     Expectation: YES for Relay-BP (per-shot early-exit -> fewer legs when the
     batch is mostly idle), ~NO for min-sum BP (fixed 30 iterations regardless).
"""
import json, numpy as np, stim, tridec
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from serve import run_load

FIX = "/Users/bledden/Documents/tridec/tests/fixtures/bb72/"
dem = stim.DetectorErrorModel.from_file(FIX+"bb72_r6_p0.003_Z.dem")
c   = stim.Circuit.from_file(FIX+"bb72_r6_p0.003_Z.stim")
pool, _ = c.compile_detector_sampler(seed=0).sample(4096, separate_observables=True)
pool = np.ascontiguousarray(pool)

T_ROUND, DURATION = 1e-3, 2.0
configs = [
    ("relay", "Relay-BP", [4, 8, 16, 32, 64, 128]),
    ("bp",    "min-sum BP", [64, 256, 512, 1024, 1536]),
]
fracs = [(1.0, "uniform (all active)", "-"), (0.1, "heterogeneous (10% active)", "--")]
colors = {"relay": "#1f77b4", "bp": "#ff7f0e"}
results, summary = {}, {}
fig, ax = plt.subplots(figsize=(9.5, 5.6))
for algo, name, Ks in configs:
    dec = tridec.from_dem(dem, algorithm=algo)
    for af, aflabel, ls in fracs:
        rows = []
        for K in Ks:
            r = run_load(dec, pool, K=K, t_round=T_ROUND, duration=DURATION, active_frac=af)
            rows.append(r)
            print(f"  {algo:5s} af={af:>3} K={K:>4} | p99 {r['p99_ms']:8.1f} ms | "
                  f"backlogR<= {r['max_backlog_rounds']:>4} | thr {r['throughput_per_s']:>8.0f}/s"
                  f"{' OVERLOADED' if r['overloaded'] else ''}")
        results[f"{algo}@{af}"] = rows
        sus = [r["K"] for r in rows if not r["overloaded"]]
        summary[f"{algo}@{af}"] = max(sus) if sus else 0
        Karr = np.array([r["K"] for r in rows]); p99 = np.array([r["p99_ms"] for r in rows])
        over = np.array([r["overloaded"] for r in rows])
        ax.plot(Karr[~over], p99[~over], ls, color=colors[algo], lw=2, marker="o",
                label=f"{name} — {aflabel}")
        if over.any():
            ax.plot(Karr[over], p99[over], "x", color=colors[algo], ms=9, mew=2)
for sla in (100, 250, 500):
    ax.axhline(sla, color="0.6", ls=":", lw=1)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("offered load (logical qubits, 1 syndrome / 1 ms round)")
ax.set_ylabel("reaction latency p99 (ms)")
ax.set_title("tridec-serve: heterogeneous load + high-K (round-batched load gen)\n"
             "× = couldn't keep up; dashed = 10% active (mostly-idle) load")
ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8, loc="upper left")
plt.tight_layout(); plt.savefig("/Users/bledden/Documents/tridec-serve/serve_hetero.png", dpi=150)
json.dump({"t_round_ms": 1.0, "duration_s": DURATION, "max_sustained": summary, "results": results},
          open("/Users/bledden/Documents/tridec-serve/serve_hetero.json", "w"), indent=2)
print("\nmax sustained logical qubits/GPU:")
for k, v in summary.items():
    print(f"  {k}: {v}")
print("saved serve_hetero.png + serve_hetero.json")
