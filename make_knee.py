"""Headline: one datacenter GPU serves ~1024 logical qubits live (min-sum BP,
single-digit-ms p99) — reached via the round-batched load generator. Relay-BP
(accurate) sits at ~8-16, compute-bound. Latency-vs-load knee on H200 + MI300X."""
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
data = {"H200": json.load(open("serve_hetero_h200.json")),
        "MI300X": json.load(open("serve_hetero_mi300x.json"))}
styles = {("H200","bp"): ("#d62728","-","o"), ("MI300X","bp"): ("#ff7f0e","-","s"),
          ("H200","relay"): ("#1f77b4","--","o"), ("MI300X","relay"): ("#17becf","--","s")}
fig, ax = plt.subplots(figsize=(9.6, 5.7))
for plat, d in data.items():
    for algo in ["bp", "relay"]:
        rows = d["results"][f"{algo}@1.0"]
        K = np.array([r["K"] for r in rows]); p99 = np.array([r["p99_ms"] for r in rows])
        over = np.array([r["overloaded"] for r in rows])
        col, ls, mk = styles[(plat, algo)]
        ax.plot(K[~over], p99[~over], ls, color=col, marker=mk, lw=2,
                label=f"{plat} — {'min-sum BP (fast)' if algo=='bp' else 'Relay-BP (accurate)'}")
        if over.any():
            ax.plot(K[over], p99[over], "x", color=col, ms=10, mew=2)
for sla in (10, 100, 1000):
    ax.axhline(sla, color="0.7", ls=":", lw=1); ax.text(4.5, sla*1.1, f"{sla} ms", color="0.5", fontsize=8)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("offered load (logical qubits, 1 syndrome / 1 ms round)")
ax.set_ylabel("reaction latency p99 (ms)")
ax.set_title("tridec-serve at scale: one datacenter GPU serves ~1024 logical qubits live\n"
             "min-sum BP holds single-digit-ms p99 to K=1024 (~1M syndromes/s); "
             "Relay-BP (accurate) ~8-16; × = couldn't keep up")
ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8, loc="upper left")
plt.tight_layout(); plt.savefig("decode_serving_knee.png", dpi=150)
print("knee summary (uniform load, max sustained / p99 at it):")
for plat, d in data.items():
    for algo in ["bp", "relay"]:
        rows = d["results"][f"{algo}@1.0"]
        sus = [r for r in rows if not r["overloaded"]]
        if sus: print(f"  {plat:7s} {algo:5s}: {sus[-1]['K']:>4} qubits @ p99 {sus[-1]['p99_ms']:.1f} ms")
print("saved decode_serving_knee.png")
