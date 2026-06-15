"""v4 figure: p99 reaction latency vs total fleet size on ONE GPU, for the
heterogeneous compositions. The knee = max sustained mixed fleet (each lane routed
to its own decoder instance). Existing benchmarks can't draw this — they decode
one homogeneous code at a time."""
import sys, json, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plat = sys.argv[1] if len(sys.argv) > 1 else "h200"
d = json.load(open(f"results_fleet_{plat}.json"))
COL = {"mixed_distance": "#1f77b4", "mixed_family": "#d62728"}
LAB = {"mixed_distance": "mixed-distance  (surface d3 + d5 + d7)",
       "mixed_family": "mixed-family  (surface d5 + BB qLDPC)"}

fig, ax = plt.subplots(figsize=(9.5, 6))
for key in ("mixed_distance", "mixed_family"):
    if key not in d:
        continue
    m = d[key]; sw = m["last_sweep"]
    xs = [r["fleet_K"] for r in sw]; ys = [max(r["p99_ms"], 0.1) for r in sw]
    ax.plot(xs, ys, "o-", color=COL[key], label=f"{LAB[key]} — sustained {m['max_fleet']}q")
    knee = m["max_fleet"]
    ky = next((r["p99_ms"] for r in sw if r["fleet_K"] == knee), None)
    if ky:
        ax.scatter([knee], [ky], s=240, facecolors="none", edgecolors=COL[key],
                   linewidths=2.2, zorder=6)
        ax.annotate(f"{knee}q sustained", (knee, ky), textcoords="offset points",
                    xytext=(6, -16), fontsize=9, color=COL[key])

for sla, lbl in [(100, "100 ms"), (250, "250 ms"), (500, "500 ms")]:
    ax.axhline(sla, ls=":", color="gray", lw=0.8)
    ax.text(ax.get_xlim()[0], sla, f" p99 {lbl} SLA", fontsize=7, color="gray", va="bottom")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("total fleet size  (logical qubits on ONE GPU, 1 ms round)  →")
ax.set_ylabel("p99 reaction latency (ms)  — lower better ↓")
ax.set_title(f"v4 — one GPU serving a HETEROGENEOUS fleet ({plat.upper()}, p=0.003)\n"
             "each lane routed to its own decoder instance; knee = max sustained mixed fleet")
ax.legend(fontsize=9, loc="upper left"); ax.grid(alpha=0.3, which="both")
plt.tight_layout(); plt.savefig(f"benchmark_fleet_{plat}.png", dpi=150)
print(f"saved benchmark_fleet_{plat}.png")
