"""The benchmark's two-axis output: accuracy (LER) vs serving capacity
(sustained qubits/GPU). Existing QEC benchmarks plot only the y-axis."""
import json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = json.load(open("results_metal.json"))
fig, ax = plt.subplots(figsize=(8.5, 5.6))
for r in d["rows"]:
    cap = r["serving"]["max_sustained_qubits"]; ler = r["ler"]*100
    col = "#1f77b4" if r["accurate"] else "#ff7f0e"
    ax.scatter(cap, ler, s=140, color=col, zorder=3, edgecolor="k", linewidth=0.5)
    ax.annotate(r["decoder"], (cap, ler), textcoords="offset points",
                xytext=(10, 6), fontsize=9)
ax.set_xlabel("serving capacity  (max sustained logical qubits / GPU, 1 ms round)")
ax.set_ylabel("logical error rate  (%, lower = better)")
ax.set_title(f"QEC decode-serving benchmark: accuracy vs serving capacity\n"
             f"({d['code']}, backend={d['backend']}) — the two axes unified "
             f"(existing benchmarks measure only accuracy)")
ax.grid(alpha=0.3); ax.margins(0.18)
plt.tight_layout(); plt.savefig("benchmark_pareto_metal.png", dpi=150)
print("rows:", [(r["decoder"], r["serving"]["max_sustained_qubits"], round(r["ler"]*100,2)) for r in d["rows"]])
print("saved benchmark_pareto_metal.png")
