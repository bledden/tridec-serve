"""Accuracy vs serving-capacity, the two axes the benchmark unifies — across the
code x decoder matrix. Lower LER = better; higher capacity = better => bottom-
right is best. Existing QEC benchmarks plot only the y-axis."""
import sys, json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plat = sys.argv[1] if len(sys.argv) > 1 else "metal"
d = json.load(open(f"results_{plat}.json"))
codes = sorted({r["code"] for r in d["rows"]})
_palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
cmap = {c: _palette[i % len(_palette)] for i, c in enumerate(codes)}
fig, ax = plt.subplots(figsize=(9.5, 6))
for r in d["rows"]:
    s = r["serving"]
    cap = max(s["max_sustained_qubits"], 0.5)   # floor for log axis
    lo = max(s.get("max_sustained_lo", cap), 0.5); hi = max(s.get("max_sustained_hi", cap), 0.5)
    ler = max(r["ler"]*100, 0.05)
    yci = r.get("ler_ci95", [r["ler"], r["ler"]])
    ylo = max(yci[0]*100, 0.04); yhi = max(yci[1]*100, ler)
    col = cmap.get(r["code"], "#555")
    mk = "o" if r["accurate"] else "^"
    # v2a: error bars -- x = multi-seed knee range, y = 95% Wilson LER CI
    ax.errorbar(cap, ler, xerr=[[max(cap-lo, 0)], [max(hi-cap, 0)]],
                yerr=[[max(ler-ylo, 0)], [max(yhi-ler, 0)]], fmt=mk, color=col,
                ms=11, zorder=3, ecolor=col, elinewidth=1.2, capsize=3,
                markeredgecolor="k", markeredgewidth=0.5)
    short = (r["decoder"].replace("tridec ", "").replace(" (accurate, CPU)", "/CPU")
             .replace(" MWPM (accurate)", "").replace(" Rust oracle (CPU, accurate)", "/CPU")
             .replace("NVIDIA CUDA-Q ", "").replace(" (GPU)", ""))
    ax.annotate(short, (cap, ler), textcoords="offset points", xytext=(8, 5), fontsize=8)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("serving capacity  (max sustained logical qubits / GPU, 1 ms round)  →  better")
ax.set_ylabel("logical error rate (%)  —  lower is better  ↓")
handles = [plt.Line2D([],[],marker='o',ls='',color=cmap[c],label=c) for c in codes if c in cmap]
handles += [plt.Line2D([],[],marker='o',ls='',color='gray',label='accurate (o) / fast (▲)')]
ax.legend(handles=handles, fontsize=8, loc="lower left")
ax.set_title(f"QEC decode-serving benchmark — accuracy vs serving capacity (backend={d['backend']})\n"
             "surface: matching dominates both axes; BB qLDPC: GPU BP-family gives the serving capacity")
ax.grid(alpha=0.3, which="both"); ax.margins(0.2)
plt.tight_layout(); plt.savefig(f"benchmark_pareto_{plat}.png", dpi=150)
print(f"saved benchmark_pareto_{plat}.png")
