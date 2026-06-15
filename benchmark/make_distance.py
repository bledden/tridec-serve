"""v3 headline figure: the accuracy<->capacity<->distance tradeoff for the surface
code. Left: LER vs distance (matching & nv-qldpc+OSD stay low; BP-family rises --
the known BP-on-topological-codes weakness). Right: serving capacity vs distance
(monotonic drop -- the serving cost of stronger protection). Error bars: Wilson
LER CI (left), multi-seed knee band (right)."""
import sys, re, json, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plat = sys.argv[1] if len(sys.argv) > 1 else "h200"
d = json.load(open(f"results_{plat}.json"))

COL = {"PyMatching MWPM": "#2ca02c", "NVIDIA CUDA-Q nv-qldpc+OSD": "#d62728",
       "tridec Relay-BP": "#1f77b4", "tridec min-sum BP": "#9467bd"}
def dist(code):
    m = re.search(r"d=(\d+)", code); return int(m.group(1)) if m else None

data = {}
for r in d["rows"]:
    dd = dist(r["code"])
    if dd is None or "surface" not in r["code"]:
        continue
    nm = r["decoder"].split(" (")[0]
    s = r["serving"]; ci = r.get("ler_ci95", [r["ler"], r["ler"]])
    data.setdefault(nm, {})[dd] = (r["ler"] * 100, ci[0] * 100, ci[1] * 100,
                                   s["max_sustained_qubits"], s.get("max_sustained_lo"),
                                   s.get("max_sustained_hi"))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
for nm in sorted(data, key=lambda n: list(COL).index(n) if n in COL else 9):
    dm = data[nm]; ds = sorted(dm); c = COL.get(nm, "#555")
    ler = np.array([dm[x][0] for x in ds])
    lo = np.array([max(dm[x][1], 0.02) for x in ds]); hi = np.array([dm[x][2] for x in ds])
    ax1.errorbar(ds, np.maximum(ler, 0.03), yerr=[ler - lo, np.maximum(hi - ler, 0)],
                 fmt="o-", color=c, capsize=3, label=nm)
    cap = np.array([max(dm[x][3], 0.5) for x in ds])
    clo = np.array([max(dm[x][4], 0.5) for x in ds]); chi = np.array([max(dm[x][5], 0.5) for x in ds])
    ax2.errorbar(ds, cap, yerr=[cap - clo, chi - cap], fmt="o-", color=c, capsize=3, label=nm)

for ax in (ax1, ax2):
    ax.set_xticks(sorted({dist(r["code"]) for r in d["rows"] if dist(r["code"])}))
    ax.set_xlabel("surface code distance  d"); ax.grid(alpha=0.3, which="both")
ax1.set_yscale("log"); ax1.set_ylabel("logical error rate (%)  — lower better ↓")
ax1.set_title("accuracy vs distance\n(matching & nv-qldpc+OSD hold; BP-family rises)")
ax2.set_yscale("log"); ax2.set_ylabel("sustained logical qubits / GPU  — higher better ↑")
ax2.set_title("serving capacity vs distance\n(bigger code → fewer qubits/GPU)")
ax1.legend(fontsize=8, loc="best")
fig.suptitle(f"QEC decode-serving — surface-code distance sweep ({plat.upper()}, p=0.003)", fontsize=12)
plt.tight_layout(); plt.savefig(f"benchmark_distance_{plat}.png", dpi=150)
print(f"saved benchmark_distance_{plat}.png")
