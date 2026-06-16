import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d=json.load(open("results_torture_mi300x.json")); rows=d["rows"]
xs=[r["total"] for r in rows]; ys=[r["p99_ms"] for r in rows]
ok=[r["total"] for r in rows if not r["overloaded"]]
fig,ax=plt.subplots(figsize=(8.5,5))
ax.plot(xs,ys,"o-",color="#d62728")
for sla in (100,250,500): ax.axhline(sla,ls=":",color="gray",lw=0.8); ax.text(xs[0],sla,f" {sla}ms",fontsize=7,color="gray",va="bottom")
ax.axvline(max(ok),ls="--",color="#1f77b4",lw=1); ax.text(max(ok),ys[-1]," backlog knee 1536q",fontsize=8,color="#1f77b4",rotation=90,va="top")
ax.set_yscale("log"); ax.set_xlabel("total fleet size (best-tool, MI300X)"); ax.set_ylabel("p99 reaction latency (ms)")
ax.set_title("v4 torture: the SLA cliff is a gradual ramp then hard overload\n(best-tool fleet, MI300X) — SLA-bound (<500ms) capacity ~1344q < backlog knee 1536q")
ax.grid(alpha=0.3,which="both"); plt.tight_layout(); plt.savefig("benchmark_torture_mi300x.png",dpi=150); print("saved benchmark_torture_mi300x.png")
