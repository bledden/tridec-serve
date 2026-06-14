"""Combined cross-vendor figure: logical-qubits-per-GPU @1ms round across
Metal / H200 / MI300X, for Relay-BP (accurate) and min-sum BP (fast).
bars = throughput model (peak B/L(B) * T_round); diamonds = live scheduler
max-sustained (BP live is capped at the sweep max K=256, not its knee)."""
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plats = ["Metal", "H200", "MI300X"]
files = {"Metal": "latency_metal.json", "H200": "latency_h200.json", "MI300X": "latency_mi300x.json"}
servef = {"Metal": "serve_measured.json", "H200": "serve_measured_h200.json", "MI300X": "serve_measured_mi300x.json"}
T = 1e-3
def peak(d, k): return max(int(b)/L for b, L in d[k].items())
model, live = {}, {}
for p in plats:
    d = json.load(open(files[p]))
    model[p] = {"relay": peak(d, "relay")*T, "bp": peak(d, "bp")*T}
    s = json.load(open(servef[p]))["max_qubits_at_sla"]
    live[p] = {"relay": s["relay"]["max_sustained_K"], "bp": s["bp"]["max_sustained_K"]}
x = np.arange(len(plats)); w = 0.36
fig, ax = plt.subplots(figsize=(9.5, 5.6))
rm = [model[p]["relay"] for p in plats]; bm = [model[p]["bp"] for p in plats]
ax.bar(x-w/2, rm, w, color="#1f77b4", label="Relay-BP (accurate) — throughput model")
ax.bar(x+w/2, bm, w, color="#ff7f0e", label="min-sum BP (fast) — throughput model")
ax.plot(x-w/2, [live[p]["relay"] for p in plats], "D", color="navy", ms=9, ls="none", label="live scheduler (max sustained)")
ax.plot(x+w/2, [live[p]["bp"] for p in plats], "D", color="saddlebrown", ms=9, ls="none")
for i, p in enumerate(plats):
    ax.text(i-w/2, rm[i]*1.15, f"{rm[i]:.0f}", ha="center", fontsize=9)
    ax.text(i+w/2, bm[i]*1.15, f"{bm[i]:.0f}", ha="center", fontsize=9)
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(plats, fontsize=11)
ax.set_ylabel("logical qubits served per GPU  (1 ms / round)")
ax.set_title("tridec decode-serving — vendor-portable capacity per GPU\n"
             "Relay-BP (accurate) is compute-bound & portable (~11–18); min-sum BP (fast) is\n"
             "launch-bound on Metal but ~1M syndromes/s on datacenter GPUs (BP live = sweep cap 256)")
ax.legend(loc="upper left", fontsize=9); ax.grid(axis="y", alpha=0.3, which="both")
ax.set_ylim(5, 3000)
plt.tight_layout(); plt.savefig("decode_serving_xvendor.png", dpi=150)
json.dump({"t_round_ms": 1.0, "model_qubits_per_gpu": model, "live_max_sustained": live},
          open("xvendor_summary.json", "w"), indent=2)
for p in plats:
    print(f"  {p:7s}: relay model {model[p]['relay']:6.1f} live {live[p]['relay']:>3}q | "
          f"bp model {model[p]['bp']:7.0f} live {live[p]['bp']:>3}q")
print("saved decode_serving_xvendor.png + xvendor_summary.json")
