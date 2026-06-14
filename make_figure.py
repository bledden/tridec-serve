"""Decode-serving model + headline figure: logical-qubits-per-GPU vs SLA.

Thesis: at scale, QEC decoding is a SERVING problem (many logical-qubit syndrome
streams, each with a reaction-latency SLA, sharing one accelerator). With
continuous batching, one GPU serves K streams where K is bounded by decode
throughput, not single-shot latency.

Model (first-order, pipelined continuous batching):
  - each logical qubit emits 1 syndrome / T_round (the QEC clock).
  - the scheduler accumulates arrivals into a batch B and decodes; while batch i
    decodes, batch i+1 fills -> steady-state reaction latency ~= 2*L(B).
  - so within an SLA, the usable batch is the largest B with 2*L(B) <= SLA, and
    the GPU sustains K = throughput(B)*T_round = (B/L(B)) * T_round logical qubits.
Empirical L(B) from tridec on the local accelerator (measure_latency.py).
"""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("/Users/bledden/Documents/tridec-serve/latency_metal.json"))
def curve(key):
    items = sorted((int(b), float(L)) for b, L in d[key].items())
    return np.array([b for b,_ in items]), np.array([L for _,L in items])
def kmax(B, L, sla, t_round):
    ok = (2.0*L) <= sla
    if not ok.any(): return 0.0
    return float((B/L)[ok].max() * t_round)     # logical qubits per GPU

SLAs = np.logspace(-5, -0.5, 80)                # 10 us .. ~300 ms reaction budget
clocks = [(1e-6, "superconducting\n(1 us / round)"),
          (1e-3, "neutral-atom / trapped-ion\n(1 ms / round)")]
decoders = [("relay", "Relay-BP (accurate, the default)", "#1f77b4"),
            ("bp",    "min-sum BP (fast, less accurate)", "#ff7f0e")]
table = {}
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
for ax, (t_round, label) in zip(axes, clocks):
    for key, name, col in decoders:
        B, L = curve(key)
        K = np.array([kmax(B, L, s, t_round) for s in SLAs])
        ax.plot(SLAs*1e3, np.maximum(K, 1e-3), color=col, lw=2.4, label=name)
        table[f"{key}@{t_round*1e6:.0f}us_round"] = {
            "sla_ms": [round(s*1e3,3) for s in SLAs[::16]],
            "qubits_per_gpu": [round(kmax(B,L,s,t_round),3) for s in SLAs[::16]]}
    ax.axhline(1, color="0.4", ls=":", lw=1.2)
    ax.text(ax.get_xlim()[0]*1.2 if False else 0.012, 1.25, "1 logical qubit",
            color="0.4", fontsize=8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(label, fontsize=11)
    ax.set_xlabel("reaction-latency SLA (ms)"); ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=9, loc="upper left")
axes[0].set_ylabel("logical qubits served per GPU")
fig.suptitle("tridec decode-serving: logical qubits per GPU vs latency SLA  "
             "(Apple M4 Max / Metal; continuous batching)", fontsize=12)
fig.text(0.5, 0.005, "Left: at a 1 us superconducting round even max throughput "
         "serves <1 qubit/GPU in real time -> the sub-us inner loop is FPGA "
         "territory. Right: at ms-scale clocks (high-rate qLDPC on neutral-atom/"
         "trapped-ion) one GPU serves 10s of logical qubits -> the GPU "
         "decode-serving regime.", ha="center", fontsize=8, color="0.3", wrap=True)
plt.tight_layout(rect=[0,0.04,1,0.96])
plt.savefig("/Users/bledden/Documents/tridec-serve/decode_serving_metal.png", dpi=150)
json.dump({"model":"reaction~2L(B); K=throughput_within_SLA*T_round",
           "backend": d["backend"], "table": table},
          open("/Users/bledden/Documents/tridec-serve/serving_metal.json","w"), indent=2)
# headline numbers
for key,name,_ in decoders:
    B,L = curve(key)
    peak = (B/L).max()
    print(f"{name}: peak throughput {peak:.0f} syndromes/s -> "
          f"{peak*1e-3:.1f} qubits/GPU @1ms-round, {peak*1e-6:.4f} @1us-round")
print("saved decode_serving_metal.png + serving_metal.json")
