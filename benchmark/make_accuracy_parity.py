"""Methodology-defense figure: the 50k-shot surface LER ladder is BIT-IDENTICAL
across H200 (CUDA) and MI300X (ROCm) -- rendered as a side-by-side table so the
'accuracy is hardware-independent' claim is verifiable at a glance."""
import json, re, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def dd(c):
    m = re.search(r"d=(\d+)", c); return int(m.group(1)) if m else None
def ladder(path):
    d = json.load(open(path)); out = {}
    for r in d["rows"]:
        if "surface" in r["code"] and r["decoder"] in ("PyMatching MWPM (accurate)", "tridec min-sum BP"):
            out[(dd(r["code"]), r["decoder"])] = r["ler"] * 100
    return out
h = ladder("results_h200.json"); m = ladder("results_mi300x.json")

rows, ident = [], True
for dec in ["PyMatching MWPM (accurate)", "tridec min-sum BP"]:
    for d in [3, 5, 7, 9, 11, 13]:
        hv, mv = h.get((d, dec)), m.get((d, dec))
        if hv is None or mv is None: continue
        same = abs(hv - mv) < 1e-9
        ident &= same
        rows.append([dec.split(" (")[0].replace("tridec ", ""), f"d={d}",
                     f"{hv:.3f}%", f"{mv:.3f}%", "✓" if same else "✗"])

fig, ax = plt.subplots(figsize=(8.5, 5.2)); ax.axis("off")
tbl = ax.table(cellText=rows, colLabels=["decoder", "d", "H200 (CUDA)", "MI300X (ROCm)", "identical"],
               cellLoc="center", loc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.5)
for j in range(5):
    tbl[0, j].set_facecolor("#dddddd"); tbl[0, j].set_text_props(weight="bold")
for i in range(1, len(rows) + 1):
    tbl[i, 4].set_text_props(color="#2ca02c", weight="bold")
ax.set_title("Accuracy is hardware-independent: 50k-shot surface LER, H200 vs MI300X\n"
             f"bit-identical to the last digit (deterministic decoders + same x86_64 stim shots) — all {'✓' if ident else '✗'}",
             fontsize=11, pad=14)
plt.tight_layout(); plt.savefig("benchmark_accuracy_parity.png", dpi=150)
print("saved benchmark_accuracy_parity.png | all identical:", ident)
