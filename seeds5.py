"""median-of-5 on the headline fleets (best-tool + mixed-family) to tighten the
abstract numbers (was median-of-3)."""
import os, json
os.environ.setdefault("TRIDEC_FIX", "/workspace/bench/fixtures/")
PLAT = os.environ.get("TRIDEC_PLAT", "gpu")
from fleetbench import surface_lane, bb_lane
from fleet_serve import max_sustained_fleet

SC = [64, 128, 256, 512, 768, 1024, 1536, 2048]
def best_tool(s): return [surface_lane(3, s, "pymatching"), surface_lane(5, s, "pymatching"), bb_lane(s, "tridec-bp")]
def mixed_family(s): return [surface_lane(5, s, "tridec-bp"), bb_lane(s, "tridec-bp")]

out = {}
for name, fn in [("best_tool", best_tool), ("mixed_family", mixed_family)]:
    m = max_sustained_fleet(fn, SC, duration=3.0, seeds=(0, 1, 2, 3, 4))
    print(f"{name} median-of-5: {m['max_fleet']}q  seeds={m['seeds']}  [{m['lo']}-{m['hi']}]", flush=True)
    out[name] = m
json.dump(out, open(f"/workspace/bench/results_seeds5_{PLAT}.json", "w"), indent=2)
print("saved")
