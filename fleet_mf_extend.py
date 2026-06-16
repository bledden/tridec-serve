"""Extend the mixed-family sweep on MI300X (it hit the sweep ceiling at 1024q) to
find the true knee. Patches results_fleet_mi300x.json."""
import os, json
os.environ.setdefault("TRIDEC_FIX", "/workspace/bench/fixtures/")
from fleetbench import surface_lane, bb_lane
from fleet_serve import max_sustained_fleet

def mixed_family(scale):
    return [surface_lane(5, scale, "tridec-bp"), bb_lane(scale, "tridec-bp")]

m = max_sustained_fleet(mixed_family, [256, 512, 768, 1024, 1536, 2048], duration=3.0)
print("mixed_family extended: max", m["max_fleet"], "seeds", m["seeds"])
for r in m["last_sweep"]:
    print(f"  total={r['fleet_K']:>5}  {'OVERLOAD' if r['overloaded'] else 'ok'}  p99={r['p99_ms']}")
RES = "/workspace/bench/results_fleet_mi300x.json"
d = json.load(open(RES)); d["mixed_family"] = m; json.dump(d, open(RES, "w"), indent=2)
print("patched", RES)
