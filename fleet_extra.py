"""Extra MI300X fleet runs (rare-AMD-window work):
(1) BEST-TOOL routing fleet — surface lanes -> PyMatching (the right tool, CPU),
    qLDPC lane -> tridec BP (GPU). Routing the RIGHT decoder per code on one box;
    the qLDPC GPU lane is exactly what nv-qldpc can't run on AMD.
(2) d=9-extended mixed-distance fleet (d3+d5+d7+d9, all tridec BP, one GPU) —
    exploits the MI300X's 192 GB to carry a code distance past the H200 sweep.
"""
import os, json
os.environ.setdefault("TRIDEC_FIX", "/workspace/bench/fixtures/")
from fleetbench import surface_lane, bb_lane
from fleet_serve import max_sustained_fleet

out = {}

def best_tool(scale):
    return [surface_lane(3, scale, "pymatching"),
            surface_lane(5, scale, "pymatching"),
            bb_lane(scale, "tridec-bp")]
bt = max_sustained_fleet(best_tool, [64, 128, 256, 512, 768, 1024], duration=3.0)
print("BEST-TOOL fleet (surf d3,d5 matching + BB qLDPC BP):", bt["max_fleet"], bt["seeds"])
for r in bt["last_sweep"]:
    print(f"  total={r['fleet_K']:>5} {'OVERLOAD' if r['overloaded'] else 'ok'} p99={r['p99_ms']}")
out["best_tool"] = bt

def md9(scale):
    return [surface_lane(d, scale, "tridec-bp") for d in (3, 5, 7, 9)]
m9 = max_sustained_fleet(md9, [16, 32, 64, 128, 256, 512], duration=3.0)
print("MIXED-DISTANCE+d9 fleet (d3+d5+d7+d9, BP):", m9["max_fleet"], m9["seeds"])
for r in m9["last_sweep"]:
    print(f"  total={r['fleet_K']:>5} {'OVERLOAD' if r['overloaded'] else 'ok'} p99={r['p99_ms']}")
out["mixed_distance_d9"] = m9

json.dump(out, open("/workspace/bench/results_fleet_extra_mi300x.json", "w"), indent=2)
print("saved results_fleet_extra_mi300x.json")
