"""Torture test: finely sample p99 vs total fleet size right around the best-tool
knee, to characterize the SLA cliff shape (gradual degradation vs sudden) -- one
figure that's hard to fake post-hoc."""
import os, json
os.environ.setdefault("TRIDEC_FIX", "/workspace/bench/fixtures/")
PLAT = os.environ.get("TRIDEC_PLAT", "mi300x")
from fleetbench import surface_lane, bb_lane
from fleet_serve import run_fleet_load

rows = []
for per in [384, 448, 512, 576, 640, 704, 768, 832, 896]:   # total = 3*per, around the ~1536 knee
    lanes = [surface_lane(3, per, "pymatching"), surface_lane(5, per, "pymatching"), bb_lane(per, "tridec-bp")]
    r = run_fleet_load(lanes, t_round=1e-3, duration=3.0, seed=0)
    rows.append({"total": 3 * per, "p99_ms": r["p99_ms"], "overloaded": r["overloaded"]})
    print(f"total={3*per:5d}  p99={r['p99_ms']:8.1f} ms  overloaded={r['overloaded']}", flush=True)
json.dump({"plat": PLAT, "rows": rows}, open(f"/workspace/bench/results_torture_{PLAT}.json", "w"), indent=2)
print("saved")
