"""v4 — mixed code-distance / mixed-decoder FLEET serving.

The real heterogeneity lever: one GPU serves a FLEET of logical qubits that are
NOT all the same code — some surface-d=3, some surface-d=5, some BB qLDPC — each
needing a DIFFERENT decoder (matching for surface, BP for qLDPC) and a different
syndrome width. Syndromes from different codes can't share a batch, so the server
keeps a per-lane queue + decoder and **time-shares the one GPU** across lanes
(a single worker cycles the lanes, decoding one lane's bucketed batch at a time).

This is the vendor-neutrality lever made operational: routing across decoder
FAMILIES on one accelerator is something a single CUDA-locked decoder
(`nv-qldpc`) structurally cannot do. tridec(BP) + PyMatching(matching) on one GPU
— on AMD too — can.

Metric: max sustained TOTAL fleet size (sum of per-lane logical qubits) at a 1 ms
round with bounded backlog, plus the per-lane p99 breakdown.
"""
import time, threading, queue
import numpy as np

DEFAULT_BUCKETS = (16, 64, 256, 1024, 4096)


class FleetServer:
    """Multi-lane decode server: one worker time-shares the GPU across lanes."""

    def __init__(self, lane_specs):
        # lane_specs: list of {name, decoder, ncols, buckets?}
        self.lanes = {}
        for s in lane_specs:
            self.lanes[s["name"]] = {
                "decoder": s["decoder"],
                "ncols": s["ncols"],
                "buckets": tuple(sorted(s.get("buckets", DEFAULT_BUCKETS))),
                "q": queue.Queue(),
                "lat": [],            # per-syndrome reaction latency (s)
                "batch_sizes": [],
            }
        self._stop = False
        self._t = threading.Thread(target=self._worker, daemon=True)

    @staticmethod
    def _bucket(buckets, n):
        for b in buckets:
            if n <= b:
                return b
        return buckets[-1]

    def warmup(self, name, row):
        ln = self.lanes[name]
        row = np.asarray(row); ok = []
        for b in ln["buckets"]:
            try:
                ln["decoder"].decode_batch(np.broadcast_to(row, (b, row.shape[0])).copy())
                ok.append(b)
            except Exception:
                break
        if ok:
            ln["buckets"] = tuple(ok)

    def start(self):
        self._t.start()

    def submit(self, name, arr, t):
        self.lanes[name]["q"].put((t, arr))

    def _drain_decode(self, ln):
        """Drain one lane's queue into a bucketed batch and decode it. Returns
        True if it did work."""
        try:
            t0, a0 = ln["q"].get_nowait()
        except queue.Empty:
            return False
        items = [(t0, a0)]; total = a0.shape[0]; maxb = ln["buckets"][-1]
        while total < maxb:
            try:
                t, a = ln["q"].get_nowait()
            except queue.Empty:
                break
            if total + a.shape[0] > maxb:
                ln["q"].put((t, a)); break
            items.append((t, a)); total += a.shape[0]
        syn = np.vstack([a for _, a in items])
        tt = np.concatenate([np.full(a.shape[0], t) for t, a in items])
        L = syn.shape[0]; b = self._bucket(ln["buckets"], L)
        if b > L:
            pad = np.empty((b, syn.shape[1]), dtype=syn.dtype)
            pad[:L] = syn; pad[L:] = syn[0]; syn = pad
        try:
            ln["decoder"].decode_batch(syn)
        except Exception:
            smaller = tuple(x for x in ln["buckets"] if x < b)
            ln["buckets"] = smaller or (ln["buckets"][0],)
            return True
        done = time.perf_counter()
        ln["lat"].extend((done - tt).tolist())
        ln["batch_sizes"].append(b)
        return True

    def _worker(self):
        while True:
            did = False
            for ln in self.lanes.values():          # round-robin: time-share the GPU
                did = self._drain_decode(ln) or did
            if not did:
                if self._stop and all(l["q"].empty() for l in self.lanes.values()):
                    return
                time.sleep(0.001)

    def stop(self):
        self._stop = True
        self._t.join()


def run_fleet_load(lanes, t_round=1e-3, duration=4.0, drain_cap=8.0, seed=0):
    """lanes: list of {name, decoder, pool, K[, buckets]}. Each lane emits K
    syndromes/round from its own pool; all share one GPU via the FleetServer.
    Returns total + per-lane sustained stats."""
    rng = np.random.default_rng(seed)
    pools = {L["name"]: np.ascontiguousarray(L["pool"]) for L in lanes}
    Ks = {L["name"]: int(L["K"]) for L in lanes}
    specs = [{"name": L["name"], "decoder": L["decoder"], "ncols": pools[L["name"]].shape[1],
              "buckets": L.get("buckets", DEFAULT_BUCKETS)} for L in lanes]
    srv = FleetServer(specs)
    for L in lanes:
        srv.warmup(L["name"], pools[L["name"]][0])
    srv.start()
    t0 = time.perf_counter(); t_end = t0 + duration; next_round = t0
    emitted = {n: 0 for n in Ks}
    while time.perf_counter() < t_end:
        now = time.perf_counter()
        if now >= next_round:
            for n, K in Ks.items():
                if K > 0:
                    p = pools[n]
                    srv.submit(n, p[rng.integers(0, p.shape[0], size=K)], now)
                    emitted[n] += K
            next_round += t_round
        else:
            time.sleep(min(0.0003, max(0.0, next_round - now)))
    decoded_during = {n: len(srv.lanes[n]["lat"]) for n in Ks}
    srv._stop = True
    drain_start = time.perf_counter()
    while any(not srv.lanes[n]["q"].empty() for n in Ks) and (time.perf_counter() - drain_start) < drain_cap:
        time.sleep(0.02)
    srv.stop()

    total_emitted = sum(emitted.values())
    total_during = sum(decoded_during.values())
    overloaded = bool(total_during < total_emitted * (1.0 - 0.5 / duration))
    per_lane = {}
    for n in Ks:
        lat = np.array(srv.lanes[n]["lat"]) * 1e3
        per_lane[n] = {
            "K": Ks[n], "emitted": emitted[n], "decoded_during": decoded_during[n],
            "p99_ms": float(np.percentile(lat, 99)) if len(lat) else None,
            "p50_ms": float(np.percentile(lat, 50)) if len(lat) else None,
            "max_batch": int(max(srv.lanes[n]["batch_sizes"])) if srv.lanes[n]["batch_sizes"] else 0,
        }
    return {
        "fleet_total_K": sum(Ks.values()),
        "composition": dict(Ks),
        "overloaded": overloaded,
        "total_emitted": total_emitted, "total_decoded_during": total_during,
        "p99_ms": max((v["p99_ms"] or 0.0) for v in per_lane.values()) if per_lane else None,
        "per_lane": per_lane,
    }


def max_sustained_fleet(make_lanes, scales, t_round=1e-3, duration=4.0, seeds=(0, 1, 2)):
    """make_lanes(scale) -> lanes list (per-lane K grows with scale). Sweep scales,
    return the largest total fleet sustained (median over seeds [min-max])."""
    per_seed_max = []
    sweep = []
    for sd in seeds:
        best = 0; rows = []
        for sc in scales:
            lanes = make_lanes(sc)
            r = run_fleet_load(lanes, t_round=t_round, duration=duration, seed=sd)
            rows.append({"scale": sc, "fleet_K": r["fleet_total_K"],
                         "overloaded": r["overloaded"], "p99_ms": r["p99_ms"]})
            if r["overloaded"]:
                break
            best = r["fleet_total_K"]
        per_seed_max.append(best); sweep = rows
    srt = sorted(per_seed_max)
    return {"max_fleet": srt[len(srt) // 2], "seeds": per_seed_max,
            "lo": min(per_seed_max), "hi": max(per_seed_max), "last_sweep": sweep}
