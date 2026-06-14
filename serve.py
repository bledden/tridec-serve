"""A live continuous-batching decode server for QEC syndrome streams.

This is the real-system version of the analytical model in make_figure.py: a
running scheduler with a live arrival process and a worker thread that does
*actual* tridec decodes, so we measure throughput + the reaction-latency tail
(p50/p99/p99.9) and whether the GPU keeps up (bounded backlog) — not a formula.

Continuous batching: the worker drains everything currently queued (up to
max_batch) and decodes it in one tridec.decode_batch call, then loops. Batch
size adapts to load automatically — exactly the lever LLM serving uses.
"""
import time, threading, queue
import numpy as np


class DecodeServer:
    # Fixed batch-size buckets (LLM-serving trick): the scheduler pads every
    # batch up to the next bucket so only a few kernel shapes ever compile —
    # otherwise variable continuous-batch sizes trigger a triton recompile per
    # new shape, and those compile stalls dominate the latency tail.
    def __init__(self, decoder, buckets=(16, 64, 256, 1024, 4096)):
        self.decoder = decoder            # any tridec decoder with .decode_batch
        self.buckets = tuple(sorted(buckets))
        self.max_batch = self.buckets[-1]
        self.q = queue.Queue()            # items: (enqueue_perf_time, syndrome_row)
        self.latencies = []               # seconds, per syndrome (enqueue -> decoded)
        self.backlog = []                 # queue depth sampled per decode cycle
        self.batch_sizes = []
        self._stop = False
        self._t = threading.Thread(target=self._worker, daemon=True)

    def _bucket(self, n):
        for b in self.buckets:
            if n <= b:
                return b
        return self.buckets[-1]

    def warmup(self, syndrome_row):
        """Compile every bucket shape once (so the load run hits no recompiles)."""
        row = np.asarray(syndrome_row)
        for b in self.buckets:
            self.decoder.decode_batch(np.broadcast_to(row, (b, row.shape[0])).copy())

    def start(self):
        self._t.start()

    def submit(self, syndrome_row, t_enqueue):
        self.q.put((t_enqueue, syndrome_row))

    def _worker(self):
        while True:
            try:
                first = self.q.get(timeout=0.002)
            except queue.Empty:
                if self._stop and self.q.empty():
                    return
                continue
            batch = [first]
            while len(batch) < self.max_batch:          # drain queued arrivals
                try:
                    batch.append(self.q.get_nowait())
                except queue.Empty:
                    break
            self.backlog.append(self.q.qsize())
            L = len(batch)
            real = np.stack([b[1] for b in batch])
            b = self._bucket(L)                         # pad up to a fixed shape
            if b > L:
                syn = np.empty((b, real.shape[1]), dtype=real.dtype)
                syn[:L] = real
                syn[L:] = real[0]                       # dummy pad rows (discarded)
            else:
                syn = real
            self.batch_sizes.append(b)
            self.decoder.decode_batch(syn)              # the real decode
            done = time.perf_counter()
            for (t0, _) in batch:                       # only the L real syndromes
                self.latencies.append(done - t0)

    def stop(self):
        self._stop = True
        self._t.join()


def run_load(decoder, syndrome_pool, K, t_round, duration,
             buckets=(16, 64, 256, 1024, 4096), drain_cap=8.0, seed=0):
    """Drive K logical-qubit streams (each emits 1 syndrome / t_round) for
    `duration` seconds, then drain (capped). Returns measured stats."""
    rng = np.random.default_rng(seed)
    srv = DecodeServer(decoder, buckets=buckets)
    srv.start()
    t0 = time.perf_counter()
    t_end = t0 + duration
    next_round = t0
    emitted = 0
    pool_n = len(syndrome_pool)
    while time.perf_counter() < t_end:
        now = time.perf_counter()
        if now >= next_round:
            idx = rng.integers(0, pool_n, size=K)        # one syndrome per qubit
            for i in idx:
                srv.submit(syndrome_pool[i], now)
            emitted += K
            next_round += t_round
        else:
            time.sleep(min(0.0003, max(0.0, next_round - now)))
    # signal stop; bound the drain so an overloaded run can't hang
    srv._stop = True
    drain_start = time.perf_counter()
    while not srv.q.empty() and (time.perf_counter() - drain_start) < drain_cap:
        time.sleep(0.02)
    srv.stop()
    L = np.array(srv.latencies) * 1e3                     # ms
    decoded = len(L)
    max_backlog = int(max(srv.backlog)) if srv.backlog else 0
    # "sustained" = real-time-keeping-up = backlog stayed bounded during the load
    # (the drain cap lets a runaway run catch up afterwards, so decoded==emitted
    # is NOT the test; a growing queue is). ~1000 = falling >~half a max-batch behind.
    overloaded = max_backlog > 1000 or decoded < emitted * 0.9
    return {
        "K": K, "t_round_ms": t_round * 1e3, "duration_s": duration,
        "emitted": emitted, "decoded": decoded,
        "throughput_per_s": decoded / max(time.perf_counter() - t0, 1e-9),
        "offered_per_s": emitted / duration,
        "p50_ms": float(np.percentile(L, 50)) if decoded else None,
        "p99_ms": float(np.percentile(L, 99)) if decoded else None,
        "p999_ms": float(np.percentile(L, 99.9)) if decoded else None,
        "max_batch_used": int(max(srv.batch_sizes)) if srv.batch_sizes else 0,
        "max_backlog": max_backlog,
        "overloaded": bool(overloaded),
    }
