"""A live continuous-batching decode server for QEC syndrome streams.

Round-batched arrivals: the producer enqueues ONE item per round (a (m, ncols)
array of that round's syndromes + a shared timestamp), so the load generator is
O(rounds/s) not O(syndromes/s) and can drive ~1M+ syndromes/s. The worker drains
round-items, concatenates them into one tridec.decode_batch (padded to a fixed
bucket shape so Triton compiles only a few kernels), and records per-syndrome
reaction latency. Heterogeneous load: a fraction `active_frac` of each round's
syndromes carry real errors; the rest are error-free ("idle" memory qubits),
which the relay megakernel's per-shot early-exit decodes nearly for free.
"""
import time, threading, queue
import numpy as np


class DecodeServer:
    def __init__(self, decoder, buckets=(16, 64, 256, 1024, 4096)):
        self.decoder = decoder
        self.buckets = tuple(sorted(buckets))
        self.max_batch = self.buckets[-1]
        self.q = queue.Queue()            # items: (enqueue_time, ndarray[m, ncols])
        self.latencies = []               # seconds, per syndrome
        self.backlog = []                 # queued *syndromes* sampled per cycle
        self.batch_sizes = []
        self._stop = False
        self._t = threading.Thread(target=self._worker, daemon=True)

    def _bucket(self, n):
        for b in self.buckets:
            if n <= b:
                return b
        return self.buckets[-1]

    def warmup(self, row):
        # Compile each bucket shape; if the device can't allocate a big bucket
        # (e.g. Metal's two-kernel BP at 4096), cap the buckets there.
        row = np.asarray(row); ok = []
        for b in self.buckets:
            try:
                self.decoder.decode_batch(np.broadcast_to(row, (b, row.shape[0])).copy())
                ok.append(b)
            except Exception:
                break
        if ok:
            self.buckets = tuple(ok)
            self.max_batch = ok[-1]

    def start(self):
        self._t.start()

    def submit_round(self, arr, t):
        self.q.put((t, arr))              # arr: (m, ncols) syndromes arriving at t

    def _qrows(self):
        # approximate queued-syndrome count (Queue of round-arrays)
        return self.q.qsize()

    def _worker(self):
        while True:
            try:
                t0, a0 = self.q.get(timeout=0.002)
            except queue.Empty:
                if self._stop and self.q.empty():
                    return
                continue
            arrs, times, total = [a0], [np.full(a0.shape[0], t0)], a0.shape[0]
            while total < self.max_batch:                   # drain more rounds
                try:
                    t, a = self.q.get_nowait()
                except queue.Empty:
                    break
                if total + a.shape[0] > self.max_batch:      # don't overflow the batch
                    self.q.put((t, a))
                    break
                arrs.append(a); times.append(np.full(a.shape[0], t)); total += a.shape[0]
            self.backlog.append(self.q.qsize())
            syn = np.vstack(arrs)
            tt = np.concatenate(times)
            L = syn.shape[0]
            b = self._bucket(L)
            if b > L:
                pad = np.empty((b, syn.shape[1]), dtype=syn.dtype)
                pad[:L] = syn; pad[L:] = syn[0]
                syn = pad
            self.batch_sizes.append(b)
            try:
                self.decoder.decode_batch(syn)
            except Exception:
                # device can't handle this batch size (e.g. Metal buffer limit):
                # cap buckets below it and drop this batch so the worker thread
                # survives instead of dying mid-run.
                smaller = tuple(x for x in self.buckets if x < b)
                self.buckets = smaller or (self.buckets[0],)
                self.max_batch = self.buckets[-1]
                continue
            done = time.perf_counter()
            self.latencies.extend((done - tt).tolist())

    def stop(self):
        self._stop = True
        self._t.join()


def run_load(decoder, syndrome_pool, K, t_round, duration,
             active_frac=1.0, buckets=(16, 64, 256, 1024, 4096),
             drain_cap=8.0, seed=0):
    """Drive K logical qubits (1 syndrome/qubit/round). `active_frac` of each
    round carries real errors (sampled from the pool); the rest are error-free
    idle rows. Returns measured stats. Backlog is in queued *rounds*."""
    rng = np.random.default_rng(seed)
    pool = np.ascontiguousarray(syndrome_pool)
    ncols = pool.shape[1]
    n_active = int(round(active_frac * K))
    srv = DecodeServer(decoder, buckets=buckets)
    srv.warmup(pool[0])
    srv.start()
    t0 = time.perf_counter(); t_end = t0 + duration; next_round = t0
    emitted = 0
    while time.perf_counter() < t_end:
        now = time.perf_counter()
        if now >= next_round:
            if n_active >= K:
                arr = pool[rng.integers(0, pool.shape[0], size=K)]
            else:
                arr = np.zeros((K, ncols), dtype=pool.dtype)
                if n_active:
                    arr[:n_active] = pool[rng.integers(0, pool.shape[0], size=n_active)]
                    rng.shuffle(arr)                          # mix active among idle
            srv.submit_round(arr, now)
            emitted += K
            next_round += t_round
        else:
            time.sleep(min(0.0003, max(0.0, next_round - now)))
    # SUSTAINED = kept up in real time: by load-end the decoder is within ~0.5 s
    # of arrivals (only a small in-flight backlog left). OVERLOADED = a large
    # undecoded backlog remains. Reproducible + robust to decoder speed / batch /
    # round-drain artifacts (unlike a latency-trend, which trips on launch jitter
    # and batch ramp-up). decoded_during_load = snapshot before the drain.
    decoded_during_load = len(srv.latencies)
    srv._stop = True
    drain_start = time.perf_counter()
    while not srv.q.empty() and (time.perf_counter() - drain_start) < drain_cap:
        time.sleep(0.02)
    srv.stop()
    Lat = np.array(srv.latencies) * 1e3                  # ms, completion order
    decoded = len(Lat)
    offered = emitted / duration
    overloaded = bool(decoded_during_load < emitted * (1.0 - 0.5 / duration))
    return {
        "K": K, "active_frac": active_frac, "t_round_ms": t_round*1e3,
        "emitted": emitted, "decoded": decoded, "decoded_during_load": decoded_during_load,
        "throughput_per_s": decoded / max(time.perf_counter() - t0, 1e-9),
        "offered_per_s": offered,
        "p50_ms": float(np.percentile(Lat, 50)) if decoded else None,
        "p99_ms": float(np.percentile(Lat, 99)) if decoded else None,
        "p999_ms": float(np.percentile(Lat, 99.9)) if decoded else None,
        "max_batch_used": int(max(srv.batch_sizes)) if srv.batch_sizes else 0,
        "max_backlog_rounds": int(max(srv.backlog)) if srv.backlog else 0,
        "overloaded": bool(overloaded),
    }
