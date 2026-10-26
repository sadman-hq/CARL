"""Open-loop load generator (separate process): drift-free schedule, scheduling-lag and CPU telemetry.
Run as a process:  python -m carl.loadgen '<json configuration>'  (started by the testbed harness)."""
import sys, json, time, asyncio, threading

import numpy as np, pandas as pd, psutil, aiohttp

from carl.common import gen_arrivals

try:
    import uvloop
    uvloop.install()
except Exception:
    pass

FIELDS = ["status", "chosen_region", "net_ms", "queue_ms", "svc_ms", "cold_ms", "cold", "util", "ci", "ci_seen",
          "carbon_g", "carbon_wall_g", "model_latency_ms", "burst_factor", "n_feasible", "fallback"]


async def run(C, cpu_samples):
    arr = gen_arrivals(C["pattern"], C["base_rps"], C["duration_s"], C["seed"], *C["burst"])
    d = C["t0_epoch"] - time.time()
    if d > 0:
        await asyncio.sleep(d)
    base = time.perf_counter()
    rows = []
    sess = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0))

    async def one(i, ts):
        send = time.perf_counter() - base
        pl = dict(mode=C["mode"], pattern=C["pattern"], request_id=i)
        try:
            async with sess.post(C["route_url"], json=pl, timeout=aiohttp.ClientTimeout(total=6)) as r:
                js = await r.json()
        except Exception:
            js = dict(status=598)
        e2e = (time.perf_counter() - base - send) * 1000.0
        rows.append(dict(request_id=i, t_rel=ts, sched_lag_ms=(send - ts) * 1000.0, e2e_ms=e2e,
                         **{k: js.get(k, np.nan) for k in FIELDS}))

    tasks = []
    for i, ts in enumerate(arr):
        now = time.perf_counter() - base
        await asyncio.sleep(ts - now if ts > now else 0)
        tasks.append(asyncio.create_task(one(i, ts)))
    await asyncio.gather(*tasks)
    wall = time.perf_counter() - base
    try:
        async with sess.get(C["route_url"].replace("/route", "/stats")) as r:
            rst = await r.json()
    except Exception:
        rst = {}
    await sess.close()
    df = pd.DataFrame(rows).sort_values("request_id")
    df.to_csv(C["out_csv"], index=False)
    cpu = [c for t, c in cpu_samples if t >= C["t0_epoch"]] or [np.nan]
    lag = df["sched_lag_ms"].to_numpy() if len(df) else np.zeros(1)
    meta = dict(duration_s=C["duration_s"], n_planned=len(arr), n_done=len(df), wall_s=wall,
                target_rps=len(arr) / C["duration_s"],
                lag_p50=float(np.percentile(lag, 50)), lag_p99=float(np.percentile(lag, 99)),
                cpu_mean=float(np.nanmean(cpu)), cpu_p95=float(np.nanpercentile(cpu, 95)), **rst)
    json.dump(meta, open(C["out_meta"], "w"))


def main():
    C = json.loads(sys.argv[1])
    cpu_samples, stop = [], threading.Event()

    def sampler():
        while not stop.is_set():
            cpu_samples.append((time.time(), psutil.cpu_percent(interval=0.5)))

    threading.Thread(target=sampler, daemon=True).start()
    try:
        asyncio.run(run(C, cpu_samples))
    finally:
        stop.set()


if __name__ == "__main__":
    main()
