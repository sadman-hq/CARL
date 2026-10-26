"""CARL router: asynchronous telemetry polling, shared policies, no per-request disk I/O.
Run as a process:  python -m carl.router   (configured through environment variables)."""
import os, time, json, random, asyncio
from collections import deque
from contextlib import asynccontextmanager

import aiohttp
import numpy as np
from fastapi import FastAPI, Request
import uvicorn

from carl import common


def create_app(cfg, regions, workers):
    """regions: id -> base url ; workers: id -> concurrency limit."""
    rng = random.Random(cfg["seed"])
    rr = {"i": 0}
    tel = {r: dict(rtt=None, util=0.0, queue=0, svc=cfg["svc_prior_ms"], ci=None, ci_mean=None, hist=deque(maxlen=64))
           for r in regions}
    pending = {r: 0 for r in regions}            # requests this router has in flight per region
    lags = []
    state = dict(session=None)

    def build_snap():
        snap = {}
        for r, t in tel.items():
            if t["rtt"] is None or t["ci"] is None:
                continue
            w = workers[r]
            pend = pending[r]
            snap[r] = dict(rtt=t["rtt"], util=min(1.0, max(t["util"], pend / w)), queue=max(t["queue"], pend - w, 0),
                           workers=w, svc=t["svc"], ci=t["ci"], ci_mean=t["ci_mean"], hist=t["hist"])
        return snap

    async def poll_once(r, url):
        t = tel[r]
        try:
            t0 = time.perf_counter()
            async with state["session"].get(f"{url}/probe", timeout=aiohttp.ClientTimeout(total=1.5)) as resp:
                js = await resp.json()
            rtt = (time.perf_counter() - t0) * 1000.0
            t["rtt"] = rtt if t["rtt"] is None else 0.7 * t["rtt"] + 0.3 * rtt
            t.update(util=js["util"], queue=js["queue"], svc=js["svc_ms"], ci=js["ci"], ci_mean=js["ci_mean"])
            t["hist"].append(min(1.0, max(js["util"], pending[r] / workers[r])))
        except Exception:
            pass

    async def poll_loop():
        while True:
            t0 = time.perf_counter()
            await asyncio.gather(*[poll_once(r, u) for r, u in regions.items()])
            await asyncio.sleep(max(0.0, cfg["poll_s"] - (time.perf_counter() - t0)))

    async def lag_monitor():                      # event-loop lag = harness-contention indicator
        while True:
            t = time.perf_counter()
            await asyncio.sleep(0.05)
            lags.append((time.perf_counter() - t - 0.05) * 1000.0)

    @asynccontextmanager
    async def lifespan(app):
        state["session"] = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0))
        tasks = [asyncio.create_task(poll_loop()), asyncio.create_task(lag_monitor())]
        yield
        for t in tasks:
            t.cancel()
        await state["session"].close()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"ok": True, "regions": len(regions)}

    @app.post("/configure")                       # change weights / beta / SLA ... without restarting
    async def configure(p: dict):
        cfg.update(p)
        return {"ok": True}

    @app.post("/reset_state")
    async def reset_state():
        rr["i"] = 0
        lags.clear()
        for r in regions:
            pending[r] = 0
            tel[r]["hist"].clear()
        return {"ok": True}

    @app.get("/stats")
    async def stats():
        a = np.array(lags) if lags else np.zeros(1)
        return dict(loop_lag_p50=float(np.percentile(a, 50)), loop_lag_p99=float(np.percentile(a, 99)),
                    loop_lag_max=float(a.max()))

    @app.post("/route")
    async def route(req: Request):
        p = await req.json()
        mode = p["mode"]
        if mode == "noop":                         # harness calibration: no region work at all
            return {"status": 200, "chosen_region": "noop"}
        snap = build_snap()
        if snap:
            chosen, info = common.choose(mode, snap, cfg, rr, rng, p.get("pattern", "continuous"))
        else:
            chosen = sorted(regions)[rr["i"] % len(regions)]
            rr["i"] += 1
            info = dict(burst_factor=1.0, n_feasible=0, fallback=1)
        pending[chosen] += 1
        try:
            async with state["session"].post(f"{regions[chosen]}/invoke", json={},
                                             timeout=aiohttp.ClientTimeout(total=cfg["router_timeout_s"])) as resp:
                js = await resp.json()
        except Exception:
            js = dict(status=599, net_ms=0.0, queue_ms=0.0, svc_ms=0.0, cold_ms=0.0, cold=False, util=0.0,
                      ci=float("nan"), carbon_g=0.0, carbon_wall_g=0.0, model_latency_ms=cfg["router_timeout_s"] * 1000.0)
        finally:
            pending[chosen] = max(0, pending[chosen] - 1)
        js.update(chosen_region=chosen, burst_factor=info["burst_factor"], n_feasible=info["n_feasible"],
                  fallback=info["fallback"], ci_seen=snap.get(chosen, {}).get("ci"))
        return js

    return app


def main():
    cfg = json.loads(os.environ["CFG_JSON"])
    app = create_app(cfg, json.loads(os.environ["REGIONS_JSON"]), json.loads(os.environ["WORKERS_JSON"]))
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ["PORT"]), log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
