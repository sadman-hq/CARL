"""Region service: serverless instance pool (warm/cold), bounded queue, emulated RTT, active-time energy.
Run as a process:  python -m carl.region_service   (configured through environment variables)."""
import os, time, json, random, asyncio
from collections import deque

from carl.common import ci_step, active_carbon_g

from fastapi import FastAPI
import uvicorn


class Throttled(Exception):
    pass


class QueueTimeout(Exception):
    pass


class Pool:
    """Serverless instance pool: <= workers instances; idle instances expire after keep_alive_s;
    a request on a fresh instance pays a cold start; excess requests wait in a bounded queue."""

    def __init__(self, cfg, workers):
        self.cfg, self.workers = cfg, workers
        self.queue_limit = int(cfg["queue_limit_factor"] * workers)
        self.reset(True)

    def reset(self, cold=True):
        self.busy = 0
        self.idle = [] if cold else [time.time()] * self.workers
        self.waiters = deque()
        self.svc_ewma = None
        self.served = self.cold_n = self.throttled = self.timeouts = 0

    def qlen(self):
        return sum(1 for f in self.waiters if not f.done())

    async def acquire(self, timeout):
        now = time.time()
        ka = self.cfg["keep_alive_s"]
        self.idle = [t for t in self.idle if now - t <= ka]
        if self.idle:
            self.idle.pop(); self.busy += 1
            return False
        if self.busy < self.workers:
            self.busy += 1
            return True                                         # scale out -> cold start
        if self.qlen() >= self.queue_limit:
            raise Throttled()
        fut = asyncio.get_running_loop().create_future()
        self.waiters.append(fut)
        try:
            await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError:
            if fut.done() and not fut.cancelled() and fut.result() is True:
                return False                                    # hand-off raced with the timeout
            raise QueueTimeout()
        return False

    def release(self):
        while self.waiters:
            f = self.waiters.popleft()
            if not f.done():
                f.set_result(True)                              # instance handed to next waiter (stays busy)
                return
        self.busy -= 1
        self.idle.append(time.time())


def create_app(cfg, region, rtt_ms, workers, speed, trace):
    pool = Pool(cfg, workers)
    clock = dict(t0=time.time(), warp=1.0, offset_h=0.0)
    trace_mean = sum(trace) / len(trace)
    app = FastAPI()

    def trace_hour():
        return clock["offset_h"] + max(0.0, time.time() - clock["t0"]) * clock["warp"] / 3600.0

    def rtt_sample():
        return max(1.0, random.gauss(rtt_ms, cfg["rtt_jitter"] * rtt_ms))

    def fail(status, net, q_ms):
        return dict(status=status, region=region, net_ms=net, queue_ms=q_ms, svc_ms=0.0, cold_ms=0.0, cold=False,
                    util=pool.busy / workers, ci=ci_step(trace, trace_hour()), carbon_g=0.0, carbon_wall_g=0.0,
                    model_latency_ms=net + q_ms)

    @app.get("/health")
    async def health():
        return {"ok": True, "region": region}

    @app.post("/reset")
    async def reset(p: dict):
        clock.update(t0=p["t0"], warp=p["warp"], offset_h=p["offset_h"])
        random.seed(p["seed"])
        pool.reset(p.get("cold", True))
        return {"ok": True}

    @app.get("/stats")
    async def stats():
        return dict(served=pool.served, cold=pool.cold_n, throttled=pool.throttled, timeouts=pool.timeouts)

    @app.get("/probe")                                          # health + telemetry in one RTT-delayed call
    async def probe():
        await asyncio.sleep(rtt_sample() / 1000.0)
        return dict(util=pool.busy / workers, queue=pool.qlen(), svc_ms=pool.svc_ewma or cfg["svc_prior_ms"],
                    ci=ci_step(trace, trace_hour()), ci_mean=trace_mean)

    @app.post("/invoke")
    async def invoke():
        net = rtt_sample()
        await asyncio.sleep(net / 2000.0)                       # request leg
        t_q = time.perf_counter()
        try:
            cold = await pool.acquire(cfg["queue_timeout_s"])
        except Throttled:
            pool.throttled += 1
            await asyncio.sleep(net / 2000.0)
            return fail(429, net, 0.0)
        except QueueTimeout:
            pool.timeouts += 1
            await asyncio.sleep(net / 2000.0)
            return fail(504, net, cfg["queue_timeout_s"] * 1000.0)
        queue_ms = (time.perf_counter() - t_q) * 1000.0
        u = pool.busy / workers
        svc = cfg["svc_base_ms"] * speed * (1.0 + cfg["alpha_contention"] * u) * random.lognormvariate(0, cfg["svc_jitter"])
        cold_ms = random.uniform(cfg["cold_ms_min"], cfg["cold_ms_max"]) if cold else 0.0
        await asyncio.sleep((svc + cold_ms) / 1000.0)           # ACTIVE execution (draws compute power)
        pool.release()
        pool.served += 1
        pool.cold_n += int(cold)
        pool.svc_ewma = svc if pool.svc_ewma is None else 0.8 * pool.svc_ewma + 0.2 * svc
        await asyncio.sleep(net / 2000.0)                       # response leg
        ci = ci_step(trace, trace_hour())
        act_s = (svc + cold_ms) / 1000.0
        return dict(status=200, region=region, net_ms=net, queue_ms=queue_ms, svc_ms=svc, cold_ms=cold_ms, cold=cold,
                    util=u, ci=ci, carbon_g=active_carbon_g(u, act_s, ci, cfg),                # active time only
                    carbon_wall_g=active_carbon_g(u, act_s + queue_ms / 1000.0, ci, cfg),      # legacy: queueing included
                    model_latency_ms=net + queue_ms + svc + cold_ms)

    return app


def main():
    cfg = json.loads(os.environ["CFG_JSON"])
    region = os.environ["REGION"]
    trace = json.load(open(os.environ["TRACE_PATH"]))[region]
    app = create_app(cfg, region, float(os.environ["RTT_MS"]), int(os.environ["WORKERS"]),
                     float(os.environ["SPEED"]), trace)
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ["PORT"]), log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
