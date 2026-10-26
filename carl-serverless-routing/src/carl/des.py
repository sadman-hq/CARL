"""Discrete-event simulator (virtual time, no CPU contention): same policies, same region model as the live testbed."""
import heapq, json, os, random
from collections import deque
from pathlib import Path

import numpy as np, pandas as pd

from carl import common


def make_profiles(base, traces, n, seed=0):
    """n regions: the 16 real profiles first, then perturbed clones (RTT +-20%, trace scale +-20%)."""
    rng = random.Random(seed)
    out = []
    for k in range(n):
        b = base[k % len(base)]
        if k < len(base):
            out.append(dict(rid=b["rid"], rtt=b["rtt"], workers=b["workers"], speed=b["speed"], trace=traces[b["rid"]]))
        else:
            sc = rng.uniform(0.8, 1.2)
            out.append(dict(rid=f"R{k + 1:03d}", rtt=b["rtt"] * rng.uniform(0.8, 1.2), workers=b["workers"],
                            speed=b["speed"], trace=[v * sc * (1 + rng.gauss(0, 0.03)) for v in traces[b["rid"]]]))
    return out


def simulate(profiles, cfg, mode, pattern, base_rps, duration_s, seed, offset_h, burst=(4.0, 20.0, 5.0)):
    rng = random.Random(seed * 7919 + 13)
    rr = {"i": 0}
    arr = common.gen_arrivals(pattern, base_rps, duration_s, seed, *burst)
    warp = cfg["trace_hours_per_trial"] * 3600.0 / duration_s
    N = len(profiles)
    rid = [p["rid"] for p in profiles]
    ridx = {x: i for i, x in enumerate(rid)}
    W = [p["workers"] for p in profiles]
    QL = [int(cfg["queue_limit_factor"] * w) for w in W]
    busy = [0] * N
    idle = [[] for _ in range(N)]
    wq = [deque() for _ in range(N)]
    qn = [0] * N
    svc_e = [None] * N
    pend = [0] * N
    tmean = [sum(p["trace"]) / len(p["trace"]) for p in profiles]
    tel = [dict(rtt=None, util=0.0, queue=0, svc=cfg["svc_prior_ms"], ci=None, hist=deque(maxlen=64)) for _ in range(N)]
    hr = lambda t: offset_h + t * warp / 3600.0
    ev, seq = [], 0

    def push(t, kind, a=None, b=None):
        nonlocal seq
        seq += 1
        heapq.heappush(ev, (t, seq, kind, a, b))

    push(0.0, 0)                                                   # first telemetry poll
    for i, t in enumerate(arr):
        push(t + 1e-6, 1, i)
    rows = {k: [] for k in ["status", "model_latency_ms", "cold", "ci", "carbon_g", "carbon_wall_g", "chosen_region",
                            "fallback", "n_feasible", "t_rel"]}

    def finish(req, t, status, queue_ms=0.0, svc=0.0, cold_ms=0.0, cold=False, u=0.0):
        r = req["r"]
        pend[r] = max(0, pend[r] - 1)
        lat = req["net"] + queue_ms + svc + cold_ms
        ci = common.ci_step(profiles[r]["trace"], hr(t + req["net"] / 2000.0))
        if status == 200 and lat > cfg["router_timeout_s"] * 1000.0:
            status = 599
        act = (svc + cold_ms) / 1000.0
        rows["status"].append(status)
        rows["model_latency_ms"].append(lat)
        rows["cold"].append(cold)
        rows["ci"].append(ci)
        rows["carbon_g"].append(common.active_carbon_g(u, act, ci, cfg) if status == 200 else 0.0)
        rows["carbon_wall_g"].append(common.active_carbon_g(u, act + queue_ms / 1000.0, ci, cfg) if status == 200 else 0.0)
        rows["chosen_region"].append(rid[r])
        rows["fallback"].append(req["fb"])
        rows["n_feasible"].append(req["nf"])
        rows["t_rel"].append(req["t_send"])

    def start(r, req, t, cold):
        u = busy[r] / W[r]
        svc = cfg["svc_base_ms"] * profiles[r]["speed"] * (1 + cfg["alpha_contention"] * u) * rng.lognormvariate(0, cfg["svc_jitter"])
        cms = rng.uniform(cfg["cold_ms_min"], cfg["cold_ms_max"]) if cold else 0.0
        req.update(q_ms=(t - req["t_at"]) * 1000.0, svc=svc, cms=cms, cold=cold, u=u)
        push(t + (svc + cms) / 1000.0, 3, r, req)

    last_t = duration_s + 10.0
    while ev:
        t, _, kind, a, b = heapq.heappop(ev)
        if kind == 0:                                              # router telemetry poll
            for r in range(N):
                s = tel[r]
                x = max(1.0, rng.gauss(profiles[r]["rtt"], cfg["rtt_jitter"] * profiles[r]["rtt"])) + 2.0
                s["rtt"] = x if s["rtt"] is None else 0.7 * s["rtt"] + 0.3 * x
                s["util"] = busy[r] / W[r]
                s["queue"] = qn[r]
                s["svc"] = svc_e[r] or cfg["svc_prior_ms"]
                s["ci"] = common.ci_step(profiles[r]["trace"], hr(t))
                s["hist"].append(min(1.0, max(s["util"], pend[r] / W[r])))
            if t < last_t:
                push(t + cfg["poll_s"], 0)
        elif kind == 1:                                            # client request arrives at router
            snap = {rid[r]: dict(rtt=tel[r]["rtt"], util=min(1.0, max(tel[r]["util"], pend[r] / W[r])),
                                 queue=max(tel[r]["queue"], pend[r] - W[r], 0), workers=W[r], svc=tel[r]["svc"],
                                 ci=tel[r]["ci"], ci_mean=tmean[r], hist=tel[r]["hist"]) for r in range(N)}
            ch, info = common.choose(mode, snap, cfg, rr, rng, pattern)
            r = ridx[ch]
            pend[r] += 1
            net = max(1.0, rng.gauss(profiles[r]["rtt"], cfg["rtt_jitter"] * profiles[r]["rtt"]))
            req = dict(r=r, net=net, t_send=t, fb=info["fallback"], nf=info["n_feasible"])
            push(t + net / 2000.0, 2, r, req)
        elif kind == 2:                                            # arrives at region
            r, req = a, b
            req["t_at"] = t
            idle[r] = [x for x in idle[r] if t - x <= cfg["keep_alive_s"]]
            if idle[r]:
                idle[r].pop()
                busy[r] += 1
                start(r, req, t, False)
            elif busy[r] < W[r]:
                busy[r] += 1
                start(r, req, t, True)
            elif qn[r] >= QL[r]:
                finish(req, t, 429)
            else:
                req["state"] = "wait"
                wq[r].append(req)
                qn[r] += 1
                push(t + cfg["queue_timeout_s"], 4, r, req)
        elif kind == 3:                                            # service finished
            r, req = a, b
            while wq[r]:
                w = wq[r].popleft()
                if w.get("state") == "wait":
                    w["state"] = "run"
                    qn[r] -= 1
                    start(r, w, t, False)
                    break
            else:
                busy[r] -= 1
                idle[r].append(t)
            svc_e[r] = req["svc"] if svc_e[r] is None else 0.8 * svc_e[r] + 0.2 * req["svc"]
            finish(req, t, 200, req["q_ms"], req["svc"], req["cms"], req["cold"], req["u"])
        else:                                                      # queue timeout
            r, req = a, b
            if req.get("state") == "wait":
                req["state"] = "timeout"
                qn[r] -= 1
                finish(req, t, 504, cfg["queue_timeout_s"] * 1000.0)
    return pd.DataFrame(rows), rid


_CACHE = {}


def _load():
    """Reads data/config.json, data/regions.json and data/carbon/traces.json from $CARL_ROOT (default: cwd)."""
    root = Path(os.environ.get("CARL_ROOT", "."))
    key = str(root)
    if _CACHE.get("root") != key:
        _CACHE.clear()
        _CACHE["root"] = key
        _CACHE["cfg"] = json.load(open(root / "data" / "config.json"))
        _CACHE["base"] = json.load(open(root / "data" / "regions.json"))
        _CACHE["traces"] = json.load(open(root / "data" / "carbon" / "traces.json"))
    return _CACHE


def des_job(spec):
    """spec: tag, level, pattern, mode, rep, rps, duration, n_regions, cfg_over (dict), trace_days, scen_id, label"""
    L = _load()
    cfg = {**L["cfg"], **spec.get("cfg_over", {})}
    n = spec.get("n_regions", 16)
    prof = make_profiles(L["base"], L["traces"], n, seed=17)
    seed = 1000 * spec["rep"] + spec.get("scen_id", 0)
    df, rid = simulate(prof, cfg, spec["mode"], spec["pattern"], spec["rps"] * n / 16.0, spec["duration"], seed,
                       24.0 * (spec["rep"] % spec.get("trace_days", 7)))
    m, share, cih = common.run_metrics(df, cfg, spec["duration"], rid)
    key = {k: spec[k] for k in ["tag", "level", "pattern", "mode", "rep", "n_regions", "label"] if k in spec}
    return {**key, **m, "share": share, "ci_hour": cih}
