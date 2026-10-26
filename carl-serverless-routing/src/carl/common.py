"""Shared core used by BOTH the live testbed and the discrete-event simulator:
routing policies, arrival generation, energy/carbon model and per-run metrics."""
import math, random
import numpy as np


# ---------------- carbon trace clock (hourly values, step-held) ----------------
def ci_step(trace, hour_f):
    return float(trace[int(hour_f) % len(trace)])


# ---------------- energy / carbon model ----------------
def active_carbon_g(util, active_s, ci, cfg):
    """Eqs. 4-5 with T_i = ACTIVE execution time only (service + cold-start init). Queueing and
    network time draw no compute power. gCO2 = P(u)[W] * t[s] / 3.6e6 * PUE * CI[gCO2/kWh]."""
    p = cfg["p_idle_w"] + (cfg["p_max_w"] - cfg["p_idle_w"]) * util
    return p * active_s / 3.6e6 * cfg["pue"] * ci


# ---------------- arrivals (open-loop Poisson / bursty NHPP via thinning) ----------------
def rate_at(t, base, pattern, mult, every, blen):
    if pattern == "bursty" and (t % every) < blen:
        return base * mult
    return base


def gen_arrivals(pattern, base_rps, duration_s, seed, mult=4.0, every=20.0, blen=5.0):
    rng = random.Random(seed)
    lam_max = base_rps * (mult if pattern == "bursty" else 1.0)
    t, out = 0.0, []
    while True:
        t += rng.expovariate(lam_max)
        if t >= duration_s:
            return out
        if rng.random() <= rate_at(t, base_rps, pattern, mult, every, blen) / lam_max:
            out.append(t)


# ---------------- routing policies ----------------
# snap[r] = dict(rtt, util, queue, workers, svc, ci, ci_mean, hist)   (all as seen by the router)
def _norm(d):
    if not d:
        return {}
    mn, mx = min(d.values()), max(d.values())
    return {k: 0.0 for k in d} if mx - mn < 1e-12 else {k: (v - mn) / (mx - mn) for k, v in d.items()}


def pred_latency(s, cfg):
    """Currently-observed latency estimate L_r = RTT + service(u) + expected queueing wait."""
    svc = s["svc"] * (1.0 + cfg["alpha_contention"] * s["util"])
    return s["rtt"] + svc + (s["queue"] / max(1, s["workers"])) * svc


def ols_slope(h):
    n = len(h)
    if n < 3:
        return 0.0
    xm, ym = (n - 1) / 2.0, sum(h) / n
    num = sum((i - xm) * (y - ym) for i, y in enumerate(h))
    den = sum((i - xm) ** 2 for i in range(n))
    return num / den


def burst_factor(hist, cfg):
    """beta_r = 1 + beta * max(0, projected load rise over the window). The same quantity is used in
    Eq. 2 and Algorithm 1: 'ols' = regression slope * (W-1); 'diff' = last - first sample."""
    h = list(hist)[-cfg["burst_window"]:]
    if len(h) < 2:
        return 1.0
    delta = (h[-1] - h[0]) if cfg["burst_estimator"] == "diff" else ols_slope(h) * (len(h) - 1)
    return 1.0 + cfg["beta"] * max(0.0, delta)


def _argmin(keys, f, rng):
    best = min(f(k) for k in keys)
    return rng.choice([k for k in keys if f(k) <= best + 1e-12])


def _rr(snap, rr):
    regs = sorted(snap)
    r = regs[rr["i"] % len(regs)]
    rr["i"] += 1
    return r


POLICIES = ["round_robin", "latency", "load", "carbon_greedy", "carbon_spill", "eps_latency",
            "carl_static", "carl", "carl_adaptive"]


def choose(mode, snap, cfg, rr, rng, pattern="continuous"):
    """Select a region. Returns (region_id, info) with info = dict(burst_factor, n_feasible, fallback)."""
    regs = list(snap)
    info = dict(burst_factor=1.0, n_feasible=len(regs), fallback=0)
    if mode == "round_robin":
        return _rr(snap, rr), info
    if mode == "latency":                                   # lowest observed network RTT
        return _argmin(regs, lambda r: snap[r]["rtt"], rng), info
    if mode == "load":                                      # lowest utilisation
        return _argmin(regs, lambda r: snap[r]["util"], rng), info
    if mode == "carbon_greedy":                             # strawman: lowest live carbon, nothing else
        return _argmin(regs, lambda r: snap[r]["ci"], rng), info

    lat = {r: pred_latency(snap[r], cfg) for r in regs}
    sla = cfg["sla_ms"]
    feas = [r for r in regs if lat[r] <= sla]
    info["n_feasible"] = len(feas)

    if mode == "carbon_spill":                              # SLA- and capacity-aware carbon fill-and-spill
        cand = [r for r in feas if snap[r]["util"] < cfg["u_cap"]]
        if cand:
            return _argmin(cand, lambda r: snap[r]["ci"], rng), info
        info["fallback"] = 1
        return (_argmin(feas, lambda r: lat[r], rng) if feas else _rr(snap, rr)), info

    if mode == "eps_latency":                               # lowest carbon within (1+eps) of best latency
        best = min(lat.values())
        cand = [r for r in feas if lat[r] <= (1.0 + cfg["eps_lat"]) * best]
        if cand:
            return _argmin(cand, lambda r: snap[r]["ci"], rng), info
        info["fallback"] = 1
        return _argmin(regs, lambda r: lat[r], rng), info

    # ---- CARL family: carl (live carbon) | carl_static (trace-mean carbon) | carl_adaptive ----
    if not feas:
        info["fallback"] = 1
        return _rr(snap, rr), info
    live = mode != "carl_static"
    ln = _norm({r: lat[r] for r in feas})
    un = _norm({r: snap[r]["util"] for r in feas})
    cn = _norm({r: (snap[r]["ci"] if live else snap[r]["ci_mean"]) for r in feas})
    wl, wu, wc = cfg["w_lat"], cfg["w_load"], cfg["w_carb"]
    if mode == "carl_adaptive":                             # shift weight from carbon to load/latency as fleet load rises
        mu = sum(snap[r]["util"] for r in regs) / len(regs)
        wc2 = wc * max(0.0, 1.0 - mu / cfg["adapt_u0"])
        k = (1.0 - wc2) / max(1e-9, wl + wu)
        wl, wu, wc = wl * k, wu * k, wc2
    use_burst = (pattern == "bursty") or cfg["burst_trigger"] == "always"
    best_r, best_s, best_bf = None, 1e18, 1.0
    for r in feas:
        bf = burst_factor(snap[r]["hist"], cfg) if use_burst else 1.0
        sc = wl * ln[r] + wu * un[r] * bf + wc * cn[r] + 1e-9 * rng.random()
        if sc < best_s:
            best_r, best_s, best_bf = r, sc, bf
    info["burst_factor"] = best_bf
    return best_r, info


# ---------------- metrics (identical for live testbed and simulator) ----------------
def jain(counts):
    c = np.asarray(counts, float)
    return float(c.sum() ** 2 / (len(c) * (c ** 2).sum() + 1e-12)) if c.sum() > 0 else 0.0


def run_metrics(df, cfg, duration_s, region_ids):
    """df columns: status, model_latency_ms, cold, ci, carbon_g, carbon_wall_g, chosen_region,
    fallback, n_feasible, t_rel.  Latency = net RTT + queue + service + cold-start (model-based,
    immune to local CPU contention); carbon = active-time model; legacy = wall-clock (queue incl.)."""
    n = len(df)
    ok = df[df["status"] == 200]
    k = len(ok)
    sla = cfg["sla_ms"]
    m = dict(n=n, ok_n=k, error_pct=100.0 * (1 - k / max(1, n)),
             throttled_pct=100.0 * (df["status"] == 429).mean() if n else np.nan,
             timeout_pct=100.0 * df["status"].isin([504, 599]).mean() if n else np.nan,
             goodput_rps=k / duration_s)
    nan = float("nan")
    if k:
        lat = ok["model_latency_ms"].to_numpy(float)
        cold = ok["cold"].astype(bool).to_numpy()
        viol = lat > sla
        m.update(lat_mean=lat.mean(), lat_p50=np.percentile(lat, 50), lat_p95=np.percentile(lat, 95),
                 lat_p99=np.percentile(lat, 99),
                 viol_warm_pct=100.0 * (viol & ~cold).sum() / k, viol_cold_pct=100.0 * (viol & cold).sum() / k,
                 cold_pct=100.0 * cold.mean(), sla_attain_all_pct=100.0 * (~viol).sum() / max(1, n),
                 carbon_total_g=ok["carbon_g"].sum(), carbon_mg_per_req=1000.0 * ok["carbon_g"].sum() / k,
                 carbon_wall_mg_per_req=1000.0 * ok["carbon_wall_g"].sum() / k, ci_chosen=ok["ci"].mean())
        fb = ok[ok["fallback"] == 1]
        m["carbon_fb_mg_per_req"] = 1000.0 * fb["carbon_g"].sum() / len(fb) if len(fb) else nan
    else:
        for key in ["lat_mean", "lat_p50", "lat_p95", "lat_p99", "viol_warm_pct", "viol_cold_pct", "cold_pct",
                    "sla_attain_all_pct", "carbon_total_g", "carbon_mg_per_req", "carbon_wall_mg_per_req",
                    "ci_chosen", "carbon_fb_mg_per_req"]:
            m[key] = nan
    cnt = df["chosen_region"].value_counts().reindex(region_ids, fill_value=0)
    m["jain"] = jain(cnt.values)
    m["fallback_pct"] = 100.0 * df["fallback"].mean() if n else nan
    m["n_feasible_mean"] = df["n_feasible"].mean() if n else nan
    share = (cnt / max(1, cnt.sum())).to_dict()
    hr = np.clip((24 * df["t_rel"].to_numpy(float) / duration_s).astype(int), 0, 23)
    ci_hour = [float(ok["ci"][hr[df["status"].to_numpy() == 200] == h].mean()) if k else nan for h in range(24)]
    return m, share, ci_hour
