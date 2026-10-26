"""Definition of all simulator experiments (main grid, sensitivity studies, scalability).
The same specification lists are used to run the experiments and to read them back for the analysis."""
from carl.config import (LEVELS_RPS, PATTERNS, SIM_MODES, SCEN_ID, DES_DURATION_S, DES_REPS, SENS_DURATION_S)


def specs_main(levels=None, patterns=None, modes=None, reps=DES_REPS, duration=DES_DURATION_S):
    levels = levels or list(LEVELS_RPS)
    patterns = patterns or PATTERNS
    modes = modes or SIM_MODES
    return [dict(tag="main", level=l, pattern=p, mode=m, rep=r, rps=LEVELS_RPS[l], duration=duration,
                 n_regions=16, scen_id=SCEN_ID[f"{l}_{p}"])
            for l in levels for p in patterns for m in modes for r in range(reps)]


def _mk(tag, level, pattern, mode, rep, label, over, duration):
    return dict(tag=tag, level=level, pattern=pattern, mode=mode, rep=rep, rps=LEVELS_RPS[level], duration=duration,
                n_regions=16, scen_id=SCEN_ID[f"{level}_{pattern}"], label=label, cfg_over=over)


def specs_sensitivity(duration=SENS_DURATION_S, rep_w=3, rep_other=5):
    """Four sensitivity studies: weights, burst penalty, SLA threshold, cold-start model."""
    simplex = [(round(a / 10, 1), round(b / 10, 1), round(1 - a / 10 - b / 10, 1))
               for a in range(11) for b in range(11 - a)]
    sp_w = [_mk("sens_w", l, p, "carl", r, f"{wl}/{wu}/{wc}", dict(w_lat=wl, w_load=wu, w_carb=wc), duration)
            for (l, p) in [("mid", "continuous"), ("high", "bursty")] for (wl, wu, wc) in simplex for r in range(rep_w)]
    sp_b = [_mk("sens_beta", l, "bursty", "carl", r, f"{est}|W{w}|b{b}",
                dict(beta=b, burst_estimator=est, burst_window=w), duration)
            for l in ["mid", "high"] for r in range(rep_other)
            for (est, w, b) in [(e, 5, bb) for e in ["ols", "diff"] for bb in [0.0, 0.2, 0.4, 0.8, 1.6, 3.2]]
            + [("ols", 3, 0.8), ("ols", 9, 0.8)]]
    sp_s = [_mk("sens_sla", l, "continuous", m, r, f"{sla:g}", dict(sla_ms=float(sla)), duration)
            for l in ["mid", "high"] for m in ["carl", "carbon_spill"] for sla in [150, 200, 250, 350, 500]
            for r in range(rep_other)]
    sp_k = [_mk("sens_cold", l, "continuous", m, r, f"ka{ka:g}|cold x{cs:g}",
                dict(keep_alive_s=float(ka), cold_ms_min=300.0 * cs, cold_ms_max=900.0 * cs), duration)
            for l in ["low", "mid"] for m in ["round_robin", "carl", "carbon_spill"] for r in range(rep_other)
            for (ka, cs) in [(10, 1), (60, 1), (300, 1), (60, 0.5), (60, 2)]]
    return dict(sens_w=sp_w, sens_beta=sp_b, sens_sla=sp_s, sens_cold=sp_k)


def specs_scale(duration=60, reps=3, ns=(16, 32, 64)):
    return [dict(tag="scale", level="mid", pattern="continuous", mode=m, rep=r, rps=LEVELS_RPS["mid"], duration=duration,
                 n_regions=n, scen_id=SCEN_ID["mid_continuous"])
            for n in ns for m in ["round_robin", "latency", "load", "carbon_spill", "carl"] for r in range(reps)]
