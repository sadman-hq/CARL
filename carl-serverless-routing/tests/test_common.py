import random

import numpy as np
import pandas as pd

from carl import common
from carl.config import DEFAULT_CFG, REGION_IDS


def _snap(**over):
    base = lambda rtt, ci, util=0.1: dict(rtt=rtt, util=util, queue=0, workers=10, svc=60.0, ci=ci, ci_mean=ci, hist=[0.1] * 5)
    s = {"A": base(30, 400), "B": base(60, 50), "C": base(90, 200)}
    s.update(over)
    return s


def test_arrivals_are_reproducible_and_have_the_right_rate():
    a = common.gen_arrivals("continuous", 100, 60, seed=1)
    assert a == common.gen_arrivals("continuous", 100, 60, seed=1)
    assert 5400 < len(a) < 6600                                   # ~ 100 req/s * 60 s
    b = common.gen_arrivals("bursty", 100, 60, seed=1)
    assert len(b) > 1.5 * len(a)                                  # bursts raise the mean rate
    assert all(0 <= t < 60 for t in b)


def test_energy_formula():
    cfg = DEFAULT_CFG
    # 300 W at full utilisation for 1 h, PUE 1.5, 100 gCO2/kWh -> 0.3 kWh * 1.5 * 100 = 45 g
    assert abs(common.active_carbon_g(1.0, 3600.0, 100.0, cfg) - 45.0) < 1e-9
    # idle power only: 50 W
    assert abs(common.active_carbon_g(0.0, 3600.0, 100.0, cfg) - 0.05 * 1.5 * 100) < 1e-9


def test_round_robin_cycles_through_all_regions():
    rr, rng = {"i": 0}, random.Random(0)
    got = [common.choose("round_robin", _snap(), DEFAULT_CFG, rr, rng)[0] for _ in range(6)]
    assert got == ["A", "B", "C", "A", "B", "C"]


def test_simple_baselines():
    rr, rng = {"i": 0}, random.Random(0)
    assert common.choose("latency", _snap(), DEFAULT_CFG, rr, rng)[0] == "A"
    assert common.choose("carbon_greedy", _snap(), DEFAULT_CFG, rr, rng)[0] == "B"
    s = _snap(C=dict(rtt=90, util=0.0, queue=0, workers=10, svc=60.0, ci=200, ci_mean=200, hist=[0.0] * 5))
    assert common.choose("load", s, DEFAULT_CFG, rr, rng)[0] == "C"


def test_carl_excludes_regions_that_violate_the_sla():
    rr, rng = {"i": 0}, random.Random(0)
    s = _snap(B=dict(rtt=400, util=0.1, queue=0, workers=10, svc=60.0, ci=1, ci_mean=1, hist=[0.1] * 5))   # greenest but too slow
    for mode in ["carl", "carl_static", "carl_adaptive", "carbon_spill", "eps_latency"]:
        region, info = common.choose(mode, s, DEFAULT_CFG, rr, rng)
        assert region != "B", mode
        assert info["fallback"] == 0


def test_carl_falls_back_to_round_robin_if_no_region_is_feasible():
    rr, rng = {"i": 0}, random.Random(0)
    slow = {k: dict(v, rtt=900) for k, v in _snap().items()}
    region, info = common.choose("carl", slow, DEFAULT_CFG, rr, rng)
    assert info["fallback"] == 1 and info["n_feasible"] == 0 and region in slow


def test_carl_prefers_green_region_when_latency_and_load_are_equal():
    rr, rng = {"i": 0}, random.Random(0)
    s = {k: dict(rtt=50, util=0.1, queue=0, workers=10, svc=60.0, ci=ci, ci_mean=ci, hist=[0.1] * 5)
         for k, ci in [("A", 500), ("B", 50), ("C", 300)]}
    assert common.choose("carl", s, DEFAULT_CFG, rr, rng)[0] == "B"


def test_burst_factor_penalises_rising_load_only():
    cfg = dict(DEFAULT_CFG)
    assert common.burst_factor([0.1, 0.1, 0.1, 0.1, 0.1], cfg) == 1.0
    assert common.burst_factor([0.5, 0.4, 0.3, 0.2, 0.1], cfg) == 1.0            # falling load: no penalty
    rising = common.burst_factor([0.1, 0.2, 0.3, 0.4, 0.5], cfg)
    assert abs(rising - (1 + 0.8 * 0.4)) < 1e-9                                  # slope 0.1/sample * (W-1) = 0.4
    cfg["burst_estimator"] = "diff"
    assert abs(common.burst_factor([0.1, 0.2, 0.3, 0.4, 0.5], cfg) - (1 + 0.8 * 0.4)) < 1e-9


def test_jain_index_bounds():
    assert abs(common.jain([5] * 16) - 1.0) < 1e-12
    assert abs(common.jain([80] + [0] * 15) - 1 / 16) < 1e-12


def test_run_metrics_basic():
    df = pd.DataFrame(dict(status=[200, 200, 200, 429], model_latency_ms=[100.0, 200.0, 400.0, 10.0],
                           cold=[False, False, True, False], ci=[100.0, 100.0, 100.0, np.nan],
                           carbon_g=[1e-3] * 3 + [0.0], carbon_wall_g=[2e-3] * 3 + [0.0],
                           chosen_region=["R01", "R01", "R02", "R02"], fallback=[0, 0, 0, 1], n_feasible=[16] * 4,
                           t_rel=[1.0, 2.0, 3.0, 4.0]))
    m, share, ci_hour = common.run_metrics(df, DEFAULT_CFG, 10.0, REGION_IDS)
    assert m["n"] == 4 and m["ok_n"] == 3 and abs(m["error_pct"] - 25.0) < 1e-9
    assert abs(m["lat_mean"] - 700 / 3) < 1e-9
    assert abs(m["viol_cold_pct"] - 100 / 3) < 1e-9 and m["viol_warm_pct"] == 0.0     # only the 400 ms cold start exceeds 250 ms
    assert abs(m["carbon_mg_per_req"] - 1.0) < 1e-9
    assert abs(share["R01"] - 0.5) < 1e-9 and len(ci_hour) == 24
