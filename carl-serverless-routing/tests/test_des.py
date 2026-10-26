import json

import pytest

from carl import des
from carl.config import REGION_PROFILES, DEFAULT_CFG


@pytest.fixture()
def root(tmp_path, monkeypatch):
    (tmp_path / "data" / "carbon").mkdir(parents=True)
    (tmp_path / "data" / "config.json").write_text(json.dumps(DEFAULT_CFG))
    (tmp_path / "data" / "regions.json").write_text(json.dumps(
        [dict(rid=r, zone=z, rtt=rtt, workers=w, speed=s, fallback_ci=ci) for r, z, rtt, w, s, ci in REGION_PROFILES]))
    traces = {r: [float(ci)] * 168 for r, _z, _rtt, _w, _s, ci in REGION_PROFILES}     # constant, non-synthetic test traces
    (tmp_path / "data" / "carbon" / "traces.json").write_text(json.dumps(traces))
    monkeypatch.setenv("CARL_ROOT", str(tmp_path))
    des._CACHE.clear()
    return tmp_path


def _spec(mode, level="low", pattern="continuous", rps=20, n=16, duration=15):
    return dict(tag="t", level=level, pattern=pattern, mode=mode, rep=0, rps=rps, duration=duration, n_regions=n, scen_id=0)


def test_simulation_is_deterministic(root):
    a, b = des.des_job(_spec("carl")), des.des_job(_spec("carl"))
    for k in ["n", "lat_mean", "carbon_mg_per_req", "error_pct"]:
        assert a[k] == b[k]


def test_low_load_has_no_errors_and_carl_beats_round_robin(root):
    rr, carl = des.des_job(_spec("round_robin")), des.des_job(_spec("carl"))
    assert rr["error_pct"] == 0 and carl["error_pct"] == 0
    assert carl["lat_mean"] < rr["lat_mean"]
    assert carl["carbon_mg_per_req"] < rr["carbon_mg_per_req"]
    assert abs(rr["jain"] - 1.0) < 0.02                       # round-robin spreads evenly


def test_carbon_greedy_collapses_under_high_load(root):
    r = des.des_job(_spec("carbon_greedy", level="high", rps=300, duration=20))
    assert r["jain"] < 0.07                                   # everything goes to one region
    assert r["error_pct"] > 20


def test_more_regions_are_supported(root):
    r = des.des_job(_spec("carl", n=32, rps=20))
    assert r["error_pct"] == 0 and len(r["share"]) == 32
