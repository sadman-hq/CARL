import numpy as np
import pandas as pd
import pytest

from carl.carbon_data import prepare_traces
from carl.config import REGION_PROFILES
from carl.paths import Paths


def _write_fake_csvs(carbon_dir, skip=()):
    """Files in the column layout of the Electricity Maps hourly datasets. The numbers are FAKE test data."""
    rng = np.random.default_rng(0)
    for rid, zone, *_rest, fb in REGION_PROFILES:
        if zone in skip:
            continue
        idx = pd.date_range("2025-01-01", "2025-12-31 23:00", freq="1h", tz="UTC")
        v = fb * (1 + 0.2 * np.sin(np.arange(len(idx)) * 2 * np.pi / 24)) + rng.normal(0, 5, len(idx))
        df = pd.DataFrame({"Datetime (UTC)": idx.strftime("%Y-%m-%d %H:%M:%S"), "Zone Id": zone,
                           "Carbon Intensity gCO₂eq/kWh (direct)": v * 0.9,
                           "Carbon Intensity gCO₂eq/kWh (life cycle assessment)": v,
                           "Data Estimated": rng.random(len(idx)) < 0.05})
        df.to_csv(carbon_dir / f"snapshots_{zone}-2025-hourly.csv", index=False)


def test_prepare_traces_builds_one_common_week(tmp_path):
    p = Paths(tmp_path).ensure()
    _write_fake_csvs(p.carbon)
    traces, src, meta, h = prepare_traces(p, verbose=False)
    assert set(traces) == {r[0] for r in REGION_PROFILES}
    assert all(len(t) == 168 for t in traces.values()) and set(src.values()) == {"real"}
    assert meta["week_start"] == "2025-12-25"                    # latest UTC-midnight week that fits in the file
    assert (p.carbon / "traces.json").exists() and (p.analysis / "carbon_data_qa.csv").exists()
    # the life-cycle column (not the direct one) was used: the mean of R01 matches the fake life-cycle values
    assert abs(np.mean(traces["R01"]) - 35) < 10


def test_missing_zone_stops_with_a_clear_error(tmp_path):
    p = Paths(tmp_path).ensure()
    _write_fake_csvs(p.carbon, skip=("ZA",))
    with pytest.raises(RuntimeError, match="ZA"):
        prepare_traces(p, verbose=False)


def test_zone_override_replaces_a_missing_zone(tmp_path):
    p = Paths(tmp_path).ensure()
    _write_fake_csvs(p.carbon, skip=("ZA",))
    traces, *_ = prepare_traces(p, zone_override={"R16": "PL"}, verbose=False)      # use the PL file for region R16
    assert len(traces["R16"]) == 168
