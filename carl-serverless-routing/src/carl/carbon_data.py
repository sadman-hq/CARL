"""Import of the Electricity Maps hourly CSV files -> aligned, time-varying carbon traces.

DATA (not included in this repository; Electricity Maps academic licence forbids redistribution):
  1. Download the HOURLY CSV of each zone listed in carl.config.REGION_PROFILES (same calendar year).
  2. Put the files (or ZIP archives of them) into data/carbon/. The zone is read from the CSV column
     "Zone Id", so file names do not matter.
The importer stops with an error if a zone is missing (synthetic traces can only be enabled for debugging).
"""
import json, hashlib, shutil, zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from carl.config import REGION_PROFILES


def unpack_archives(carbon_dir):
    for z in Path(carbon_dir).glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            for m in zf.namelist():
                if m.lower().endswith(".csv"):
                    (Path(carbon_dir) / Path(m).name).write_bytes(zf.read(m))


def read_portal_csv(path, zone_candidates, ci_column="lca"):
    """Returns (zone_id, hourly raw series (UTC), 'estimated' flag series, name of the carbon column)."""
    df = pd.read_csv(path)
    low = {c: c.lower() for c in df.columns}
    tcol = next((c for c, l in low.items() if "datetime" in l), df.columns[0])
    zcol = next((c for c, l in low.items() if l.replace("_", " ").strip() == "zone id"), None)
    cand = [c for c, l in low.items() if "carbon" in l and "intensity" in l]
    keys = ("lca", "life cycle") if ci_column == "lca" else ("direct",)
    vcol = next((c for c in cand if any(k in low[c] for k in keys)), None)
    if vcol is None:
        raise ValueError(f"{Path(path).name}: no '{ci_column}' carbon-intensity column found in {list(df.columns)}")
    zone = str(df[zcol].dropna().iloc[0]) if zcol and df[zcol].notna().any() else \
        next((z for z in zone_candidates if z.lower() in Path(path).name.lower()), None)
    idx = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    ok = ~idx.isna()
    s = pd.Series(pd.to_numeric(df[vcol], errors="coerce").values, index=idx)[ok.values]
    ecol = next((c for c, l in low.items() if "estimated" in l and "method" not in l), None)
    e = pd.Series(df[ecol].astype(str).str.lower().eq("true").astype(float).values, index=idx)[ok.values] if ecol else s * 0.0
    return zone, s, e, vcol


TZ_OFFSET = [1, 1, 1, 2, 1, 0, -8, -5, 1, 1, 9, 9, 8, 10, 1, 2]


def synthetic_trace(mean, k, hours, seed=0):
    """DEBUGGING ONLY: synthetic diurnal trace (never use for reported results)."""
    rng = np.random.default_rng(seed + k)
    h = np.arange(hours)
    lt = (h + TZ_OFFSET[k]) % 24
    rel = 0.10 + 0.20 * min(1.0, mean / 400.0)
    solar = np.exp(-((lt - 13) / 3.5) ** 2)
    evening = np.exp(-((lt - 19.5) / 2.0) ** 2)
    ar = np.zeros(hours)
    for i in range(1, hours):
        ar[i] = 0.85 * ar[i - 1] + rng.normal(0, 0.03)
    tr = np.clip(mean * (1 - rel * solar + 0.5 * rel * evening + ar), 0.2 * mean, None)
    return tr * (mean / tr.mean())


def prepare_traces(paths, ci_column="lca", trace_days=7, start_date=None, zone_override=None,
                   require_real=True, allow_synthetic=False, verbose=True):
    """Builds data/carbon/traces.json (+ trace_meta.json, ATTRIBUTION.txt, analysis/carbon_data_qa.csv).
    Returns (traces, trace_src, trace_meta, trace_hash)."""
    zone_override = zone_override or {}
    hours = 24 * trace_days
    carbon_dir = Path(paths.carbon)
    carbon_dir.mkdir(parents=True, exist_ok=True)
    unpack_archives(carbon_dir)
    need = {rid: zone_override.get(rid, zone) for rid, zone, *_ in REGION_PROFILES}

    raw, est, files, used_col = {}, {}, {}, {}
    for f in sorted(carbon_dir.glob("*.csv")):
        try:
            zone, s, e, vcol = read_portal_csv(f, set(need.values()), ci_column)
        except Exception as ex:
            print("skipped", f.name, "->", ex)
            continue
        if zone is None:
            print("skipped", f.name, "-> zone could not be identified")
            continue
        raw.setdefault(zone, []).append(s)
        est.setdefault(zone, []).append(e)
        files.setdefault(zone, []).append(f.name)
        used_col[zone] = vcol
    hourly, hourly_raw, est_h = {}, {}, {}
    for z in raw:
        s = pd.concat(raw[z])
        s = s[~s.index.duplicated()].sort_index().resample("1h").mean()
        e = pd.concat(est[z])
        e = e[~e.index.duplicated()].sort_index().resample("1h").mean()
        hourly_raw[z], hourly[z], est_h[z] = s, s.interpolate(limit=6), e      # gaps <= 6 h are interpolated

    missing = sorted({z for z in need.values() if z not in hourly})
    if missing and require_real and not allow_synthetic:
        raise RuntimeError(f"No real data for zones {missing}. Zones found in {carbon_dir}: {sorted(hourly)}.\n"
                           f"Download them, or swap the zone via zone_override (e.g. {{'R02': '<available zone>'}}).")
    real_zones = [z for z in set(need.values()) if z in hourly]
    if not real_zones:
        raise RuntimeError(f"No carbon CSV files found in {carbon_dir} (see the instructions in data/carbon/README.md).")

    def window(s, start):
        return s.reindex(pd.date_range(start, periods=hours, freq="1h", tz="UTC"))

    if start_date:
        start = pd.Timestamp(start_date, tz="UTC")
        bad = [z for z in real_zones if window(hourly[z], start).isna().any()]
        if bad:
            raise RuntimeError(f"The week starting {start_date} is incomplete for zones {bad}.")
    else:
        common_idx = hourly[real_zones[0]].dropna().index
        for z in real_zones[1:]:
            common_idx = common_idx.intersection(hourly[z].dropna().index)
        start = next((t for t in sorted(common_idx[common_idx.hour == 0], reverse=True)
                      if all(not window(hourly[z], t).isna().any() for z in real_zones)), None)
        if start is None:
            raise RuntimeError("No 7-day window is complete for all zones; download the same year for every zone "
                               "or use zone_override / start_date.")

    traces, trace_src, qa = {}, {}, []
    for k, (rid, _zone, _rtt, _w, _s, fb) in enumerate(REGION_PROFILES):
        z = need[rid]
        if z in hourly:
            w = window(hourly[z], start)
            traces[rid] = w.values.astype(float).tolist()
            trace_src[rid] = "real"
            raw_w = window(hourly_raw[z], start)
            qa.append(dict(region=rid, zone=z, files=";".join(files[z]), column=used_col[z], week_start=str(start.date()),
                           src_first=str(hourly_raw[z].dropna().index.min().date()),
                           src_last=str(hourly_raw[z].dropna().index.max().date()),
                           missing_pct=100 * raw_w.isna().mean(), estimated_pct=100 * float(window(est_h[z], start).mean()),
                           ci_mean=w.mean(), ci_min=w.min(), ci_max=w.max()))
        else:
            traces[rid] = synthetic_trace(fb, k, hours).tolist()
            trace_src[rid] = "SYNTHETIC"
    paths.analysis.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(qa).to_csv(Path(paths.analysis) / "carbon_data_qa.csv", index=False)

    meta = dict(provider="Electricity Maps Data Portal (academic licence)", week_start=str(start.date()), days=trace_days,
                ci_column_pref=ci_column, zones=need)
    (carbon_dir / "traces.json").write_text(json.dumps(traces))
    (carbon_dir / "trace_meta.json").write_text(json.dumps(meta, indent=1))
    (carbon_dir / "ATTRIBUTION.txt").write_text(
        "Carbon-intensity data: Electricity Maps (https://app.electricitymaps.com). Obtained under the academic licence; "
        "redistribution of the data is not permitted. Cite Electricity Maps in any publication.\n")
    trace_hash = hashlib.sha256(json.dumps(traces, sort_keys=True).encode()).hexdigest()[:10]
    archive_stale_runs(paths, trace_hash)

    if verbose:
        n_syn = sum(v == "SYNTHETIC" for v in trace_src.values())
        print(f"Carbon dataset: week starting {start.date()} (UTC), {hours} hourly values/region | "
              f"real={len(traces) - n_syn} synthetic={n_syn} | trace id {trace_hash}")
        if n_syn:
            print("WARNING: SYNTHETIC traces for:", [r for r, v in trace_src.items() if v == "SYNTHETIC"],
                  "- do not report as real data")
        for q in qa:
            print(f"  {q['region']} {q['zone']:<12s} mean {q['ci_mean']:6.0f}  min {q['ci_min']:6.0f}  max {q['ci_max']:6.0f} gCO2/kWh | "
                  f"missing {q['missing_pct']:.1f}% estimated {q['estimated_pct']:.0f}% | {q['files']}")
    return traces, trace_src, meta, trace_hash


def archive_stale_runs(paths, trace_hash):
    """Never mix results produced with different traces: archive previous results automatically."""
    runs = Path(paths.runs)
    runs.mkdir(parents=True, exist_ok=True)
    hp = runs / ".trace_hash"
    old = hp.read_text() if hp.exists() else None
    if old and old != trace_hash and any(runs.glob("*/*")):
        shutil.move(str(runs), str(Path(paths.root) / f"runs_stale_{old}"))
        print(f"Traces changed -> previous results archived to runs_stale_{old}/")
    for d in [paths.live, paths.des]:
        Path(d).mkdir(parents=True, exist_ok=True)
    hp.write_text(trace_hash)


def load_traces(paths):
    f = Path(paths.carbon) / "traces.json"
    if not f.exists():
        raise FileNotFoundError(f"{f} not found: run scripts/01_prepare_carbon_data.py first")
    return json.loads(f.read_text())


def load_trace_meta(paths):
    f = Path(paths.carbon) / "trace_meta.json"
    return json.loads(f.read_text()) if f.exists() else {"week_start": "n/a"}
