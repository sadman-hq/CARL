#!/usr/bin/env python
"""Step 1: import the Electricity Maps hourly CSV files from data/carbon/ and build the aligned 7-day carbon traces.

The CSV files are NOT part of this repository (licence). See data/carbon/README.md for how to obtain them.
"""
import argparse
import _bootstrap
from carl.paths import Paths
from carl.config import write_runtime_files
from carl.carbon_data import prepare_traces

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--root", default=str(_bootstrap.ROOT), help="project root (default: repository root)")
ap.add_argument("--ci-column", default="lca", choices=["lca", "direct"], help="life-cycle (lca) or direct carbon intensity")
ap.add_argument("--days", type=int, default=7, help="length of the trace window in days")
ap.add_argument("--start-date", default=None, help="UTC midnight of the window, e.g. 2025-12-25 (default: latest window complete for all zones)")
ap.add_argument("--zone-override", nargs="*", default=[], metavar="RID=ZONE", help="e.g. R02=NO-NO2")
ap.add_argument("--allow-synthetic", action="store_true", help="DEBUGGING ONLY: synthetic traces for missing zones")
a = ap.parse_args()

paths = Paths(a.root).ensure()
write_runtime_files(paths)
override = dict(x.split("=", 1) for x in a.zone_override)
prepare_traces(paths, ci_column=a.ci_column, trace_days=a.days, start_date=a.start_date, zone_override=override,
               require_real=not a.allow_synthetic, allow_synthetic=a.allow_synthetic)
