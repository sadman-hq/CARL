#!/usr/bin/env python
"""Step 4: figures A1-A12 and the statistics table A8 from the runs in runs/ (live and/or simulator).
Use --source des for the paper figures of the bursty scenarios (the live testbed is invalid there)."""
import argparse
import _bootstrap
from carl.paths import Paths
from carl.config import DES_DURATION_S, DES_REPS
from carl.analysis import Analysis

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--root", default=str(_bootstrap.ROOT))
ap.add_argument("--source", default="des", choices=["live", "des"], help="data behind the main figures")
ap.add_argument("--exclude-invalid", action="store_true", help="drop live runs flagged as harness-invalid")
ap.add_argument("--only", nargs="*", default=None, help="subset, e.g. a1 a4 a9a a12")
ap.add_argument("--des-duration", type=float, default=DES_DURATION_S)
ap.add_argument("--des-reps", type=int, default=DES_REPS)
ap.add_argument("--quick", action="store_true", help="match the settings of `03_run_simulator.py --quick`")
a = ap.parse_args()

kw = {}
if a.quick:
    a.des_duration, a.des_reps = 20, 2
    kw = dict(sens_kwargs=dict(duration=10, rep_w=1, rep_other=1), scale_kwargs=dict(duration=10, reps=1))
Analysis(Paths(a.root), a.source, a.exclude_invalid, a.des_duration, a.des_reps, **kw).run_all(a.only)
