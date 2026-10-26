#!/usr/bin/env python
"""Step 3: discrete-event simulator: main grid (6 scenarios x 9 policies x 10 repetitions x 180 s),
sensitivity studies (weights, burst penalty, SLA threshold, cold-start model), scalability and decision-time benchmark.
Parallel and resumable (results are cached in runs/des/)."""
import argparse
import _bootstrap
from carl.paths import Paths
from carl.config import write_runtime_files, LEVELS_RPS, PATTERNS, DES_DURATION_S, DES_REPS, SENS_DURATION_S
from carl.experiments import specs_main, specs_sensitivity, specs_scale
from carl.simulation import run_des, decision_time_benchmark

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--root", default=str(_bootstrap.ROOT))
ap.add_argument("--only", nargs="*", default=["main", "sens", "scale", "bench"], choices=["main", "sens", "scale", "bench"])
ap.add_argument("--workers", type=int, default=None, help="parallel processes (default: all CPU cores)")
ap.add_argument("--levels", nargs="*", default=list(LEVELS_RPS))
ap.add_argument("--patterns", nargs="*", default=PATTERNS)
ap.add_argument("--duration", type=float, default=DES_DURATION_S, help="trial length of the main grid (s)")
ap.add_argument("--reps", type=int, default=DES_REPS, help="repetitions of the main grid")
ap.add_argument("--sens-duration", type=float, default=SENS_DURATION_S)
ap.add_argument("--quick", action="store_true", help="tiny settings for a fast functional check (NOT the paper setting)")
a = ap.parse_args()

if a.quick:
    a.duration, a.reps, a.sens_duration = 20, 2, 10
paths = Paths(a.root).ensure()
write_runtime_files(paths)
if "main" in a.only:
    run_des(paths, specs_main(a.levels, a.patterns, None, a.reps, a.duration), "main", a.workers)
if "sens" in a.only:
    kw = dict(duration=a.sens_duration, rep_w=1, rep_other=1) if a.quick else dict(duration=a.sens_duration)
    for name, specs in specs_sensitivity(**kw).items():
        run_des(paths, specs, name, a.workers)
if "scale" in a.only:
    kw = dict(duration=10, reps=1) if a.quick else {}
    run_des(paths, specs_scale(**kw), "scale", a.workers)
if "bench" in a.only:
    print(decision_time_benchmark(paths, number=200 if a.quick else 2000).pivot(index="n_regions", columns="mode", values="decision_us").round(1))
