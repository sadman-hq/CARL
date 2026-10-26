#!/usr/bin/env python
"""Step 2 (optional, Linux): live single-host testbed = 16 region processes + router + open-loop load generator.

Full paper setting (default): 6 scenarios x 8 policies x 5 repetitions x 60 s  ~ 4-5 hours. Resumable.
"""
import argparse
import _bootstrap
from carl.paths import Paths
from carl.config import write_runtime_files, LEVELS_RPS, PATTERNS, LIVE_MODES, DURATION_S, REPS
from carl.testbed import LiveTestbed

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--root", default=str(_bootstrap.ROOT))
ap.add_argument("--levels", nargs="*", default=list(LEVELS_RPS), choices=list(LEVELS_RPS))
ap.add_argument("--patterns", nargs="*", default=PATTERNS, choices=PATTERNS)
ap.add_argument("--modes", nargs="*", default=LIVE_MODES)
ap.add_argument("--reps", type=int, default=REPS)
ap.add_argument("--duration", type=float, default=DURATION_S, help="seconds per trial")
ap.add_argument("--smoke-only", action="store_true", help="only start the testbed and run a 4 s smoke test of every policy")
ap.add_argument("--skip-calibration", action="store_true")
ap.add_argument("--calibration-duration", type=float, default=15, help="seconds per calibration run")
ap.add_argument("--keep-running", action="store_true", help="do not stop the processes at the end")
a = ap.parse_args()

paths = Paths(a.root).ensure()
write_runtime_files(paths)
tb = LiveTestbed(paths)
try:
    tb.launch()
    tb.smoke_test()
    if not a.smoke_only:
        if not a.skip_calibration:
            tb.calibration(a.levels, a.patterns, a.calibration_duration)
        tb.sweep(a.levels, a.patterns, a.modes, a.reps, a.duration)
finally:
    if not a.keep_running:
        tb.kill()
