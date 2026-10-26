#!/usr/bin/env python
"""Regenerates the paper's tables (LaTeX rows) and headline numbers from results/metrics_per_run.csv.
Needs neither the carbon data nor any experiment run."""
import argparse
import _bootstrap
from carl import paper_tables as pt

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--metrics", default=str(_bootstrap.ROOT / "results" / "metrics_per_run.csv"))
ap.add_argument("--out", default=str(_bootstrap.ROOT / "results" / "paper_tables"))
a = ap.parse_args()

d, paired, welch = pt.write_all(a.metrics, a.out)
print("== CARL vs round-robin (simulator) and non-dominated policies (error rate < 1%)")
print(pt.headline(d).round(2).to_string(index=False))
print("\n== live vs simulator (valid live runs)")
import json
print(json.dumps(json.load(open(f"{a.out}/validation.json")), indent=1)[:1500])
print("\n== wall-clock / active-time carbon ratio per policy (simulator)")
print(pt.energy_accounting(d).round(3))
print("\n== CARL-static / CARL-adaptive relative to CARL")
print(pt.adaptive_vs_carl(d).round(2).to_string(index=False))
print(f"\n== live Welch tests (valid scenarios, Holm): significant {int((welch.p_holm < .05).sum())} of {len(welch)}")
print(f"\nLaTeX table rows written to {a.out}")
