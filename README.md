# CARL: Carbon-Aware Request Routing for Multi-Region Serverless Platforms

Code for the paper on **CARL**, a routing layer that sends each serverless request to a region whose estimated latency meets an SLA threshold, and scores the feasible regions by a weighted combination of normalized latency, load, and carbon intensity (with a load-trend penalty under bursty traffic and a round-robin fallback).

The repository contains a live single-host testbed (16 region services, a router, and an open-loop load generator), a discrete-event simulator that uses the same policy code and region model, and the analysis (figures, statistics, paper tables).

> **Data:** the Electricity Maps carbon-intensity data are **not included** (academic licence, no redistribution). See [`data/carbon/README.md`](data/carbon/README.md) for how to obtain the files used in the paper.

## Setup

Python 3.10+. The live testbed needs Linux (or Google Colab); the simulator and analysis run anywhere.

```bash
pip install -r requirements.txt
python -m pytest -q          # optional unit tests, no data needed
```

## Reproduce

```bash
python scripts/01_prepare_carbon_data.py      # import the carbon CSVs from data/carbon/
python scripts/02_run_live.py                 # live testbed (optional, Linux, ~4-5 h, resumable)
python scripts/03_run_simulator.py            # simulator, sensitivity, scalability (~1-2 h, resumable)
python scripts/04_analyze.py --source des     # figures A1-A12 and statistics
python scripts/05_paper_tables.py             # paper tables from results/metrics_per_run.csv (no data needed)
```

Add `--quick` to steps 3 and 4 for a fast functional check (not the paper setting), or `--smoke-only` to step 2 for a short testbed check.

## What is in the repository

| Path | Content |
|---|---|
| `src/carl/` | policies, region service, router, load generator, testbed harness, simulator, carbon import, analysis |
| `scripts/` | the five steps above |
| `tests/` | unit tests (policies, energy model, metrics, simulator, data import) |
| `results/` | per-run metrics of all live and simulated runs, and the decision-time benchmark |

## Notes

- The live testbed is valid only where the host is not saturated (each run is flagged); the high-intensity bursty scenario and part of the mid-intensity bursty scenario are taken from the simulator.
- Results come from an emulated environment and a single seven-day carbon trace (25-31 December 2025, UTC); they are not measurements of a real cloud platform.
- The `carbon_spill` and `eps_latency` baselines follow the principle of carbon-aware schedulers but are not re-implementations of published systems.

## Citation and licence
