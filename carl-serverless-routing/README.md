# CARL: carbon-aware request routing for multi-region serverless platforms

Code for the paper on **CARL**, a routing layer that sends every serverless request to a region whose estimated latency
meets an SLA threshold and that scores the feasible regions by a weighted combination of normalized latency, load and
carbon intensity (with a load-trend penalty under bursty traffic and a round-robin fallback).

The repository contains

* a **live single-host testbed**: 16 region services, the router and an open-loop load generator as separate processes,
* a **discrete-event simulator** that uses the *same* policy code and region model,
* the **analysis** (figures A1-A12, statistics) and a script that regenerates the paper's tables from the released per-run results.

> **Data.** The Electricity Maps carbon-intensity data are *not* included (academic licence, no redistribution).
> See [`data/carbon/README.md`](data/carbon/README.md) for how to obtain exactly the files used in the paper.

## Installation

Python >= 3.10. The live testbed needs **Linux** (it uses `os.sched_setaffinity` and `pkill`); Google Colab works.
The simulator and the analysis also run elsewhere.

```bash
git clone <this repository>
cd carl-serverless-routing
pip install -r requirements.txt        # or: pip install -e .
python -m pytest -q                    # optional: unit tests (no data needed)
```

## Reproducing the paper

| Step | Command | Needs | Time (paper setting) |
|---|---|---|---|
| 1. carbon traces | `python scripts/01_prepare_carbon_data.py` | the 16 CSV files in `data/carbon/` | seconds |
| 2. live testbed (optional) | `python scripts/02_run_live.py` | step 1, Linux | ~4-5 h (240 runs of 60 s, resumable) |
| 3. simulator | `python scripts/03_run_simulator.py` | step 1 | ~1-2 h on 2 cores (parallel, resumable) |
| 4. figures + statistics | `python scripts/04_analyze.py --source des` | steps 1-3 | minutes |
| 5. paper tables | `python scripts/05_paper_tables.py` | `results/metrics_per_run.csv` only | seconds |

Useful options: `02_run_live.py --smoke-only` (4 s smoke test of every policy), `--levels/--patterns/--modes/--reps/--duration`
(subsets), `03_run_simulator.py --quick` and `04_analyze.py --quick` (tiny functional check, *not* the paper setting).
Interrupted runs continue where they stopped (finished runs are skipped). If the carbon traces change, earlier results are
archived automatically to `runs_stale_<id>/` so that results from different traces are never mixed.

The live testbed is valid only where the host is not saturated; every live run is flagged (`harness_ok`) by CPU load,
scheduling lag and event-loop lag. In the paper, the high-intensity bursty scenario and part of the mid-intensity bursty
scenario are invalid and are taken from the simulator (`04_analyze.py --source des`).

### Google Colab

```python
!git clone <this repository> && cd carl-serverless-routing && pip -q install -r requirements.txt
# upload the 16 carbon CSV files to carl-serverless-routing/data/carbon/ (file browser), then:
!cd carl-serverless-routing && python scripts/01_prepare_carbon_data.py
!cd carl-serverless-routing && python scripts/03_run_simulator.py
!cd carl-serverless-routing && python scripts/04_analyze.py --source des
```

## Repository layout

```
src/carl/
  config.py          region profiles, default configuration, experiment design (all paper parameters)
  common.py          routing policies, arrival generator, energy/carbon model, per-run metrics  (shared core)
  region_service.py  region process: instance pool, cold starts, bounded queue, emulated RTT, telemetry
  router.py          CARL router: telemetry polling, policy selection, request forwarding
  loadgen.py         open-loop load generator with scheduling-lag and CPU telemetry
  testbed.py         harness: start/stop processes, trials, calibration, resumable sweep
  des.py             discrete-event simulator
  experiments.py     specifications of the simulator experiments (main grid, sensitivity, scalability)
  simulation.py      parallel, resumable execution; decision-time benchmark
  carbon_data.py     import of the Electricity Maps CSV files -> aligned 7-day traces
  analysis.py        per-run table, figures A1-A12, statistics table A8
  paper_tables.py    tables and headline numbers of the paper from metrics_per_run.csv
scripts/             01_prepare_carbon_data, 02_run_live, 03_run_simulator, 04_analyze, 05_paper_tables
tests/               unit tests (policies, energy model, metrics, simulator, data import)
results/             released per-run metrics and the decision-time benchmark
data/carbon/         (empty) put the downloaded CSV files here
```

## Policies (`carl.common.choose`)

`round_robin`, `latency`, `load`, `carbon_greedy` (classic baselines); `carbon_spill` and `eps_latency` (carbon-aware
baselines written for this study following the principle of carbon-prioritizing schedulers, *not* re-implementations of
published systems); `carl` (live hourly carbon), `carl_static` (time-invariant mean carbon), `carl_adaptive` (simulator only).

## Model summary

* Latency of a request = network RTT + queueing delay + service time + cold-start penalty (composed from the model, so it is
  not distorted by host contention); the client-side latency is recorded in addition to quantify the harness overhead.
* Carbon of a request = idle-plus-dynamic power at the region's utilization x **active execution time** (service + cold-start
  initialization, no queueing) x PUE x carbon intensity at the time of execution. The wall-clock variant (queueing included)
  is also stored (`carbon_wall_*`) for the ablation. Carbon is reported per *successful* request.
* Carbon traces are replayed time-compressed: one trial replays 24 hours of the selected week.

## Citation

```
[ADD paper reference]
```

When using the carbon data, cite Electricity Maps as required by its academic licence.

## License

MIT for the code (see `LICENSE`; add the author names). The carbon-intensity data are not covered by this licence.
