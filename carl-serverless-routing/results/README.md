# Released results

| File | Content |
|---|---|
| `metrics_per_run.csv` | One row per run (240 live runs, 540 main simulator runs): per-run metrics computed by `carl.common.run_metrics` (latency, error rate, throughput, SLA violations, cold starts, carbon per request, Jain index, harness validity flags for the live runs). |
| `decision_time.csv` | Measured wall-clock cost of one routing decision (microseconds) for 16-512 candidate regions. |

`python scripts/05_paper_tables.py` regenerates the paper's table rows and headline numbers from `metrics_per_run.csv`
into `results/paper_tables/` without the carbon data and without re-running any experiment.

The files contain computed metrics only (no raw carbon-intensity series).
