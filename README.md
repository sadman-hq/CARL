# CARL: Sustainable Serverless Auto-Scaling Framework for Carbon-Intensity-Aware Cloud Routing

This repository contains the experimental testbed used to evaluate **CARL**, a
carbon-, latency-, and load-aware request router for multi-region serverless
deployments. It is implemented as a single, self-contained Jupyter/Colab
notebook (`traffic_CARL.ipynb`) that spins up a synthetic multi-region cloud
environment, runs a full factorial load-testing campaign against five routing
policies, and produces all tables, statistical tests, and figures reported in
the paper.

> **Reproducibility note:** everything below — the region simulators, the
> router, the load generator, and the analysis pipeline — is generated and
> executed *by the notebook itself*. There is no external infrastructure to
> provision; the whole experiment runs on a single machine (or a free Colab
> instance).

---

## 1. What this experiment measures

CARL routes each incoming request to one of *N* simulated cloud regions by
minimizing a weighted score over three normalized signals:

```
score(r) = W_LAT · latency_norm(r) + W_LOAD · load_norm(r) · burst_factor(r) + W_CARB · carbon_norm(r)
```

- `latency_norm` / `load_norm` — min-max normalized, live-polled health-check
  latency and in-flight-request load for each region.
- `carbon_norm` — min-max normalized static grid carbon intensity per region
  (gCO₂/kWh).
- `burst_factor` — only applied under bursty traffic; inflates the load term
  when a region's short-term load history is trending upward
  (`1 + β · max(0, Δload)`), so CARL reacts to incipient hotspots rather than
  only steady-state load.
- Regions whose polled latency exceeds an SLA threshold (`SLA_MS`, default
  120 ms) are excluded from selection entirely.

CARL is compared against four baselines:

| Mode | Policy |
|---|---|
| `round_robin` | Cycles through regions in fixed order |
| `latency` | Always picks the lowest-latency healthy region |
| `load` | Always picks the least-loaded region |
| `carbon_greedy` | Always picks the lowest-carbon-intensity region, ignoring latency/load |
| `carl` | Weighted multi-objective policy described above |

---

## 2. System architecture

The notebook programmatically writes and launches two FastAPI microservices
as local subprocesses (`localhost`, distinct ports), then drives synthetic
traffic against them with an async load generator.

```
                     ┌───────────────────────────┐
   load generator ─▶ │   carl_router  (:8000)     │
   (aiohttp, N reqs/s)│   POST /route             │
                     │   - polls /health,/load    │──┐
                     │     every 1s (bg thread)   │  │
                     │   - scores & picks a region│  │
                     └───────────────────────────┘  │
                                                       │ POST /invoke
                     ┌───────────────────────────┐  │
                     │  region_service R01..R16   │◀─┘
                     │  (:9001-:9016)             │
                     │  - simulates queueing delay │
                     │  - simulates cold starts     │
                     │  - simulates failures under  │
                     │    pressure                   │
                     │  - computes per-request      │
                     │    energy/carbon               │
                     └───────────────────────────┘
```

### 2.1 Region service (`region_service/app.py`)

Each of the 16 simulated regions models:

- **Queueing delay:** `delay_ms = BASE_DELAY_MS · (1 + 4·pressure) + noise + cold_penalty`,
  where `pressure = in-flight / MAX_INFLIGHT`.
- **Failure under pressure:** requests fail (HTTP 500) with probability
  `max(0, pressure − 0.75) · 0.6`, i.e. failures only appear once a region is
  more than 75% saturated.
- **Cold starts:** a region is "cold" if it hasn't been invoked for more than
  `KEEP_ALIVE_S` (default 60s); cold invocations add a uniform 300–900 ms
  penalty, mimicking serverless container cold-start behavior.
- **Energy/carbon model:** per-request energy is computed from an
  idle+dynamic power model, `P = P_IDLE + (P_MAX − P_IDLE) · pressure`,
  converted to kWh over the execution time, adjusted by a data-center
  PUE (1.5), and multiplied by the region's static grid carbon intensity
  (gCO₂/kWh) to yield `carbon_g` per request.

### 2.2 Router (`carl_router/app.py`)

- Polls every region's `/health` (for latency) and `/load` (for load score)
  once per second on a background thread, kept deliberately synchronous and
  off the asyncio event loop so it cannot block request handling.
- Exposes `POST /route`, which selects a region under the requested policy,
  forwards the request to that region's `/invoke` endpoint over a shared
  `aiohttp` client session (async, unbounded connection pool — required to
  sustain hundreds of req/s without serializing requests), and logs a full
  per-request record to `logs/requests.csv`.
- Logged fields include chosen region, status code, end-to-end latency,
  function execution time, cold-start flag, the region's latency/load/carbon
  snapshot at decision time, per-request carbon emissions, the bursty-mode
  burst factor, and a switch flag.

### 2.3 Region carbon/latency profile

16 regions are defined with a **deliberate mismatch** between proximity
(lowest base latency) and carbon intensity, so that no routing policy can
trivially optimize both objectives at once — the nearest region (`R11`, 15 ms
base delay) is also the second-dirtiest (380 gCO₂/kWh), forcing a genuine
latency–carbon trade-off:

| Region | Carbon (gCO₂/kWh) | Base delay (ms) | Max in-flight |
|---|---|---|---|
| R01 | 35  | 95  | 40 |
| R02 | 45  | 110 | 35 |
| R03 | 60  | 80  | 50 |
| R04 | 90  | 55  | 60 |
| R05 | 120 | 40  | 70 |
| R06 | 150 | 65  | 55 |
| R07 | 190 | 30  | 80 |
| R08 | 210 | 45  | 65 |
| R09 | 260 | 20  | 90 |
| R10 | 310 | 50  | 60 |
| R11 | 380 | 15  | 100 |
| R12 | 430 | 70  | 50 |
| R13 | 480 | 100 | 45 |
| R14 | 560 | 85  | 40 |
| R15 | 650 | 120 | 35 |
| R16 | 720 | 140 | 30 |

> **Carbon values are static, illustrative numbers loosely inspired by
> publicly reported regional grid-mix averages — they are *not* pulled from a
> live carbon-intensity API.** This is a synthetic benchmark for
> algorithmic comparison, not a claim about real-world deployment carbon
> savings.

---

## 3. Experiment design

A full factorial sweep over **traffic level × traffic pattern × routing
mode × repetition**:

- **Traffic levels:** `low` = 20 req/s, `mid` = 100 req/s, `high` = 300 req/s
- **Traffic patterns:**
  - `continuous` — constant target request rate
  - `bursty` — periodic spikes at 4× the base rate for 5s every 20s
- **Routing modes:** `round_robin`, `latency`, `load`, `carbon_greedy`, `carl`
- **Repetitions:** 3 per (level, pattern, mode) cell by default (paper-quality
  runs should use 5–10; see §5)
- **Duration:** 45s per run

Total runs in the default configuration: 3 levels × 2 patterns × 5 modes × 3
reps = **90 independent load-test runs**, each producing a full per-request
CSV log.

A separate **weight-sensitivity sweep** varies CARL's carbon weight
`W_CARB ∈ {0.0, 0.2, 0.3, 0.5, 0.7, 1.0}` (splitting the remainder evenly
between latency and load) at fixed mid-level continuous traffic, to trace out
the empirical carbon–latency Pareto frontier.

---

## 4. Metrics and analysis pipeline

Per run, and aggregated per (level, pattern, mode) cell across repetitions:

- **Average and P95 end-to-end latency (ms)**
- **Throughput (successful req/s)**
- **SLA violation rate** — % of successful requests exceeding `SLA_MS` (120 ms)
- **Error rate (%)**
- **Cold-start rate (%)**
- **Total carbon emissions (gCO₂)**
- **Average carbon intensity of chosen regions**
- **Region load-balance fairness** — Jain's fairness index over the
  distribution of chosen regions (1.0 = perfectly even, 1/N = fully
  concentrated)
- **Carbon reduction vs. round-robin (%)** — computed per (level, pattern)
  cell relative to the `round_robin` baseline

**Statistical significance:** for every (level, pattern) cell, Welch's
two-sample t-test compares CARL against each baseline on both average latency
and total carbon, with results (t-statistic, p-value, significance at
α = 0.05) written to `analysis/significance_tests.csv`.

**Outputs written to `analysis/`:**

| File | Contents |
|---|---|
| `all_requests_raw.csv` | Concatenated per-request logs across all runs |
| `metrics_per_rep.csv` | Per-repetition aggregated metrics |
| `metrics_summary.csv` | Mean/std metrics per (level, pattern, mode) — canonical results table |
| `significance_tests.csv` | CARL vs. baseline t-tests per scenario |
| `weight_sensitivity_pareto.csv` | Carbon/latency trade-off across `W_CARB` sweep |
| `plots/*.png` | Bar charts (one per metric × traffic pattern) and the carbon–latency Pareto frontier plot |

---

## 5. Reproducing the experiment

### Requirements

- Python 3.9+
- Packages: `fastapi`, `uvicorn`, `requests`, `pandas`, `numpy`, `matplotlib`,
  `scipy`, `psutil`, `aiohttp`

Install with:

```bash
pip install fastapi uvicorn requests pandas numpy matplotlib scipy psutil aiohttp
```

### Running

The experiment is designed to be run top-to-bottom as a Jupyter/Colab
notebook:

```bash
jupyter notebook traffic_CARL.ipynb
```

Execution order matters — cells must be run sequentially:

1. **Setup** — install dependencies; create working directories
   (`region_service/`, `carl_router/`, `analysis/`, `logs/`, `runs/`).
2. **Service generation** — writes `region_service/app.py` and
   `carl_router/app.py` to disk from embedded source strings.
3. **Cleanup** — kills any stale processes on the ports this notebook uses
   (`8000`, `9001`–`9016`).
4. **Launch** — starts 16 region services and the router as subprocesses and
   waits for all `/health` checks to pass.
5. **Smoke test** — verifies a handful of regions and a single `/route` call
   work before committing to a full run.
6. **Load-test harness** — defines the async request generator
   (`run_load`) and per-scenario driver (`run_scenario`).
7. **Full experiment sweep** — executes the full factorial campaign
   (§3). This is the long-running cell; expect roughly
   `3 levels × 2 patterns × 3 reps × 5 modes × 45s ≈ 40 minutes` at the
   default settings.
8. **Load & aggregate results** — concatenates all `runs/**/requests.csv`
   into `analysis/all_requests_raw.csv`.
9. **Per-repetition metrics** — computes the metric table in §4.
10. **Summary table + carbon-reduction-vs-baseline** —
    `analysis/metrics_summary.csv`.
11. **Significance testing** — `analysis/significance_tests.csv`.
12. **Weight-sensitivity sweep** — restarts the router with different
    `W_CARB` values and re-runs a short load test for each, producing the
    Pareto-frontier data.
13. **Plotting** — generates all figures in `analysis/plots/`.
14. **Archive results** — zips the `analysis/` directory
    (Colab-specific download step; safe to skip or replace with a plain
    `shutil.make_archive` call when running locally).

### Configuration knobs

All key experiment parameters are environment variables or top-level
notebook constants and can be adjusted without touching the service code:

| Parameter | Default | Where |
|---|---|---|
| `SLA_MS` | 120 | Router — latency SLA threshold |
| `W_LAT`, `W_LOAD`, `W_CARB` | 0.4, 0.3, 0.3 | Router — CARL score weights (must sum to 1.0) |
| `BETA_BURST` | 0.8 | Router — burst-factor sensitivity |
| `LOAD_HISTORY_LEN` | 5 | Router — samples kept for burst-slope detection |
| `KEEP_ALIVE_S` | 60 | Region — cold-start warm window |
| `LEVELS_RPS` | `{low: 20, mid: 100, high: 300}` | Notebook Cell 8 |
| `DURATION_S` | 45 | Notebook Cell 8 — seconds per run |
| `REPS` | 3 | Notebook Cell 8 — increase to 5–10 for publication-grade statistics |

### Scaling up for a paper-quality run

The default `REPS = 3` and `DURATION_S = 45` are chosen to keep total runtime
reasonable on a free Colab instance. For the statistics reported in the
paper, increase `REPS` to 5–10 (Cell 8) and re-run the full sweep; the rest
of the pipeline (aggregation, significance testing, plotting) requires no
changes.

---

## 6. Repository structure (generated at runtime)

```
.
├── traffic_CARL.ipynb        # the entire experiment (this is the only source file)
├── region_service/app.py     # generated: simulated serverless region
├── carl_router/app.py        # generated: CARL router + baselines
├── logs/                     # per-region and router stdout/stderr, live requests.csv
├── runs/
│   ├── <level>_<pattern>/<mode>/rep<k>/requests.csv
│   └── sensitivity/wcarb_<w>/rep<k>/requests.csv
└── analysis/
    ├── all_requests_raw.csv
    ├── metrics_per_rep.csv
    ├── metrics_summary.csv        # canonical results table
    ├── significance_tests.csv
    ├── weight_sensitivity_pareto.csv
    └── plots/*.png
```

---

## 7. Limitations

- All region topology, latency, load-capacity, and carbon-intensity values
  are **synthetic** and constructed to guarantee a latency–carbon
  trade-off exists; they are not measurements of any real cloud provider or
  live grid carbon feed.
- Traffic patterns (`continuous`, `bursty`) are parametric approximations of
  real-world workloads, not traces replayed from production systems.
- All experiments run on a single host, so results reflect the routing
  *policies'* relative behavior under identical simulated conditions rather
  than absolute performance numbers transferable to production
  infrastructure.

---
  year    = {2026}
}
```
