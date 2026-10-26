# Mapping of the original Colab notebook to this repository

| Notebook cells | Repository |
|---|---|
| 0-1 (installs, folders, region profiles, configuration) | `requirements.txt`, `src/carl/config.py`, `src/carl/paths.py` |
| 2 (Electricity Maps import, traces) | `src/carl/carbon_data.py`, `scripts/01_prepare_carbon_data.py` |
| 3 (shared module) | `src/carl/common.py` |
| 4 (region service) | `src/carl/region_service.py` |
| 5 (router) | `src/carl/router.py` |
| 6 (load generator) | `src/carl/loadgen.py` |
| 7-12 (launch, smoke test, calibration, live sweep, stop) | `src/carl/testbed.py`, `scripts/02_run_live.py` |
| 13 (discrete-event simulator) | `src/carl/des.py` |
| 14-16 (simulator runs, sensitivity, scalability, decision time) | `src/carl/experiments.py`, `src/carl/simulation.py`, `scripts/03_run_simulator.py` |
| 17-29 (per-run table, figures A1-A12, statistics A8) | `src/carl/analysis.py`, `scripts/04_analyze.py` |
| (analysis of the paper's tables) | `src/carl/paper_tables.py`, `scripts/05_paper_tables.py` |
| 30 (reproducibility bundle) | not needed: the repository is the release (raw carbon data excluded) |
