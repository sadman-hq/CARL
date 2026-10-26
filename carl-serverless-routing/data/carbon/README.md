# Carbon-intensity data (not included)

The experiments use hourly carbon-intensity data from **Electricity Maps** (https://app.electricitymaps.com).
The data are provided under an **academic licence that does not permit redistribution**, so they are *not* part of
this repository. Anyone who wants to reproduce the traces can obtain the same files as follows:

1. Create an Electricity Maps account with an institutional e-mail address and activate the academic access
   (Settings -> Access -> Academic). Cite Electricity Maps in any publication that uses the data.
2. Open the **Datasets** page of the application. For each of the 16 zones below choose granularity **Hourly**,
   year **2025**, and the version dated **10 February 2026**, and download the CSV file:

   `SE-SE3, CH, FR, FI, ES, GB, US-CAL-CISO, US-NY-NYIS, IT-NO, DE, KR, JP-TK, SG, AU-NSW, PL, ZA`

   (If a zone is not available to you, use `--zone-override R02=<other zone>` in step 1; the results will then differ.)
3. Put the 16 CSV files (or one ZIP archive of them) into this folder. File names do not matter, the zone is read
   from the `Zone Id` column inside each file.
4. Run `python scripts/01_prepare_carbon_data.py`.

The importer selects the latest 7-day window (UTC midnight to midnight) that is complete for all 16 zones, linearly
interpolates gaps of up to six hours, uses the life-cycle carbon-intensity column (`--ci-column direct` for the direct
column), and writes `traces.json`, `trace_meta.json` and a data-quality report (`analysis/carbon_data_qa.csv`).
The paper used the window 25-31 December 2025 (UTC). Because the data are updated over time, keep the version date fixed.

Everything in this folder except this README is ignored by git (see `.gitignore`).
