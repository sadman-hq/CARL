"""Region profiles, default configuration and experiment design (identical to the values used in the paper)."""
import json

# (id, Electricity Maps zone id, RTT ms, workers (instances), hardware slowdown factor, fallback mean CI gCO2/kWh)
# The zone id must match the "Zone Id" column of the downloaded CSV files (or be swapped via --zone-override).
# The fallback mean is only used for the synthetic debugging traces (never for results).
REGION_PROFILES = [
    ("R01", "SE-SE3",       95,  8, 1.00,  35),
    ("R02", "CH",          110,  7, 1.10,  45),
    ("R03", "FR",           80, 10, 0.90,  60),
    ("R04", "FI",           55, 12, 1.00,  90),
    ("R05", "ES",           40, 14, 1.20, 120),
    ("R06", "GB",           65, 11, 1.00, 150),
    ("R07", "US-CAL-CISO",  30, 16, 0.80, 190),
    ("R08", "US-NY-NYIS",   45, 13, 1.10, 210),
    ("R09", "IT-NO",        20, 18, 0.90, 260),
    ("R10", "DE",           50, 12, 1.00, 310),
    ("R11", "KR",           15, 20, 0.85, 380),
    ("R12", "JP-TK",        70, 10, 1.20, 430),
    ("R13", "SG",          100,  9, 1.10, 480),
    ("R14", "AU-NSW",       85,  8, 1.30, 560),
    ("R15", "PL",          120,  7, 1.00, 650),
    ("R16", "ZA",          140,  6, 1.20, 720),
]
REGION_IDS = [p[0] for p in REGION_PROFILES]

DEFAULT_CFG = dict(
    # ---- CARL / routing ----
    sla_ms=250.0,
    w_lat=0.4, w_load=0.3, w_carb=0.3,
    beta=0.8, burst_window=5, burst_estimator="ols",   # "ols" (regression slope) or "diff" (last - first)
    burst_trigger="pattern",                           # "pattern" (bursty traffic only) or "always"
    poll_s=0.5, u_cap=0.7, eps_lat=0.5, adapt_u0=0.5, seed=42,
    # ---- region model ----
    svc_base_ms=60.0, svc_prior_ms=60.0, alpha_contention=0.5, svc_jitter=0.10, rtt_jitter=0.08,
    queue_limit_factor=4, queue_timeout_s=2.0, router_timeout_s=3.0,
    keep_alive_s=60.0, cold_ms_min=300.0, cold_ms_max=900.0,
    # ---- power / carbon ----
    p_idle_w=50.0, p_max_w=300.0, pue=1.5,
    # ---- one trial replays this many hours of the carbon trace ----
    trace_hours_per_trial=24,
)

# ---- experiment design ----
LEVELS_RPS = {"low": 20, "mid": 100, "high": 300}
PATTERNS = ["continuous", "bursty"]
LIVE_MODES = ["round_robin", "latency", "load", "carbon_greedy", "carbon_spill", "eps_latency", "carl_static", "carl"]
SIM_MODES = LIVE_MODES + ["carl_adaptive"]
BURST = (4.0, 20.0, 5.0)        # rate multiplier, period (s), burst length (s)
TRACE_DAYS = 7
DURATION_S = 60                 # live trial length
REPS = 5                        # live repetitions
SETTLE_S = 2.5
DES_DURATION_S = 180            # simulator trial length
DES_REPS = 10                   # simulator repetitions
SENS_DURATION_S = 120

# a live run is flagged invalid if any limit is exceeded
HARNESS_LIMITS = dict(cpu_mean=90.0, lag_p99=25.0, loop_lag_p99=25.0)

# scenario id (fixes the random seeds): identical to the notebook numbering
SCEN_ID = {f"{l}_{p}": i for i, (l, p) in enumerate((l, p) for l in LEVELS_RPS for p in PATTERNS)}


def write_runtime_files(paths, cfg=None):
    """data/config.json and data/regions.json are read by the simulator worker processes."""
    cfg = cfg or DEFAULT_CFG
    paths.data.mkdir(parents=True, exist_ok=True)
    (paths.data / "config.json").write_text(json.dumps(cfg, indent=1))
    (paths.data / "regions.json").write_text(json.dumps(
        [dict(rid=r, zone=z, rtt=rtt, workers=w, speed=s, fallback_ci=ci) for r, z, rtt, w, s, ci in REGION_PROFILES], indent=1))
