"""Live testbed harness: starts the 16 region processes and the router (CPU-pinned), runs trials with the
open-loop load generator, measures the harness itself, and executes the resumable experiment sweep.
Linux only (uses os.sched_setaffinity and pkill)."""
import json, os, random, subprocess, sys, time
from pathlib import Path

import numpy as np, pandas as pd, requests
from tqdm.auto import tqdm

from carl.config import (REGION_PROFILES, DEFAULT_CFG, LEVELS_RPS, PATTERNS, LIVE_MODES, BURST, DURATION_S, REPS,
                         SETTLE_S, TRACE_DAYS, SCEN_ID)

_SRC_DIR = str(Path(__file__).resolve().parents[1])          # makes `python -m carl.xxx` work without installation


def _pin(cores):
    return lambda: os.sched_setaffinity(0, cores)


class LiveTestbed:
    def __init__(self, paths, cfg=None, base_port=9001, router_port=8000):
        self.paths = paths.ensure()
        self.cfg = dict(cfg or DEFAULT_CFG)
        self.base_port, self.router_port = base_port, router_port
        self.router = f"http://127.0.0.1:{router_port}"
        ncpu = os.cpu_count() or 1
        if ncpu >= 4:
            self.router_cores, self.loadgen_cores, self.region_cores = {0}, {1}, set(range(2, ncpu))
        else:                       # e.g. free Colab (2 vCPU): router and load generator on separate cores
            self.router_cores, self.loadgen_cores, self.region_cores = {0}, {ncpu - 1}, set(range(ncpu))
        self.region_urls, self.workers_map, self.procs = {}, {}, []

    # ------------------------------------------------------------------ processes
    def _env(self, **extra):
        env = {**os.environ, **{k: str(v) for k, v in extra.items()}}
        env["PYTHONPATH"] = _SRC_DIR + os.pathsep + env.get("PYTHONPATH", "")
        return env

    def kill(self):
        for pat in ["carl.region_service", "carl.router", "carl.loadgen"]:
            subprocess.run(["pkill", "-f", pat], stderr=subprocess.DEVNULL)
        time.sleep(1.5)

    @staticmethod
    def _wait_healthy(url, timeout=30):
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                if requests.get(f"{url}/health", timeout=1).status_code == 200:
                    return True
            except Exception:
                pass
            time.sleep(0.3)
        return False

    def launch(self):
        self.kill()
        traces = self.paths.carbon / "traces.json"
        if not traces.exists():
            raise FileNotFoundError(f"{traces} not found: run scripts/01_prepare_carbon_data.py first")
        self.region_urls, self.workers_map, self.procs = {}, {}, []
        for i, (rid, zone, rtt, workers, speed, _) in enumerate(REGION_PROFILES):
            port = self.base_port + i
            env = self._env(CFG_JSON=json.dumps(self.cfg), REGION=rid, PORT=port, RTT_MS=rtt, WORKERS=workers,
                            SPEED=speed, TRACE_PATH=traces)
            self.procs.append(subprocess.Popen([sys.executable, "-m", "carl.region_service"], env=env,
                                               preexec_fn=_pin(self.region_cores),
                                               stdout=open(self.paths.logs / f"{rid}.out", "w"), stderr=subprocess.STDOUT))
            self.region_urls[rid] = f"http://127.0.0.1:{port}"
            self.workers_map[rid] = workers
        bad = [r for r, u in self.region_urls.items() if not self._wait_healthy(u)]
        if bad:
            raise RuntimeError(f"Regions failed to start: {bad} (see {self.paths.logs}/<id>.out)")
        env = self._env(CFG_JSON=json.dumps(self.cfg), REGIONS_JSON=json.dumps(self.region_urls),
                        WORKERS_JSON=json.dumps(self.workers_map), PORT=self.router_port)
        self.procs.append(subprocess.Popen([sys.executable, "-m", "carl.router"], env=env,
                                           preexec_fn=_pin(self.router_cores),
                                           stdout=open(self.paths.logs / "router.out", "w"), stderr=subprocess.STDOUT))
        if not self._wait_healthy(self.router):
            raise RuntimeError(f"Router failed to start (see {self.paths.logs}/router.out)")
        print(f"{len(self.region_urls)} regions + router up | cores: router={self.router_cores} "
              f"loadgen={self.loadgen_cores} regions={self.region_cores}")

    # ------------------------------------------------------------------ one trial
    def reset_all(self, t0, warp, offset_h, seed, cold=True):
        for i, u in enumerate(self.region_urls.values()):
            requests.post(f"{u}/reset", json=dict(t0=t0, warp=warp, offset_h=offset_h, seed=seed + i, cold=cold), timeout=5)
        requests.post(f"{self.router}/reset_state", timeout=5)

    def run_loadgen(self, mode, pattern, base_rps, duration_s, seed, t0_epoch, out_csv, out_meta, burst=BURST):
        cfgj = dict(mode=mode, pattern=pattern, base_rps=base_rps, duration_s=duration_s, seed=seed, t0_epoch=t0_epoch,
                    out_csv=str(out_csv), out_meta=str(out_meta), route_url=f"{self.router}/route", burst=list(burst))
        r = subprocess.run([sys.executable, "-m", "carl.loadgen", json.dumps(cfgj)], env=self._env(),
                           preexec_fn=_pin(self.loadgen_cores), capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-1500:])

    def run_one(self, level, pattern, mode, rep, duration_s=DURATION_S, settle_s=SETTLE_S):
        scen = f"{level}_{pattern}"
        out = self.paths.live / scen / mode / f"rep{rep}"
        if (out / "meta.json").exists():
            return "skip"
        out.mkdir(parents=True, exist_ok=True)
        seed = 1000 * rep + SCEN_ID[scen]              # common random numbers: every mode sees the same arrivals
        t0 = time.time() + settle_s
        self.reset_all(t0, self.cfg["trace_hours_per_trial"] * 3600.0 / duration_s, 24.0 * (rep % TRACE_DAYS), seed)
        time.sleep(settle_s)
        self.run_loadgen(mode, pattern, LEVELS_RPS[level], duration_s, seed, t0, out / "requests.csv", out / "meta.json")
        return "ok"

    # ------------------------------------------------------------------ checks and sweeps
    def smoke_test(self, duration_s=4, rps=20):
        self.reset_all(time.time() + 1.0, 24 * 3600 / 5.0, 0.0, 1)
        time.sleep(2.5)
        tmp = self.paths.logs
        for m in ["round_robin", "latency", "load", "carbon_greedy", "carbon_spill", "eps_latency",
                  "carl_static", "carl", "carl_adaptive"]:
            self.run_loadgen(m, "continuous", rps, duration_s, 1, time.time() + 0.5, tmp / "smoke.csv", tmp / "smoke.json")
            d = pd.read_csv(tmp / "smoke.csv")
            print(f"{m:14s} n={len(d):3d} ok={int((d.status == 200).sum()):3d} regions_used={d.chosen_region.nunique():2d} "
                  f"model_lat_mean={d.model_latency_ms.mean():6.1f}ms  e2e_mean={d.e2e_ms.mean():6.1f}ms")

    def calibration(self, levels=None, patterns=None, duration_s=15):
        """Harness cost with NO region work ('noop' route) + accuracy of the emulated RTT timers on an idle system."""
        levels = levels or list(LEVELS_RPS)
        patterns = patterns or PATTERNS
        tmp = self.paths.logs
        cal = []
        for level in levels:
            for pattern in patterns:
                self.reset_all(time.time() + 1, 1.0, 0.0, 7)
                time.sleep(1.5)
                self.run_loadgen("noop", pattern, LEVELS_RPS[level], duration_s, 7, time.time() + 0.5,
                                 tmp / "cal.csv", tmp / "cal.json")
                d, m = pd.read_csv(tmp / "cal.csv"), json.load(open(tmp / "cal.json"))
                cal.append(dict(level=level, pattern=pattern, noop_e2e_p50=d.e2e_ms.quantile(.5),
                                noop_e2e_p99=d.e2e_ms.quantile(.99), noop_lag_p99=m["lag_p99"],
                                noop_cpu_mean=m["cpu_mean"], noop_loop_lag_p99=m.get("loop_lag_p99", np.nan)))
        self.paths.analysis.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(cal).to_csv(self.paths.analysis / "harness_calibration_noop.csv", index=False)
        fid = []
        for rid, u in self.region_urls.items():
            xs = []
            for _ in range(15):
                t = time.perf_counter()
                requests.get(f"{u}/probe", timeout=3)
                xs.append((time.perf_counter() - t) * 1000)
            nominal = [p[2] for p in REGION_PROFILES if p[0] == rid][0]
            fid.append(dict(region=rid, nominal_ms=nominal, measured_median_ms=float(np.median(xs))))
        pd.DataFrame(fid).to_csv(self.paths.analysis / "rtt_timer_fidelity.csv", index=False)
        print("Calibration saved to analysis/.")

    def sweep(self, levels=None, patterns=None, modes=None, reps=REPS, duration_s=DURATION_S, settle_s=SETTLE_S):
        """Resumable: finished runs (meta.json present) are skipped, so an interrupted session can simply be restarted."""
        levels = levels or list(LEVELS_RPS)
        patterns = patterns or PATTERNS
        modes = modes or LIVE_MODES
        rng_order = random.Random(123)
        jobs = []
        for rep in range(reps):
            for level in levels:
                for pattern in patterns:
                    ms = list(modes)
                    rng_order.shuffle(ms)                       # randomised mode order removes drift bias
                    jobs += [(level, pattern, m, rep) for m in ms]
        todo = [j for j in jobs if not (self.paths.live / f"{j[0]}_{j[1]}" / j[2] / f"rep{j[3]}" / "meta.json").exists()]
        print(f"{len(jobs)} runs total, {len(todo)} remaining, ~{len(todo) * (duration_s + settle_s + 4) / 3600:.1f} h")
        for j in tqdm(todo):
            self.run_one(*j, duration_s=duration_s, settle_s=settle_s)
        print("Live runs complete.")
