"""Parallel, resumable execution of simulator experiments + per-decision cost benchmark."""
import json, os, pickle, random, timeit
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from carl import common, des


def _key(spec):
    return json.dumps(spec, sort_keys=True)


def _cache_file(paths, name):
    return Path(paths.des) / f"{name}.pkl"


def run_des(paths, specs, name, workers=None):
    """Runs all specs that are not cached yet (results are cached in runs/des/<name>.pkl, keyed by the full spec)
    and returns a DataFrame with one row per spec (in the order of `specs`)."""
    os.environ["CARL_ROOT"] = str(paths.root)
    p = _cache_file(paths, name)
    done = pickle.load(open(p, "rb")) if p.exists() else {}
    todo = [s for s in specs if _key(s) not in done]
    print(f"[{name}] {len(specs)} runs, {len(todo)} to do")
    if todo:
        workers = workers or os.cpu_count() or 1
        if workers == 1:
            for s in tqdm(todo):
                done[_key(s)] = des.des_job(s)
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as ex:
                for s, r in zip(todo, tqdm(ex.map(des.des_job, todo, chunksize=2), total=len(todo))):
                    done[_key(s)] = r
        pickle.dump(done, open(p, "wb"))
    return pd.DataFrame([done[_key(s)] for s in specs])


def read_des(paths, specs, name):
    """Reads cached results for `specs` (no computation); raises if some are missing."""
    p = _cache_file(paths, name)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found (run scripts/03_run_simulator.py first)")
    done = pickle.load(open(p, "rb"))
    missing = [s for s in specs if _key(s) not in done]
    if missing:
        raise KeyError(f"{len(missing)} of {len(specs)} runs missing in {p}")
    return pd.DataFrame([done[_key(s)] for s in specs])


def decision_time_benchmark(paths, ns=(16, 32, 64, 128, 256, 512), number=2000):
    """Wall-clock cost of one routing decision (measured) for increasing numbers of candidate regions."""
    cfgd = json.load(open(Path(paths.data) / "config.json"))
    rows = []
    for n in ns:
        g = random.Random(n)
        snap = {f"R{i}": dict(rtt=g.uniform(15, 140), util=g.random(), queue=g.randint(0, 5), workers=10, svc=60.0,
                              ci=g.uniform(30, 700), ci_mean=g.uniform(30, 700), hist=[g.random() * 0.5 for _ in range(5)])
                for i in range(n)}
        for mode in ["round_robin", "load", "carbon_spill", "carl"]:
            rr, rg = {"i": 0}, random.Random(1)
            t = timeit.timeit(lambda: common.choose(mode, snap, cfgd, rr, rg, "bursty"), number=number) / number
            rows.append(dict(n_regions=n, mode=mode, decision_us=t * 1e6))
    df = pd.DataFrame(rows)
    Path(paths.analysis).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(paths.analysis) / "decision_time.csv", index=False)
    return df
