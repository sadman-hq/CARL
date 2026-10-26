"""Collects live and simulator runs into one per-run metrics table and produces all figures (A1-A12)
and the statistics table (A8) of the paper. Every analysis is either a figure with its values printed on it
or a table, never both."""
import json
from pathlib import Path

import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sps

from carl import common
from carl.config import (REGION_IDS, DEFAULT_CFG, LEVELS_RPS, PATTERNS, SIM_MODES, TRACE_DAYS, HARNESS_LIMITS,
                         DES_DURATION_S, DES_REPS)
from carl.experiments import specs_main, specs_sensitivity, specs_scale
from carl.simulation import read_des
from carl.carbon_data import load_traces, load_trace_meta

MODE_ORDER = ["round_robin", "latency", "load", "carbon_greedy", "carbon_spill", "eps_latency", "carl_static", "carl",
              "carl_adaptive"]
MODE_NAME = dict(round_robin="RR", latency="Latency", load="Load", carbon_greedy="C-greedy", carbon_spill="C-spill",
                 eps_latency="Eps-lat", carl_static="CARL-static", carl="CARL", carl_adaptive="CARL-adapt")
MODE_COLOR = dict(zip(MODE_ORDER, plt.cm.tab10.colors))
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .25, "font.size": 8})


def ci95(x):
    x = pd.Series(x).dropna()
    return float(sps.t.ppf(.975, len(x) - 1) * x.sem()) if len(x) > 1 else 0.0


def load_live(paths, cfg):
    """One row of per-run metrics for every finished live run (runs/live/<scenario>/<mode>/rep<k>/)."""
    rows = []
    for mp_ in sorted(Path(paths.live).glob("*/*/rep*/meta.json")):
        d = mp_.parent
        scen, mode, rep = d.parent.parent.name, d.parent.name, int(d.name[3:])
        level, pattern = scen.split("_")
        meta = json.load(open(mp_))
        df = pd.read_csv(d / "requests.csv")
        df["status"] = pd.to_numeric(df["status"], errors="coerce").fillna(598).astype(int)
        df["cold"] = df["cold"].astype(str).str.lower().eq("true")
        df["fallback"] = pd.to_numeric(df["fallback"], errors="coerce").fillna(0)
        df["n_feasible"] = pd.to_numeric(df["n_feasible"], errors="coerce").fillna(0)
        df["chosen_region"] = df["chosen_region"].astype(str)
        m, share, cih = common.run_metrics(df, cfg, meta["duration_s"], REGION_IDS)
        ok = df[df.status == 200]
        ovh = ok["e2e_ms"] - ok["model_latency_ms"]
        bad = (meta["cpu_mean"] > HARNESS_LIMITS["cpu_mean"] or meta["lag_p99"] > HARNESS_LIMITS["lag_p99"]
               or meta.get("loop_lag_p99", 0) > HARNESS_LIMITS["loop_lag_p99"])
        rows.append(dict(source="live", tag="main", level=level, pattern=pattern, mode=mode, rep=rep, **m, share=share,
                         ci_hour=cih, ovh_p50=ovh.median(), ovh_p95=ovh.quantile(.95), cpu_mean=meta["cpu_mean"],
                         lag_p99=meta["lag_p99"], loop_lag_p99=meta.get("loop_lag_p99", np.nan), harness_ok=not bad))
    return pd.DataFrame(rows)


class Analysis:
    def __init__(self, paths, source="live", exclude_invalid=False, des_duration=DES_DURATION_S, des_reps=DES_REPS,
                 levels=None, patterns=None, sens_kwargs=None, scale_kwargs=None):
        self.paths = paths
        paths.plots.mkdir(parents=True, exist_ok=True)
        paths.tables.mkdir(parents=True, exist_ok=True)
        self.source, self.exclude_invalid = source, exclude_invalid
        self.levels = levels or list(LEVELS_RPS)
        self.patterns = patterns or PATTERNS
        self.cfg = json.load(open(paths.data / "config.json")) if (paths.data / "config.json").exists() else DEFAULT_CFG
        try:
            self.traces = load_traces(paths)
            self.trace_meta = load_trace_meta(paths)
        except FileNotFoundError:
            self.traces, self.trace_meta = None, {"week_start": "n/a"}
        self.live = load_live(paths, self.cfg)
        try:
            self.des_main = read_des(paths, specs_main(self.levels, self.patterns, None, des_reps, des_duration), "main")
        except (FileNotFoundError, KeyError) as e:
            print("simulator main grid not available:", e)
            self.des_main = pd.DataFrame()
        self.sens = {}
        for name, specs in specs_sensitivity(**(sens_kwargs or {})).items():
            try:
                self.sens[name] = read_des(paths, specs, name)
            except (FileNotFoundError, KeyError):
                pass
        try:
            self.scale = read_des(paths, specs_scale(**(scale_kwargs or {})), "scale")
        except (FileNotFoundError, KeyError):
            self.scale = pd.DataFrame()
        parts = [d for d in [self.live, self.des_main.assign(source="des") if len(self.des_main) else None]
                 if d is not None and len(d)]
        self.M = pd.concat(parts, ignore_index=True)
        self.M.drop(columns=["share", "ci_hour"]).to_csv(paths.analysis / "metrics_per_run.csv", index=False)
        print("runs:", self.M.groupby("source").size().to_dict(),
              "| live runs flagged harness-invalid:", int((~self.live.harness_ok).sum()) if len(self.live) else 0)
        self.LV = "mid" if "mid" in self.levels else self.levels[0]

    # ------------------------------------------------------------------ helpers
    def main_data(self, source=None):
        src = source or self.source
        d = self.M[(self.M.source == src) & (self.M.tag == "main")]
        if self.exclude_invalid and src == "live":
            d = d[d.harness_ok]
        return d

    @staticmethod
    def modes_in(d):
        return [m for m in MODE_ORDER if m in set(d["mode"])]

    def _save(self, fig, name):
        fig.savefig(self.paths.plots / f"{name}.png", dpi=200)
        plt.close(fig)
        print("saved", self.paths.plots / f"{name}.png")

    def facet_bars(self, metric, ylabel, title, fname, text_fn=None, stack=None, marker=None, logy=False, fmt="{:.1f}"):
        """Rows = traffic pattern, cols = load level, x = routing mode. Values are printed ON the figure."""
        d = self.main_data()
        modes = self.modes_in(d)
        fig, axs = plt.subplots(len(self.patterns), len(self.levels),
                                figsize=(4.6 * len(self.levels), 3.6 * len(self.patterns)), squeeze=False)
        for i, pat in enumerate(self.patterns):
            for j, lv in enumerate(self.levels):
                ax = axs[i][j]
                g = d[(d.level == lv) & (d.pattern == pat)]
                for k, mo in enumerate(modes):
                    gm = g[g["mode"] == mo]
                    if not len(gm):
                        continue
                    if stack:
                        b = 0
                        for s, a in zip(stack, [1.0, .55]):
                            v = gm[s].mean()
                            ax.bar(k, v, bottom=b, color=MODE_COLOR[mo], alpha=a, hatch="//" if s == stack[1] else None)
                            b += v
                        top, err = b, 0
                    else:
                        top, err = gm[metric].mean(), ci95(gm[metric])
                        ax.bar(k, top, yerr=err, color=MODE_COLOR[mo], capsize=2)
                        if marker:
                            ax.plot(k, gm[marker].mean(), "kD", ms=3)
                    txt = text_fn(gm) if text_fn else fmt.format(top)
                    ax.annotate(txt, (k, top + err), ha="center", va="bottom", fontsize=6, xytext=(0, 2),
                                textcoords="offset points")
                ax.set_xticks(range(len(modes)))
                ax.set_xticklabels([MODE_NAME[m] for m in modes], rotation=60, ha="right")
                ax.set_title(f"{lv} ({LEVELS_RPS[lv]} RPS) | {pat}", fontsize=8)
                if logy:
                    ax.set_yscale("log")
                if j == 0:
                    ax.set_ylabel(ylabel)
                ax.margins(y=.25)
        fig.suptitle(f"{title}  [{self.source}, mean +/- 95% CI over reps]", fontsize=10)
        fig.tight_layout()
        self._save(fig, fname)

    # ------------------------------------------------------------------ A1-A5 (main metrics)
    def a1_latency(self):
        self.facet_bars("lat_mean", "Latency (ms)", "A1  Latency: bar = mean, diamond = P95, label = mean / P95", "A1_latency",
                        text_fn=lambda g: f"{g.lat_mean.mean():.0f}/{g.lat_p95.mean():.0f}", marker="lat_p95")

    def a2_reliability(self):
        self.facet_bars("error_pct", "Error rate (%)",
                        "A2  Reliability: bar = error rate (429 + 504/599), label = error% | goodput req/s", "A2_reliability",
                        text_fn=lambda g: f"{g.error_pct.mean():.1f}%\n{g.goodput_rps.mean():.0f}/s")

    def a3_sla(self):
        sla = self.cfg["sla_ms"]
        self.facet_bars(None, "SLA violations (% of successful req.)",
                        f"A3  SLA ({sla:.0f} ms) violations: solid = warm, hatched = cold start; label = total | cold-start rate",
                        "A3_sla_decomposition", stack=("viol_warm_pct", "viol_cold_pct"),
                        text_fn=lambda g: f"{g.viol_warm_pct.mean() + g.viol_cold_pct.mean():.1f}\ncold {g.cold_pct.mean():.1f}%")

    def a4_carbon(self):
        d = self.main_data()
        rr = d[d["mode"] == "round_robin"].groupby(["level", "pattern"])["carbon_mg_per_req"].mean()

        def txt(g):
            lv, pt = g["level"].iloc[0], g["pattern"].iloc[0]
            v = g.carbon_mg_per_req.mean()
            e = g.error_pct.mean()
            return f"{v:.2f}\n{100 * (v / rr[(lv, pt)] - 1):+.0f}%" + (f"\nERR {e:.0f}%" if e >= 5 else "")

        # a policy that drops most requests looks 'green' per successful request: always read together with A2
        self.facet_bars("carbon_mg_per_req", "mgCO2 per successful request",
                        "A4  Carbon (active-time model): label = mg/request and change vs round-robin", "A4_carbon", text_fn=txt)

    def a5_energy_ablation(self):
        d = self.main_data()
        modes = [m for m in self.modes_in(d) if m != "round_robin"]
        fig, axs = plt.subplots(len(self.patterns), len(self.levels),
                                figsize=(4.6 * len(self.levels), 3.6 * len(self.patterns)), squeeze=False)
        for i, pat in enumerate(self.patterns):
            for j, lv in enumerate(self.levels):
                ax = axs[i][j]
                g = d[(d.level == lv) & (d.pattern == pat)]
                rr = g[g["mode"] == "round_robin"]
                if not len(rr):
                    continue
                for k, mo in enumerate(modes):
                    gm = g[g["mode"] == mo]
                    if not len(gm):
                        continue
                    for off, col, lab, c in [(-.2, "carbon_wall_mg_per_req", "wall-clock (queueing included)", "tab:gray"),
                                             (.2, "carbon_mg_per_req", "active-time (used in the paper)", "tab:green")]:
                        v = 100 * (gm[col].mean() / rr[col].mean() - 1)
                        ax.bar(k + off, v, .38, color=c, label=lab if k == 0 else None)
                        ax.annotate(f"{v:+.0f}", (k + off, v), ha="center", va="bottom" if v >= 0 else "top", fontsize=6)
                ax.axhline(0, color="k", lw=.7)
                ax.set_xticks(range(len(modes)))
                ax.set_xticklabels([MODE_NAME[m] for m in modes], rotation=60, ha="right")
                ax.set_title(f"{lv} | {pat}", fontsize=8)
                ax.margins(y=.2)
                if j == 0:
                    ax.set_ylabel("carbon vs round-robin (%)")
                if i == 0 and j == 0:
                    ax.legend(fontsize=6)
        fig.suptitle(f"A5  Energy-model ablation: negative = greener than round-robin  [{self.source}]", fontsize=10)
        fig.tight_layout()
        self._save(fig, "A5_energy_model_ablation")

    # ------------------------------------------------------------------ A6-A7 (needs the carbon traces)
    def a6_distribution(self):
        if self.traces is None:
            print("A6 skipped: carbon traces not available")
            return
        order = sorted(self.traces, key=lambda r: np.mean(self.traces[r]))
        d = self.main_data()
        modes = self.modes_in(d)
        fig, axs = plt.subplots(1, len(self.patterns), figsize=(8.5 * len(self.patterns), 0.55 * len(modes) + 2.2),
                                squeeze=False)
        for ax, pat in zip(axs[0], self.patterns):
            g = d[(d.level == self.LV) & (d.pattern == pat)]
            mat = np.array([[np.mean([s.get(r, 0) for s in g[g["mode"] == mo]["share"]]) * 100 for r in order] for mo in modes])
            ax.imshow(mat, aspect="auto", cmap="viridis_r", vmin=0, vmax=max(30, mat.max()))
            for a in range(mat.shape[0]):
                for b in range(mat.shape[1]):
                    if mat[a, b] >= 1:
                        ax.text(b, a, f"{mat[a, b]:.0f}", ha="center", va="center", fontsize=6,
                                color="w" if mat[a, b] > .55 * mat.max() else "k")
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels([f"{r}\n{np.mean(self.traces[r]):.0f}" for r in order], fontsize=6)
            ax.set_yticks(range(len(modes)))
            ax.set_yticklabels([f"{MODE_NAME[m]} (Jain {g[g['mode'] == m]['jain'].mean():.2f})" for m in modes])
            ax.set_title(f"{self.LV} load | {pat}   (x: region and its mean gCO2/kWh; cell = % of requests)", fontsize=8)
            ax.grid(False)
        fig.suptitle(f"A6  Request distribution across regions  [{self.source}]", fontsize=10)
        fig.tight_layout()
        self._save(fig, "A6_region_distribution")

    def a7_carbon_tracking(self):
        if self.traces is None:
            print("A7 skipped: carbon traces not available")
            return
        d = self.main_data()
        fleet = lambda rep: np.array([np.mean([self.traces[r][(24 * (rep % TRACE_DAYS) + h) % len(self.traces[r])]
                                               for r in self.traces]) for h in range(24)])
        sel = [m for m in ["round_robin", "carbon_spill", "carl_static", "carl"] if m in set(d["mode"])]
        fig, axs = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
        for ax, (lv, pat) in zip(axs, [(self.LV, "continuous"), (self.levels[-1], "bursty")]):
            for mo in sel:
                g = d[(d.level == lv) & (d.pattern == pat) & (d["mode"] == mo)]
                if not len(g):
                    continue
                R = np.array([np.array(r.ci_hour) / fleet(r.rep) for r in g.itertuples()])
                mu = np.nanmean(R, axis=0)
                ax.plot(range(24), mu, marker="o", ms=2.5, color=MODE_COLOR[mo],
                        label=f"{MODE_NAME[mo]} (mean {np.nanmean(mu):.2f})")
            ax.axhline(1, color="k", lw=.7, ls="--")
            ax.set_xlabel(f"hour of the replayed 24 h carbon trace (week starting {self.trace_meta['week_start']} UTC)")
            ax.set_title(f"{lv} | {pat}", fontsize=8)
            ax.legend(fontsize=6)
        axs[0].set_ylabel("CI of chosen region / fleet-mean CI  (<1 = greener than average)")
        fig.suptitle(f"A7  Carbon tracking over time: legend gives the day-average ratio  [{self.source}]", fontsize=10)
        fig.tight_layout()
        self._save(fig, "A7_carbon_tracking")

    # ------------------------------------------------------------------ A8 (table: inferential statistics only)
    def a8_statistics(self):
        d = self.main_data()
        mets = dict(lat_mean="Latency", carbon_mg_per_req="Carbon/req", error_pct="Error rate")
        rows = []
        for (lv, pat), g in d.groupby(["level", "pattern"]):
            c = g[g["mode"] == "carl"]
            for bm in [m for m in self.modes_in(d) if m != "carl"]:
                b = g[g["mode"] == bm]
                if len(c) < 2 or len(b) < 2:
                    continue
                r = dict(level=lv, pattern=pat, baseline=MODE_NAME[bm])
                for k in mets:
                    x, y = c[k].dropna().to_numpy(), b[k].dropna().to_numpy()
                    if len(x) < 2 or len(y) < 2:
                        r[k + "_p"] = np.nan
                        r[k + "_d"] = np.nan
                        continue
                    r[k + "_p"] = sps.ttest_ind(x, y, equal_var=False).pvalue
                    sp = np.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / (len(x) + len(y) - 2))
                    r[k + "_d"] = (x.mean() - y.mean()) / sp if sp > 1e-12 else 0.0
                rows.append(r)
        T = pd.DataFrame(rows)
        if not len(T):
            print("A8 skipped: need >= 2 repetitions per configuration")
            return None
        for k in mets:                                          # Holm step-down within each metric family
            p = T[k + "_p"].to_numpy(float)
            idx = np.argsort(np.where(np.isnan(p), 2, p))
            adj = np.full(len(p), np.nan)
            run_max = 0.0
            n_valid = int((~np.isnan(p)).sum())
            for rank, i in enumerate(idx[:n_valid]):
                run_max = max(run_max, min(1.0, (n_valid - rank) * p[i]))
                adj[i] = run_max
            T[k + "_p"] = adj
        fmt_p = lambda v: "n/a" if np.isnan(v) else ("<0.001" if v < 1e-3 else f"{v:.3f}")
        out = T[["level", "pattern", "baseline"]].copy()
        for k, nm in mets.items():
            out[f"{nm}: d"] = T[k + "_d"].round(2)
            out[f"{nm}: p(Holm)"] = T[k + "_p"].map(fmt_p)
        out.to_csv(self.paths.analysis / "A8_significance_tests.csv", index=False)
        print("A8: CARL vs each baseline [" + self.source + "]. Cohen's d < 0 means CARL has the LOWER value. n/a = zero variance.")
        print(out.to_string(index=False))
        return out

    # ------------------------------------------------------------------ A9 (sensitivity, simulator)
    @staticmethod
    def _pareto_mask(P):
        keep = np.ones(len(P), bool)
        for i in range(len(P)):
            keep[i] = not any((P[j] <= P[i]).all() and (P[j] < P[i]).any() for j in range(len(P)) if j != i)
        return keep

    def a9a_weights(self):
        if "sens_w" not in self.sens:
            print("A9a skipped: weight sweep not available")
            return
        W = self.sens["sens_w"]
        fig, axs = plt.subplots(1, 2, figsize=(13, 5))
        for ax, (lv, pat) in zip(axs, [("mid", "continuous"), ("high", "bursty")]):
            g = W[(W.level == lv) & (W.pattern == pat)].groupby("label")[["carbon_mg_per_req", "lat_mean", "error_pct"]].mean().dropna()
            g = g[g.error_pct < 5]
            if not len(g):
                continue
            P = g[["carbon_mg_per_req", "lat_mean"]].to_numpy()
            pm = self._pareto_mask(P)
            ax.scatter(P[:, 0], P[:, 1], c="lightgray", s=14, label="all weightings")
            front = g[pm].sort_values("carbon_mg_per_req")
            ax.plot(front.carbon_mg_per_req, front.lat_mean, "-o", color="tab:green", ms=4, label="Pareto front")
            for lab, r in front.iterrows():
                ax.annotate(lab, (r.carbon_mg_per_req, r.lat_mean), fontsize=5, xytext=(3, 3), textcoords="offset points")
            if "0.4/0.3/0.3" in g.index:
                r = g.loc["0.4/0.3/0.3"]
                ax.scatter(r.carbon_mg_per_req, r.lat_mean, marker="*", s=140, c="red", label="default 0.4/0.3/0.3", zorder=5)
            b = self.main_data("des")
            b = b[(b.level == lv) & (b.pattern == pat)] if len(b) else b
            for mo, mk in [("round_robin", "x"), ("carbon_spill", "s"), ("latency", "^"), ("load", "v")]:
                gm = b[b["mode"] == mo] if len(b) else b
                if len(gm):
                    ax.scatter(gm.carbon_mg_per_req.mean(), gm.lat_mean.mean(), marker=mk, s=45, c="k")
                    ax.annotate(f"{MODE_NAME[mo]} ({gm.carbon_mg_per_req.mean():.2f}, {gm.lat_mean.mean():.0f})",
                                (gm.carbon_mg_per_req.mean(), gm.lat_mean.mean()), fontsize=6, xytext=(4, -8),
                                textcoords="offset points")
            ax.set_xlabel("mgCO2 per successful request")
            ax.set_ylabel("mean latency (ms)")
            ax.set_title(f"{lv} | {pat}", fontsize=8)
            ax.legend(fontsize=6)
        fig.suptitle("A9a  Weight sensitivity (labels = w_lat/w_load/w_carb)  [des]", fontsize=10)
        fig.tight_layout()
        self._save(fig, "A9a_weight_pareto")

    def a9b_burst(self):
        if "sens_beta" not in self.sens:
            print("A9b skipped")
            return
        B = self.sens["sens_beta"].copy()
        B["est"] = B.label.str.split("|").str[0]
        B["win"] = B.label.str.split("|").str[1]
        B["beta_v"] = B.label.str.split("|").str[2].str[1:].astype(float)
        betas = sorted(B.beta_v.unique())
        fig, axs = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
        for j, lv in enumerate(["mid", "high"]):
            for i, (met, yl) in enumerate([("lat_p95", "P95 latency (ms)"), ("carbon_mg_per_req", "mgCO2/request")]):
                ax = axs[i][j]
                f = (lambda v: f"{v:.0f}") if met == "lat_p95" else (lambda v: f"{v:.2f}")
                for est, col in [("ols", "tab:blue"), ("diff", "tab:orange")]:
                    g = B[(B.level == lv) & (B.est == est) & (B.win == "W5")].groupby("beta_v")[met].mean()
                    ax.plot(range(len(g)), g.values, "-o", color=col, label=f"{est} (W=5)")
                    for k, v in enumerate(g.values):
                        ax.annotate(f(v), (k, v), fontsize=5.5, xytext=(0, 4 if est == "ols" else -9),
                                    textcoords="offset points", ha="center")
                for w, mk in [("W3", "s"), ("W9", "^")]:
                    g = B[(B.level == lv) & (B.win == w)][met].mean()
                    k0 = betas.index(0.8)
                    ax.scatter(k0, g, marker=mk, c="k", s=30, label=f"ols {w}, beta=0.8")
                    ax.annotate(f(g), (k0, g), fontsize=5.5, xytext=(6, 0), textcoords="offset points")
                ax.set_xticks(range(len(betas)))
                ax.set_xticklabels(betas)
                ax.set_ylabel(yl)
                ax.set_title(f"{lv} | bursty", fontsize=8)
                if i == 1:
                    ax.set_xlabel("beta")
                if i == 0 and j == 0:
                    ax.legend(fontsize=6)
        fig.suptitle("A9b  Burst-penalty sensitivity (regression slope vs difference, window length)  [des]", fontsize=10)
        fig.tight_layout()
        self._save(fig, "A9b_burst_sensitivity")

    def a9c_sla(self):
        if "sens_sla" not in self.sens:
            print("A9c skipped")
            return
        S = self.sens["sens_sla"].copy()
        S["sla"] = S.label.astype(float)
        fig, axs = plt.subplots(3, 2, figsize=(10, 8), sharex=True)
        for j, lv in enumerate(["mid", "high"]):
            for i, (met, yl) in enumerate([("fallback_pct", "fallback to RR / spill (%)"),
                                           ("sla_attain_all_pct", "requests within SLA (%)"),
                                           ("carbon_mg_per_req", "mgCO2/request")]):
                ax = axs[i][j]
                for mo in ["carl", "carbon_spill"]:
                    g = S[(S.level == lv) & (S["mode"] == mo)].groupby("sla")[met].mean()
                    ax.plot(g.index, g.values, "-o", color=MODE_COLOR[mo], label=MODE_NAME[mo])
                    for x, v in g.items():
                        ax.annotate(f"{v:.1f}" if met != "carbon_mg_per_req" else f"{v:.2f}", (x, v), fontsize=5.5,
                                    xytext=(0, 4 if mo == "carl" else -9), textcoords="offset points", ha="center")
                ax.set_ylabel(yl)
                ax.set_title(f"{lv} | continuous", fontsize=8)
                if i == 2:
                    ax.set_xlabel("SLA threshold (ms)")
                if i == 0 and j == 0:
                    ax.legend(fontsize=6)
        fig.suptitle("A9c  SLA-threshold sensitivity  [des]", fontsize=10)
        fig.tight_layout()
        self._save(fig, "A9c_sla_sensitivity")

    def a9d_cold(self):
        if "sens_cold" not in self.sens:
            print("A9d skipped")
            return
        C = self.sens["sens_cold"]
        cfgs = ["ka10|cold x1", "ka60|cold x1", "ka300|cold x1", "ka60|cold x0.5", "ka60|cold x2"]
        fig, axs = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
        for ax, lv in zip(axs, ["low", "mid"]):
            for k, mo in enumerate(["round_robin", "carl", "carbon_spill"]):
                g = C[(C.level == lv) & (C["mode"] == mo)].groupby("label")[["lat_p95", "cold_pct"]].mean().reindex(cfgs)
                ax.bar(np.arange(len(cfgs)) + (k - 1) * .27, g.lat_p95, .27, color=MODE_COLOR[mo], label=MODE_NAME[mo])
                for x, (v, c) in enumerate(zip(g.lat_p95, g.cold_pct)):
                    ax.annotate(f"{v:.0f}\n{c:.1f}%", (x + (k - 1) * .27, v), ha="center", va="bottom", fontsize=5.5)
            ax.set_xticks(range(len(cfgs)))
            ax.set_xticklabels(cfgs, rotation=20)
            ax.set_title(f"{lv} | continuous", fontsize=8)
            ax.margins(y=.25)
        axs[0].set_ylabel("P95 latency (ms)")
        axs[0].legend(fontsize=6)
        fig.suptitle("A9d  Cold-start sensitivity: label = P95 latency (ms) / cold-start rate  [des]", fontsize=10)
        fig.tight_layout()
        self._save(fig, "A9d_cold_start_sensitivity")

    # ------------------------------------------------------------------ A10-A12
    def a10_fallback(self):
        d = self.main_data()
        g = d[d["mode"] == "carl"]
        keys = [(l, p) for l in self.levels for p in self.patterns]
        lab = [f"{l}\n{p[:5]}" for l, p in keys]
        fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
        fb = [g[(g.level == l) & (g.pattern == p)].fallback_pct.mean() for l, p in keys]
        axs[0].bar(range(len(keys)), fb, color=MODE_COLOR["carl"])
        for x, v in enumerate(fb):
            axs[0].annotate(f"{v:.2f}%", (x, v), ha="center", va="bottom", fontsize=7)
        axs[0].set_xticks(range(len(keys)))
        axs[0].set_xticklabels(lab)
        axs[0].set_ylabel("requests handled by the fallback (%)")
        axs[0].margins(y=.25)
        ratio = []
        for l, p in keys:
            x = g[(g.level == l) & (g.pattern == p)]
            ratio.append((x.carbon_fb_mg_per_req / x.carbon_mg_per_req).mean())
        axs[1].bar(range(len(keys)), [0 if np.isnan(r) else r for r in ratio], color="tab:gray")
        for x, v in enumerate(ratio):
            axs[1].annotate("no fallback" if np.isnan(v) else f"{v:.2f}x", (x, 0 if np.isnan(v) else v), ha="center",
                            va="bottom", fontsize=7)
        axs[1].axhline(1, color="k", lw=.7, ls="--")
        axs[1].set_xticks(range(len(keys)))
        axs[1].set_xticklabels(lab)
        axs[1].set_ylabel("carbon/req of fallback requests  /  CARL average")
        axs[1].margins(y=.25)
        fig.suptitle(f"A10  CARL fallback behaviour (SLA {self.cfg['sla_ms']:.0f} ms)  [{self.source}]", fontsize=10)
        fig.tight_layout()
        self._save(fig, "A10_fallback")

    def a11_scalability(self):
        f = self.paths.analysis / "decision_time.csv"
        if not f.exists() or not len(self.scale):
            print("A11 skipped: run the scalability study and the decision-time benchmark first")
            return
        dt, sc = pd.read_csv(f), self.scale
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        for mo in dt["mode"].unique():
            g = dt[dt["mode"] == mo]
            axs[0].plot(g.n_regions, g.decision_us, "-o", color=MODE_COLOR[mo], label=MODE_NAME[mo])
            for _, r in g[g.n_regions.isin([g.n_regions.min(), g.n_regions.max()])].iterrows():
                axs[0].annotate(f"{r.decision_us:.0f}", (r.n_regions, r.decision_us), fontsize=6, xytext=(0, 4),
                                textcoords="offset points", ha="center")
        axs[0].set_xscale("log", base=2)
        axs[0].set_yscale("log")
        axs[0].set_xlabel("candidate regions N")
        axs[0].set_ylabel("routing decision time (us)")
        axs[0].set_title("per-decision cost (measured; linear in N)", fontsize=8)
        axs[0].legend(fontsize=6)
        ns = sorted(sc.n_regions.unique())
        smodes = [m for m in MODE_ORDER if m in set(sc["mode"])]
        for ax, (met, yl, fm) in zip(axs[1:], [("lat_mean", "mean latency (ms)", "{:.0f}"),
                                               ("carbon_mg_per_req", "mgCO2/request", "{:.2f}")]):
            for k, mo in enumerate(smodes):
                v = [sc[(sc.n_regions == n) & (sc["mode"] == mo)][met].mean() for n in ns]
                xs = np.arange(len(ns)) + (k - len(smodes) / 2 + .5) * .16
                ax.bar(xs, v, .16, color=MODE_COLOR[mo], label=MODE_NAME[mo])
                for x, y in zip(xs, v):
                    ax.annotate(fm.format(y), (x, y), ha="center", va="bottom", fontsize=5.5, rotation=90)
            ax.set_xticks(range(len(ns)))
            ax.set_xticklabels([f"N={n}\n({int(LEVELS_RPS['mid'] * n / 16)} RPS)" for n in ns])
            ax.set_ylabel(yl)
            ax.margins(y=.25)
            ax.set_title("mid load per region held constant  [des]", fontsize=8)
        axs[1].legend(fontsize=6)
        fig.suptitle("A11  Scalability of CARL", fontsize=10)
        fig.tight_layout()
        self._save(fig, "A11_scalability")

    def a12_validity(self):
        lv_ = self.live
        if not len(lv_):
            print("A12 skipped: no live runs")
            return
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        keys = [(l, p) for l in self.levels for p in self.patterns]
        lab = [f"{l}\n{p[:5]}" for l, p in keys]
        sel = lambda l, p: lv_[(lv_.level == l) & (lv_.pattern == p)]
        cpu = [sel(l, p).cpu_mean.mean() for l, p in keys]
        inv = [100 * (~sel(l, p).harness_ok).mean() if len(sel(l, p)) else np.nan for l, p in keys]
        axs[0][0].bar(range(len(keys)), cpu, color=["tab:red" if (i and i > 0) else "tab:green" for i in inv])
        for x, (c, i) in enumerate(zip(cpu, inv)):
            axs[0][0].annotate(f"CPU {c:.0f}%\ninvalid {i:.0f}%", (x, c), ha="center", va="bottom", fontsize=6.5)
        axs[0][0].axhline(HARNESS_LIMITS["cpu_mean"], color="k", ls="--", lw=.8)
        axs[0][0].set_xticks(range(len(keys)))
        axs[0][0].set_xticklabels(lab)
        axs[0][0].set_ylabel("host CPU during trial (%)")
        axs[0][0].set_title("(a) host load; red = some runs exceed harness limits", fontsize=8)
        axs[0][0].margins(y=.3)
        w, x = .38, np.arange(len(keys))
        p50 = [sel(l, p).ovh_p50.mean() for l, p in keys]
        p95 = [sel(l, p).ovh_p95.mean() for l, p in keys]
        axs[0][1].bar(x - w / 2, p50, w, label="median")
        axs[0][1].bar(x + w / 2, p95, w, label="P95")
        for xi, a, b in zip(x, p50, p95):
            axs[0][1].annotate(f"{a:.1f}", (xi - w / 2, a), ha="center", va="bottom", fontsize=6.5)
            axs[0][1].annotate(f"{b:.1f}", (xi + w / 2, b), ha="center", va="bottom", fontsize=6.5)
        ff = self.paths.analysis / "rtt_timer_fidelity.csv"
        err = (lambda f: (f.measured_median_ms - f.nominal_ms).median())(pd.read_csv(ff)) if ff.exists() else float("nan")
        axs[0][1].set_xticks(x)
        axs[0][1].set_xticklabels(lab)
        axs[0][1].set_ylabel("measured - model latency (ms)")
        axs[0][1].legend(fontsize=6)
        axs[0][1].margins(y=.25)
        axs[0][1].set_title(f"(b) harness overhead per request (idle RTT-timer error: median {err:+.1f} ms)", fontsize=8)
        A = self.M[(self.M.tag == "main")].groupby(["source", "level", "pattern", "mode"])[["lat_mean", "carbon_mg_per_req"]].mean().unstack(0).dropna()
        okmask = lv_.groupby(["level", "pattern"]).harness_ok.all()
        for ax, met, nm in [(axs[1][0], "lat_mean", "mean latency (ms)"), (axs[1][1], "carbon_mg_per_req", "mgCO2/request")]:
            if len(A) < 3 or ("live" not in A[met]) or ("des" not in A[met]):
                ax.text(.5, .5, "need both live and simulator runs", ha="center")
                continue
            xs, ys = A[met]["des"].to_numpy(), A[met]["live"].to_numpy()
            valid = np.array([okmask.get((l, p), False) for l, p, _ in A.index])
            ax.scatter(xs[valid], ys[valid], s=14, c="tab:blue", label="harness valid")
            ax.scatter(xs[~valid], ys[~valid], s=18, facecolors="none", edgecolors="tab:red", label="harness saturated")
            lim = [min(xs.min(), ys.min()), max(xs.max(), ys.max())]
            ax.plot(lim, lim, "k--", lw=.7)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(f"simulator: {nm}")
            ax.set_ylabel(f"live testbed: {nm}")
            r = np.corrcoef(np.log(xs), np.log(ys))[0, 1]
            rho = sps.spearmanr(xs, ys).correlation
            ax.set_title(f"({'c' if met == 'lat_mean' else 'd'}) agreement: log-Pearson r={r:.2f}, "
                         f"Spearman rho={rho:.2f} ({len(xs)} configs)", fontsize=8)
            ax.legend(fontsize=6)
        fig.suptitle("A12  Live-testbed validity (single-host emulation) and simulator cross-validation", fontsize=10)
        fig.tight_layout()
        self._save(fig, "A12_harness_validity")

    # ------------------------------------------------------------------ everything
    def run_all(self, only=None):
        steps = dict(a1=self.a1_latency, a2=self.a2_reliability, a3=self.a3_sla, a4=self.a4_carbon,
                     a5=self.a5_energy_ablation, a6=self.a6_distribution, a7=self.a7_carbon_tracking,
                     a8=self.a8_statistics, a9a=self.a9a_weights, a9b=self.a9b_burst, a9c=self.a9c_sla, a9d=self.a9d_cold,
                     a10=self.a10_fallback, a11=self.a11_scalability, a12=self.a12_validity)
        for name, fn in steps.items():
            if only and name not in only:
                continue
            try:
                fn()
            except Exception as e:                     # one failing figure must not stop the others
                print(f"[{name}] failed: {type(e).__name__}: {e}")
