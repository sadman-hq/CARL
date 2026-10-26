"""Regenerates the tables and headline numbers of the paper from the per-run metrics table
(results/metrics_per_run.csv, columns as written by carl.analysis). No experiment has to be re-run."""
from pathlib import Path

import numpy as np, pandas as pd
from scipy import stats

MODES = ["round_robin", "latency", "load", "carbon_greedy", "carbon_spill", "eps_latency", "carl"]
NAME = dict(round_robin="Round-robin", latency="Latency-based", load="Load-based", carbon_greedy="Carbon-greedy",
            carbon_spill="Carbon-spill", eps_latency="$\\varepsilon$-latency", carl="\\textbf{CARL}")
LEVELS = ["low", "mid", "high"]


def load_metrics(csv):
    d = pd.read_csv(csv)
    d["viol"] = d.viol_warm_pct + d.viol_cold_pct
    return d


def _fmt(x, n=1):
    return f"{x:.{n}f}"


def main_tables(d):
    """The four LaTeX table bodies (performance and carbon, continuous and bursty traffic), simulator results."""
    des = d[d.source == "des"]
    t = des.groupby(["level", "pattern", "mode"]).agg(
        lat=("lat_mean", "mean"), p95=("lat_p95", "mean"), thr=("goodput_rps", "mean"), viol=("viol", "mean"),
        err=("error_pct", "mean"), co2=("carbon_mg_per_req", "mean"), jain=("jain", "mean"), cold=("cold_pct", "mean"))
    rr = t.xs("round_robin", level="mode")["co2"]
    out = {}
    for pat in ["continuous", "bursty"]:
        perf, carb = [], []
        for lv in LEVELS:
            for i, m in enumerate(MODES):
                r = t.loc[(lv, pat, m)]
                pre = f"\\multirow{{{len(MODES)}}}{{*}}{{{lv.capitalize()}}}" if i == 0 else ""
                c1 = [_fmt(r.lat), _fmt(r.p95), _fmt(r.thr), _fmt(r.viol), _fmt(r.err, 2)]
                red = 100 * (1 - r.co2 / rr.loc[(lv, pat)])
                c2 = [f"{r.co2:.3f}", "--" if m == "round_robin" else f"{red:.1f}", f"{r.jain:.3f}", _fmt(r.cold, 2)]
                if m == "carl":
                    c1, c2 = [f"\\textbf{{{c}}}" for c in c1], [f"\\textbf{{{c}}}" for c in c2]
                perf.append(f" {pre} & {NAME[m]} & " + " & ".join(c1) + " \\\\")
                carb.append(f" {pre} & {NAME[m]} & " + " & ".join(c2) + " \\\\")
            perf.append("\\midrule")
            carb.append("\\midrule")
        out[f"{pat}_performance"] = "\n".join(perf[:-1])
        out[f"{pat}_carbon"] = "\n".join(carb[:-1])
    return out


def _pci(a, b):
    x = a - b
    n = len(x)
    return x.mean(), stats.t.ppf(.975, n - 1) * x.std(ddof=1) / np.sqrt(n)


def paired_differences(d):
    """CARL minus other policies, paired by repetition (all policies see the same arrivals), simulator results."""
    des = d[d.source == "des"]
    rows = []
    for lv in LEVELS:
        for pat in ["continuous", "bursty"]:
            g = des[(des.level == lv) & (des.pattern == pat)]
            c = g[g["mode"] == "carl"].sort_values("rep")
            row = dict(scenario=f"{lv}/{pat}")
            for o in ["round_robin", "carbon_spill", "eps_latency", "carl_static", "carl_adaptive"]:
                x = g[g["mode"] == o].sort_values("rep")
                if x.empty or c.empty:
                    continue
                dl = _pci(c.lat_mean.values, x.lat_mean.values)
                dk = _pci(c.carbon_mg_per_req.values, x.carbon_mg_per_req.values)
                row.update({f"{o}_dlat": dl[0], f"{o}_dlat_ci": dl[1], f"{o}_dco2": dk[0], f"{o}_dco2_ci": dk[1]})
            rows.append(row)
    return pd.DataFrame(rows)


def paired_latex(p):
    lines = []
    for _, r in p.iterrows():
        cells = [r.scenario.title()]
        for o in ["round_robin", "carbon_spill", "eps_latency", "carl_static"]:
            cells += [f"${r[o + '_dlat']:+.1f}\\pm{r[o + '_dlat_ci']:.1f}$", f"${r[o + '_dco2']:+.3f}\\pm{r[o + '_dco2_ci']:.3f}$"]
        lines.append(" & ".join(cells) + " \\\\")
    return "\n".join(lines)


def headline(d):
    """CARL vs round-robin (simulator means) and non-dominated sets among policies with < 1 % errors."""
    des = d[d.source == "des"]
    rows = []
    for (lv, pat), g in des.groupby(["level", "pattern"]):
        m = g.groupby("mode")[["lat_mean", "carbon_mg_per_req", "error_pct"]].mean()
        ok = m[m.error_pct < 1]
        nd = [a for a in ok.index if not any(
            (ok.loc[b, "lat_mean"] <= ok.loc[a, "lat_mean"] and ok.loc[b, "carbon_mg_per_req"] <= ok.loc[a, "carbon_mg_per_req"]
             and (ok.loc[b, "lat_mean"] < ok.loc[a, "lat_mean"] or ok.loc[b, "carbon_mg_per_req"] < ok.loc[a, "carbon_mg_per_req"]))
            for b in ok.index if b != a)]
        rows.append(dict(level=lv, pattern=pat,
                         latency_reduction_pct=100 * (1 - m.loc["carl", "lat_mean"] / m.loc["round_robin", "lat_mean"]),
                         carbon_reduction_pct=100 * (1 - m.loc["carl", "carbon_mg_per_req"] / m.loc["round_robin", "carbon_mg_per_req"]),
                         carl_in_nondominated_set="carl" in nd, nondominated=",".join(nd)))
    return pd.DataFrame(rows)


def validation(d):
    """Live (valid runs only) vs simulator agreement."""
    live = d[(d.source == "live") & d.harness_ok.astype(bool)]
    des = d[d.source == "des"]
    lv = live.groupby(["level", "pattern", "mode"])[["lat_mean", "carbon_mg_per_req"]].mean()
    dd = des.groupby(["level", "pattern", "mode"])[["lat_mean", "carbon_mg_per_req"]].mean()
    j = lv.join(dd, rsuffix="_des", how="inner")
    res = dict(n_configs=len(j))
    for m in ["lat_mean", "carbon_mg_per_req"]:
        rel = 100 * abs(j[m] / j[m + "_des"] - 1)
        res[m] = dict(log_pearson=float(np.corrcoef(np.log(j[m + "_des"]), np.log(j[m]))[0, 1]),
                      spearman=float(stats.spearmanr(j[m + "_des"], j[m]).correlation),
                      mean_abs_pct_diff=float(rel.mean()), max_abs_pct_diff=float(rel.max()))
    red = []
    for pat in ["continuous", "bursty"]:
        for lvl in LEVELS:
            a = live[(live.level == lvl) & (live.pattern == pat)]
            b = des[(des.level == lvl) & (des.pattern == pat)]
            if a[a["mode"] == "carl"].empty or b.empty:
                continue
            ra = 1 - a[a["mode"] == "carl"].carbon_mg_per_req.mean() / a[a["mode"] == "round_robin"].carbon_mg_per_req.mean()
            rb = 1 - b[b["mode"] == "carl"].carbon_mg_per_req.mean() / b[b["mode"] == "round_robin"].carbon_mg_per_req.mean()
            red.append(dict(scenario=f"{lvl}/{pat}", live_reduction_pct=100 * ra, sim_reduction_pct=100 * rb,
                            n_valid_live_carl_runs=len(a[a["mode"] == "carl"])))
    res["carl_carbon_reduction_vs_rr"] = red
    inv = d[d.source == "live"].groupby(["level", "pattern"]).harness_ok.agg(lambda s: int((~s.astype(bool)).sum()))
    res["invalid_live_runs"] = {f"{k[0]}/{k[1]}": int(v) for k, v in inv.items()}
    return res


def energy_accounting(d):
    """Ratio of wall-clock (queueing included) to active-time carbon, per policy (simulator)."""
    des = d[d.source == "des"].assign(ratio=lambda x: x.carbon_wall_mg_per_req / x.carbon_mg_per_req)
    return des.groupby("mode").ratio.agg(["mean", "max"])


def adaptive_vs_carl(d):
    des = d[d.source == "des"]
    rows = []
    for (lv, pt), g in des.groupby(["level", "pattern"]):
        c = g[g["mode"] == "carl"]
        for o in ["carl_adaptive", "carl_static"]:
            x = g[g["mode"] == o]
            if x.empty:
                continue
            rows.append(dict(scenario=f"{lv}/{pt}", other=o,
                             carbon_pct_vs_carl=100 * (x.carbon_mg_per_req.mean() / c.carbon_mg_per_req.mean() - 1),
                             latency_diff_ms=x.lat_mean.mean() - c.lat_mean.mean()))
    return pd.DataFrame(rows)


def live_welch(d):
    """Live testbed, valid scenarios only: Welch t-tests CARL vs every other policy with Holm correction."""
    live = d[(d.source == "live") & d.harness_ok.astype(bool)]
    rows = []
    for (lv, pt), g in live.groupby(["level", "pattern"]):
        c = g[g["mode"] == "carl"]
        if len(c) < 2:
            continue
        for m in [x for x in g["mode"].unique() if x != "carl"]:
            x = g[g["mode"] == m]
            for k in ["lat_mean", "carbon_mg_per_req"]:
                a, b = c[k].values, x[k].values
                p = stats.ttest_ind(a, b, equal_var=False).pvalue if (a.std() + b.std() > 0 and len(b) > 1) else np.nan
                rows.append(dict(scenario=f"{lv}/{pt}", baseline=m, metric=k, p=p, diff=a.mean() - b.mean()))
    T = pd.DataFrame(rows)
    if T.empty:
        return T
    T["p_holm"] = np.nan
    for k in T.metric.unique():
        s = T[T.metric == k]
        p = s.p.values
        order = np.argsort(np.where(np.isnan(p), 2, p))
        adj = np.full(len(p), np.nan)
        mx, nv = 0.0, int((~np.isnan(p)).sum())
        for rank, i in enumerate(order[:nv]):
            mx = max(mx, min(1.0, (nv - rank) * p[i]))
            adj[i] = mx
        T.loc[s.index, "p_holm"] = adj
    return T


def write_all(metrics_csv, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d = load_metrics(metrics_csv)
    for k, v in main_tables(d).items():
        (out / f"table_{k}.tex").write_text(v + "\n")
    p = paired_differences(d)
    p.to_csv(out / "paired_differences.csv", index=False)
    (out / "table_paired_differences.tex").write_text(paired_latex(p) + "\n")
    headline(d).to_csv(out / "headline.csv", index=False)
    energy_accounting(d).to_csv(out / "energy_accounting_ratio.csv")
    adaptive_vs_carl(d).to_csv(out / "adaptive_static_vs_carl.csv", index=False)
    lw = live_welch(d)
    lw.to_csv(out / "live_welch_tests.csv", index=False)
    import json
    (out / "validation.json").write_text(json.dumps(validation(d), indent=1))
    return d, p, lw
