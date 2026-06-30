"""
06_temporal_analysis.py

Stage 06 - TIME-SERIES LAYER (the addition that makes this a Convergence of
Time-Series Analytics and Social Media Intelligence study).

Builds claim-level time series from each claim's created_at (carried from the
source Bluesky post), then runs four analyses that mirror the special-issue
themes (time-series analysis, anomaly detection, predictive modelling,
social-media intelligence):

  RQ-T1  Temporal trend: weekly volume + weekly share of non-supported claims
         + weekly mean stigma score, over the dense analysis window.
  RQ-T2  Anomaly / changepoint detection: weeks whose misinformation rate or
         stigma score deviate beyond a robust threshold (median +/- k*MAD),
         plus a CUSUM changepoint scan. The March-2025 volume surge is the
         showcase.
  RQ-T3  Forecasting: a simple, honest predictive model (Holt-Winters /
         exponential smoothing, with a seasonal-naive fallback) on the weekly
         misinformation-rate series, evaluated on a held-out tail (MAE/RMSE vs
         naive baseline).
  RQ-T4  Per-condition trajectories: weekly stigma series per condition and
         their cross-correlation (do PCOS vs endometriosis vs infertility move
         together or diverge over time?).

No API calls. Pure analysis over data/03_classified/classified_claims.jsonl.

Outputs -> data/04_evaluation/results/
  temporal_stats.json
  figT1_weekly_trends.png
  figT2_anomaly_detection.png
  figT3_forecast.png
  figT4_condition_trajectories.png

Design choices made explicit for the paper:
  * Window: full series is described, but inferential/forecast analysis is
    restricted to the dense window (default: first week with >= MIN_WEEK_CLAIMS
    onward) so a sparse early tail does not dominate. Both are reported.
  * Sanity filter: created_at must parse and fall within [2023-01-01, today];
    out-of-range timestamps (client-supplied on Bluesky) are dropped and counted.
  * Weekly binning (W-MON) balances resolution against per-bin noise given the
    claim density.
"""

import json
from collections import defaultdict, Counter
from datetime import date, datetime, timezone

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 13,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.dpi": 200,
})
GRAD = plt.cm.viridis(np.linspace(0.15, 0.85, 6))

# ----------------------------- tunables ----------------------------------- #
MIN_WEEK_CLAIMS = 5      # a week needs >= this many claims to enter the dense window
MAD_K = 3.0              # robust anomaly threshold: median +/- K * (1.4826*MAD)
FORECAST_HOLDOUT = 8     # weeks held out at the tail for forecast evaluation
NONSUPPORTED = {"UNSUPPORTED", "EXAGGERATED", "CONTRADICTED", "DANGEROUS"}
LO_DATE = date(2023, 1, 1)


# ----------------------------- helpers ------------------------------------ #
def gv(c):
    return c.get("classification", {}).get("llm_grounded", {}).get("veracity", "UNSUPPORTED")

def stigma(c):
    s = c.get("classification", {}).get("stigma_sentiment", {}).get("stigma_score")
    return s if isinstance(s, (int, float)) else None

def norm_condition(cl):
    cond = (cl.get("target_condition", "general health") or "general health").lower().strip()
    if "pcos" in cond or "polycystic" in cond:
        return "pcos"
    if "endometrios" in cond:
        return "endometriosis"
    if "infertil" in cond:
        return "infertility"
    if cond.startswith("fertility") or "conception" in cond:
        return "fertility"
    return cond

def parse_day(s):
    """Parse an ISO created_at to a date, with sanity bounds. None if bad."""
    if not s or not isinstance(s, str):
        return None
    txt = s.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(txt)
    except ValueError:
        try:
            dt = datetime.fromisoformat(txt[:10])
        except ValueError:
            return None
    d = dt.date()
    today = datetime.now(timezone.utc).date()
    if d < LO_DATE or d > today:
        return None
    return d

def week_start(d):
    """Monday of the ISO week containing date d."""
    return d.fromordinal(d.toordinal() - d.weekday())

def load():
    return [json.loads(l) for l in open(C.CLASSIFIED_CLAIMS, encoding="utf-8") if l.strip()]


# ----------------------------- series build ------------------------------- #
def build_weekly(claims):
    """Return ordered weeks and per-week aggregates over the DENSE window."""
    dated, n_bad = [], 0
    for c in claims:
        d = parse_day(c.get("created_at"))
        if d is None:
            n_bad += 1
            continue
        dated.append((week_start(d), c))

    by_week_all = defaultdict(list)
    for w, c in dated:
        by_week_all[w].append(c)

    weeks_sorted = sorted(by_week_all)
    # dense window: from the first week meeting MIN_WEEK_CLAIMS to the end
    dense_start = None
    for w in weeks_sorted:
        if len(by_week_all[w]) >= MIN_WEEK_CLAIMS:
            dense_start = w
            break
    if dense_start is None:
        dense_start = weeks_sorted[0] if weeks_sorted else None

    # build a complete weekly index (fill gaps) from dense_start..last
    weeks = []
    if dense_start is not None:
        last = weeks_sorted[-1]
        w = dense_start
        while w <= last:
            weeks.append(w)
            w = date.fromordinal(w.toordinal() + 7)

    rows = []
    for w in weeks:
        cs = by_week_all.get(w, [])
        n = len(cs)
        if n == 0:
            rows.append({"week": w, "n": 0, "misinfo_rate": np.nan,
                         "mean_stigma": np.nan})
            continue
        nonsupp = sum(1 for c in cs if gv(c) in NONSUPPORTED)
        stig_vals = [stigma(c) for c in cs if stigma(c) is not None]
        rows.append({
            "week": w, "n": n,
            "misinfo_rate": nonsupp / n,
            "mean_stigma": float(np.mean(stig_vals)) if stig_vals else np.nan,
        })
    return rows, by_week_all, weeks_sorted, n_bad


def interp_nan(y):
    """Linear-interpolate internal NaNs so smoothing/forecasting can run."""
    y = np.asarray(y, float)
    idx = np.arange(len(y))
    good = ~np.isnan(y)
    if good.sum() < 2:
        return np.nan_to_num(y, nan=float(np.nanmean(y)) if good.any() else 0.0)
    y[~good] = np.interp(idx[~good], idx[good], y[good])
    return y


# ----------------------------- anomaly / changepoint ---------------------- #
def mad_anomalies(values):
    v = np.asarray(values, float)
    med = np.nanmedian(v)
    mad = np.nanmedian(np.abs(v - med)) * 1.4826
    if mad == 0 or np.isnan(mad):
        return med, mad, np.zeros(len(v), bool)
    hi = med + MAD_K * mad
    lo = med - MAD_K * mad
    flags = (v > hi) | (v < lo)
    return med, mad, flags

def cusum_changepoints(values, threshold_sd=4.0):
    """Simple two-sided CUSUM; returns indices where cumulative drift resets."""
    v = np.asarray(values, float)
    mu = np.nanmean(v)
    sd = np.nanstd(v) or 1.0
    k = 0.5 * sd
    sp = sn = 0.0
    cps = []
    for i, x in enumerate(v):
        sp = max(0, sp + (x - mu) - k)
        sn = min(0, sn + (x - mu) + k)
        if sp > threshold_sd * sd or sn < -threshold_sd * sd:
            cps.append(i)
            sp = sn = 0.0
    return cps


# ----------------------------- forecasting -------------------------------- #
def holt_winters_forecast(y, horizon):
    """Holt linear-trend exponential smoothing (no seasonality, short series).
    Falls back gracefully. Returns (fitted, forecast)."""
    y = np.asarray(y, float)
    n = len(y)
    if n < 4:
        return np.full(n, np.nan), np.full(horizon, y[-1] if n else 0.0)
    alpha, beta = 0.4, 0.2
    level = y[0]
    trend = y[1] - y[0]
    fitted = [level]
    for t in range(1, n):
        prev_level = level
        level = alpha * y[t] + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
        fitted.append(level)
    fc = [level + (h + 1) * trend for h in range(horizon)]
    return np.array(fitted), np.clip(np.array(fc), 0, 1)

def seasonal_naive(y, horizon):
    y = np.asarray(y, float)
    return np.full(horizon, y[-1] if len(y) else 0.0)

def eval_forecast(y):
    """Hold out the tail, fit on the head, report MAE/RMSE vs naive."""
    y = interp_nan(y)
    n = len(y)
    h = min(FORECAST_HOLDOUT, max(2, n // 4))
    if n - h < 4:
        return None
    train, test = y[:n - h], y[n - h:]
    _, fc = holt_winters_forecast(train, h)
    nv = seasonal_naive(train, h)
    def mae(a, b): return float(np.mean(np.abs(a - b)))
    def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
    return {
        "holdout_weeks": h,
        "hw_mae": mae(test, fc), "hw_rmse": rmse(test, fc),
        "naive_mae": mae(test, nv), "naive_rmse": rmse(test, nv),
        "test": test.tolist(), "hw_pred": fc.tolist(), "naive_pred": nv.tolist(),
        "improvement_over_naive_mae": float(mae(test, nv) - mae(test, fc)),
    }


# ----------------------------- figures ------------------------------------ #
def figT1(rows, out):
    weeks = [r["week"] for r in rows]
    n = [r["n"] for r in rows]
    mis = [r["misinfo_rate"] for r in rows]
    stg = [r["mean_stigma"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(weeks, n, width=5, color=GRAD[1], alpha=0.5, label="Claims/week")
    ax1.set_ylabel("Claims per week")
    ax1.set_xlabel("Week")
    ax2 = ax1.twinx()
    ax2.plot(weeks, mis, color=GRAD[5], lw=2, label="Non-supported rate")
    ax2.plot(weeks, stg, color=GRAD[0], lw=2, ls="--", label="Mean stigma")
    ax2.set_ylabel("Rate / stigma (0-1)")
    ax2.set_ylim(0, 1)
    lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labs = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(lines, labs, loc="upper left", fontsize=10)
    plt.title("Weekly claim volume, non-supported rate, and stigma")
    plt.tight_layout()
    plt.savefig(out / "figT1_weekly_trends.png")
    plt.close()

def figT2(rows, out):
    weeks = [r["week"] for r in rows]
    mis = interp_nan([r["misinfo_rate"] for r in rows])
    vol = np.array([r["n"] for r in rows], float)
    med, mad, flags = mad_anomalies(mis)
    vmed, vmad, vflags = mad_anomalies(vol)
    cps = cusum_changepoints(mis)
    fig, (a, b) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    a.plot(weeks, vol, color=GRAD[1], lw=2)
    a.scatter([weeks[i] for i in np.where(vflags)[0]],
              vol[vflags], color="red", zorder=5, label="volume anomaly")
    a.axhline(vmed, color="gray", ls=":")
    a.set_ylabel("Claims/week"); a.legend(fontsize=10)
    a.set_title("Volume anomalies (robust MAD threshold)")
    b.plot(weeks, mis, color=GRAD[5], lw=2)
    b.scatter([weeks[i] for i in np.where(flags)[0]],
              mis[flags], color="red", zorder=5, label="rate anomaly")
    b.axhline(med, color="gray", ls=":")
    b.axhline(med + MAD_K * mad, color="gray", ls="--", lw=0.8)
    for i in cps:
        b.axvline(weeks[i], color=GRAD[0], ls="-.", lw=1, alpha=0.7)
    b.set_ylabel("Non-supported rate"); b.set_ylim(0, 1); b.legend(fontsize=10)
    b.set_title("Misinformation-rate anomalies + CUSUM changepoints (vertical)")
    plt.tight_layout()
    plt.savefig(out / "figT2_anomaly_detection.png")
    plt.close()
    return med, mad, flags, vflags, cps

def figT3(rows, out, ev):
    if not ev:
        return
    weeks = [r["week"] for r in rows]
    mis = interp_nan([r["misinfo_rate"] for r in rows])
    h = ev["holdout_weeks"]
    fitted, _ = holt_winters_forecast(mis[:len(mis) - h], h)
    plt.figure(figsize=(12, 6))
    plt.plot(weeks, mis, color=GRAD[1], lw=2, label="Actual")
    tail = weeks[len(weeks) - h:]
    plt.plot(tail, ev["hw_pred"], color=GRAD[5], lw=2, marker="o",
             label=f"Holt-Winters (MAE={ev['hw_mae']:.3f})")
    plt.plot(tail, ev["naive_pred"], color=GRAD[0], lw=2, ls="--",
             label=f"Naive (MAE={ev['naive_mae']:.3f})")
    plt.axvline(tail[0], color="gray", ls=":")
    plt.ylim(0, 1); plt.ylabel("Non-supported rate"); plt.xlabel("Week")
    plt.title("Forecasting weekly misinformation rate (held-out tail)")
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(out / "figT3_forecast.png")
    plt.close()

def figT4(claims, out, top_n=4):
    dated = []
    for c in claims:
        d = parse_day(c.get("created_at"))
        s = stigma(c)
        if d is not None and s is not None:
            dated.append((week_start(d), norm_condition(c), s))
    conds = [c for c, _ in Counter(x[1] for x in dated).most_common(top_n)]
    series = {cond: defaultdict(list) for cond in conds}
    for w, cond, s in dated:
        if cond in series:
            series[cond][w].append(s)
    all_weeks = sorted({w for _, w_set in
                        [(c, series[c]) for c in conds] for w in w_set})
    plt.figure(figsize=(12, 6))
    traj = {}
    for i, cond in enumerate(conds):
        ys = [np.mean(series[cond][w]) if series[cond].get(w) else np.nan
              for w in all_weeks]
        traj[cond] = interp_nan(ys)
        plt.plot(all_weeks, traj[cond], lw=2, color=GRAD[i], label=cond)
    plt.ylim(0, 1); plt.ylabel("Mean stigma (0-1)"); plt.xlabel("Week")
    plt.title("Per-condition stigma trajectories")
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(out / "figT4_condition_trajectories.png")
    plt.close()
    # cross-correlation at lag 0
    xcorr = {}
    names = list(traj)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = traj[names[i]], traj[names[j]]
            if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0:
                xcorr[f"{names[i]}~{names[j]}"] = float(np.corrcoef(a, b)[0, 1])
    return xcorr


def main():
    out = C.RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    if not C.CLASSIFIED_CLAIMS.exists():
        raise SystemExit(f"No input at {C.CLASSIFIED_CLAIMS}")
    claims = load()
    n_with_date = sum(1 for c in claims if parse_day(c.get("created_at")))
    if n_with_date == 0:
        raise SystemExit("No usable created_at timestamps. Re-run stages 00 and 01 "
                         "with the patched collector first.")

    rows, by_week_all, weeks_sorted, n_bad = build_weekly(claims)
    med, mad, flags, vflags, cps = figT2(rows, out)
    figT1(rows, out)
    ev = eval_forecast([r["misinfo_rate"] for r in rows])
    figT3(rows, out, ev)
    xcorr = figT4(claims, out)

    anom_weeks = [str(rows[i]["week"]) for i in np.where(flags)[0]]
    vol_anom_weeks = [str(rows[i]["week"]) for i in np.where(vflags)[0]]
    cp_weeks = [str(rows[i]["week"]) for i in cps]

    stats = {
        "n_claims_total": len(claims),
        "n_claims_with_valid_date": n_with_date,
        "n_claims_dropped_bad_date": n_bad,
        "full_span": {
            "earliest_week": str(weeks_sorted[0]) if weeks_sorted else None,
            "latest_week": str(weeks_sorted[-1]) if weeks_sorted else None,
            "n_distinct_weeks_observed": len(weeks_sorted),
        },
        "dense_window": {
            "start_week": str(rows[0]["week"]) if rows else None,
            "end_week": str(rows[-1]["week"]) if rows else None,
            "n_weeks": len(rows),
            "min_week_claims_threshold": MIN_WEEK_CLAIMS,
        },
        "anomaly_detection": {
            "misinfo_rate_median": float(med),
            "misinfo_rate_mad": float(mad),
            "mad_k": MAD_K,
            "rate_anomaly_weeks": anom_weeks,
            "volume_anomaly_weeks": vol_anom_weeks,
            "cusum_changepoint_weeks": cp_weeks,
        },
        "forecast": ev,
        "condition_trajectory_xcorr_lag0": xcorr,
    }
    with open(out / "temporal_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False, default=str)

    print(f"Done. temporal stage over {n_with_date} dated claims "
          f"({n_bad} dropped for bad/out-of-range dates).")
    print(f"  dense window: {stats['dense_window']['start_week']} .. "
          f"{stats['dense_window']['end_week']} ({len(rows)} weeks)")
    print(f"  rate anomalies: {anom_weeks}")
    print(f"  volume anomalies: {vol_anom_weeks}")
    print(f"  CUSUM changepoints: {cp_weeks}")
    if ev:
        print(f"  forecast MAE  Holt-Winters={ev['hw_mae']:.3f}  "
              f"naive={ev['naive_mae']:.3f}  "
              f"(improvement={ev['improvement_over_naive_mae']:+.3f})")
    print(f"  4 figures + temporal_stats.json -> {out}")


if __name__ == "__main__":
    main()
