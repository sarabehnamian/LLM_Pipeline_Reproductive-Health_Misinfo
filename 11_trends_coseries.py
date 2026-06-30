"""
07_trends_coseries.py  (v2 - granularity-correct + detrended)

Relate Google Trends search interest to the Bluesky misinformation/stigma signal,
at MATCHED MONTHLY resolution, with a detrended (first-difference) test that
controls for shared growth.

WHY MONTHLY: Google Trends returns MONTHLY points for multi-year ranges (weekly
only for <~5y, daily for <~9mo). v1 mislabelled monthly points as weeks. v2
aggregates BOTH series to calendar month so the comparison is apples-to-apples.

WHY DETRENDING: both series rise over 2024-2026 (Bluesky grows; search drifts).
A raw correlation can be high just because "both went up." First-differencing
(month-to-month change) removes the shared trend and tests genuine co-movement.
The detrended result is the one to trust for the paper.

Run:
  pip install pytrends
  python 07_trends_coseries.py --auto
Or with manual monthly CSVs (one per term) in ./trends_csv:
  python 07_trends_coseries.py --csv-dir trends_csv

Reads dated classified claims if available (gives misinfo_rate & stigma),
else falls back to all_posts.jsonl (volume only) for a FREE preview.
"""

import argparse
import glob
import json
import os
from collections import defaultdict
from datetime import date, datetime, timezone

import numpy as np
import config as C

TERMS = ["pcos cure", "pcos natural", "endometriosis diet", "seed cycling",
         "inositol pcos", "period detox", "fertility supplements",
         "spearmint tea pcos", "birth control infertility"]

LO = date(2020, 1, 1)
NONSUPPORTED = {"UNSUPPORTED", "EXAGGERATED", "CONTRADICTED", "DANGEROUS"}


def parse_day(s):
    if not isinstance(s, str) or not s.strip():
        return None
    t = s.strip().replace("Z", "+00:00")
    for cand in (t, t[:10]):
        try:
            d = datetime.fromisoformat(cand).date()
            today = datetime.now(timezone.utc).date()
            return d if LO <= d <= today else None
        except ValueError:
            continue
    return None

def ym(d):
    """Calendar-month key as a date (first of month)."""
    return date(d.year, d.month, 1)


# --------------------------- Bluesky side (MONTHLY) ----------------------- #
def load_social_monthly():
    src = C.CLASSIFIED_CLAIMS
    have_labels = True
    rows = []
    if src.exists():
        rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    if not rows or not any(r.get("created_at") for r in rows[:50]):
        src, have_labels = C.ALL_POSTS, False
        rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]

    by_m = defaultdict(list)
    for r in rows:
        d = parse_day(r.get("created_at"))
        if d:
            by_m[ym(d)].append(r)

    out = {}
    for m, items in by_m.items():
        rec = {"n": len(items), "misinfo_rate": np.nan, "mean_stigma": np.nan}
        if have_labels:
            labs = [it.get("classification", {}).get("llm_grounded", {}).get("veracity")
                    for it in items]
            labs = [l for l in labs if l]
            if labs:
                rec["misinfo_rate"] = sum(1 for l in labs if l in NONSUPPORTED)/len(labs)
            st = [it.get("classification", {}).get("stigma_sentiment", {}).get("stigma_score")
                  for it in items]
            st = [s for s in st if isinstance(s, (int, float))]
            if st:
                rec["mean_stigma"] = float(np.mean(st))
        out[m] = rec
    print(f"[social] source={src.name}  labels={'yes' if have_labels else 'NO (volume only)'}  "
          f"months={len(out)}")
    return out, have_labels


# --------------------------- Trends side (MONTHLY) ------------------------ #
def load_trends_monthly_csv(csv_dir):
    files = glob.glob(os.path.join(csv_dir, "*.csv"))
    if not files:
        raise SystemExit(f"No CSVs in {csv_dir}.")
    per_m = defaultdict(list)
    for fp in files:
        for ln in open(fp, encoding="utf-8").read().splitlines():
            parts = ln.split(",")
            if len(parts) < 2:
                continue
            try:
                d = datetime.fromisoformat(parts[0].strip()[:10]).date()
            except ValueError:
                continue
            try:
                v = float(parts[1].strip().replace("<1", "0"))
            except ValueError:
                continue
            per_m[ym(d)].append(v)
    return {m: float(np.mean(v)) for m, v in per_m.items() if v}

def load_trends_monthly_auto():
    from pytrends.request import TrendReq
    pt = TrendReq(hl="en-US", tz=0)
    per_m = defaultdict(list)
    for i in range(0, len(TERMS), 5):
        batch = TERMS[i:i+5]
        try:
            pt.build_payload(batch, timeframe="2020-01-01 " +
                             datetime.now().strftime("%Y-%m-%d"))
            df = pt.interest_over_time()
        except Exception as e:
            print(f"  ! pytrends failed for {batch}: {e}")
            continue
        if df is None or df.empty:
            continue
        for ts, row in df.iterrows():
            m = ym(ts.date())
            for term in batch:
                if term in row:
                    per_m[m].append(float(row[term]))
    if not per_m:
        raise SystemExit("pytrends returned nothing (rate-limited?). Try --csv-dir.")
    return {m: float(np.mean(v)) for m, v in per_m.items() if v}


# --------------------------- stats ---------------------------------------- #
def spearman(x, y):
    import math
    n = len(x)
    if n < 6:
        return float("nan"), float("nan")
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0]*len(v); i = 0
        while i < len(v):
            j = i
            while j+1 < len(v) and v[order[j+1]] == v[order[i]]:
                j += 1
            avg = (i+j)/2.0 + 1
            for k in range(i, j+1):
                r[order[k]] = avg
            i = j+1
        return r
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx)/n, sum(ry)/n
    cov = sum((a-mx)*(b-my) for a, b in zip(rx, ry))
    vx = sum((a-mx)**2 for a in rx); vy = sum((b-my)**2 for b in ry)
    if vx == 0 or vy == 0:
        return float("nan"), float("nan")
    rho = cov/math.sqrt(vx*vy)
    t = rho*math.sqrt((n-2)/max(1e-12, 1-rho**2))
    return rho, math.erfc(abs(t)/math.sqrt(2))

def diff(series):
    return [series[i] - series[i-1] for i in range(1, len(series))]


def report(name, a, b):
    pair = [(x, y) for x, y in zip(a, b) if not (np.isnan(x) or np.isnan(y))]
    if len(pair) < 6:
        print(f"  {name:>28}: n={len(pair)} too few (need labels/re-run?)")
        return None
    xs, ys = zip(*pair)
    rho, p = spearman(list(xs), list(ys))
    tag = "REAL" if (p < 0.05 and abs(rho) >= 0.2) else "weak/null"
    print(f"  {name:>28}: rho={rho:+.3f}  p={p:.4f}  n={len(pair)}  -> {tag}")
    return rho, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--csv-dir", default=None)
    args = ap.parse_args()

    social, have_labels = load_social_monthly()
    if args.csv_dir:
        trends = load_trends_monthly_csv(args.csv_dir)
    elif args.auto:
        trends = load_trends_monthly_auto()
    else:
        raise SystemExit("Use --auto or --csv-dir trends_csv")
    print(f"[trends] months={len(trends)}")

    months = sorted(set(social) & set(trends))
    if len(months) < 8:
        raise SystemExit(f"Only {len(months)} overlapping months; need >=8.")
    t = [trends[m] for m in months]
    vol = [social[m]["n"] for m in months]
    mis = [social[m]["misinfo_rate"] for m in months]
    stg = [social[m]["mean_stigma"] for m in months]

    print(f"\n=== TRENDS x BLUESKY  (MONTHLY, {len(months)} months "
          f"{months[0]}..{months[-1]}) ===\n")

    print("RAW levels (can be inflated by shared upward trend):")
    report("trends ~ volume", t, vol)
    if have_labels:
        report("trends ~ misinfo_rate", t, mis)
        report("trends ~ mean_stigma", t, stg)

    print("\nDETRENDED (month-to-month change; controls for shared growth) "
          "<-- TRUST THIS:")
    rv = report("d.trends ~ d.volume", diff(t), diff(vol))
    rm = rs = None
    if have_labels:
        rm = report("d.trends ~ d.misinfo_rate", diff(t), diff(mis))
        rs = report("d.trends ~ d.mean_stigma", diff(t), diff(stg))

    print("\nVERDICT (based on DETRENDED results):")
    sig = lambda r: r is not None and abs(r[0]) >= 0.2 and r[1] < 0.05
    if not have_labels:
        if sig(rv):
            print("  Detrended trends<->VOLUME co-move beyond shared growth.")
            print("  -> Real signal at volume level. Re-run unlocks rate & stigma,")
            print("     which are the growth-independent, publishable targets.")
        else:
            print("  After detrending, trends<->volume link is weak/null.")
            print("  -> The raw correlation was mostly shared growth. The convergence")
            print("     story is NOT supported even at volume level. Reconsider before")
            print("     spending on the re-run.")
    else:
        if sig(rm) or sig(rs):
            print("  Detrended search interest co-moves with the misinformation/stigma")
            print("  RATE beyond shared growth. This is the strong, honest result.")
        elif sig(rv):
            print("  Search co-moves with VOLUME but NOT with misinfo/stigma rate.")
            print("  Honest finding: search tracks how MUCH is posted, not how much of")
            print("  it is false or stigmatizing. Publishable, but a modest claim.")
        else:
            print("  No detrended co-movement. Do not build the convergence story.")

    print("\nNOTE: lead-lag intentionally omitted here - with short monthly series")
    print("a flat all-lag correlation reflects shared trend, not genuine lead. If")
    print("needed, run lead-lag on the DETRENDED series only, and report cautiously.")


if __name__ == "__main__":
    main()
