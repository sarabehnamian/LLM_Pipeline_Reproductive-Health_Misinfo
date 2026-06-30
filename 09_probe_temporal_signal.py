"""
probe_temporal_signal.py

FREE pre-flight probe. No API calls, no LLM. Answers one question before you
spend any credits on a re-run:

    Is there ANY temporal signal in the raw collected posts, or is only the
    post VOLUME moving (which just tracks Bluesky's growth, not discourse)?

Reads data/00_raw/all_posts.jsonl (the 1,580 dated posts already collected).
For each ISO week it computes, from raw text only:
    - n posts
    - stigma_kw_rate : fraction of posts containing >=1 stigma lexicon word
    - claim_kw_rate  : fraction containing >=1 strong claim/cure word
    - mean_stigma_hits per post (intensity, not just presence)

Then it does the honest test: a Spearman trend of each RATE against week index
over the dense window. If the rates are flat (rho ~ 0, p large) while only
volume rises, the time-series angle is being forced and you should NOT build a
temporal paper on this data. If a rate genuinely trends or spikes, there may be
a real signal worth the paid re-run.

This uses surface keywords (same lexicons as config), so it is a LOWER BOUND on
signal, not a substitute for the LLM scoring. Flat here => almost certainly flat
after the expensive run. Signal here => worth confirming with the full pipeline.
"""

import json
from collections import defaultdict
from datetime import date, datetime, timezone

import config as C

LO = date(2023, 1, 1)
MIN_WEEK = 5   # dense-window threshold, same spirit as stage 06

# flatten the stigma lexicon (all dimensions incl. empowerment) for presence test
STIGMA_WORDS = sorted({w.lower() for words in C.STIGMA_KEYWORDS.values() for w in words})
CLAIM_WORDS = ["cure", "cures", "reverse", "heal", "miracle", "guaranteed",
               "100%", "detox", "flush out", "stop taking", "instead of",
               "permanently", "completely", "always", "never fails"]


def parse_day(s):
    if not isinstance(s, str) or not s.strip():
        return None
    t = s.strip().replace("Z", "+00:00")
    try:
        d = datetime.fromisoformat(t).date()
    except ValueError:
        try:
            d = datetime.fromisoformat(t[:10]).date()
        except ValueError:
            return None
    today = datetime.now(timezone.utc).date()
    return d if LO <= d <= today else None


def week_start(d):
    return d.fromordinal(d.toordinal() - d.weekday())


def spearman(x, y):
    """Dependency-free Spearman rho + a rough two-sided p via normal approx."""
    import math
    n = len(x)
    if n < 4:
        return float("nan"), float("nan")
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return float("nan"), float("nan")
    rho = cov / math.sqrt(vx * vy)
    # t approximation
    t = rho * math.sqrt((n - 2) / max(1e-12, 1 - rho ** 2))
    # crude two-sided p from normal tail
    p = math.erfc(abs(t) / math.sqrt(2))
    return rho, p


def main():
    src = C.ALL_POSTS
    if not src.exists():
        raise SystemExit(f"No raw posts at {src}. Run stage 00 first.")

    posts = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    by_week = defaultdict(list)
    n_bad = 0
    for p in posts:
        d = parse_day(p.get("created_at"))
        if d is None:
            n_bad += 1
            continue
        by_week[week_start(d)].append((p.get("text", "") or "").lower())

    weeks = sorted(by_week)
    # dense window
    dense = [w for w in weeks if len(by_week[w]) >= MIN_WEEK]
    if not dense:
        raise SystemExit("No dense weeks; corpus too sparse for any temporal claim.")
    start = dense[0]
    window = [w for w in weeks if w >= start]

    rows = []
    for w in window:
        texts = by_week[w]
        n = len(texts)
        if n == 0:
            continue
        stig_present = sum(1 for t in texts if any(kw in t for kw in STIGMA_WORDS))
        claim_present = sum(1 for t in texts if any(kw in t for kw in CLAIM_WORDS))
        stig_hits = sum(sum(1 for kw in STIGMA_WORDS if kw in t) for t in texts)
        rows.append({
            "week": w, "n": n,
            "stigma_kw_rate": stig_present / n,
            "claim_kw_rate": claim_present / n,
            "mean_stigma_hits": stig_hits / n,
        })

    idx = list(range(len(rows)))
    vol = [r["n"] for r in rows]
    stig_rate = [r["stigma_kw_rate"] for r in rows]
    claim_rate = [r["claim_kw_rate"] for r in rows]
    intensity = [r["mean_stigma_hits"] for r in rows]

    print(f"=== FREE TEMPORAL-SIGNAL PROBE (raw text only) ===")
    print(f"posts={len(posts)}  bad_dates={n_bad}  "
          f"dense weeks={len(rows)}  window={rows[0]['week']}..{rows[-1]['week']}\n")

    print(f"{'series':>18}  {'Spearman_rho':>12}  {'p':>10}  reading")
    for name, series in [("volume", vol),
                         ("stigma_kw_rate", stig_rate),
                         ("claim_kw_rate", claim_rate),
                         ("stigma_intensity", intensity)]:
        rho, p = spearman(idx, series)
        if name == "volume":
            note = "(expected to rise = Bluesky growth, NOT a discourse signal)"
        else:
            if p == p and p < 0.05 and abs(rho) >= 0.2:
                note = "<-- real trend: signal worth the paid run"
            else:
                note = "flat: no trend in this measure"
        print(f"{name:>18}  {rho:>12.3f}  {p:>10.4f}  {note}")

    # simple range check on rates (is there even spread to model?)
    def spread(s): return max(s) - min(s)
    print("\nRate spread (max-min over weeks):")
    print(f"  stigma_kw_rate : {spread(stig_rate):.3f}")
    print(f"  claim_kw_rate  : {spread(claim_rate):.3f}")

    print("\nVERDICT GUIDE:")
    print("  * If ONLY 'volume' moves and the three rate series are flat")
    print("    (|rho|<0.2 or p>0.05, small spread) -> the temporal angle is")
    print("    forced. Do NOT build a time-series paper on this; submit the")
    print("    cross-sectional paper to a fitting venue instead.")
    print("  * If a rate series genuinely trends or has clear spread -> there")
    print("    may be real signal; the paid re-run + stage 06 is justified.")
    print("  NOTE: this is surface-keyword signal (a lower bound). The LLM")
    print("  scoring could reveal a bit more, but flat here usually = flat there.")


if __name__ == "__main__":
    main()
