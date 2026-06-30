"""
05_validation_and_stats.py

Stage 05 - statistical validation for the menstrual/reproductive-health
misinformation + stigma study. Two jobs, run as subcommands:

  python 05_validation_and_stats.py stats
      Inferential tests on the existing classified claims:
        - RQ2: Kruskal-Wallis (stigma_score across veracity groups)
               + Spearman (stigma vs ordinal veracity) + epsilon-squared
        - RQ2 pairwise: Mann-Whitney U (SUPPORTED vs each misinfo tier)
                        with Holm correction + rank-biserial effect size
        - RQ3: Kruskal-Wallis (stigma_score across conditions)
        - RQ4: Spearman (stigma vs engagement) - reported honestly
      Writes data/04_evaluation/results/inferential_stats.json and prints a summary.

  python 05_validation_and_stats.py make-sheet [N]
      Samples N claims (default 150), stratified by veracity, and writes a BLIND
      human-coding sheet:
        data/05_validation/validation_sheet.csv
      The LLM labels are stored separately (NOT shown to the coder) in
        data/05_validation/validation_key.csv
      You (or a second coder) fill in the human columns, then run 'kappa'.

  python 05_validation_and_stats.py kappa
      Reads the filled validation_sheet.csv + the hidden key and computes
      Cohen's kappa (veracity) and quadratic-weighted kappa + ICC-style
      agreement (stigma), plus confusion matrices.
      Writes data/05_validation/validation_results.json

Dependencies: numpy, scipy, pandas.  (pip install scipy pandas)
No LLM calls, no cost. Safe to re-run.
"""

import sys
import json
import csv
import random
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np

import config as C

VALID_DIR = C.DATA / "05_validation"
SHEET = VALID_DIR / "validation_sheet.csv"
KEY = VALID_DIR / "validation_key.csv"
VALID_RESULTS = VALID_DIR / "validation_results.json"
STATS_OUT = C.RESULTS_DIR / "inferential_stats.json"

# Ordinal scale for veracity (false-ness increases left to right).
VERACITY_ORDINAL = {
    "SUPPORTED": 0,
    "UNSUPPORTED": 1,
    "EXAGGERATED": 2,
    "CONTRADICTED": 3,
    "DANGEROUS": 4,
}


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def load_claims():
    if not C.CLASSIFIED_CLAIMS.exists():
        raise SystemExit(f"No classified claims at {C.CLASSIFIED_CLAIMS}")
    return [json.loads(l) for l in open(C.CLASSIFIED_CLAIMS, encoding="utf-8")
            if l.strip()]


def veracity(c):
    return c.get("classification", {}).get("llm_grounded", {}).get("veracity")


def stigma(c):
    s = c.get("classification", {}).get("stigma_sentiment", {}).get("stigma_score")
    return s if isinstance(s, (int, float)) else None


# --------------------------------------------------------------------------- #
# STATS subcommand
# --------------------------------------------------------------------------- #
def epsilon_squared(H, n, k):
    """Effect size for Kruskal-Wallis: eps^2 = (H - k + 1) / (n - k)."""
    if n - k <= 0:
        return float("nan")
    return (H - k + 1) / (n - k)


def rank_biserial_from_u(U, n1, n2):
    """Rank-biserial correlation effect size for Mann-Whitney U."""
    return 1.0 - (2.0 * U) / (n1 * n2)


def holm_correction(pvals):
    """Holm-Bonferroni. Returns adjusted p-values in original order."""
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adj = [0.0] * m
    prev = 0.0
    for rank, i in enumerate(idx):
        val = (m - rank) * pvals[i]
        val = min(1.0, max(val, prev))
        adj[i] = val
        prev = val
    return adj


def run_stats():
    from scipy import stats

    claims = load_claims()
    out = {"n_claims": len(claims)}

    # ---- RQ2: stigma across veracity groups -------------------------------
    groups, labels = [], []
    for label in C.VERACITY_LABELS:
        vals = [stigma(c) for c in claims
                if veracity(c) == label and stigma(c) is not None]
        if vals:
            groups.append(vals)
            labels.append(label)
    H, p = stats.kruskal(*groups)
    n_total = sum(len(g) for g in groups)
    out["RQ2_kruskal_wallis"] = {
        "groups": labels,
        "group_n": {l: len(g) for l, g in zip(labels, groups)},
        "group_mean_stigma": {l: float(np.mean(g)) for l, g in zip(labels, groups)},
        "H": float(H), "p_value": float(p),
        "epsilon_squared": float(epsilon_squared(H, n_total, len(groups))),
        "df": len(groups) - 1,
    }

    # ---- RQ2: Spearman stigma vs ordinal veracity -------------------------
    xs, ys = [], []
    for c in claims:
        v = veracity(c); s = stigma(c)
        if v in VERACITY_ORDINAL and s is not None:
            xs.append(VERACITY_ORDINAL[v]); ys.append(s)
    rho, p_rho = stats.spearmanr(xs, ys)
    out["RQ2_spearman_stigma_vs_veracity"] = {
        "n": len(xs), "rho": float(rho), "p_value": float(p_rho),
        "note": "veracity coded 0=SUPPORTED..4=DANGEROUS; positive rho = "
                "more stigma as claims become less true",
    }

    # ---- RQ2: pairwise SUPPORTED vs each misinfo tier ---------------------
    support_vals = [stigma(c) for c in claims
                    if veracity(c) == "SUPPORTED" and stigma(c) is not None]
    pairwise, raw_p = [], []
    for label in ["UNSUPPORTED", "EXAGGERATED", "CONTRADICTED", "DANGEROUS"]:
        vals = [stigma(c) for c in claims
                if veracity(c) == label and stigma(c) is not None]
        if len(vals) < 3:
            pairwise.append({"comparison": f"SUPPORTED_vs_{label}",
                             "n2": len(vals), "skipped": "n<3"})
            continue
        U, p_u = stats.mannwhitneyu(support_vals, vals, alternative="two-sided")
        rb = rank_biserial_from_u(U, len(support_vals), len(vals))
        pairwise.append({
            "comparison": f"SUPPORTED_vs_{label}",
            "n1": len(support_vals), "n2": len(vals),
            "U": float(U), "p_raw": float(p_u),
            "rank_biserial": float(rb),
        })
        raw_p.append(p_u)
    adj = holm_correction(raw_p)
    ai = 0
    for pw in pairwise:
        if "p_raw" in pw:
            pw["p_holm"] = float(adj[ai]); ai += 1
    out["RQ2_pairwise_vs_supported"] = pairwise

    # ---- RQ3: stigma across conditions (top 6 by n) -----------------------
    by_cond = defaultdict(list)
    for c in claims:
        cond = (c.get("target_condition", "") or "general health").lower()
        s = stigma(c)
        if s is not None:
            by_cond[cond].append(s)
    top = [k for k, _ in Counter(
        {k: len(v) for k, v in by_cond.items()}).most_common(6)]
    cond_groups = [by_cond[k] for k in top if len(by_cond[k]) >= 5]
    cond_labels = [k for k in top if len(by_cond[k]) >= 5]
    if len(cond_groups) >= 2:
        Hc, pc = stats.kruskal(*cond_groups)
        nc = sum(len(g) for g in cond_groups)
        out["RQ3_kruskal_wallis_conditions"] = {
            "conditions": cond_labels,
            "condition_n": {l: len(g) for l, g in zip(cond_labels, cond_groups)},
            "condition_mean_stigma": {l: float(np.mean(g))
                                      for l, g in zip(cond_labels, cond_groups)},
            "H": float(Hc), "p_value": float(pc),
            "epsilon_squared": float(epsilon_squared(Hc, nc, len(cond_groups))),
        }

    # ---- RQ4: stigma vs engagement (honest) -------------------------------
    xs, ys = [], []
    for c in claims:
        s = stigma(c); e = c.get("post_score")
        if s is not None and isinstance(e, (int, float)):
            xs.append(s); ys.append(e)
    if len(xs) > 2:
        rho4, p4 = stats.spearmanr(xs, ys)
        out["RQ4_spearman_stigma_vs_engagement"] = {
            "n": len(xs), "spearman_rho": float(rho4), "p_value": float(p4),
            "interpretation": "weak/with caution; Bluesky like counts are sparse",
        }

    STATS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(STATS_OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # Console summary
    k2 = out["RQ2_kruskal_wallis"]; sp = out["RQ2_spearman_stigma_vs_veracity"]
    print("\n=== RQ2: does misinformation carry more stigma? ===")
    print(f"  Kruskal-Wallis H={k2['H']:.2f}, df={k2['df']}, "
          f"p={k2['p_value']:.2e}, eps^2={k2['epsilon_squared']:.3f}")
    print(f"  Spearman stigma vs veracity rho={sp['rho']:.3f}, p={sp['p_value']:.2e}")
    print("  group means:", {k: round(v, 3)
                             for k, v in k2["group_mean_stigma"].items()})
    print("\n  pairwise vs SUPPORTED (Holm-adjusted):")
    for pw in out["RQ2_pairwise_vs_supported"]:
        if "p_holm" in pw:
            print(f"    {pw['comparison']}: p={pw['p_holm']:.2e}, "
                  f"rank-biserial={pw['rank_biserial']:.2f} (n2={pw['n2']})")
        else:
            print(f"    {pw['comparison']}: skipped ({pw.get('skipped')})")
    if "RQ3_kruskal_wallis_conditions" in out:
        k3 = out["RQ3_kruskal_wallis_conditions"]
        print(f"\n=== RQ3: stigma differs by condition? ===")
        print(f"  Kruskal-Wallis H={k3['H']:.2f}, p={k3['p_value']:.2e}, "
              f"eps^2={k3['epsilon_squared']:.3f}")
    if "RQ4_spearman_stigma_vs_engagement" in out:
        k4 = out["RQ4_spearman_stigma_vs_engagement"]
        print(f"\n=== RQ4: stigma vs engagement (honest) ===")
        print(f"  Spearman rho={k4['spearman_rho']:.3f}, p={k4['p_value']:.2e} "
              f"(weak; sparse like counts)")
    print(f"\nWrote {STATS_OUT}")


# --------------------------------------------------------------------------- #
# MAKE-SHEET subcommand
# --------------------------------------------------------------------------- #
def make_sheet(n=150):
    claims = load_claims()
    random.seed(42)

    # Stratify by veracity so small classes (e.g. DANGEROUS) are represented.
    by_label = defaultdict(list)
    for c in claims:
        v = veracity(c)
        if v:
            by_label[v].append(c)

    present = [l for l in C.VERACITY_LABELS if by_label[l]]
    per = max(1, n // len(present))
    sample = []
    for label in present:
        pool = by_label[label][:]
        random.shuffle(pool)
        sample.extend(pool[:per])
    # top up to n if rounding left us short
    if len(sample) < n:
        rest = [c for c in claims if c not in sample]
        random.shuffle(rest)
        sample.extend(rest[:n - len(sample)])
    random.shuffle(sample)
    sample = sample[:n]

    VALID_DIR.mkdir(parents=True, exist_ok=True)

    # Blind coding sheet: NO llm labels shown. Coder fills human_* columns.
    with open(SHEET, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "product", "claimed_effect", "target_condition",
                    "quote",
                    "human_veracity (SUPPORTED/UNSUPPORTED/EXAGGERATED/"
                    "CONTRADICTED/DANGEROUS)",
                    "human_stigma_0_to_1"])
        for i, c in enumerate(sample):
            w.writerow([i, c.get("product", ""), c.get("claimed_effect", ""),
                        c.get("target_condition", ""),
                        (c.get("verbatim_quote", "") or "")[:300], "", ""])

    # Hidden key: the LLM labels, matched by row_id. Not shown to the coder.
    with open(KEY, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "llm_veracity", "llm_stigma"])
        for i, c in enumerate(sample):
            w.writerow([i, veracity(c), stigma(c)])

    print(f"Wrote blind coding sheet: {SHEET}  ({len(sample)} rows)")
    print(f"Wrote hidden key:         {KEY}")
    print("\nNext: open validation_sheet.csv, fill in the two human_* columns")
    print("for every row (ideally a second coder too), save, then run:")
    print("  python 05_validation_and_stats.py kappa")


# --------------------------------------------------------------------------- #
# KAPPA subcommand
# --------------------------------------------------------------------------- #
def cohen_kappa(y1, y2, labels):
    """Unweighted Cohen's kappa for categorical labels."""
    idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    m = np.zeros((k, k))
    for a, b in zip(y1, y2):
        if a in idx and b in idx:
            m[idx[a], idx[b]] += 1
    n = m.sum()
    if n == 0:
        return float("nan"), m
    po = np.trace(m) / n
    row = m.sum(axis=1) / n
    col = m.sum(axis=0) / n
    pe = float(np.sum(row * col))
    kappa = (po - pe) / (1 - pe) if pe != 1 else float("nan")
    return float(kappa), m


def quadratic_weighted_kappa(y1, y2, ordinal_map):
    """QWK for ordinal labels (veracity)."""
    a = np.array([ordinal_map[x] for x in y1])
    b = np.array([ordinal_map[x] for x in y2])
    k = max(ordinal_map.values()) + 1
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[x, y] += 1
    W = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            W[i, j] = (i - j) ** 2 / (k - 1) ** 2
    act = np.histogram(a, bins=k, range=(0, k))[0]
    pred = np.histogram(b, bins=k, range=(0, k))[0]
    E = np.outer(act, pred) / O.sum()
    num = (W * O).sum(); den = (W * E).sum()
    return float(1 - num / den) if den else float("nan")


def run_kappa():
    import pandas as pd
    if not SHEET.exists() or not KEY.exists():
        raise SystemExit("Run 'make-sheet' first, then fill in the human columns.")

    sheet = pd.read_csv(SHEET)
    key = pd.read_csv(KEY)
    df = sheet.merge(key, on="row_id")

    vcol = [c for c in df.columns if c.startswith("human_veracity")][0]
    scol = [c for c in df.columns if c.startswith("human_stigma")][0]

    coded = df[df[vcol].notna() & (df[vcol].astype(str).str.strip() != "")]
    if len(coded) < 10:
        raise SystemExit(f"Only {len(coded)} rows coded. Fill in more first.")

    hv = coded[vcol].astype(str).str.strip().str.upper().tolist()
    lv = coded["llm_veracity"].astype(str).str.strip().str.upper().tolist()

    kappa, cm = cohen_kappa(lv, hv, C.VERACITY_LABELS)
    qwk = quadratic_weighted_kappa(
        [x for x in lv], [x for x in hv], VERACITY_ORDINAL) \
        if all(x in VERACITY_ORDINAL for x in lv + hv) else None

    # Stigma agreement (continuous): Spearman + mean abs error
    res = {"n_coded": int(len(coded)),
           "veracity_cohen_kappa": kappa,
           "veracity_quadratic_weighted_kappa": qwk,
           "veracity_confusion_matrix": {
               "labels": C.VERACITY_LABELS, "matrix": cm.astype(int).tolist()},
           "veracity_exact_agreement":
               float(np.mean([a == b for a, b in zip(hv, lv)]))}

    try:
        hs = pd.to_numeric(coded[scol], errors="coerce")
        ls = pd.to_numeric(coded["llm_stigma"], errors="coerce")
        mask = hs.notna() & ls.notna()
        if mask.sum() >= 10:
            from scipy import stats
            rho, p = stats.spearmanr(hs[mask], ls[mask])
            res["stigma_spearman_rho"] = float(rho)
            res["stigma_spearman_p"] = float(p)
            res["stigma_mean_abs_error"] = float(np.mean(np.abs(hs[mask] - ls[mask])))
            res["stigma_n"] = int(mask.sum())
    except Exception as e:
        res["stigma_error"] = str(e)

    with open(VALID_RESULTS, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print(f"\n=== Human validation (n={res['n_coded']}) ===")
    print(f"  Veracity Cohen's kappa:           {kappa:.3f}")
    if qwk is not None:
        print(f"  Veracity quadratic-weighted kappa: {qwk:.3f}")
    print(f"  Veracity exact agreement:          {res['veracity_exact_agreement']:.1%}")
    if "stigma_spearman_rho" in res:
        print(f"  Stigma Spearman rho:               {res['stigma_spearman_rho']:.3f} "
              f"(p={res['stigma_spearman_p']:.2e}, n={res['stigma_n']})")
        print(f"  Stigma mean abs error:             {res['stigma_mean_abs_error']:.3f}")
    print("\n  kappa guide: <0.20 poor, 0.21-0.40 fair, 0.41-0.60 moderate,")
    print("               0.61-0.80 substantial, 0.81-1.0 almost perfect")
    print(f"\nWrote {VALID_RESULTS}")


# --------------------------------------------------------------------------- #
def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("stats", "make-sheet", "kappa"):
        print("Usage:")
        print("  python 05_validation_and_stats.py stats")
        print("  python 05_validation_and_stats.py make-sheet [N]")
        print("  python 05_validation_and_stats.py kappa")
        raise SystemExit(1)
    cmd = sys.argv[1]
    if cmd == "stats":
        run_stats()
    elif cmd == "make-sheet":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 150
        make_sheet(n)
    elif cmd == "kappa":
        run_kappa()


if __name__ == "__main__":
    main()
