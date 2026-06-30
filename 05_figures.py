"""
04_figures.py

Stage 04 - descriptive stats and publication figures answering RQ1-RQ7.

Input : data/03_classified/classified_claims.jsonl
Output: data/04_evaluation/results/
  descriptive_stats.json
  fig1_veracity_distribution.png        (RQ1)
  fig2_risk_category_distribution.png   (RQ1)
  fig3_veracity_vs_stigma.png           (RQ2)  <- core novel result
  fig4_condition_stigma_heatmap.png     (RQ3)
  fig5_stigma_vs_engagement.png         (RQ4)
  fig6_llm_vs_baseline_agreement.png    (RQ5)
  fig7_language_comparison.png          (RQ6, only if >1 language present)
  examples_<VERACITY>.txt

Plot style: matplotlib Agg, serif fonts, white background, 200 DPI.
"""

import json
from collections import defaultdict, Counter

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


def load():
    return [json.loads(l) for l in open(C.CLASSIFIED_CLAIMS, encoding="utf-8")
            if l.strip()]


def gv(c):  # grounded veracity label
    return c.get("classification", {}).get("llm_grounded", {}).get("veracity", "UNSUPPORTED")


def stig(c):
    return c.get("classification", {}).get("stigma_sentiment", {})


def norm_condition(cl):
    """Normalize target_condition so label variants collapse to one group.
    Merges 'pcos (polycystic ovary syndrome)' and similar into 'pcos'."""
    cond = (cl.get("target_condition", "general health") or "general health").lower().strip()
    if "pcos" in cond or "polycystic" in cond:
        return "pcos"
    if "endometrios" in cond:
        return "endometriosis"
    if "infertil" in cond:
        return "infertility"
    if cond.startswith("fertility") or cond == "fertility/conception" or "conception" in cond:
        return "fertility"
    return cond


def fig1_veracity(claims, out):
    counts = Counter(gv(c) for c in claims)
    labels = [l for l in C.VERACITY_LABELS if counts.get(l, 0)]
    vals = [counts[l] for l in labels]
    plt.figure(figsize=(9, 6))
    plt.bar(labels, vals, color=GRAD[:len(labels)], edgecolor="black")
    plt.title("Veracity distribution of reproductive-health claims")
    plt.ylabel("Number of claims")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out / "fig1_veracity_distribution.png")
    plt.close()


def fig2_categories(claims, out):
    counts = Counter(c.get("risk_category", "other") for c in claims)
    labels = [l for l in C.RISK_CATEGORIES if counts.get(l, 0)]
    vals = [counts[l] for l in labels]
    plt.figure(figsize=(10, 6))
    plt.barh(labels, vals, color=plt.cm.viridis(np.linspace(0.15, 0.85, len(labels))),
             edgecolor="black")
    plt.title("Claims by risk category")
    plt.xlabel("Number of claims")
    plt.tight_layout()
    plt.savefig(out / "fig2_risk_category_distribution.png")
    plt.close()


def fig3_veracity_vs_stigma(claims, out):
    """RQ2: mean stigma_score per veracity label."""
    by = defaultdict(list)
    for c in claims:
        s = stig(c).get("stigma_score")
        if isinstance(s, (int, float)):
            by[gv(c)].append(s)
    labels = [l for l in C.VERACITY_LABELS if by.get(l)]
    means = [np.mean(by[l]) for l in labels]
    errs = [np.std(by[l]) / max(1, np.sqrt(len(by[l]))) for l in labels]
    plt.figure(figsize=(9, 6))
    plt.bar(labels, means, yerr=errs, capsize=5,
            color=GRAD[:len(labels)], edgecolor="black")
    plt.title("Mean stigma score by veracity label")
    plt.ylabel("Mean stigma score (0-1)")
    plt.ylim(0, 1)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out / "fig3_veracity_vs_stigma.png")
    plt.close()


def fig4_condition_heatmap(claims, out, top_n=6):
    """RQ3: condition x stigma-dimension mean intensity.
    Conditions are normalized (norm_condition) and restricted to the same
    top_n most frequent groups used by the inferential test, so the figure
    and the Kruskal-Wallis test describe an identical set of conditions."""
    conds = [c for c, _ in Counter(
        norm_condition(cl) for cl in claims).most_common(top_n)]
    dims = C.STIGMA_DIMENSIONS
    mat = np.zeros((len(conds), len(dims)))
    cnt = np.zeros((len(conds), len(dims)))
    cond_idx = {c: i for i, c in enumerate(conds)}
    for cl in claims:
        cond = norm_condition(cl)
        if cond not in cond_idx:
            continue
        d = stig(cl).get("dimensions", {})
        for j, dim in enumerate(dims):
            v = d.get(dim)
            if isinstance(v, (int, float)):
                mat[cond_idx[cond], j] += v
                cnt[cond_idx[cond], j] += 1
    with np.errstate(invalid="ignore"):
        mat = np.where(cnt > 0, mat / np.maximum(cnt, 1), 0)
    plt.figure(figsize=(10, 7))
    im = plt.imshow(mat, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, label="Mean intensity (0-1)")
    plt.xticks(range(len(dims)), dims, rotation=30, ha="right")
    plt.yticks(range(len(conds)), conds)
    plt.title("Stigma framing by condition")
    plt.tight_layout()
    plt.savefig(out / "fig4_condition_stigma_heatmap.png")
    plt.close()


def fig5_stigma_engagement(claims, out):
    """RQ4: stigma score vs post score (engagement)."""
    xs, ys = [], []
    for c in claims:
        s = stig(c).get("stigma_score")
        e = c.get("post_score")
        if isinstance(s, (int, float)) and isinstance(e, (int, float)):
            xs.append(s); ys.append(max(0, e))
    plt.figure(figsize=(9, 6))
    if xs:
        plt.scatter(xs, ys, alpha=0.4, color=GRAD[3], edgecolor="black", linewidth=0.3)
        if len(xs) > 2 and np.std(xs) > 0:
            b, a = np.polyfit(xs, ys, 1)
            xr = np.linspace(min(xs), max(xs), 50)
            plt.plot(xr, a + b * xr, color=GRAD[0], lw=2,
                     label=f"slope={b:.1f}")
            r = np.corrcoef(xs, ys)[0, 1]
            plt.legend(title=f"Pearson r={r:.2f}")
    plt.title("Stigma score vs engagement (post score)")
    plt.xlabel("Stigma score (0-1)")
    plt.ylabel("Post score (likes)")
    plt.tight_layout()
    plt.savefig(out / "fig5_stigma_vs_engagement.png")
    plt.close()


def fig6_agreement(claims, out):
    """RQ5: LLM vs keyword baseline agreement on veracity."""
    agree = sum(1 for c in claims
                if gv(c) == c.get("classification", {})
                .get("keyword_baseline", {}).get("veracity"))
    total = len(claims)
    disagree = total - agree
    plt.figure(figsize=(7, 6))
    plt.bar(["Agree", "Disagree"], [agree, disagree],
            color=[GRAD[1], GRAD[4]], edgecolor="black")
    plt.title(f"LLM-grounded vs keyword baseline\nagreement = {agree}/{total} "
              f"({100*agree/max(1,total):.0f}%)")
    plt.ylabel("Number of claims")
    plt.tight_layout()
    plt.savefig(out / "fig6_llm_vs_baseline_agreement.png")
    plt.close()


def fig7_language(claims, out):
    """RQ6: stigma by language, only if >1 language present."""
    langs = sorted({c.get("language", "en") for c in claims})
    if len(langs) < 2:
        return False
    means, errs = [], []
    for lg in langs:
        vals = [stig(c).get("stigma_score") for c in claims
                if c.get("language") == lg
                and isinstance(stig(c).get("stigma_score"), (int, float))]
        means.append(np.mean(vals) if vals else 0)
        errs.append(np.std(vals) / max(1, np.sqrt(len(vals))) if vals else 0)
    plt.figure(figsize=(8, 6))
    plt.bar(langs, means, yerr=errs, capsize=5,
            color=GRAD[:len(langs)], edgecolor="black")
    plt.title("Stigma score by language/culture")
    plt.ylabel("Mean stigma score (0-1)")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out / "fig7_language_comparison.png")
    plt.close()
    return True


def write_examples(claims, out):
    by = defaultdict(list)
    for c in claims:
        by[gv(c)].append(c)
    for label, items in by.items():
        items.sort(key=lambda c: c.get("classification", {})
                   .get("llm_grounded", {}).get("confidence", 0), reverse=True)
        with open(out / f"examples_{label}.txt", "w", encoding="utf-8") as f:
            for c in items[:20]:
                f.write(f"[{c.get('risk_category')}] {c.get('product')} -> "
                        f"{c.get('claimed_effect')}\n")
                f.write(f"  stigma={stig(c).get('stigma_score')} "
                        f"frame={stig(c).get('dominant_frame')}\n")
                f.write(f"  quote: {c.get('verbatim_quote','')[:200]}\n\n")


def descriptive_stats(claims):
    return {
        "n_claims": len(claims),
        "n_languages": len({c.get("language", "en") for c in claims}),
        "veracity_counts": dict(Counter(gv(c) for c in claims)),
        "risk_category_counts": dict(Counter(c.get("risk_category", "other")
                                             for c in claims)),
        "dominant_frame_counts": dict(Counter(
            stig(c).get("dominant_frame", "neutral") for c in claims)),
        "mean_stigma_by_veracity": {
            v: float(np.mean([stig(c).get("stigma_score", 0) for c in claims
                              if gv(c) == v]
                             or [0]))
            for v in C.VERACITY_LABELS
        },
    }


def main():
    out = C.RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    if not C.CLASSIFIED_CLAIMS.exists():
        raise SystemExit(f"No input at {C.CLASSIFIED_CLAIMS}")
    claims = load()
    if not claims:
        raise SystemExit("No claims to plot.")

    fig1_veracity(claims, out)
    fig2_categories(claims, out)
    fig3_veracity_vs_stigma(claims, out)
    fig4_condition_heatmap(claims, out)
    fig5_stigma_engagement(claims, out)
    fig6_agreement(claims, out)
    has_lang = fig7_language(claims, out)
    write_examples(claims, out)

    with open(out / "descriptive_stats.json", "w", encoding="utf-8") as f:
        json.dump(descriptive_stats(claims), f, indent=2, ensure_ascii=False)

    figs = 6 + (1 if has_lang else 0)
    print(f"Done. {len(claims)} claims, {figs} figures -> {out}")


if __name__ == "__main__":
    main()
