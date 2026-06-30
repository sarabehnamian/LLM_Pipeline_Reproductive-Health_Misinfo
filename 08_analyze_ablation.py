"""
analyze_ablation.py

Quantifies what the retrieval layer contributes by comparing the grounded
classifier (classification["llm_grounded"]) against the no-evidence ablation
(classification["llm_no_evidence"]) on the same 937 claims.

Reports:
  1. Overall agreement between grounded and ungrounded labels.
  2. Confusion matrix (grounded rows x no-evidence cols).
  3. Directional shifts: where ungrounded inflates SUPPORTED or loses
     CONTRADICTED/DANGEROUS (the safety-relevant failures of un-grounded LLMs).
  4. Label distribution side by side.
  5. If a human-validated subset exists (validation_sheet.csv with filled
     human_veracity), per-method accuracy vs human on that subset -> the
     accuracy claim, not just a divergence claim.

No API calls.
"""

import csv
import json
import os

import config as C
from collections import Counter, defaultdict

LABELS = ["SUPPORTED", "UNSUPPORTED", "EXAGGERATED", "CONTRADICTED", "DANGEROUS"]
ABLATION_PATH = str(C.CLASSIFIED_DIR / "classified_ablation.jsonl")
GROUNDED_PATH = str(C.CLASSIFIED_CLAIMS)   # classified_claims.jsonl (real pipeline output)
KEY_PATH = "validation_key.csv"          # row_id -> llm label (for subset mapping)
SHEET_PATH = "validation_sheet.csv"      # row_id -> human label (may be empty)

# Map the 5 labels to a coarse "trustworthy vs problematic" axis for one summary.
PROBLEMATIC = {"UNSUPPORTED", "EXAGGERATED", "CONTRADICTED", "DANGEROUS"}


def load():
    """Merge by line order: both files derive from the same claims_with_evidence
    in identical order, so position i is the same claim in both."""
    ablation = []
    with open(ABLATION_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ablation.append(json.loads(line))
    grounded = []
    with open(GROUNDED_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                grounded.append(json.loads(line))
    if len(ablation) != len(grounded):
        print(f"WARNING: line counts differ (ablation={len(ablation)}, "
              f"grounded={len(grounded)}); zipping to the shorter.")
    # attach the grounded label onto each ablation record by position
    for ab, gr in zip(ablation, grounded):
        gl = gr.get("classification", {}).get("llm_grounded")
        if gl:
            ab.setdefault("classification", {})["llm_grounded"] = gl
    return ablation


def lab(claim, key):
    return claim.get("classification", {}).get(key, {}).get("veracity")


def main():
    if not os.path.exists(ABLATION_PATH):
        raise SystemExit(f"Run 03c_ablation_no_evidence.py first ({ABLATION_PATH} missing).")
    claims = load()
    paired = [(lab(c, "llm_grounded"), lab(c, "llm_no_evidence")) for c in claims]
    paired = [(g, n) for g, n in paired if g in LABELS and n in LABELS]
    total = len(paired)

    # 1. Agreement
    agree = sum(g == n for g, n in paired)
    print(f"=== Grounded vs No-Evidence ablation (n={total}) ===")
    print(f"Exact label agreement: {agree}/{total} = {agree/total:.1%}\n")

    # 2. Confusion matrix
    cm = defaultdict(lambda: Counter())
    for g, n in paired:
        cm[g][n] += 1
    print("Confusion matrix (rows=GROUNDED, cols=NO-EVIDENCE):")
    print("            " + "".join(f"{l[:5]:>8}" for l in LABELS))
    for g in LABELS:
        row = "".join(f"{cm[g][n]:>8}" for n in LABELS)
        print(f"{g:>11} {row}")
    print()

    # 3. Distributions
    dg = Counter(g for g, _ in paired)
    dn = Counter(n for _, n in paired)
    print("Label distribution:")
    print(f"{'label':>13}{'GROUNDED':>12}{'NO-EVIDENCE':>14}")
    for l in LABELS:
        print(f"{l:>13}{dg[l]:>12}{dn[l]:>14}")
    print()

    # 4. Safety-relevant directional shifts
    # Ungrounded model "upgrades" a problematic claim to SUPPORTED:
    up_to_supported = sum(1 for g, n in paired if g in PROBLEMATIC and n == "SUPPORTED")
    # Ungrounded model misses a CONTRADICTED/DANGEROUS that grounding caught:
    lost_warnings = sum(1 for g, n in paired
                        if g in {"CONTRADICTED", "DANGEROUS"}
                        and n not in {"CONTRADICTED", "DANGEROUS"})
    print("Safety-relevant divergences (grounded -> no-evidence):")
    print(f"  Problematic claim relabelled SUPPORTED without evidence: {up_to_supported}")
    print(f"  CONTRADICTED/DANGEROUS downgraded without evidence:      {lost_warnings}")
    print()

    # 5. Accuracy vs human, IF the validation subset is coded
    accuracy_vs_human(claims)


def accuracy_vs_human(claims):
    if not (os.path.exists(KEY_PATH) and os.path.exists(SHEET_PATH)):
        print("(No validation files -> skipping accuracy-vs-human.)")
        return
    human = {}
    with open(SHEET_PATH, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            hv = (r.get("human_veracity (SUPPORTED/UNSUPPORTED/EXAGGERATED/CONTRADICTED/DANGEROUS)") or "").strip().upper()
            if hv in LABELS:
                human[r["row_id"]] = hv
    if not human:
        print("(validation_sheet.csv human column is empty -> code it to get the "
              "accuracy comparison. Divergence stats above still hold.)")
        return

    # validation_key row_id maps to the same claim order used for sampling;
    # the grounded label is in the key, the no-evidence label must be matched
    # by the same row_id ordering used when the validation sample was drawn.
    # Here we assume row_id indexes into the classified dataset order.
    key_grounded = {}
    with open(KEY_PATH, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key_grounded[r["row_id"]] = r["llm_veracity"].strip().upper()

    g_correct = n_correct = compared = 0
    no_ev = [lab(c, "llm_no_evidence") for c in claims]
    for rid, hv in human.items():
        g = key_grounded.get(rid)
        try:
            n = no_ev[int(rid)]
        except (ValueError, IndexError):
            n = None
        if g in LABELS and n in LABELS:
            compared += 1
            g_correct += (g == hv)
            n_correct += (n == hv)
    if compared:
        print(f"Accuracy vs human on validated subset (n={compared}):")
        print(f"  GROUNDED   : {g_correct}/{compared} = {g_correct/compared:.1%}")
        print(f"  NO-EVIDENCE: {n_correct}/{compared} = {n_correct/compared:.1%}")
        print("  -> the gap is the measured value of the retrieval layer.")


if __name__ == "__main__":
    main()
