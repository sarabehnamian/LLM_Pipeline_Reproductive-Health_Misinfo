"""
03c_ablation_no_evidence.py

ABLATION for RQ6-style methodological evaluation: re-classify each claim using
the SAME model and SAME prompt as the grounded classifier, but WITHOUT any
retrieved evidence. The model must fall back on its parametric knowledge alone.

This isolates the contribution of the retrieval layer. Everything is held
identical to 03_veracity_classification.py except the evidence block, so any
label divergence is attributable to grounding and nothing else.

Input : claims_with_evidence.jsonl   (same input as stage 03)
Output: classified_ablation.jsonl    (adds classification["llm_no_evidence"])

Run normally, or with --batch to use the Batch API (~50% cheaper, async).
"""

import argparse
import json
import os
import time

import anthropic

import config as C

# ----------------------------------------------------------------------------- #
# Held identical to stage 03. The ONLY changes are:
#   (1) the EVIDENCE section is removed from the prompt
#   (2) the model is told to rely on its own knowledge
# Model, system message, label set, parsing, and fallbacks are unchanged.
# ----------------------------------------------------------------------------- #
MODEL = C.ANTHROPIC_MODEL                     # same as stage 03
MAX_TOKENS = C.MAX_TOKENS_CLASSIFY           # same as stage 03
SYSTEM = "You are a careful biomedical fact-checker. Output only JSON."

PROMPT_NO_EV = """Assess the veracity of a single health claim using your own
biomedical knowledge. No external evidence is provided for this assessment.

CLAIM
- product: {product}
- claimed_effect: {claimed_effect}
- target_condition: {target_condition}
- claim_strength: {claim_strength}
- quote: {verbatim_quote}

Classify into exactly one veracity label:
- SUPPORTED: the claim is backed by established biomedical knowledge
- UNSUPPORTED: no adequate basis either way
- EXAGGERATED: a kernel of truth overstated
- CONTRADICTED: established knowledge contradicts the claim
- DANGEROUS: claim could cause harm if followed

Return ONLY this JSON object:
{{"veracity":"<label>","confidence":<0..1>,"risk_tier":"LOW|MODERATE|HIGH|CRITICAL",
"reasoning":"<2 sentences>","key_evidence":"<short>","recommendation":"<short>"}}
"""

IN_PATH = str(C.CLAIMS_WITH_EVIDENCE)              # data/02_evidence/claims_with_evidence.jsonl
OUT_PATH = str(C.CLASSIFIED_DIR / "classified_ablation.jsonl")

FALLBACK = {"veracity": "UNSUPPORTED", "confidence": 0.0, "risk_tier": "LOW",
            "reasoning": "parse_failed", "key_evidence": "", "recommendation": ""}


def build_prompt(claim):
    return PROMPT_NO_EV.format(
        product=claim.get("product", ""),
        claimed_effect=claim.get("claimed_effect", ""),
        target_condition=claim.get("target_condition", ""),
        claim_strength=claim.get("claim_strength", ""),
        verbatim_quote=(claim.get("verbatim_quote", "") or "")[:500],
    )


def parse_obj(raw):
    """Identical to stage 03's parser."""
    raw = raw.strip().strip("`")
    if raw.lower().startswith("json"):
        raw = raw[4:]
    s, e = raw.find("{"), raw.rfind("}")
    if s == -1 or e == -1:
        return None
    try:
        return json.loads(raw[s:e + 1])
    except json.JSONDecodeError:
        return None


def load_claims():
    claims = []
    with open(IN_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                claims.append(json.loads(line))
    return claims


# --------------------------------------------------------------------------- #
# Sequential mode (mirror of stage 03's loop)
# --------------------------------------------------------------------------- #
def run_sequential(client, claims):
    out = []
    for n, claim in enumerate(claims, 1):
        try:
            msg = client.messages.create(
                model=MODEL, max_tokens=MAX_TOKENS, system=SYSTEM,
                messages=[{"role": "user", "content": build_prompt(claim)}],
            )
            text = "".join(b.text for b in msg.content if b.type == "text")
            obj = parse_obj(text) or dict(FALLBACK)
        except anthropic.APIError as exc:
            print(f"  ! API error: {exc}")
            if "credit" in str(exc).lower():
                print("  Stopping: out of credit.")
                break
            obj = dict(FALLBACK, reasoning="api_error")
        claim.setdefault("classification", {})["llm_no_evidence"] = obj
        out.append(claim)
        time.sleep(0.5)
        if n % 25 == 0:
            print(f"  {n}/{len(claims)} classified")
    return out


# --------------------------------------------------------------------------- #
# Batch mode (~50% cheaper). Submits all claims, polls, then maps results back.
# --------------------------------------------------------------------------- #
def run_batch(client, claims):
    from anthropic.types.messages.batch_create_params import Request
    from anthropic.types.messages import MessageCreateParamsNonStreaming

    requests = [
        Request(
            custom_id=f"claim-{i}",
            params=MessageCreateParamsNonStreaming(
                model=MODEL, max_tokens=MAX_TOKENS, system=SYSTEM,
                messages=[{"role": "user", "content": build_prompt(c)}],
            ),
        )
        for i, c in enumerate(claims)
    ]
    batch = client.messages.batches.create(requests=requests)
    print(f"  Submitted batch {batch.id} ({len(requests)} requests). Polling...")

    while True:
        b = client.messages.batches.retrieve(batch.id)
        if b.processing_status == "ended":
            break
        print(f"    status={b.processing_status} ... waiting 30s")
        time.sleep(30)

    results = {}
    for r in client.messages.batches.results(batch.id):
        idx = int(r.custom_id.split("-")[1])
        if r.result.type == "succeeded":
            text = "".join(b.text for b in r.result.message.content
                           if b.type == "text")
            results[idx] = parse_obj(text) or dict(FALLBACK)
        else:
            results[idx] = dict(FALLBACK, reasoning=f"batch_{r.result.type}")

    for i, claim in enumerate(claims):
        claim.setdefault("classification", {})["llm_no_evidence"] = \
            results.get(i, dict(FALLBACK, reasoning="missing_result"))
    return claims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", action="store_true",
                    help="Use the Batch API (~50%% cheaper, async)")
    args = ap.parse_args()

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("Set ANTHROPIC_API_KEY.")
    if not os.path.exists(IN_PATH):
        raise SystemExit(f"No input at {IN_PATH}")

    client = anthropic.Anthropic(api_key=key)
    claims = load_claims()
    print(f"Loaded {len(claims)} claims. Mode: {'batch' if args.batch else 'sequential'}")

    out = run_batch(client, claims) if args.batch else run_sequential(client, claims)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for claim in out:
            f.write(json.dumps(claim, ensure_ascii=False) + "\n")
    print(f"\nDone. {len(out)} claims -> {OUT_PATH}")


if __name__ == "__main__":
    main()
