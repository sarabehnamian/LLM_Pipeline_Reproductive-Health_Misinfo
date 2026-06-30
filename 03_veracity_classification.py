"""
03_veracity_classification.py

Stage 03 - classify each claim's veracity using the retrieved evidence (RAG),
plus a transparent keyword baseline for comparison.

Input : data/02_evidence/claims_with_evidence.jsonl
Output: data/03_classified/classified_claims.jsonl

Adds claim["classification"]["llm_grounded"] and ["keyword_baseline"].
The stigma layer is added separately by 03b.
"""

import json
import time

import anthropic

import config as C

SYSTEM = "You are a careful biomedical fact-checker. Output only JSON."

PROMPT = """Assess the veracity of a single health claim using ONLY the evidence provided.

CLAIM
- product: {product}
- claimed_effect: {claimed_effect}
- target_condition: {target_condition}
- claim_strength: {claim_strength}
- quote: {verbatim_quote}

EVIDENCE
{evidence_block}

Classify into exactly one veracity label:
- SUPPORTED: evidence backs the claim
- UNSUPPORTED: no adequate evidence either way
- EXAGGERATED: a kernel of truth overstated
- CONTRADICTED: evidence contradicts the claim
- DANGEROUS: claim could cause harm if followed

Return ONLY this JSON object:
{{"veracity":"<label>","confidence":<0..1>,"risk_tier":"LOW|MODERATE|HIGH|CRITICAL",
"reasoning":"<2 sentences>","key_evidence":"<short>","recommendation":"<short>"}}
"""

DANGEROUS_KEYWORDS = [
    "cure", "cures", "stop taking", "instead of", "no need for doctor",
    "reverse", "guaranteed", "100%", "miracle", "detox your", "flush out",
]
EXAGGERATION_KEYWORDS = [
    "always", "completely", "totally", "permanently", "instantly",
    "every woman", "all women", "never fails",
]


def evidence_block(claim):
    ev = claim.get("evidence", {})
    lines = []
    for a in ev.get("pubmed_articles", [])[:5]:
        ptypes = ", ".join(a.get("publication_types", [])[:2])
        lines.append(f"- PubMed {a.get('year','')} [{ptypes}]: {a.get('title','')}. "
                     f"{a.get('abstract','')[:400]}")
    for e in ev.get("fda_events", [])[:3]:
        lines.append(f"- FDA adverse-event report: reactions={e.get('reactions')}")
    ref = ev.get("authoritative_reference")
    if ref:
        lines.append(f"- {ref.get('source')}: {ref.get('note')} ({ref.get('reference_url')})")
    return "\n".join(lines) if lines else "(no evidence retrieved)"


def parse_obj(raw):
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


def llm_grounded(client, claim):
    prompt = PROMPT.format(
        product=claim.get("product", ""),
        claimed_effect=claim.get("claimed_effect", ""),
        target_condition=claim.get("target_condition", ""),
        claim_strength=claim.get("claim_strength", ""),
        verbatim_quote=(claim.get("verbatim_quote", "") or "")[:500],
        evidence_block=evidence_block(claim),
    )
    msg = client.messages.create(
        model=C.ANTHROPIC_MODEL, max_tokens=C.MAX_TOKENS_CLASSIFY,
        system=SYSTEM, messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in msg.content if b.type == "text")
    obj = parse_obj(text)
    if not obj:
        return {"veracity": "UNSUPPORTED", "confidence": 0.0,
                "risk_tier": "LOW", "reasoning": "parse_failed",
                "key_evidence": "", "recommendation": ""}
    return obj


def keyword_baseline(claim):
    text = f"{claim.get('claimed_effect','')} {claim.get('verbatim_quote','')}".lower()
    if any(k in text for k in DANGEROUS_KEYWORDS):
        v, tier = "DANGEROUS", "HIGH"
    elif any(k in text for k in EXAGGERATION_KEYWORDS):
        v, tier = "EXAGGERATED", "MODERATE"
    else:
        v, tier = "UNSUPPORTED", "LOW"
    return {"veracity": v, "confidence": 0.5, "risk_tier": tier,
            "reasoning": "keyword match", "method": "keyword_baseline"}


def main():
    C.CLASSIFIED_DIR.mkdir(parents=True, exist_ok=True)
    if not C.CLAIMS_WITH_EVIDENCE.exists():
        raise SystemExit(f"No input at {C.CLAIMS_WITH_EVIDENCE}")
    if not C.ANTHROPIC_API_KEY:
        raise SystemExit("Set ANTHROPIC_API_KEY.")

    client = anthropic.Anthropic(api_key=C.ANTHROPIC_API_KEY)
    n = 0
    with open(C.CLAIMS_WITH_EVIDENCE, encoding="utf-8") as fin, \
         open(C.CLASSIFIED_CLAIMS, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            claim = json.loads(line)
            try:
                grounded = llm_grounded(client, claim)
            except anthropic.APIError as e:
                print(f"  ! API error: {e}")
                if "credit" in str(e).lower():
                    break
                grounded = {"veracity": "UNSUPPORTED", "confidence": 0.0,
                            "risk_tier": "LOW", "reasoning": "api_error",
                            "key_evidence": "", "recommendation": ""}
            claim.setdefault("classification", {})
            claim["classification"]["llm_grounded"] = grounded
            claim["classification"]["keyword_baseline"] = keyword_baseline(claim)
            fout.write(json.dumps(claim, ensure_ascii=False) + "\n")
            n += 1
            time.sleep(0.5)
            if n % 25 == 0:
                print(f"  {n} classified")

    print(f"\nDone. {n} claims -> {C.CLASSIFIED_CLAIMS}")


if __name__ == "__main__":
    main()
