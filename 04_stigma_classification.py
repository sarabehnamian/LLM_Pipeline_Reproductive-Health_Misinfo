"""
03b_stigma_classification.py

Stage 03b - THE NOVEL LAYER. Score each claim's stigma-sentiment framing on six
dimensions (shame, secrecy, disgust, fear, blame, empowerment), independent of
whether the claim is true. Adds an LLM scorer and a lexicon baseline, parallel
to the veracity stage.

Runs AFTER 03 and edits the same file in place (reads classified_claims.jsonl,
rewrites it with classification.stigma_sentiment added). Safe to re-run.

Input/Output: data/03_classified/classified_claims.jsonl
"""

import json
import time

import anthropic

import config as C

SYSTEM = "You analyze stigma and emotional framing in menstrual/reproductive-health text. Output only JSON."

PROMPT = """Analyze the STIGMA and emotional framing of this menstrual/reproductive-health
social-media quote. This is about ATTITUDE, not truth. The quote may be English or Persian.

QUOTE: {quote}
(context: product={product}, condition={target_condition})

Score each dimension from 0.0 (absent) to 1.0 (strongly present):
- shame: embarrassment about one's own body or period
- secrecy: concealment, the idea it must be hidden
- disgust: framing periods/bodies as dirty, impure, unclean
- fear: catastrophizing, dread about symptoms or outcomes
- blame: moralizing, "your fault", lifestyle blame for a condition
- empowerment: normalizing, destigmatizing, openness (POSITIVE framing)

Then give the single dominant_frame (one of: shame, secrecy, disgust, fear,
blame, empowerment, neutral) and an overall stigma_score from 0.0 to 1.0
(0 = no stigma / neutral or empowering, 1 = heavily stigmatizing).
Empowerment does NOT add to stigma_score.

Return ONLY this JSON:
{{"dimensions":{{"shame":0.0,"secrecy":0.0,"disgust":0.0,"fear":0.0,"blame":0.0,"empowerment":0.0}},
"dominant_frame":"<frame>","stigma_score":0.0,"reasoning":"<1-2 sentences>"}}
"""


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


def llm_stigma(client, claim):
    prompt = PROMPT.format(
        quote=(claim.get("verbatim_quote", "") or "")[:600],
        product=claim.get("product", ""),
        target_condition=claim.get("target_condition", ""),
    )
    msg = client.messages.create(
        model=C.ANTHROPIC_MODEL, max_tokens=C.MAX_TOKENS_CLASSIFY,
        system=SYSTEM, messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in msg.content if b.type == "text")
    obj = parse_obj(text)
    if not obj or "dimensions" not in obj:
        return {"dimensions": {d: 0.0 for d in C.STIGMA_DIMENSIONS},
                "dominant_frame": "neutral", "stigma_score": 0.0,
                "reasoning": "parse_failed"}
    for d in C.STIGMA_DIMENSIONS:
        obj["dimensions"].setdefault(d, 0.0)
    return obj


def lexicon_baseline(claim):
    text = (claim.get("verbatim_quote", "") or "").lower()
    scores = {}
    for dim, words in C.STIGMA_KEYWORDS.items():
        hits = sum(1 for w in words if w in text)
        scores[dim] = min(1.0, hits / 2.0)
    stig = {d: scores[d] for d in C.STIGMA_DIMENSIONS if d != "empowerment"}
    dominant = max(scores, key=scores.get) if any(scores.values()) else "neutral"
    stigma_score = max(stig.values()) if stig else 0.0
    return {"dimensions": scores, "dominant_frame": dominant,
            "stigma_score": stigma_score, "method": "lexicon_baseline"}


def main():
    if not C.CLASSIFIED_CLAIMS.exists():
        raise SystemExit(f"Run stage 03 first; missing {C.CLASSIFIED_CLAIMS}")
    if not C.ANTHROPIC_API_KEY:
        raise SystemExit("Set ANTHROPIC_API_KEY.")

    client = anthropic.Anthropic(api_key=C.ANTHROPIC_API_KEY)

    claims = [json.loads(l) for l in open(C.CLASSIFIED_CLAIMS, encoding="utf-8")
              if l.strip()]

    n = 0
    for claim in claims:
        try:
            grounded = llm_stigma(client, claim)
        except anthropic.APIError as e:
            print(f"  ! API error: {e}")
            if "credit" in str(e).lower():
                break
            grounded = {"dimensions": {d: 0.0 for d in C.STIGMA_DIMENSIONS},
                        "dominant_frame": "neutral", "stigma_score": 0.0,
                        "reasoning": "api_error"}
        claim.setdefault("classification", {})
        claim["classification"]["stigma_sentiment"] = grounded
        claim["classification"]["stigma_baseline"] = lexicon_baseline(claim)
        n += 1
        time.sleep(0.5)
        if n % 25 == 0:
            print(f"  {n} stigma-scored")

    with open(C.CLASSIFIED_CLAIMS, "w", encoding="utf-8") as fout:
        for claim in claims:
            fout.write(json.dumps(claim, ensure_ascii=False) + "\n")

    print(f"\nDone. {n} claims stigma-scored -> {C.CLASSIFIED_CLAIMS}")


if __name__ == "__main__":
    main()
