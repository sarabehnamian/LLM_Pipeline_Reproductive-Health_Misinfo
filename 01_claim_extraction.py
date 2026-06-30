"""
01_claim_extraction.py

Stage 01 - extract structured health claims from each collected post/comment.

Uses Anthropic (claude-haiku) by default. The prompt is language-aware: posts
may be English or Persian; the model normalizes the structured fields to English
so the English and Persian arms share one schema, while verbatim_quote stays in
the original language.

Input : data/00_raw/all_posts.jsonl  (fallback: reddit_posts.jsonl)
Output: data/01_claims/extracted_claims.jsonl   (one JSON object per claim)
"""

import json
import time

import anthropic

import config as C

SYSTEM = "You extract structured health claims from social media text. Output only JSON."

PROMPT_TEMPLATE = """You are given one social-media post about menstrual or reproductive health.
The post may be in English or Persian (Farsi). Extract every distinct health CLAIM it makes.

A claim asserts that some product, food, supplement, behaviour, or intervention
has an effect on a menstrual/reproductive condition or on general health.

Return a JSON array. Each element has these fields, all written in ENGLISH
except verbatim_quote which stays in the post's original language:

- product: substance/product/behaviour named (e.g. "spearmint tea", "inositol")
- claimed_effect: the alleged effect (e.g. "regulates periods")
- target_condition: condition named, or "general health"
- claim_strength: one of {strengths}
- verbatim_quote: <=100 words copied from the post, ORIGINAL language
- risk_category: one of {categories}

If the post makes no health claim, return an empty array: []
Output ONLY the JSON array, no prose, no markdown fences.

POST (platform={platform}, language={language}):
\"\"\"{text}\"\"\"
"""


def build_prompt(post):
    combined = f"{post.get('title','')}\n{post.get('text','')}".strip()
    return PROMPT_TEMPLATE.format(
        strengths=", ".join(C.CLAIM_STRENGTHS),
        categories=", ".join(C.RISK_CATEGORIES),
        platform=post.get("platform", ""),
        language=post.get("language", ""),
        text=combined[:6000],
    )


def parse_json_array(raw):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lstrip().lower().startswith("json"):
            raw = raw.lstrip()[4:]
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        data = json.loads(raw[start:end + 1])
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def extract_claims(client, post):
    msg = client.messages.create(
        model=C.ANTHROPIC_MODEL,
        max_tokens=C.MAX_TOKENS_EXTRACT,
        system=SYSTEM,
        messages=[{"role": "user", "content": build_prompt(post)}],
    )
    text = "".join(b.text for b in msg.content if b.type == "text")
    return parse_json_array(text)


def add_provenance(claim, post):
    claim["source_post_id"] = post.get("id", "")
    claim["source_platform"] = post.get("platform", "")
    claim["created_at"] = post.get("created_at", "")   # temporal anchor for time-series stage
    claim["source_subreddit"] = post.get("subreddit", "")
    claim["source_url"] = post.get("url", "")
    claim["post_score"] = post.get("score", 0)
    claim["post_num_comments"] = post.get("num_comments", 0)
    claim["language"] = post.get("language", "")
    return claim


def main():
    C.CLAIMS_DIR.mkdir(parents=True, exist_ok=True)
    src = C.ALL_POSTS if C.ALL_POSTS.exists() else C.REDDIT_POSTS
    if not src.exists():
        raise SystemExit(f"No input posts found at {C.ALL_POSTS} or {C.REDDIT_POSTS}")
    if not C.ANTHROPIC_API_KEY:
        raise SystemExit("Set ANTHROPIC_API_KEY in your environment.")

    client = anthropic.Anthropic(api_key=C.ANTHROPIC_API_KEY)

    n_posts = n_claims = 0
    with open(src, encoding="utf-8") as fin, \
         open(C.EXTRACTED_CLAIMS, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            post = json.loads(line)
            combined = f"{post.get('title','')}{post.get('text','')}"
            if len(combined.strip()) < C.MIN_TEXT_LEN:
                continue
            n_posts += 1
            try:
                claims = extract_claims(client, post)
            except anthropic.APIError as e:
                print(f"  ! API error on {post.get('id')}: {e}")
                if "credit" in str(e).lower():
                    print("  stopping (credit error).")
                    break
                continue
            for claim in claims:
                fout.write(json.dumps(add_provenance(claim, post),
                                      ensure_ascii=False) + "\n")
                n_claims += 1
            time.sleep(0.5)
            if n_posts % 25 == 0:
                print(f"  {n_posts} posts -> {n_claims} claims")

    print(f"\nDone. {n_posts} posts -> {n_claims} claims")
    print(f"  -> {C.EXTRACTED_CLAIMS}")


if __name__ == "__main__":
    main()
