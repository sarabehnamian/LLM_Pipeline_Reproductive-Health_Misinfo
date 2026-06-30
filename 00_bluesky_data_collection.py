"""
00_bluesky_data_collection.py

Stage 00 (English arm) - Bluesky collector.

Why Bluesky instead of Reddit: Reddit's Responsible Builder Policy requires
research access to go through its Researcher Program (RFR) and prohibits
inferring sensitive characteristics (incl. health) from data collected outside
it. Bluesky is built on the open AT Protocol with research-friendly public
access, so it is the compliant English source for this study.

Auth: uses a free Bluesky App Password (Settings -> Privacy and Security ->
App Passwords -> Add). This is NOT your main password; it is a revocable token.
Set:
    $env:BLUESKY_HANDLE       = "yourname.bsky.social"
    $env:BLUESKY_APP_PASSWORD = "xxxx-xxxx-xxxx-xxxx"
Then: pip install atproto

Behaviour mirrors the rest of the pipeline:
  - searches SEARCH_QUERIES, paginates to MAX_POSTS_PER_QUERY per query
  - keyword gate against HEALTH_CLAIM_KEYWORDS
  - outputs data/00_raw/bluesky_posts.jsonl and a copy in all_posts.jsonl

Every record carries platform="bluesky" and language="en". No author handles,
DIDs, or other personal identifiers are stored - only post text + engagement
counts + a non-identifying hashed id.

Ethics: collect only under Bluesky's terms and your institution's ethics
approval. Outputs are aggregate; raw text is not redistributed.
"""

import hashlib
import json
import time

import config as C

SEARCH_QUERIES = [
    "pcos cure",
    "pcos natural",
    "endometriosis cure",
    "endometriosis diet",
    "regulate my period",
    "period cramps remedy",
    "seed cycling",
    "inositol pcos",
    "boost fertility",
    "birth control infertility",
    "spearmint tea pcos",
    "period detox",
    "fertility supplements",
    "menopause natural remedy",
]

MAX_POSTS_PER_QUERY = 300   # widened for temporal coverage
PAGE_SIZE = 100            # max per page; fewer round-trips
SLEEP_BETWEEN_PAGES = 0.6


def matches_keywords(text):
    if not text:
        return False
    low = text.lower()
    return any(kw in low for kw in C.HEALTH_CLAIM_KEYWORDS)


def hashed_id(uri):
    """Non-identifying stable id derived from the post AT-URI."""
    h = hashlib.sha1(uri.encode("utf-8")).hexdigest()[:16]
    return f"bsky_{h}"


def post_record(post):
    """Build a normalized record from an atproto post view. No identifiers."""
    record = getattr(post, "record", None)
    text = getattr(record, "text", "") or ""
    uri = getattr(post, "uri", "") or ""
    # Bluesky post creation time (ISO 8601). Falls back to indexed_at on the
    # post view if the record field is missing. This is the temporal anchor
    # for all downstream time-series analysis.
    created_at = getattr(record, "created_at", "") or getattr(post, "indexed_at", "") or ""
    return {
        "id": hashed_id(uri),
        "platform": "bluesky",
        "language": "en",
        "created_at": created_at,
        "subreddit": "",                       # kept for schema compatibility
        "title": "",
        "text": text,
        "score": int(getattr(post, "like_count", 0) or 0),
        "num_comments": int(getattr(post, "reply_count", 0) or 0),
        "url": "",                             # omitted to avoid identifiability
        "kind": "post",
    }


def get_client():
    try:
        from atproto import Client
    except ImportError:
        raise SystemExit("pip install atproto")
    handle = getattr(C, "BLUESKY_HANDLE", "")
    app_pw = getattr(C, "BLUESKY_APP_PASSWORD", "")
    if not handle or not app_pw:
        raise SystemExit(
            "Set BLUESKY_HANDLE and BLUESKY_APP_PASSWORD environment variables. "
            "Create an App Password in Bluesky: Settings -> Privacy and Security "
            "-> App Passwords."
        )
    client = Client()
    client.login(handle, app_pw)
    return client


def search_query(client, query):
    """Paginate searchPosts for one query; return list of post views."""
    out = []
    cursor = None
    while len(out) < MAX_POSTS_PER_QUERY:
        params = {"q": query, "limit": PAGE_SIZE, "lang": "en"}
        if cursor:
            params["cursor"] = cursor
        try:
            resp = client.app.bsky.feed.search_posts(params)
        except Exception as e:
            print(f"  ! search failed for '{query}': {e}")
            break
        posts = getattr(resp, "posts", []) or []
        out.extend(posts)
        cursor = getattr(resp, "cursor", None)
        if not cursor or not posts:
            break
        time.sleep(SLEEP_BETWEEN_PAGES)
    return out[:MAX_POSTS_PER_QUERY]


def collect():
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    client = get_client()

    seen = set()
    records = []
    for query in SEARCH_QUERIES:
        print(f"\n=== query: {query} ===")
        posts = search_query(client, query)
        kept = 0
        for post in posts:
            rec = post_record(post)
            if rec["id"] in seen:
                continue
            if len(rec["text"].strip()) < C.MIN_TEXT_LEN:
                continue
            if not matches_keywords(rec["text"]):
                continue
            seen.add(rec["id"])
            records.append(rec)
            kept += 1
        print(f"  kept {kept} / {len(posts)}")
        time.sleep(0.5)

    out_path = C.RAW_DIR / "bluesky_posts.jsonl"
    for path in (out_path, C.ALL_POSTS):
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # --- Temporal span report (decides whether a time-series paper is viable) ---
    dates = sorted(r["created_at"][:10] for r in records if r.get("created_at"))
    print(f"\nDone. {len(records)} bluesky posts")
    print(f"  -> {out_path}")
    print(f"  -> {C.ALL_POSTS}")
    if dates:
        from collections import Counter
        span_days = "?"
        try:
            from datetime import date
            d0 = date.fromisoformat(dates[0]); d1 = date.fromisoformat(dates[-1])
            span_days = (d1 - d0).days
        except Exception:
            pass
        by_month = Counter(d[:7] for d in dates)
        with_ts = len(dates)
        print("\n=== TEMPORAL SPAN REPORT ===")
        print(f"  posts with timestamp : {with_ts}/{len(records)}")
        print(f"  earliest             : {dates[0]}")
        print(f"  latest               : {dates[-1]}")
        print(f"  span (days)          : {span_days}")
        print(f"  distinct months      : {len(by_month)}")
        print("  posts per month:")
        for m in sorted(by_month):
            print(f"    {m}: {by_month[m]}")
        print("  RULE OF THUMB: >=3 months and >=8 non-empty weeks -> time-series viable.")
    else:
        print("\n!! No timestamps captured - check atproto field names.")


if __name__ == "__main__":
    collect()
