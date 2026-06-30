"""
config.py

Shared configuration for the menstrual / reproductive-health misinformation
and stigma-sentiment pipeline. Imported by every stage so the English (Reddit)
and Persian (Ninisite) arms stay consistent.

Credentials are read from environment variables. Never hard-code secrets here.

  Windows PowerShell (current session):
    $env:BLUESKY_HANDLE       = "yourname.bsky.social"
    $env:BLUESKY_APP_PASSWORD = "xxxx-xxxx-xxxx-xxxx"   # App Password, not main pw
    $env:ANTHROPIC_API_KEY    = "sk-ant-..."
    $env:ENTREZ_EMAIL         = "you@university.edu"
    $env:YOUTUBE_API_KEY      = "..."        # optional, stage 00b

  To persist across sessions use [Environment]::SetEnvironmentVariable(...,'User').
"""

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths  (all relative to wherever you run the scripts; created as needed)
# --------------------------------------------------------------------------- #
DATA = Path("data")
RAW_DIR = DATA / "00_raw"
CLAIMS_DIR = DATA / "01_claims"
EVIDENCE_DIR = DATA / "02_evidence"
CLASSIFIED_DIR = DATA / "03_classified"
RESULTS_DIR = DATA / "04_evaluation" / "results"

ALL_POSTS = RAW_DIR / "all_posts.jsonl"
REDDIT_POSTS = RAW_DIR / "reddit_posts.jsonl"
BLUESKY_POSTS = RAW_DIR / "bluesky_posts.jsonl"
YOUTUBE_POSTS = RAW_DIR / "youtube_posts.jsonl"
EXTRACTED_CLAIMS = CLAIMS_DIR / "extracted_claims.jsonl"
CLAIMS_WITH_EVIDENCE = EVIDENCE_DIR / "claims_with_evidence.jsonl"
CLASSIFIED_CLAIMS = CLASSIFIED_DIR / "classified_claims.jsonl"

# --------------------------------------------------------------------------- #
# Credentials (from env)
# --------------------------------------------------------------------------- #
REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.environ.get(
    "REDDIT_USER_AGENT",
    "menstrual-misinfo-research/0.1 by u/CHANGE_ME (academic research)",
)
# Bluesky (English arm). App Password, created in Bluesky settings - not the
# account's main password. Free, revocable, no captcha.
BLUESKY_HANDLE = os.environ.get("BLUESKY_HANDLE", "")
BLUESKY_APP_PASSWORD = os.environ.get("BLUESKY_APP_PASSWORD", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "")

# NCBI Entrez (PubMed). Replace with your real email for your own runs.
ENTREZ_EMAIL = os.environ.get("ENTREZ_EMAIL", "your.email@university.edu")
ENTREZ_TOOL = "menstrual-misinfo-pipeline"

# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"   # extraction + classification
MAX_TOKENS_EXTRACT = 2000
MAX_TOKENS_CLASSIFY = 1000

# --------------------------------------------------------------------------- #
# Reddit collection (English arm)
# --------------------------------------------------------------------------- #
SUBREDDITS = [
    "PCOS",
    "endometriosis",
    "Endo",
    "Periods",
    "Menstruation",
    "PMDD",
    "birthcontrol",
    "TryingForABaby",
    "Fertility",
    "WomensHealth",
    "Healthyhooha",
    "TwoXChromosomes",
    "pregnant",
    "BabyBumps",
]

LISTINGS = ["hot", "new", "top"]
LIMIT_PER_LISTING = 100
TOP_TIME_FILTER = "year"

MAX_COMMENT_POSTS = 5
COMMENT_POST_MIN_SCORE = 5
COMMENT_LIMIT = 20

MIN_TEXT_LEN = 20   # skip posts whose title+text is shorter than this

# A post/comment is kept only if it matches >= 1 keyword (case-insensitive).
HEALTH_CLAIM_KEYWORDS = [
    # conditions
    "pcos", "endometriosis", "endo", "period", "menstrual", "menstruation",
    "cramps", "fertility", "infertility", "ovulation", "cycle", "pregnan",
    "menopause", "pmdd", "pms", "fibroid", "cyst",
    # products / interventions tied to claims
    "supplement", "seed cycling", "inositol", "spearmint", "detox", "cleanse",
    "tampon", "pad", "menstrual cup", "birth control", "the pill", "iud",
    "fertility tea", "vitamin", "herbal", "natural remedy",
    # claim / treatment vocabulary
    "cure", "reverse", "heal", "regulate", "balance hormones",
    "boost fertility", "get pregnant", "stop bleeding", "free bleeding",
    "shrink",
]

# --------------------------------------------------------------------------- #
# Claim schema vocabulary (stage 01)
# --------------------------------------------------------------------------- #
CLAIM_STRENGTHS = ["definitive", "suggestive", "anecdotal"]
RISK_CATEGORIES = [
    "supplement_efficacy",
    "menstrual_management",
    "fertility_conception",
    "pregnancy_prenatal",
    "condition_cure",        # PCOS/endo "cures", cyst/fibroid shrinking
    "hormonal_birth_control",
    "detox",
    "other",
]

# --------------------------------------------------------------------------- #
# Veracity labels (stage 03)
# --------------------------------------------------------------------------- #
VERACITY_LABELS = ["SUPPORTED", "UNSUPPORTED", "EXAGGERATED",
                   "CONTRADICTED", "DANGEROUS"]
RISK_TIERS = ["LOW", "MODERATE", "HIGH", "CRITICAL"]

# --------------------------------------------------------------------------- #
# Stigma-sentiment dimensions (stage 03b)  -- the novel layer
# --------------------------------------------------------------------------- #
STIGMA_DIMENSIONS = [
    "shame",          # embarrassment about one's own body/period
    "secrecy",        # concealment, "no one should know"
    "disgust",        # impurity, "dirty", uncleanliness
    "fear",           # catastrophizing, dread about symptoms
    "blame",          # moralizing, "your fault", lifestyle blame
    "empowerment",    # destigmatizing, normalizing, openness (positive)
]
STIGMA_FRAMES = STIGMA_DIMENSIONS + ["neutral"]

# Lexicon baseline for stage 03b (parallel to the keyword veracity baseline).
STIGMA_KEYWORDS = {
    "shame": ["embarrassing", "embarrassed", "ashamed", "shame", "gross",
              "disgusting myself", "so gross", "humiliating"],
    "secrecy": ["hide", "hidden", "secret", "no one should know", "discreet",
                "sneak", "conceal", "don't tell"],
    "disgust": ["dirty", "unclean", "impure", "filthy", "nasty", "icky"],
    "fear": ["scared", "terrified", "afraid", "panic", "something is wrong",
             "dying", "worst", "dread"],
    "blame": ["your fault", "her fault", "should have", "deserve", "lazy",
              "just lose weight", "if you ate better", "brought it on"],
    "empowerment": ["normal", "natural", "talk about", "no shame", "proud",
                    "break the stigma", "nothing to be ashamed", "open up"],
}
