# Reproductive-Health Misinformation & Stigma Pipeline

An evidence-grounded **LLM + retrieval (RAG)** pipeline that detects
reproductive-health misinformation on Bluesky and, in a parallel layer, scores
the **stigma framing** of each claim independently of its truth value. From 991
English-language posts the pipeline extracts 937 distinct health claims, grounds
each in PubMed / openFDA / clinical-guideline evidence, classifies veracity into
five categories, and scores six stigma dimensions.

Companion code for the paper *"An LLM and Retrieval Pipeline for
Reproductive-Health Misinformation and Stigma"* (submitted to *Information*, MDPI).

## Research questions

- **RQ1** What is the veracity distribution of reproductive-health claims when grounded in biomedical evidence?
- **RQ2** Do claims that depart from the evidence carry stronger stigmatizing framing? *(core, novel)*
- **RQ3** Does stigma framing vary across reproductive-health conditions?
- **RQ4** Is stigma associated with post-level engagement?
- **RQ5** What is the added value of evidence-grounded classification over a keyword baseline?
- **RQ6** How reliably does the automated pipeline agree with human annotation?
- **RQ7** What does retrieval grounding contribute relative to the same model without evidence?

## Setup

```powershell
pip install -r requirements.txt

# Bluesky (collection). Create a free App Password in Bluesky:
# Settings -> Privacy and Security -> App Passwords -> Add.
$env:BLUESKY_HANDLE       = "yourname.bsky.social"
$env:BLUESKY_APP_PASSWORD = "xxxx-xxxx-xxxx-xxxx"

$env:ANTHROPIC_API_KEY    = "sk-ant-..."
$env:ENTREZ_EMAIL         = "you@university.edu"
```

## Pipeline

Scripts are numbered in run order. All share `config.py` (paths, vocabularies,
model, thresholds). Intermediate data flows as JSONL under `data/`.

### Core pipeline (paper Sections 3–4)

```powershell
python 00_bluesky_data_collection.py   # collect posts -> data/00_raw/
python 01_claim_extraction.py          # LLM extracts structured claims
python 02_evidence_retrieval.py        # PubMed + openFDA + ACOG/NICE/NHS/WHO
python 03_veracity_classification.py   # LLM+RAG veracity + keyword baseline
python 04_stigma_classification.py     # NOVEL stigma layer + lexicon baseline
python 05_figures.py                   # descriptive stats + figures (RQ1-RQ4)
```

### Evaluation

```powershell
python 06_validation_and_stats.py      # inferential stats + human validation (RQ2-RQ6)
python 07_ablation_no_evidence.py      # re-classify with evidence removed (RQ7)
python 08_analyze_ablation.py          # grounded vs no-evidence comparison (RQ7)
```

### Temporal robustness check (auxiliary, paper Section 4.11)

```powershell
python 09_probe_temporal_signal.py     # FREE pre-flight: is there any temporal signal?
python 10_temporal_analysis.py         # monthly aggregation of dated claims
python 11_trends_coseries.py           # Bluesky vs Google Trends, detrended
```

## Output schema (per claim, after stage 04)

```json
"classification": {
  "llm_grounded":     {"veracity": "...", "confidence": 0.0, "risk_tier": "..."},
  "keyword_baseline": {"veracity": "..."},
  "stigma_sentiment": {
    "dimensions": {"shame":0.0,"secrecy":0.0,"disgust":0.0,"fear":0.0,"blame":0.0,"empowerment":0.0},
    "dominant_frame": "...", "stigma_score": 0.0
  },
  "stigma_baseline":  {"...": "..."}
}
```

## Why Bluesky, not Reddit

Reddit's Responsible Builder Policy requires research access via its Researcher
Program (RFR) and prohibits inferring sensitive characteristics (including
health) from data collected outside it. Reproductive-health analysis falls under
that restriction, so collection uses Bluesky, built on the open AT Protocol with
research-friendly public access.

## Ethics & data

No usernames, handles, DIDs, profile metadata, or post URLs are stored. Only
post text, engagement counts, and a non-identifying hashed ID are retained. Raw
third-party text is not redistributed; downstream outputs are aggregate. Run
collectors only where platform terms and your institution's ethics rules permit.

## Citation

```bibtex
@article{behnamian2026reproductive,
  title   = {An LLM and Retrieval Pipeline for Reproductive-Health Misinformation and Stigma},
  author  = {Behnamian, Sara and Shahbazi, Zeinab and Fogh, Fatemeh and
             Baghestani, Bita and Khani, Ameneh and Darzian Rostami, Arash},
  journal = {Information},
  year    = {2026},
  note    = {Submitted}
}
```
