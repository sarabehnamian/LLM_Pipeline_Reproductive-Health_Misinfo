"""
02_evidence_retrieval.py

Stage 02 - attach grounding evidence to each claim.

Sources:
  - PubMed (NCBI Entrez): systematic reviews / meta-analyses / RCTs for
    (product + condition) or (product + effect).
  - openFDA drug adverse events: api.fda.gov/drug/event.json on the product.
  - Authoritative reproductive-health reference map (ACOG / NHS / NICE / WHO /
    Cochrane / NIH ODS) -> a static URL pointer when the condition matches.

For Persian claims, the structured product/condition fields are already in
English (stage 01), so the same English evidence sources apply. Only
verbatim_quote stayed Persian.

Input : data/01_claims/extracted_claims.jsonl
Output: data/02_evidence/claims_with_evidence.jsonl
"""

import json
import time
import urllib.parse
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET

import config as C

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
FDA_EVENT = "https://api.fda.gov/drug/event.json"

PUBLICATION_FILTER = (
    '("systematic review"[Publication Type] OR "meta-analysis"[Publication Type]'
    ' OR "randomized controlled trial"[Publication Type])'
)

# Curated authoritative references keyed by substrings of target_condition.
AUTHORITATIVE_REFS = {
    "pcos": {
        "source": "ACOG / NICE",
        "reference_url": "https://www.acog.org/womens-health/faqs/polycystic-ovary-syndrome-pcos",
        "note": "Evidence-based guidance on PCOS diagnosis and management.",
    },
    "endometriosis": {
        "source": "NICE NG73 / ACOG",
        "reference_url": "https://www.nice.org.uk/guidance/ng73",
        "note": "NICE guideline on endometriosis diagnosis and management.",
    },
    "fertility": {
        "source": "NICE / ACOG",
        "reference_url": "https://www.nice.org.uk/guidance/cg156",
        "note": "Fertility assessment and treatment guidance.",
    },
    "pregnan": {
        "source": "NHS / ACOG",
        "reference_url": "https://www.nhs.uk/pregnancy/",
        "note": "NHS pregnancy guidance.",
    },
    "menopause": {
        "source": "NICE NG23",
        "reference_url": "https://www.nice.org.uk/guidance/ng23",
        "note": "Menopause diagnosis and management.",
    },
    "period": {
        "source": "NHS / ACOG",
        "reference_url": "https://www.nhs.uk/conditions/periods/",
        "note": "General menstruation guidance.",
    },
    "menstrual": {
        "source": "NHS / ACOG",
        "reference_url": "https://www.nhs.uk/conditions/periods/",
        "note": "General menstruation guidance.",
    },
    "general health": {
        "source": "NIH Office of Dietary Supplements",
        "reference_url": "https://ods.od.nih.gov/factsheets/list-all/",
        "note": "Dietary supplement fact sheets.",
    },
}


def _get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": C.ENTREZ_TOOL})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def pubmed_search(query, retmax=5):
    params = urllib.parse.urlencode({
        "db": "pubmed", "term": f"({query}) AND {PUBLICATION_FILTER}",
        "retmax": retmax, "retmode": "json",
        "email": C.ENTREZ_EMAIL, "tool": C.ENTREZ_TOOL,
    })
    try:
        data = json.loads(_get(f"{EUTILS}/esearch.fcgi?{params}"))
        return data.get("esearchresult", {}).get("idlist", [])
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
        print(f"    ! pubmed search failed: {e}")
        return []


def pubmed_fetch(pmids):
    if not pmids:
        return []
    params = urllib.parse.urlencode({
        "db": "pubmed", "id": ",".join(pmids), "retmode": "xml",
        "email": C.ENTREZ_EMAIL, "tool": C.ENTREZ_TOOL,
    })
    try:
        xml = _get(f"{EUTILS}/efetch.fcgi?{params}")
        root = ET.fromstring(xml)
    except (urllib.error.URLError, ET.ParseError, TimeoutError) as e:
        print(f"    ! pubmed fetch failed: {e}")
        return []
    out = []
    for art in root.findall(".//PubmedArticle"):
        title = art.findtext(".//ArticleTitle") or ""
        abstract = " ".join(t.text or "" for t in art.findall(".//AbstractText"))
        year = art.findtext(".//PubDate/Year") or ""
        pmid = art.findtext(".//PMID") or ""
        ptypes = [p.text for p in art.findall(".//PublicationType") if p.text]
        out.append({
            "pmid": pmid, "title": title, "year": year,
            "abstract": abstract[:1200],
            "publication_types": ptypes,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })
    return out


def fda_events(product, limit=3):
    params = urllib.parse.urlencode({
        "search": f'patient.drug.medicinalproduct:"{product}"',
        "limit": limit,
    })
    try:
        data = json.loads(_get(f"{FDA_EVENT}?{params}"))
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        return []   # 404 = no events, treated as empty
    events = []
    for r in data.get("results", []):
        reactions = [x.get("reactionmeddrapt", "")
                     for x in r.get("patient", {}).get("reaction", [])]
        events.append({"reactions": [x for x in reactions if x],
                       "serious": r.get("serious", "")})
    return events


def authoritative_ref(condition):
    cond = (condition or "").lower()
    for key, ref in AUTHORITATIVE_REFS.items():
        if key in cond:
            return ref
    return None


def build_query(claim):
    product = claim.get("product", "") or ""
    condition = claim.get("target_condition", "") or ""
    effect = claim.get("claimed_effect", "") or ""
    if condition and condition.lower() != "general health":
        return f"{product} {condition}".strip()
    return f"{product} {effect}".strip()


def main():
    C.EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    if not C.EXTRACTED_CLAIMS.exists():
        raise SystemExit(f"No claims at {C.EXTRACTED_CLAIMS}")

    n = 0
    with open(C.EXTRACTED_CLAIMS, encoding="utf-8") as fin, \
         open(C.CLAIMS_WITH_EVIDENCE, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            claim = json.loads(line)
            query = build_query(claim)

            pmids = pubmed_search(query)
            time.sleep(0.4)
            articles = pubmed_fetch(pmids)
            time.sleep(0.4)

            product = claim.get("product", "") or ""
            events = fda_events(product) if product else []
            time.sleep(0.3)

            ref = authoritative_ref(claim.get("target_condition", ""))

            claim["evidence"] = {
                "pubmed_query": query,
                "pubmed_articles": articles,
                "fda_events": events,
                "authoritative_reference": ref,
                "evidence_count": len(articles) + len(events) + (1 if ref else 0),
            }
            fout.write(json.dumps(claim, ensure_ascii=False) + "\n")
            n += 1
            if n % 25 == 0:
                print(f"  {n} claims grounded")

    print(f"\nDone. {n} claims -> {C.CLAIMS_WITH_EVIDENCE}")


if __name__ == "__main__":
    main()
