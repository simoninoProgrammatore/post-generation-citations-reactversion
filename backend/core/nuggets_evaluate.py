"""
Nugget-based evaluation for post-hoc citation generation.

Implements two metrics:
  - Nugget Precision:  Of the nuggets covered by generated claims,
                       how many are actually cited with a supporting passage?
  - Nugget Recall:     Of all required nuggets in the dataset,
                       how many are both covered by a claim AND have a citation?

A nugget is considered "covered" by a claim if keyword overlap or
NLI entailment is above threshold.

A nugget is considered "cited" if at least one of the claim's supporting
passages actually contains evidence for it (via keyword match or NLI).

Dataset format expected (asqa_gold_nuggets.json):
[
  {
    "question": "...",
    "nuggets": [
      {
        "nugget_id": "n0",
        "text": "Josef Bican holds the record...",
        "keywords": ["Bican", "Josef Bican"],
        "golden_passage_title": "Josef Bican",
        "golden_evidence": "...",
        "required": true
      },
      ...
    ],
    "docs": [
      {
        "title": "...",
        "text": "...",
        "is_gold": true,
        "support_level": "full",
        "evidence_sentence": "..."
      },
      ...
    ],
    ...
  }
]

Pipeline matched_claims format (from Step 4 / retrieve):
[
  {
    "claim": "Josef Bican is the all-time leading scorer.",
    "supporting_passages": [
      {
        "title": "Josef Bican",
        "text": "...",
        "entailment_score": 0.85,
        "best_sentence": "..."
      }
    ]
  },
  ...
]
"""

import re
import json
import argparse
import numpy as np
from pathlib import Path
from functools import lru_cache
from typing import Optional


# ──────────────────────────────────────────────
# Text utilities
# ──────────────────────────────────────────────

STOPWORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to',
    'for', 'of', 'and', 'or', 'but', 'with', 'as', 'his', 'her', 'their',
    'its', 'has', 'have', 'had', 'by', 'it', 'this', 'that', 'from', 'not',
    'be', 'been', 'who', 'which', 'what', 'how', 'when', 'where',
}


def _tokenize(text: str) -> set[str]:
    tokens = re.sub(r"[^\w\s]", "", text.lower()).split()
    return {t for t in tokens if t not in STOPWORDS and len(t) > 1}


def keyword_overlap(text_a: str, text_b: str) -> float:
    """Jaccard-like overlap between content words of two strings."""
    a = _tokenize(text_a)
    b = _tokenize(text_b)
    if not a:
        return 0.0
    return len(a & b) / len(a)


def keywords_present(keywords: list[str], text: str) -> bool:
    """Return True if ANY of the nugget keywords appear in text (case-insensitive)."""
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


# ──────────────────────────────────────────────
# NLI helpers (optional, lazy-loaded)
# ──────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_nli_model(model_name: str = "cross-encoder/nli-deberta-v3-large"):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name)


def _nli_score(premise: str, hypothesis: str, model_name: str) -> float:
    """P(premise ⊨ hypothesis) using NLI cross-encoder."""
    model = _load_nli_model(model_name)
    logits = np.array(model.predict([(premise, hypothesis)])[0])
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()
    return float(probs[1])  # index 1 = entailment


# ──────────────────────────────────────────────
# Core: nugget ↔ claim matching
# ──────────────────────────────────────────────

def nugget_covered_by_claim(
    nugget: dict,
    claim_text: str,
    use_nli: bool = False,
    nli_model: str = "cross-encoder/nli-deberta-v3-large",
    keyword_threshold: float = 0.3,
    nli_threshold: float = 0.5,
) -> bool:
    """
    Determine if a nugget is 'covered' by a claim.

    Strategy (in order):
    1. Keyword match — any nugget keyword present in the claim
    2. Lexical overlap >= keyword_threshold
    3. (Optional) NLI entailment score >= nli_threshold
    """
    # 1. Direct keyword hit
    if keywords_present(nugget.get("keywords", []), claim_text):
        return True

    # 2. Lexical overlap between nugget text and claim
    overlap = keyword_overlap(nugget["text"], claim_text)
    if overlap >= keyword_threshold:
        return True

    # 3. NLI (optional, expensive)
    if use_nli:
        score = _nli_score(claim_text, nugget["text"], nli_model)
        if score >= nli_threshold:
            return True

    return False


def nugget_cited_in_passages(
    nugget: dict,
    supporting_passages: list[dict],
    use_nli: bool = False,
    nli_model: str = "cross-encoder/nli-deberta-v3-large",
    keyword_threshold: float = 0.25,
    nli_threshold: float = 0.45,
) -> tuple[bool, Optional[dict]]:
    """
    Check if any of the supporting passages actually contain evidence
    for this nugget.

    Returns (is_cited, best_evidence_passage | None).

    Strategy:
    1. Title match with golden_passage_title
    2. Keyword match in passage text
    3. Lexical overlap between golden_evidence and passage text
    4. (Optional) NLI: passage entails nugget text
    """
    golden_title = (nugget.get("golden_passage_title") or "").lower().strip()
    golden_evidence = nugget.get("golden_evidence") or ""
    keywords = nugget.get("keywords", [])

    best = None
    best_score = -1.0

    for p in supporting_passages:
        p_title = (p.get("title") or "").lower().strip()
        p_text = p.get("text") or ""

        score = 0.0

        # 1. Title match
        if golden_title and golden_title in p_title:
            score += 0.6

        # 2. Keyword match in passage text
        if keywords_present(keywords, p_text):
            score += 0.4

        # 3. Overlap with golden evidence
        if golden_evidence:
            ev_overlap = keyword_overlap(golden_evidence, p_text)
            score += ev_overlap * 0.3

        # 4. General overlap nugget text → passage
        nug_overlap = keyword_overlap(nugget["text"], p_text)
        score += nug_overlap * 0.2

        # NLI boost (optional)
        if use_nli and score > 0:
            nli = _nli_score(p_text, nugget["text"], nli_model)
            score += nli * 0.5

        if score > best_score:
            best_score = score
            best = p

    # A passage is considered a valid citation if its composite score is
    # above a reasonable threshold (empirically tuned on ASQA structure)
    CITE_THRESHOLD = 0.35
    if best_score >= CITE_THRESHOLD:
        return True, best
    return False, None


# ──────────────────────────────────────────────
# Main metrics
# ──────────────────────────────────────────────

def compute_nugget_metrics(
    nuggets: list[dict],
    matched_claims: list[dict],
    use_nli: bool = False,
    nli_model: str = "cross-encoder/nli-deberta-v3-large",
    required_only: bool = False,
) -> dict:
    """
    Compute Nugget Precision and Nugget Recall for one example.

    Args:
        nuggets:        List of nugget dicts from the dataset.
        matched_claims: List of matched claim dicts from the pipeline.
        use_nli:        Whether to use NLI for coverage/citation checks.
        nli_model:      NLI model name (if use_nli=True).
        required_only:  If True, only evaluate on nuggets marked required=True.

    Returns dict with:
        nugget_precision    — of covered+cited nuggets / covered nuggets
        nugget_recall       — of covered+cited nuggets / total (required) nuggets
        nugget_coverage     — of nuggets covered by at least one claim / total nuggets
        n_nuggets           — total nuggets evaluated
        n_covered           — nuggets covered by at least one claim
        n_cited             — covered nuggets with at least one valid citation
        per_nugget          — per-nugget breakdown list
    """
    if required_only:
        nuggets = [n for n in nuggets if n.get("required", True)]

    if not nuggets or not matched_claims:
        return {
            "nugget_precision": 0.0,
            "nugget_recall": 0.0,
            "nugget_coverage": 0.0,
            "n_nuggets": len(nuggets),
            "n_covered": 0,
            "n_cited": 0,
            "per_nugget": [],
        }

    per_nugget = []

    for nug in nuggets:
        # Find claims that cover this nugget
        covering_claims = []
        for mc in matched_claims:
            if nugget_covered_by_claim(nug, mc["claim"], use_nli=use_nli, nli_model=nli_model):
                covering_claims.append(mc)

        covered = len(covering_claims) > 0

        # Among covering claims, check if any cite a passage that contains evidence
        cited = False
        best_evidence_passage = None
        best_covering_claim = None

        for mc in covering_claims:
            passages = mc.get("supporting_passages", [])
            is_cited, best_p = nugget_cited_in_passages(
                nug, passages, use_nli=use_nli, nli_model=nli_model
            )
            if is_cited:
                cited = True
                best_evidence_passage = best_p
                best_covering_claim = mc["claim"]
                break

        per_nugget.append({
            "nugget_id": nug.get("nugget_id", "?"),
            "nugget_text": nug["text"],
            "required": nug.get("required", True),
            "keywords": nug.get("keywords", []),
            "golden_passage_title": nug.get("golden_passage_title"),
            "golden_evidence": nug.get("golden_evidence"),
            "covered": covered,
            "cited": cited,
            "n_covering_claims": len(covering_claims),
            "best_covering_claim": best_covering_claim,
            "best_evidence_passage_title": (
                best_evidence_passage.get("title") if best_evidence_passage else None
            ),
            "best_evidence_passage_text": (
                best_evidence_passage.get("text", "")[:200]
                if best_evidence_passage else None
            ),
        })

    n_covered = sum(1 for r in per_nugget if r["covered"])
    n_cited   = sum(1 for r in per_nugget if r["cited"])
    n_total   = len(per_nugget)

    nugget_precision = n_cited / n_covered if n_covered > 0 else 0.0
    nugget_recall    = n_cited / n_total   if n_total > 0  else 0.0
    nugget_coverage  = n_covered / n_total if n_total > 0  else 0.0

    return {
        "nugget_precision": round(nugget_precision, 4),
        "nugget_recall":    round(nugget_recall,    4),
        "nugget_coverage":  round(nugget_coverage,  4),
        "n_nuggets":  n_total,
        "n_covered":  n_covered,
        "n_cited":    n_cited,
        "per_nugget": per_nugget,
    }


# ──────────────────────────────────────────────
# API-friendly wrapper (called from Flask/FastAPI)
# ──────────────────────────────────────────────

def evaluate_nuggets_api(payload: dict) -> dict:
    """
    Wrapper for the backend API endpoint.

    Expected payload:
    {
        "matched_claims": [...],   # from pipeline step 4
        "nuggets": [...],          # from the nugget dataset for this example
        "use_nli": false,          # optional
        "required_only": false     # optional, filter only required nuggets
    }

    Returns the same dict as compute_nugget_metrics.
    """
    matched_claims = payload.get("matched_claims", [])
    nuggets        = payload.get("nuggets", [])
    use_nli        = payload.get("use_nli", False)
    required_only  = payload.get("required_only", False)
    nli_model      = payload.get("nli_model", "cross-encoder/nli-deberta-v3-large")

    return compute_nugget_metrics(
        nuggets=nuggets,
        matched_claims=matched_claims,
        use_nli=use_nli,
        nli_model=nli_model,
        required_only=required_only,
    )


# ──────────────────────────────────────────────
# Batch runner (CLI)
# ──────────────────────────────────────────────

def run_batch(
    pipeline_results_path: str,
    nugget_dataset_path: str,
    output_path: str,
    use_nli: bool = False,
    required_only: bool = False,
):
    """
    Batch evaluation over a full pipeline results file.

    pipeline_results_path: JSON with list of {question, matched_claims, ...}
    nugget_dataset_path:   JSON with list of {question, nuggets, ...}
    """
    with open(pipeline_results_path) as f:
        pipeline_data = json.load(f)

    with open(nugget_dataset_path) as f:
        nugget_data = json.load(f)

    # Index nugget dataset by question text (normalize whitespace)
    nugget_index = {
        ex["question"].strip().lower(): ex["nuggets"]
        for ex in nugget_data
        if "nuggets" in ex
    }

    per_example = []
    all_precision = []
    all_recall    = []
    all_coverage  = []

    for ex in pipeline_data:
        question = ex.get("question", "").strip().lower()
        matched  = ex.get("matched_claims", [])
        nuggets  = nugget_index.get(question, [])

        if not nuggets:
            print(f"  ⚠ No nuggets found for: {question[:60]}")
            continue

        metrics = compute_nugget_metrics(
            nuggets=nuggets,
            matched_claims=matched,
            use_nli=use_nli,
            required_only=required_only,
        )
        per_example.append({
            "question": ex.get("question", ""),
            **metrics,
        })
        all_precision.append(metrics["nugget_precision"])
        all_recall.append(metrics["nugget_recall"])
        all_coverage.append(metrics["nugget_coverage"])

    results = {
        "num_examples": len(per_example),
        "metrics": {
            "nugget_precision": round(float(np.mean(all_precision)), 4) if all_precision else 0.0,
            "nugget_recall":    round(float(np.mean(all_recall)),    4) if all_recall    else 0.0,
            "nugget_coverage":  round(float(np.mean(all_coverage)),  4) if all_coverage  else 0.0,
        },
        "per_example": per_example,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nNugget evaluation saved → {output_path}")
    for k, v in results["metrics"].items():
        print(f"  {k}: {v:.4f}")


# ──────────────────────────────────────────────
# Quick smoke test
# ──────────────────────────────────────────────

def _smoke_test():
    nuggets = [
        {
            "nugget_id": "n0",
            "text": "Josef Bican holds the record for the highest number of goals all-time in men's football.",
            "keywords": ["Bican", "Josef Bican"],
            "golden_passage_title": "Josef Bican",
            "golden_evidence": "RSSSF estimates that he scored at least 805 goals in all competitive matches.",
            "required": True,
        },
        {
            "nugget_id": "n1",
            "text": "Ali Daei holds the record for the highest goals in men's international football.",
            "keywords": ["Daei", "Ali Daei"],
            "golden_passage_title": None,
            "golden_evidence": None,
            "required": True,
        },
        {
            "nugget_id": "n2",
            "text": "Christine Sinclair holds the record for the highest goals in women's international football.",
            "keywords": ["Sinclair", "Christine Sinclair"],
            "golden_passage_title": None,
            "golden_evidence": None,
            "required": False,
        },
    ]

    matched_claims = [
        {
            "claim": "Josef Bican scored at least 805 goals and is the all-time leading scorer in men's football.",
            "supporting_passages": [
                {
                    "title": "Josef Bican",
                    "text": "RSSSF estimates that he scored at least 805 goals in all competitive matches, which would make him the most prolific scorer of all time.",
                    "entailment_score": 0.91,
                }
            ],
        },
        {
            "claim": "Ali Daei has the most international goals in men's football history.",
            "supporting_passages": [],  # no citation
        },
    ]

    result = compute_nugget_metrics(nuggets, matched_claims)
    print("\n=== Smoke test ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))


# ──────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nugget-based evaluation")
    subparsers = parser.add_subparsers(dest="command")

    # batch
    batch_p = subparsers.add_parser("batch", help="Run batch evaluation")
    batch_p.add_argument("--pipeline",  required=True, help="Pipeline results JSON")
    batch_p.add_argument("--nuggets",   required=True, help="Nugget dataset JSON")
    batch_p.add_argument("--output",    default="results/nugget_eval.json")
    batch_p.add_argument("--use-nli",   action="store_true")
    batch_p.add_argument("--required-only", action="store_true")

    # test
    subparsers.add_parser("test", help="Run smoke test")

    args = parser.parse_args()

    if args.command == "batch":
        run_batch(
            args.pipeline,
            args.nuggets,
            args.output,
            use_nli=args.use_nli,
            required_only=args.required_only,
        )
    elif args.command == "test":
        _smoke_test()
    else:
        parser.print_help()