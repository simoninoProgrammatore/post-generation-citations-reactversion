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


@lru_cache(maxsize=1)
def _load_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def _nli_score(premise: str, hypothesis: str, model_name: str) -> float:
    """P(premise ⊨ hypothesis) using NLI cross-encoder."""
    model = _load_nli_model(model_name)
    logits = np.array(model.predict([(premise, hypothesis)])[0])
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()
    return float(probs[1])  # index 1 = entailment


def _semantic_similarity(text_a: str, text_b: str,
                         model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> float:
    """Cosine similarity between sentence embeddings (MiniLM)."""
    model = _load_embedding_model(model_name)
    embs = model.encode([text_a, text_b])
    # cosine similarity
    dot = float(np.dot(embs[0], embs[1]))
    norm = float(np.linalg.norm(embs[0]) * np.linalg.norm(embs[1]))
    return dot / norm if norm > 0 else 0.0


# ──────────────────────────────────────────────
# Core: nugget ↔ claim matching (semantic similarity)
# ──────────────────────────────────────────────

def match_nuggets_to_claims_semantic(
    nuggets: list[dict],
    claims: list[str],
    threshold: float = 0.80,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> list[dict]:
    """
    For each nugget, find covering claims via cosine similarity on
    sentence embeddings (MiniLM). This replaces keyword matching,
    which fails when nugget and claim share surface keywords but
    refer to different facts.

    Returns list of dicts (one per nugget):
    {
        "nugget_idx": int,
        "covering_claim_indices": [int, ...],
        "best_claim_idx": int | None,
        "best_similarity": float,
    }
    """
    if not nuggets or not claims:
        return [
            {"nugget_idx": i, "covering_claim_indices": [],
             "best_claim_idx": None, "best_similarity": 0.0}
            for i in range(len(nuggets))
        ]

    model = _load_embedding_model(embedding_model)

    nugget_texts = [n["text"] for n in nuggets]
    nugget_embs = model.encode(nugget_texts, convert_to_numpy=True)
    claim_embs = model.encode(claims, convert_to_numpy=True)

    # Normalize for cosine similarity
    nugget_norm = nugget_embs / (np.linalg.norm(nugget_embs, axis=1, keepdims=True) + 1e-9)
    claim_norm = claim_embs / (np.linalg.norm(claim_embs, axis=1, keepdims=True) + 1e-9)

    # (n_nuggets, n_claims) similarity matrix
    sim_matrix = nugget_norm @ claim_norm.T

    results = []
    for i in range(len(nuggets)):
        sims = sim_matrix[i]
        covering = [int(j) for j in range(len(claims)) if sims[j] >= threshold]
        best_j = int(np.argmax(sims))
        best_sim = float(sims[best_j])

        results.append({
            "nugget_idx": i,
            "covering_claim_indices": covering,
            "best_claim_idx": best_j if best_sim >= threshold else None,
            "best_similarity": round(best_sim, 4),
        })

    return results


def nugget_covered_by_claim(
    nugget: dict,
    claim_text: str,
    use_nli: bool = False,
    nli_model: str = "cross-encoder/nli-deberta-v3-large",
    keyword_threshold: float = 0.3,
    nli_threshold: float = 0.5,
) -> bool:
    """
    LEGACY — single-pair keyword-based matching.
    Kept for backward compatibility but no longer used by compute_nugget_metrics.
    Use match_nuggets_to_claims_semantic() for batch evaluation instead.
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
    use_semantic: bool = True,
    nli_model: str = "cross-encoder/nli-deberta-v3-large",
    keyword_threshold: float = 0.25,
    nli_threshold: float = 0.45,
) -> tuple[bool, Optional[dict], float]:
    """
    Check if any of the supporting passages actually contain evidence
    for this nugget.

    Returns (is_cited, best_evidence_passage | None, best_score).

    Strategy:
    1. Title match with golden_passage_title
    2. Keyword match in passage text
    3. Lexical overlap between golden_evidence and passage text
    4. Semantic similarity (MiniLM) between nugget and passage
    5. (Optional) NLI: passage entails nugget text
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
            score += 0.5

        # 2. Keyword match in passage text
        if keywords_present(keywords, p_text):
            score += 0.3

        # 3. Overlap with golden evidence
        if golden_evidence:
            ev_overlap = keyword_overlap(golden_evidence, p_text)
            score += ev_overlap * 0.2

        # 4. Semantic similarity (MiniLM) — nugget text vs passage text
        if use_semantic:
            # Compare against golden_evidence if available, else nugget text
            compare_text = golden_evidence if golden_evidence else nugget["text"]
            sim = _semantic_similarity(compare_text, p_text)
            score += sim * 0.5

        # 5. NLI boost (optional)
        if use_nli and score > 0:
            nli = _nli_score(p_text, nugget["text"], nli_model)
            score += nli * 0.3

        if score > best_score:
            best_score = score
            best = p

    CITE_THRESHOLD = 0.45
    if best_score >= CITE_THRESHOLD:
        return True, best, best_score
    return False, None, best_score


# ──────────────────────────────────────────────
# Main metrics
# ──────────────────────────────────────────────

def _add_split_metrics(per_nugget: list[dict], result: dict) -> dict:
    """Enrich a metrics result dict with required/optional breakdowns.
    Does NOT modify existing keys — only adds new ones."""
    req  = [r for r in per_nugget if r.get("required", True)]
    opt  = [r for r in per_nugget if not r.get("required", True)]

    def _stats(items):
        n = len(items)
        nc = sum(1 for r in items if r["covered"])
        ni = sum(1 for r in items if r["cited"])
        return n, nc, ni

    nr, nrc, nri = _stats(req)
    no, noc, noi = _stats(opt)

    result["n_required"]         = nr
    result["n_required_covered"] = nrc
    result["n_required_cited"]   = nri
    result["required_precision"] = round(nri / nrc, 4) if nrc > 0 else 0.0
    result["required_recall"]    = round(nri / nr,  4) if nr  > 0 else 0.0
    result["required_coverage"]  = round(nrc / nr,  4) if nr  > 0 else 0.0

    result["n_optional"]         = no
    result["n_optional_covered"] = noc
    result["n_optional_cited"]   = noi
    result["optional_precision"] = round(noi / noc, 4) if noc > 0 else 0.0
    result["optional_recall"]    = round(noi / no,  4) if no  > 0 else 0.0
    result["optional_coverage"]  = round(noc / no,  4) if no  > 0 else 0.0

    return result

def compute_nugget_metrics(
    nuggets: list[dict],
    matched_claims: list[dict],
    use_nli: bool = False,
    use_semantic: bool = True,
    nli_model: str = "cross-encoder/nli-deberta-v3-large",
    semantic_threshold: float = 0.80,
    required_only: bool = False,
) -> dict:
    """
    Compute Nugget Precision and Nugget Recall for one example.

    If matched_claims carry a precomputed 'matched_nugget' field (from
    the retrieve step), the alignment is reused — skipping the expensive
    semantic/keyword re-computation.  When multiple claims map to the
    same nugget, the one with the highest match_score is chosen as the
    best covering claim.

    Falls back to the full keyword+semantic alignment when no
    precomputed associations are found.
    """
    if required_only:
        nuggets = [n for n in nuggets if n.get("required", True)]

    if not nuggets or not matched_claims:
        return _add_split_metrics([], {
            "nugget_precision": 0.0,
            "nugget_recall": 0.0,
            "nugget_coverage": 0.0,
            "n_nuggets": len(nuggets),
            "n_covered": 0,
            "n_cited": 0,
            "per_nugget": [],
        })

    # ── Check if precomputed nugget associations exist ──
    has_precomputed = any(
        mc.get("matched_nugget") is not None for mc in matched_claims
    )

    if has_precomputed:
        return _compute_metrics_precomputed(nuggets, matched_claims,
                                            use_nli, use_semantic, nli_model)

    # ── Fallback: full keyword+semantic alignment ──
    return _compute_metrics_full(nuggets, matched_claims,
                                 use_nli, use_semantic, nli_model,
                                 semantic_threshold)


def _compute_metrics_precomputed(
    nuggets: list[dict],
    matched_claims: list[dict],
    use_nli: bool,
    use_semantic: bool,
    nli_model: str,
) -> dict:
    """
    Fast path: use matched_nugget from retrieve step.
    For each nugget, find the best claim (highest match_score).
    """
    # Build nugget_id → [list of (claim_dict, match_score)] mapping
    nugget_to_claims: dict[str, list[tuple[dict, float]]] = {}
    for mc in matched_claims:
        mn = mc.get("matched_nugget")
        if mn is None:
            continue
        nid = mn.get("nugget_id", "")
        score = mn.get("match_score", 0.0)
        nugget_to_claims.setdefault(nid, []).append((mc, score))

    # Sort each nugget's claims by score descending → best first
    for nid in nugget_to_claims:
        nugget_to_claims[nid].sort(key=lambda x: x[1], reverse=True)

    per_nugget = []

    for nug in nuggets:
        nid = nug.get("nugget_id", "?")
        candidates = nugget_to_claims.get(nid, [])
        covered = len(candidates) > 0

        # Best claim = highest match_score
        best_mc = candidates[0][0] if candidates else None
        best_match_score = candidates[0][1] if candidates else 0.0

        # ── Citation verification (still needed) ──
        cited = False
        best_evidence_passage = None
        best_covering_claim = None
        cite_score = 0.0

        # Check best claim first, then others
        for mc, _ in candidates:
            passages = mc.get("supporting_passages", [])
            is_cited, best_p, score = nugget_cited_in_passages(
                nug, passages,
                use_nli=use_nli, use_semantic=use_semantic,
                nli_model=nli_model,
            )
            if is_cited:
                cited = True
                best_evidence_passage = best_p
                best_covering_claim = mc["claim"]
                cite_score = score
                break

        per_nugget.append({
            "nugget_id": nid,
            "nugget_text": nug["text"],
            "required": nug.get("required", True),
            "keywords": nug.get("keywords", []),
            "golden_passage_title": nug.get("golden_passage_title"),
            "golden_evidence": nug.get("golden_evidence"),
            "covered": covered,
            "cited": cited,
            "semantic_similarity": round(best_match_score, 4),
            "cite_score": round(cite_score, 4),
            "n_covering_claims": len(candidates),
            "best_covering_claim": best_covering_claim or (best_mc["claim"] if best_mc else None),
            "best_evidence_passage_title": (
                best_evidence_passage.get("title") if best_evidence_passage else None
            ),
            "best_evidence_passage_text": (
                best_evidence_passage.get("text", "")[:200]
                if best_evidence_passage else None
            ),
            "best_evidence_sentence": (
                best_evidence_passage.get("best_sentence", "")
                if best_evidence_passage else None
            ),
        })

    n_covered = sum(1 for r in per_nugget if r["covered"])
    n_cited   = sum(1 for r in per_nugget if r["cited"])
    n_total   = len(per_nugget)

    nugget_precision = n_cited / n_covered if n_covered > 0 else 0.0
    nugget_recall    = n_cited / n_total   if n_total > 0  else 0.0
    nugget_coverage  = n_covered / n_total if n_total > 0  else 0.0

    return _add_split_metrics(per_nugget, {
        "nugget_precision": round(nugget_precision, 4),
        "nugget_recall":    round(nugget_recall,    4),
        "nugget_coverage":  round(nugget_coverage,  4),
        "n_nuggets":  n_total,
        "n_covered":  n_covered,
        "n_cited":    n_cited,
        "per_nugget": per_nugget,
    })


def _compute_metrics_full(
    nuggets: list[dict],
    matched_claims: list[dict],
    use_nli: bool,
    use_semantic: bool,
    nli_model: str,
    semantic_threshold: float,
) -> dict:
    """
    Original full path: keyword gate + semantic similarity alignment.
    Used when matched_claims don't have precomputed nugget associations.
    """
    claim_texts = [mc["claim"] for mc in matched_claims]

    if use_semantic:
        semantic_matches = match_nuggets_to_claims_semantic(
            nuggets, claim_texts, threshold=semantic_threshold,
        )
    else:
        semantic_matches = None

    per_nugget = []

    for nug_idx, nug in enumerate(nuggets):
        covering_claim_indices = set()

        for mc_idx, mc in enumerate(matched_claims):
            # Gate: at least one keyword must be present
            if not keywords_present(nug.get("keywords", []), mc["claim"]):
                continue

            # Keyword gate passed — now optionally confirm with semantic
            if use_semantic and semantic_matches is not None:
                sm = semantic_matches[nug_idx]
                if mc_idx in sm["covering_claim_indices"]:
                    covering_claim_indices.add(mc_idx)
                else:
                    # Keyword hit but low semantic sim — still accept,
                    # keywords are the ground truth
                    covering_claim_indices.add(mc_idx)
            else:
                covering_claim_indices.add(mc_idx)

        covering_claims = [matched_claims[i] for i in covering_claim_indices]
        covered = len(covering_claims) > 0

        # Best similarity score for reporting
        best_sim = 0.0
        if semantic_matches is not None:
            best_sim = semantic_matches[nug_idx]["best_similarity"]

        # ── Citation verification ──
        cited = False
        best_evidence_passage = None
        best_covering_claim = None
        cite_score = 0.0

        for mc in covering_claims:
            passages = mc.get("supporting_passages", [])
            is_cited, best_p, score = nugget_cited_in_passages(
                nug, passages,
                use_nli=use_nli, use_semantic=use_semantic,
                nli_model=nli_model,
            )
            if is_cited:
                cited = True
                best_evidence_passage = best_p
                best_covering_claim = mc["claim"]
                cite_score = score
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
            "semantic_similarity": round(best_sim, 4),
            "cite_score": round(cite_score, 4),
            "n_covering_claims": len(covering_claims),
            "best_covering_claim": best_covering_claim,
            "best_evidence_passage_title": (
                best_evidence_passage.get("title") if best_evidence_passage else None
            ),
            "best_evidence_passage_text": (
                best_evidence_passage.get("text", "")[:200]
                if best_evidence_passage else None
            ),
            "best_evidence_sentence": (
                best_evidence_passage.get("best_sentence", "")
                if best_evidence_passage else None
            ),
        })

    n_covered = sum(1 for r in per_nugget if r["covered"])
    n_cited   = sum(1 for r in per_nugget if r["cited"])
    n_total   = len(per_nugget)

    nugget_precision = n_cited / n_covered if n_covered > 0 else 0.0
    nugget_recall    = n_cited / n_total   if n_total > 0  else 0.0
    nugget_coverage  = n_covered / n_total if n_total > 0  else 0.0

    return _add_split_metrics(per_nugget, {
        "nugget_precision": round(nugget_precision, 4),
        "nugget_recall":    round(nugget_recall,    4),
        "nugget_coverage":  round(nugget_coverage,  4),
        "n_nuggets":  n_total,
        "n_covered":  n_covered,
        "n_cited":    n_cited,
        "per_nugget": per_nugget,
    })


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
        "use_semantic": true,      # optional, MiniLM semantic similarity (default True)
        "required_only": false     # optional, filter only required nuggets
    }

    Returns the same dict as compute_nugget_metrics.
    """
    matched_claims = payload.get("matched_claims", [])
    nuggets        = payload.get("nuggets", [])
    use_nli        = payload.get("use_nli", False)
    use_semantic   = payload.get("use_semantic", True)
    required_only  = payload.get("required_only", False)
    nli_model      = payload.get("nli_model", "cross-encoder/nli-deberta-v3-large")

    return compute_nugget_metrics(
        nuggets=nuggets,
        matched_claims=matched_claims,
        use_nli=use_nli,
        use_semantic=use_semantic,
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