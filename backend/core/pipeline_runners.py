"""Funzioni che orchestrano gli step del pipeline.

Versione FastAPI: rimosso @st.cache_resource, usiamo functools.lru_cache
per il caching dei modelli NLI e embedding.

Ogni runner è un thin wrapper: la logica vera (prompt, matching, citazione)
vive nei moduli core/. I runner si limitano a orchestrare e a passare i
parametri. Questo evita la duplicazione dei prompt fra core/ e i runner.
"""

import re
import numpy as np
from functools import lru_cache


# ──────────────────────────────────────────────
# Text utilities condivise (una sola copia per tutto il modulo)
# ──────────────────────────────────────────────

# Peso lessicale del tie-break match claim<->nugget: 0.2 lex / 0.8 sem.
# NB: deve restare coerente con MATCH_LEXICAL_WEIGHT in core.nuggets_evaluate.
LEXICAL_WEIGHT = 0.2

# Soglia di covering: STESSO valore del gate in MatchedView (frontend) e di
# COVERAGE_THRESHOLD in core.nuggets_evaluate.
COVERAGE_THRESHOLD = 0.6

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
    """Overlap fra content-word di due stringhe (asimmetrico su a)."""
    a = _tokenize(text_a)
    b = _tokenize(text_b)
    if not a:
        return 0.0
    return len(a & b) / len(a)


def count_matched_keywords(keywords: list[str], text: str) -> int:
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw.lower() in text_lower)


def _cosine(embedding_model, text_a: str, text_b: str) -> float:
    """Cosine singola fra due testi (path non batched)."""
    embs = embedding_model.encode([text_a, text_b], convert_to_numpy=True)
    dot = float(np.dot(embs[0], embs[1]))
    norm = float(np.linalg.norm(embs[0]) * np.linalg.norm(embs[1]))
    return dot / norm if norm > 0 else 0.0


# ──────────────────────────────────────────────
# Model loaders (lazy + cached)
# ──────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_nli_model(model_name: str):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name)


@lru_cache(maxsize=1)
def get_embedding_model(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


# ──────────────────────────────────────────────
# Pipeline steps
# ──────────────────────────────────────────────

def run_generate(query: str, model: str, passages: list[dict] | None = None) -> str:
    """Step 1 — delega interamente a core.generate (single source of truth
    per il prompt di generazione)."""
    from core.generate import generate_response
    return generate_response(query, passages=passages, model=model)


def run_decompose(response: str, model: str) -> list[str]:
    from core.decompose import decompose_with_llm
    return decompose_with_llm(response, model=model)


def run_retrieve(
    claims: list[str],
    passages: list[dict],
    method: str,
    threshold: float,
    top_k: int,
    nuggets: list[dict] | None = None,
    pre_filter_k: int = 0,
    model: str = "claude-haiku-4-5-20251001",
) -> tuple[list[dict], list[dict]]:
    from core.retrieve import match_with_nli, match_with_llm, extract_evidence

    matched = []
    debug_data = []

    for claim in claims:
        sentence_scores = []

        if method == "nli":
            matches, sentence_scores = match_with_nli(
                claim, passages, threshold=threshold, top_k=top_k,
                return_all_scores=True, pre_filter_k=pre_filter_k,
            )
        elif method == "llm":
            matches = match_with_llm(claim, passages, threshold=threshold, top_k=top_k, model=model)
        else:
            matches = match_with_similarity(claim, passages, top_k=top_k)

        for match in matches:
            ev = extract_evidence(
                claim,
                match.get("text", ""),
                best_sentence=match.get("best_sentence", ""),
                extraction_start=match.get("extraction_start", -1),
                extraction_end=match.get("extraction_end", -1),
            )
            match["extraction"] = ev["extraction"]
            match["extraction_start"] = ev["extraction_start"]
            match["extraction_end"] = ev["extraction_end"]
            match["summary"] = ev["summary"]

        matches = [m for m in matches if m.get("extraction", "").strip()]

        # ── Nugget matching (keyword-based, anticipato) ──
        matched_nugget = None
        if nuggets:
            matched_nugget = _find_best_nugget(claim, nuggets, matches)

        entry = {"claim": claim, "supporting_passages": matches}
        if matched_nugget:
            entry["matched_nugget"] = matched_nugget

        matched.append(entry)
        debug_data.append({"claim": claim, "sentence_scores": sentence_scores})

    return matched, debug_data


# ──────────────────────────────────────────────
# Nugget matching
# ──────────────────────────────────────────────

def _claim_nugget_match_score(nugget: dict, claim_text: str,
                              embedding_model) -> tuple[int, float]:
    """(n_keyword PRIMARY, tie SECONDARY). tie = LEXICAL_WEIGHT * overlap
    lessicale(nugget.text, claim) + (1-LEXICAL_WEIGHT) * cosine(nugget.text, claim)."""
    keywords = nugget.get("keywords", [])
    n_kw = count_matched_keywords(keywords, claim_text)
    lexical = keyword_overlap(nugget.get("text", ""), claim_text)
    semantic = _cosine(embedding_model, nugget.get("text", ""), claim_text)
    tie = round(LEXICAL_WEIGHT * lexical + (1.0 - LEXICAL_WEIGHT) * semantic, 4)
    return n_kw, tie


def _find_best_nugget(
    claim: str,
    nuggets: list[dict],
    supporting_passages: list[dict],
    embedding_model=None,
    min_keywords_matched: int = 1,
    coverage_threshold: float = COVERAGE_THRESHOLD,
) -> dict | None:
    """Claim-centered: per UN claim, il nugget migliore (n_kw PRIMARY, tie
    SECONDARY) che supera i due gate. Ritorna None se nessuno passa."""
    if not nuggets:
        return None

    if embedding_model is None:
        embedding_model = get_embedding_model("all-MiniLM-L6-v2")

    best = None
    best_score = (-1, -1.0)

    for nug in nuggets:
        keywords = nug.get("keywords", [])
        if not keywords:
            continue

        n_kw, tie = _claim_nugget_match_score(nug, claim, embedding_model)

        # Gate 1: almeno min_keywords_matched keyword esatte nel claim
        if n_kw < min_keywords_matched:
            continue
        # Gate 2: tie sopra coverage_threshold
        if tie < coverage_threshold:
            continue

        if (n_kw, tie) > best_score:
            best_score = (n_kw, tie)
            best = {
                "nugget_id":              nug.get("nugget_id", ""),
                "text":                   nug.get("text", ""),
                "keywords":               keywords,
                "required":               nug.get("required", True),
                "golden_passage_title":   nug.get("golden_passage_title"),
                "golden_evidence":        nug.get("golden_evidence"),
                "match_score":            tie,
                "matched_keywords_count": n_kw,
            }

    return best


def find_covering_claims_for_nugget(
    nugget: dict,
    claims: list[dict],  # matched_claims con "claim" e "supporting_passages"
    embedding_model=None,
    min_keywords_matched: int = 1,
    coverage_threshold: float = COVERAGE_THRESHOLD,
) -> list[dict]:
    """UNUSED — conservata per riferimento. Lo Step 6 usa
    core.nuggets_evaluate._group_like_frontend (fonte di verita' = MatchedView).

    Nugget-centered: per UN nugget, tutti i claim che lo coprono, ordinati per
    score decrescente (n_kw PRIMARY, tie SECONDARY). Usa encode batched."""
    if embedding_model is None:
        embedding_model = get_embedding_model("all-MiniLM-L6-v2")

    keywords = nugget.get("keywords", [])
    nugget_text = nugget.get("text", "")
    claim_texts = [mc.get("claim", "") for mc in claims]

    if not claim_texts:
        return []

    # Encode batched: nugget + tutti i claim in una chiamata sola.
    # normalize_embeddings=True ⇒ la cosine si riduce a un prodotto scalare.
    all_embs = embedding_model.encode(
        [nugget_text] + claim_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    nugget_emb = all_embs[0]
    claim_embs = all_embs[1:]
    semantic_scores = claim_embs @ nugget_emb  # shape (n_claims,)

    covering = []
    for idx, mc in enumerate(claims):
        claim_text = claim_texts[idx]
        n_kw = count_matched_keywords(keywords, claim_text)

        # Gate 1: almeno 1 keyword esatta
        if n_kw < min_keywords_matched:
            continue

        lexical = keyword_overlap(nugget_text, claim_text)
        semantic = float(semantic_scores[idx])
        tie = round(LEXICAL_WEIGHT * lexical + (1.0 - LEXICAL_WEIGHT) * semantic, 4)

        # Gate 2: tie sopra threshold
        if tie < coverage_threshold:
            continue

        covering.append({
            "claim":                  claim_text,
            "supporting_passages":    mc.get("supporting_passages", []),
            "matched_keywords_count": n_kw,
            "match_score":            tie,
        })

    covering.sort(key=lambda x: (x["matched_keywords_count"], x["match_score"]), reverse=True)
    return covering


# ──────────────────────────────────────────────
# Cite / Evaluate
# ──────────────────────────────────────────────

def run_cite(response: str, matched_claims: list[dict]) -> tuple[str, list[dict]]:
    from core.cite import build_citation_map, insert_citations
    citation_map = build_citation_map(matched_claims)
    cited, refs = insert_citations(response, matched_claims, citation_map)
    return cited, refs


def run_evaluate_deepseek(matched_claims: list[dict], model: str = "deepseek-v4-flash") -> dict:
    """Step 6 (modalita DeepSeek) — LLM-as-judge sincrono.

    Thin wrapper su core.deepseek_evaluate. NB: usare la variante _async
    dentro un endpoint FastAPI (loop gia' attivo); questo wrapper sincrono
    serve a CLI / contesti non-async.
    """
    from core.deepseek_evaluate import evaluate_matched_deepseek
    return evaluate_matched_deepseek(matched_claims, model=model)