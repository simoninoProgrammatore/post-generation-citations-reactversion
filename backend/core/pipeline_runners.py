"""Funzioni che orchestrano gli step del pipeline.

Versione FastAPI: rimosso @st.cache_resource, usiamo functools.lru_cache
per il caching dei modelli NLI e embedding.
"""

from functools import lru_cache


@lru_cache(maxsize=1)
def get_nli_model(model_name: str):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name)


@lru_cache(maxsize=1)
def get_embedding_model(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def run_generate(query: str, model: str, passages: list[dict] | None = None) -> str:
    from core.llm_client import call_llm

    if passages:
        passages_text = "\n\n".join([
            f"[{i+1}] {p.get('title', 'N/A')}:\n{p.get('text', '')}"
            for i, p in enumerate(passages[:10])
        ])
        prompt = (
            "You are a knowledgeable assistant. "
            "Answer the question using the information provided in the passages below.\n\n"
            "IMPORTANT RULES:\n"
            "- Use ONLY the information from the passages. Do NOT use external knowledge.\n"
            "- Write a DETAILED answer of at least 3-5 sentences.\n"
            "- Include specific facts, names, dates, and numbers from the passages.\n"
            "- Do NOT include citations, references, or source numbers like [1] or [2].\n"
            "- Write in plain text without markdown formatting.\n"
            "- If the passages contain multiple relevant details, include ALL of them.\n\n"
            f"Passages:\n{passages_text}\n\n"
            f"Question: {query}\n\n"
            "Provide a detailed, informative answer:"
        )
    else:
        prompt = (
            "You are a knowledgeable assistant. "
            "Answer the question with a detailed, informative response.\n\n"
            "IMPORTANT RULES:\n"
            "- Write at least 3-5 sentences with specific facts and details.\n"
            "- Do NOT use markdown formatting, headers, or bullet points.\n"
            "- Do NOT include any citations or references.\n"
            "- Write in plain, flowing prose.\n\n"
            f"Question: {query}\n\n"
            "Provide a detailed, informative answer:"
        )

    return call_llm(prompt, model=model)


def run_decompose(response: str, model: str) -> list[str]:
    from core.llm_client import call_llm_json
    prompt = f"""\
Break the following text into independent atomic facts.

RULES:
- Each fact must contain exactly ONE piece of information
- Each fact must be a COMPLETE sentence, understandable on its own without any context
- Include the SUBJECT in every fact (never use pronouns like "he", "it", "they")
- Include specific details: names, dates, numbers, locations

EXAMPLE:
Text: "Albert Einstein was born in Ulm, Germany in 1879. He developed the theory of relativity and won the Nobel Prize in Physics in 1921."
Output: ["Albert Einstein was born in Ulm, Germany.", "Albert Einstein was born in 1879.", "Albert Einstein developed the theory of relativity.", "Albert Einstein won the Nobel Prize in Physics.", "Albert Einstein won the Nobel Prize in 1921."]

Now do the same for this text. Return ONLY a JSON array of strings, nothing else.

Text:
{response}
"""
    return call_llm_json(prompt, model=model)


def run_retrieve(
    claims: list[str],
    passages: list[dict],
    method: str,
    threshold: float,
    top_k: int,
    nuggets: list[dict] | None = None,
    pre_filter_k: int = 0,
) -> tuple[list[dict], list[dict]]:
    from core.retrieve import (
        match_with_nli, match_with_similarity, match_with_llm, extract_evidence
    )

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
            matches = match_with_llm(claim, passages, threshold=threshold, top_k=top_k)
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


def _find_best_nugget(
    claim: str,
    nuggets: list[dict],
    supporting_passages: list[dict],
) -> dict | None:
    """
    Trova il nugget migliore per un claim usando keyword matching.
    Se più nugget matchano, sceglie quello con più keyword overlap
    e, a parità, con entailment score più alto dai passaggi.
    """
    import re

    claim_lower = claim.lower()
    best = None
    best_score = -1.0

    for nug in nuggets:
        keywords = nug.get("keywords", [])
        if not keywords:
            continue

        # Check which keywords match
        matched_kws = [kw for kw in keywords if kw.lower() in claim_lower]
        if not matched_kws:
            continue

        # Score: fraction of keywords matched + bonus for total chars matched
        kw_fraction = len(matched_kws) / len(keywords)
        char_coverage = sum(len(kw) for kw in matched_kws) / max(len(claim), 1)

        # Entailment boost: best score from supporting passages
        ent_boost = 0.0
        if supporting_passages:
            ent_boost = max(
                (p.get("entailment_score", 0) or 0) for p in supporting_passages
            ) * 0.1  # small weight

        score = kw_fraction * 0.6 + char_coverage * 0.3 + ent_boost

        if score > best_score:
            best_score = score
            best = {
                "nugget_id": nug.get("nugget_id", ""),
                "text": nug.get("text", ""),
                "keywords": keywords,
                "required": nug.get("required", True),
                "golden_passage_title": nug.get("golden_passage_title"),
                "golden_evidence": nug.get("golden_evidence"),
                "match_score": round(score, 4),
                "matched_keywords": matched_kws,
            }

    return best


def run_cite(response: str, matched_claims: list[dict]) -> tuple[str, list[dict]]:
    from core.cite import build_citation_map, insert_citations
    citation_map = build_citation_map(matched_claims)
    cited, refs = insert_citations(response, matched_claims, citation_map)
    return cited, refs