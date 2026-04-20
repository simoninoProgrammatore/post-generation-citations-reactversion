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
            "Answer the question ONLY using the information provided in the passages below. "
            "Do NOT use any external knowledge. "
            "If the passages do not contain enough information to answer, say so. "
            "Do NOT include any citations or references in your response. "
            "Write in plain text without markdown formatting.\n\n"
            f"Passages:\n{passages_text}\n\n"
            f"Question: {query}\n\nAnswer:"
        )
    else:
        prompt = (
            "You are a knowledgeable assistant. "
            "Answer the question clearly and factually in plain text. "
            "Do NOT use markdown formatting, headers, or bullet points. "
            "Do NOT include any citations or references in your response.\n\n"
            f"Question: {query}\n\nAnswer:"
        )

    return call_llm(prompt, model=model)


def run_decompose(response: str, model: str) -> list[str]:
    from core.llm_client import call_llm_json
    prompt = f"""\
Break the following text into independent atomic facts.
Each fact must:
- Contain exactly one piece of information
- Be self-contained and understandable without context
- Be a complete declarative sentence

Return ONLY a JSON array of strings, no preamble.

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
                claim, passages, threshold=threshold, top_k=top_k, return_all_scores=True
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
        matched.append({"claim": claim, "supporting_passages": matches})
        debug_data.append({"claim": claim, "sentence_scores": sentence_scores})

    return matched, debug_data


def run_cite(response: str, matched_claims: list[dict]) -> tuple[str, list[dict]]:
    from core.cite import build_citation_map, insert_citations
    citation_map = build_citation_map(matched_claims)
    cited, refs = insert_citations(response, matched_claims, citation_map)
    return cited, refs