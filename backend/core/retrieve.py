"""
Step 3: Match atomic claims to supporting evidence passages.

Given a list of claims and a pool of candidate passages (provided 
by ALCE), this module determines which passages support each claim.
"""

import re
import json
import argparse
from pathlib import Path
from functools import lru_cache


# ──────────────────────────────────────────────
# Evidence extraction
# ──────────────────────────────────────────────

def extract_evidence(claim: str, passage_text: str, best_sentence: str = "",
                     extraction_start: int = -1, extraction_end: int = -1,
                     model_name: str = "cross-encoder/nli-deberta-v3-large") -> dict:
    """
    Extract the supporting sentence from the passage.
    If best_sentence and span are provided (from match_with_nli), use them directly.
    Otherwise fall back to NLI scoring.
    """
    # Se abbiamo già span dal matching, usali direttamente
    if best_sentence and extraction_start >= 0:
        return {
            "extraction": best_sentence,
            "extraction_start": extraction_start,
            "extraction_end": extraction_end,
            "summary": f"Matched at [{extraction_start}:{extraction_end}]",
        }

    # Fallback per similarity/llm methods
    import numpy as np

    sentences = _split_passage_into_sentences(passage_text)
    if not sentences:
        return {"extraction": "", "extraction_start": -1, "extraction_end": -1, "summary": "No sentences found."}

    model = _load_nli_model(model_name)
    pairs = [(s, claim) for s in sentences]
    scores = model.predict(pairs)
    scores = np.array(scores)
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)
    exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    entailment_scores = probs[:, 1]

    best_idx = int(np.argmax(entailment_scores))
    best_score = float(entailment_scores[best_idx])
    best_sentence = sentences[best_idx]

    if best_score < 0.5:
        return {"extraction": "", "extraction_start": -1, "extraction_end": -1, "summary": "No direct support found."}

    start = passage_text.find(best_sentence)
    end = start + len(best_sentence) if start != -1 else -1

    return {
        "extraction": best_sentence,
        "extraction_start": start,
        "extraction_end": end,
        "summary": f"Entailment score: {best_score:.3f}",
    }


# ──────────────────────────────────────────────
# Sentence splitting
# ──────────────────────────────────────────────

def _split_passage_into_sentences(text: str) -> list[str]:
    """
    Split a passage into individual sentences.
    Kept for backward compatibility with similarity/llm methods.
    """
    protected = text
    abbreviations = ["Mr.", "Mrs.", "Ms.", "Dr.", "Jr.", "Sr.", "Prof.",
                     "Inc.", "Ltd.", "Corp.", "vs.", "etc.", "approx.",
                     "U.S.", "U.K.", "E.U."]
    placeholders = {}
    for i, abbr in enumerate(abbreviations):
        placeholder = f"__ABBR{i}__"
        placeholders[placeholder] = abbr
        protected = protected.replace(abbr, placeholder)

    sentences = re.split(r'(?<=[.!?])\s+', protected.strip())

    restored = []
    for sent in sentences:
        for placeholder, abbr in placeholders.items():
            sent = sent.replace(placeholder, abbr)
        sent = sent.strip()
        if sent:
            restored.append(sent)

    return restored


def _split_passage_with_spans(text: str) -> list[tuple[str, int, int]]:
    """
    Split a passage into sentences, returning (sentence, start, end) tuples
    with positions relative to the original text.
    """
    abbreviations = ["Mr.", "Mrs.", "Ms.", "Dr.", "Jr.", "Sr.", "Prof.",
                     "Inc.", "Ltd.", "Corp.", "vs.", "etc.", "approx.",
                     "U.S.", "U.K.", "E.U."]

    protected = text
    placeholders = {}
    for i, abbr in enumerate(abbreviations):
        placeholder = f"__ABBR{i}__"
        placeholders[placeholder] = abbr
        protected = protected.replace(abbr, placeholder)

    results = []
    for match in re.finditer(r'[^.!?]*[.!?]+', protected):
        sent_protected = match.group().strip()
        if not sent_protected:
            continue

        start = match.start()
        end = match.end()

        original_sent = text[start:end].strip()

        stripped_start = start + (len(text[start:end]) - len(text[start:end].lstrip()))
        stripped_end = stripped_start + len(original_sent)

        if original_sent:
            results.append((original_sent, stripped_start, stripped_end))

    return results


# ──────────────────────────────────────────────
# NLI model loading
# ──────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_nli_model(model_name: str):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name)


# ──────────────────────────────────────────────
# Embedding model for pre-filtering (reranker)
# ──────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_reranker_embedding(model_name: str = "BAAI/bge-base-en-v1.5"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def _pre_filter_sentences(
    claim: str,
    all_sentences: list[str],
    pair_to_passage: list[int],
    pair_to_span: list[tuple[str, int, int]],
    pre_filter_k: int = 10,
    model_name: str = "BAAI/bge-base-en-v1.5",
) -> tuple[list[str], list[int], list[tuple[str, int, int]]]:
    """
    Pre-filter sentences using fast embedding cosine similarity.
    Returns only the top-K most similar sentences (with their metadata).
    
    Uses BAAI/bge-base-en-v1.5: strong retrieval-oriented embeddings,
    768-dim, ~110M params. Much faster than NLI cross-encoder but
    accurate enough for candidate selection.
    """
    import numpy as np

    if len(all_sentences) <= pre_filter_k:
        # No need to filter if we already have fewer sentences
        return all_sentences, pair_to_passage, pair_to_span

    model = _load_reranker_embedding(model_name)

    # Encode claim and all sentences in one batch
    claim_emb = model.encode([claim], normalize_embeddings=True)
    sent_embs = model.encode(all_sentences, normalize_embeddings=True)

    # Cosine similarity (already normalized)
    sims = (sent_embs @ claim_emb.T).flatten()

    # Get top-K indices
    top_indices = np.argsort(sims)[::-1][:pre_filter_k]
    # Keep original order for stability
    top_indices = sorted(top_indices)

    filtered_sents = [all_sentences[i] for i in top_indices]
    filtered_passages = [pair_to_passage[i] for i in top_indices]
    filtered_spans = [pair_to_span[i] for i in top_indices]

    return filtered_sents, filtered_passages, filtered_spans


# ──────────────────────────────────────────────
# NLI-based matching (sentence-level)
# ──────────────────────────────────────────────

def match_with_nli(
    claim: str,
    passages: list[dict],
    model_name: str = "cross-encoder/nli-deberta-v3-large",
    threshold: float = 0.5,
    top_k: int = 3,
    return_all_scores: bool = False,
    pre_filter_k: int = 0,              # 0 = no pre-filtering
    reranker_model: str = "BAAI/bge-base-en-v1.5",
) -> list[dict] | tuple[list[dict], list[dict]]:
    if not passages:
        return []

    import numpy as np

    model = _load_nli_model(model_name)

    all_sents = []
    pair_to_passage = []
    pair_to_span = []  # (sentence_text, start, end)

    for p_idx, p in enumerate(passages):
        spans = _split_passage_with_spans(p.get("text", ""))
        if not spans:
            continue
        for sent, start, end in spans:
            all_sents.append(sent)
            pair_to_passage.append(p_idx)
            pair_to_span.append((sent, start, end))

    if not all_sents:
        return []

    n_total = len(all_sents)

    # ── Pre-filtering with embedding reranker ──
    if pre_filter_k > 0 and n_total > pre_filter_k:
        filtered_sents, filtered_passages, filtered_spans = _pre_filter_sentences(
            claim, all_sents, pair_to_passage, pair_to_span,
            pre_filter_k=pre_filter_k,
            model_name=reranker_model,
        )
    else:
        filtered_sents = all_sents
        filtered_passages = pair_to_passage
        filtered_spans = pair_to_span

    # ── NLI on filtered sentences ──
    all_pairs = [(sent, claim) for sent in filtered_sents]

    scores = model.predict(all_pairs)
    scores = np.array(scores)
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)

    exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    entailment_scores = probs[:, 1]

    # Per ogni passaggio, traccia score migliore E span corrispondente
    passage_best: dict[int, tuple[float, str, int, int]] = {}
    for i, score in enumerate(entailment_scores):
        p_idx = filtered_passages[i]
        sent, start, end = filtered_spans[i]
        if p_idx not in passage_best or score > passage_best[p_idx][0]:
            passage_best[p_idx] = (float(score), sent, start, end)

    results = []
    for p_idx, (best_score, best_sent, start, end) in passage_best.items():
        if best_score >= threshold:
            results.append({
                **passages[p_idx],
                "entailment_score": best_score,
                "best_sentence": best_sent,
                "extraction_start": start,
                "extraction_end": end,
            })

    results.sort(key=lambda x: x["entailment_score"], reverse=True)
    
    if return_all_scores:
        # Costruisci struttura debug per la UI
        # Include ALL sentences (filtered get NLI scores, rest get 0)
        all_scores_by_passage: dict[int, list[dict]] = {}

        # Sentences that went through NLI
        filtered_set = set()
        for i, score in enumerate(entailment_scores):
            p_idx = filtered_passages[i]
            sent, start, end = filtered_spans[i]
            best_score_for_passage = passage_best.get(p_idx, (0,))[0]
            all_scores_by_passage.setdefault(p_idx, []).append({
                "text": sent,
                "score": float(score),
                "is_best": float(score) == best_score_for_passage,
                "pre_filtered": True,
            })
            filtered_set.add((p_idx, sent))

        # Add non-filtered sentences with score 0 (if pre-filtering was active)
        if pre_filter_k > 0 and n_total > pre_filter_k:
            for i, sent in enumerate(all_sents):
                p_idx = pair_to_passage[i]
                if (p_idx, sent) not in filtered_set:
                    all_scores_by_passage.setdefault(p_idx, []).append({
                        "text": sent,
                        "score": 0.0,
                        "is_best": False,
                        "pre_filtered": False,
                    })

        sentence_scores = [
            {
                "title": passages[p_idx].get("title", f"Passage {p_idx}"),
                "sentences": sents,
                "n_total_sentences": sum(1 for s in all_sents if pair_to_passage[all_sents.index(s)] == p_idx) if pre_filter_k > 0 else len(sents),
            }
            for p_idx, sents in all_scores_by_passage.items()
        ]

        return results[:top_k], sentence_scores

    return results[:top_k]


# ──────────────────────────────────────────────
# Similarity-based matching (baseline)
# ──────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_embedding_model(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def match_with_similarity(
    claim: str,
    passages: list[dict],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    top_k: int = 3,
) -> list[dict]:
    if not passages:
        return []

    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    model = _load_embedding_model(model_name)
    claim_emb = model.encode([claim])
    passage_embs = model.encode([p["text"] for p in passages])
    sims = cosine_similarity(claim_emb, passage_embs)[0]

    ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:top_k]
    return [{**passages[i], "similarity_score": float(score)} for i, score in ranked]


# ──────────────────────────────────────────────
# LLM-based matching (Claude re-ranker)
# ──────────────────────────────────────────────

def match_with_llm(
    claim: str,
    passages: list[dict],
    threshold: float = 0.5,
    top_k: int = 3,
) -> list[dict]:
    from core.llm_client import call_llm_json

    if not passages:
        return []

    # Step 1: pre-filtra con similarity → top 10
    candidates = match_with_similarity(claim, passages, top_k=10)
    if not candidates:
        return []

    # Step 2: LLM re-ranking + extraction in un'unica chiamata
    passages_text = "\n\n".join([
        f"[{i}] {p.get('title', 'N/A')}: {p.get('text', '')[:500]}"
        for i, p in enumerate(candidates)
    ])

    prompt = f"""You are a fact-checking assistant.

Claim: "{claim}"

For each passage below, decide if it SUPPORTS the claim (entails it), CONTRADICTS it, or is NEUTRAL.
If it SUPPORTS, extract the EXACT sentence from the passage that best supports the claim.
Copy the sentence verbatim — do not paraphrase.

Return ONLY a JSON array like:
[{{"idx": 0, "label": "supports", "score": 0.95, "evidence": "The exact sentence from the passage."}}, ...]

For non-supporting passages, set "evidence" to "".

Passages:
{passages_text}
"""

    try:
        results = call_llm_json(prompt)
    except Exception:
        return candidates[:top_k]

    scored = []
    for r in results:
        idx = r.get("idx")
        if (
            isinstance(idx, int)
            and 0 <= idx < len(candidates)
            and r.get("label") == "supports"
            and r.get("score", 0) >= threshold
        ):
            p = candidates[idx]
            evidence = r.get("evidence", "").strip()
            passage_text = p.get("text", "")

            # Trova lo span dell'evidenza nel passaggio
            extraction_start = -1
            extraction_end = -1
            if evidence:
                extraction_start = passage_text.find(evidence)
                if extraction_start != -1:
                    extraction_end = extraction_start + len(evidence)

            scored.append({
                **p,
                "entailment_score": float(r["score"]),
                "best_sentence": evidence,
                "extraction_start": extraction_start,
                "extraction_end": extraction_end,
            })

    scored.sort(key=lambda x: x["entailment_score"], reverse=True)
    return scored[:top_k]


# ──────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────

def run(input_path: str, output_path: str, method: str = "nli", extract: bool = True):
    """
    Match claims to evidence passages and optionally extract
    the exact supporting sentence from each passage.

    Args:
        input_path:  Path to claims JSON (from Step 2).
        output_path: Path to save matched claims.
        method:      Matching method ('nli', 'similarity', 'llm').
        extract:     If True, run evidence extraction on each match.
    """
    with open(input_path, "r") as f:
        data = json.load(f)

    match_fn = {
        "nli": match_with_nli,
        "similarity": match_with_similarity,
        "llm": match_with_llm,
    }[method]

    for example in data:
        passages = example.get("passages", [])
        matched_claims = []

        for claim in example["claims"]:
            matches = match_fn(claim, passages)

            if extract:
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

            matched_claims.append({
                "claim": claim,
                "supporting_passages": matches,
            })

        example["matched_claims"] = matched_claims

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    supported = sum(
        1 for ex in data
        for mc in ex["matched_claims"]
        if mc["supporting_passages"]
    )
    total = sum(len(ex["matched_claims"]) for ex in data)
    print(f"Matched {supported}/{total} claims with evidence -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match claims to evidence")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/matched.json")
    parser.add_argument("--method", type=str, default="nli", choices=["nli", "similarity", "llm"])
    parser.add_argument("--no-extract", action="store_true",
                        help="Skip evidence extraction")
    args = parser.parse_args()
    run(args.input, args.output, args.method, extract=not args.no_extract)