"""
Nugget-based evaluation for post-hoc citation generation.

DUE NOZIONI DI MATCH, TENUTE DISTINTE:

  (1) COVERING claim<->nugget  (fonte di verita' = MatchedView, frontend).
      Un claim "matcha" un nugget se _claim_covers_nugget e' vero
      (matched_nugget.nugget_id corrisponde OPPURE una keyword del nugget e'
      nel testo del claim) E passa il gate match_score >= COVERAGE_THRESHOLD,
      con lo score preso COSI' COM'E' da matched_nugget.match_score (mai
      ricalcolato, come nel frontend). Replicato da _group_like_frontend.
      Questo definisce QUALI CLAIM SONO IN GIOCO: i "claim matched" sono i
      claim che coprono >= 1 nugget.

  (2) CORRETTEZZA della coppia (claim, span)  (soglia tau).
      Una coppia e' CORRETTA se il suo span supera MATCH_THRESHOLD contro la
      golden evidence di almeno uno dei nugget che il claim COPRE (nozione 1),
      con lo score ibrido (EVIDENCE_LEXICAL_WEIGHT lessicale + resto semantico).

METRICHE (calcolate SOLO sui claim matched — i claim che non coprono nessun
nugget sono esclusi da numeratore E denominatore):

    citation precision = #coppie corrette / #coppie totali        (sui claim matched)
    citation recall    = #claim matched con >=1 coppia corretta
                         / #claim matched

  recall NON degenera a 1.0: un claim puo' coprire un nugget per keyword/
  match_score (nozione 1) ma avere span che non superano tau contro la golden
  (nozione 2) -> sta al denominatore ma non al numeratore.

DIAGNOSTICA: coverage dei nugget (di quali fatti del gold il sistema parla,
con evidenza che matcha), spezzata su required e optional. Un nugget e'
coperto se almeno una coppia corretta di un claim che lo copre lo matcha.
I nugget senza golden_evidence sono esclusi dal totale (non matchabili).

  Aggregazione sul dataset: media per domanda (macro), gestita dal chiamante.
"""

import re
import json
import argparse
import numpy as np
from .cite import word_tokens, idf_weights, containment as idf_containment
from functools import lru_cache
from typing import Optional


# ──────────────────────────────────────────────
# Costanti (manopole di tuning — toccare QUI)
# ──────────────────────────────────────────────

# Gate del covering claim<->nugget. STESSO valore del gate in MatchedView
# (frontend):  if (match_score >= COVERAGE_THRESHOLD) covering.push(...)
# Cambiare qui E nel frontend insieme.
COVERAGE_THRESHOLD = 0.6

# Soglia unica tau sul match span_estratto <-> golden_evidence.
MATCH_THRESHOLD = 0.5

# Score ibrido per il match evidenza: 0.2 lessicale / 0.8 semantico.
EVIDENCE_LEXICAL_WEIGHT = 0.2


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
    """Overlap fra content-word di due stringhe (asimmetrico su a)."""
    a = _tokenize(text_a)
    b = _tokenize(text_b)
    if not a:
        return 0.0
    return len(a & b) / len(a)


def count_matched_keywords(keywords: list[str], text: str) -> int:
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw.lower() in text_lower)


# ──────────────────────────────────────────────
# Model loaders (lazy + cached)
# ──────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


# ──────────────────────────────────────────────
# Embedding cache batched (per-esempio)
# ──────────────────────────────────────────────

class _EmbCache:
    """Encoda ogni testo UNA sola volta (batch) e serve cosine da cache."""

    def __init__(self, texts: list[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        uniq = sorted({t for t in texts if t and t.strip()})
        self._idx = {t: i for i, t in enumerate(uniq)}
        if uniq:
            model = _load_embedding_model(model_name)
            self._embs = model.encode(uniq, convert_to_numpy=True, normalize_embeddings=True)
        else:
            self._embs = np.zeros((0, 1))

    def cos(self, a: str, b: str) -> float:
        if not a or not b or not a.strip() or not b.strip():
            return 0.0
        ia = self._idx.get(a)
        ib = self._idx.get(b)
        if ia is None or ib is None:
            return 0.0
        return float(self._embs[ia] @ self._embs[ib])


def _semantic_similarity(text_a: str, text_b: str,
                         model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> float:
    """Cosine singola (path legacy / fuori dal batch)."""
    model = _load_embedding_model(model_name)
    embs = model.encode([text_a, text_b], convert_to_numpy=True, normalize_embeddings=True)
    return float(embs[0] @ embs[1])


# ──────────────────────────────────────────────
# Match evidenza: UN solo confronto span_estratto <-> golden_evidence
# ──────────────────────────────────────────────

def _extracted_span(passage: dict) -> str:
    """Lo span di evidenza prodotto dal retrieve per un passaggio."""
    return (passage.get("extraction", "") or passage.get("best_sentence", "") or "").strip()


def _match_score(span, golden_evidence, emb, idf=None, use_semantic=True):
    if not span or not golden_evidence:
        return 0.0
    
    if idf is not None:
        # IDF-weighted containment (asimmetrico su golden_evidence)
        claim_toks = word_tokens(golden_evidence)
        span_toks  = word_tokens(span)
        lexical = idf_containment(claim_toks, span_toks, idf)
    else:
        lexical = keyword_overlap(golden_evidence, span)  # fallback
    
    if use_semantic:
        semantic = emb.cos(golden_evidence, span) if emb else _semantic_similarity(golden_evidence, span)
    else:
        semantic = 0.0
    
    return round(EVIDENCE_LEXICAL_WEIGHT * lexical + (1.0 - EVIDENCE_LEXICAL_WEIGHT) * semantic, 4)

# ──────────────────────────────────────────────
# Covering claim<->nugget: fonte di verita' = MatchedView (frontend)
# ──────────────────────────────────────────────

def _claim_covers_nugget(mc: dict, nugget: dict) -> bool:
    """Replica esatta di claimCoversNugget (MatchedView):
    matched_nugget.nugget_id corrisponde OPPURE una keyword e' nel testo."""
    mn = mc.get("matched_nugget")
    if mn and mn.get("nugget_id") == nugget.get("nugget_id"):
        return True
    claim_text = (mc.get("claim", "") or "").lower()
    return any(kw.lower() in claim_text for kw in nugget.get("keywords", []))


def _group_like_frontend(nuggets: list[dict], matched_claims: list[dict]) -> dict:
    out: dict[str, list[dict]] = {n.get("nugget_id", "?"): [] for n in nuggets}

    for mc in matched_claims:
        best_nid = None
        best_score = -1.0
        for nug in nuggets:
            if not _claim_covers_nugget(mc, nug):
                continue
            score = (mc.get("matched_nugget") or {}).get("match_score", 0.0) or 0.0
            if score >= COVERAGE_THRESHOLD and score > best_score:
                best_score = score
                best_nid = nug.get("nugget_id", "?")
        if best_nid is not None:
            out[best_nid].append(mc)

    # Ordina per score desc (come prima)
    for nid in out:
        out[nid].sort(
            key=lambda m: (m.get("matched_nugget") or {}).get("match_score", 0.0) or 0.0,
            reverse=True,
        )
    return out

# ──────────────────────────────────────────────
# Noise / emb cache
# ──────────────────────────────────────────────

def _count_noise_usage(matched_claims: list[dict]) -> dict:
    total_supporting = 0
    noise_supporting = 0
    claims_with_noise = 0
    for mc in matched_claims:
        passages = mc.get("supporting_passages", [])
        n_noise = sum(1 for p in passages if p.get("is_noise", False))
        total_supporting += len(passages)
        noise_supporting += n_noise
        if n_noise > 0:
            claims_with_noise += 1
    return {
        "total_supporting_passages": total_supporting,
        "noise_supporting_passages": noise_supporting,
        "claims_citing_noise": claims_with_noise,
        "noise_ratio": round(noise_supporting / total_supporting, 4) if total_supporting > 0 else 0.0,
    }


def _build_emb_cache(nuggets: list[dict], matched_claims: list[dict],
                     use_semantic: bool) -> Optional[_EmbCache]:
    if not use_semantic:
        return None
    texts = []
    for n in nuggets:
        if n.get("golden_evidence"):
            texts.append(n["golden_evidence"])
    for mc in matched_claims:
        for p in mc.get("supporting_passages", []):
            texts.append(_extracted_span(p))
    return _EmbCache(texts)


# ──────────────────────────────────────────────
# per_nugget (per la UI / il dettaglio)
# ──────────────────────────────────────────────

def _build_per_nugget(nuggets_g: list[dict],
                      nugget_to_claims: dict,
                      matches_by_nugget: dict) -> list[dict]:
    out = []
    for n in nuggets_g:
        nid = n.get("nugget_id", "?")
        covering_claims = nugget_to_claims.get(nid, [])          # nozione (1)
        matches = sorted(matches_by_nugget.get(nid, []),         # nozione (2)
                         key=lambda x: x["evidence_score"], reverse=True)
        # "covered" ai fini metriche = esiste >=1 coppia corretta su questo nugget.
        covered_for_metrics = len(matches) > 0
        best = matches[0] if matches else None
        out.append({
            "nugget_id": nid,
            "nugget_text": n.get("text", ""),
            "required": n.get("required", True),
            "keywords": n.get("keywords", []),
            "golden_passage_title": n.get("golden_passage_title"),
            "golden_evidence": n.get("golden_evidence"),
            # almeno un claim COPRE il nugget (covering, anche senza evidenza valida)
            "covered": len(covering_claims) > 0,
            # coperto da evidenza reale (>= tau): cio' che entra nella coverage
            "covered_for_metrics": covered_for_metrics,
            "cited": covered_for_metrics,
            # miglior match-score dell'evidenza (non una precision).
            "nugget_precision_score": (best["evidence_score"] if best else None),
            "excluded_no_golden": False,
            "cite_score": best["evidence_score"] if best else 0.0,
            "n_covering_claims": len(covering_claims),
            "best_covering_claim": (
                best["claim"] if best
                else (covering_claims[0].get("claim") if covering_claims else None)
            ),
            "best_evidence_passage_title": best["passage_title"] if best else None,
            "best_evidence_passage_text": None,
            "best_evidence_sentence": best["span"] if best else None,
            "cited_from_noise": (best["is_noise"] if best else False),
            "all_evidence": matches,
        })
    return out


def _empty_result(matched_claims: list[dict]) -> dict:
    return {
        "nugget_precision": 0.0, "nugget_precision_all": 0.0,
        "nugget_recall": 0.0, "nugget_coverage": 0.0,
        "n_claims": len(matched_claims), "n_matched_claims": 0, "n_claims_covered": 0,
        "n_pairs": 0, "n_pairs_total": 0, "n_pairs_correct": 0,
        "n_nuggets": 0, "n_covered": 0, "n_cited": 0,
        "per_nugget": [],
        "noise_usage": _count_noise_usage(matched_claims),
        "n_cited_from_noise": 0,
        "n_required": 0, "n_required_covered": 0, "required_coverage": 0.0,
        "n_optional": 0, "n_optional_covered": 0, "optional_coverage": 0.0,
        "n_pairs_from_noise": 0, "n_pairs_correct_from_noise": 0,
    }


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def compute_nugget_metrics(
    nuggets: list[dict],
    matched_claims: list[dict],
    nugget_covering: dict | None = None,   # ignorato (covering ricostruito internamente)
    use_nli: bool = False,                 # retrocompat: ignorato
    use_semantic: bool = True,
    nli_model: str = "cross-encoder/nli-deberta-v3-large",
    semantic_threshold: float = 0.80,      # retrocompat: ignorato
    match_threshold: float = MATCH_THRESHOLD,
    required_only: bool = False,
) -> dict:
    """citation precision (coppie) e recall (claim) calcolate SOLO sui claim
    matched (claim che coprono >= 1 nugget), piu' coverage dei nugget (split)."""
    nuggets_g = [n for n in nuggets if (n.get("golden_evidence") or "").strip()]
    if required_only:
        nuggets_g = [n for n in nuggets_g if n.get("required", True)]

    if not nuggets_g or not matched_claims:
        return _empty_result(matched_claims)

    emb = _build_emb_cache(nuggets_g, matched_claims, use_semantic)

    all_texts = (
    [n.get("golden_evidence", "") for n in nuggets_g]
    + [_extracted_span(p)
       for mc in matched_claims
       for p in mc.get("supporting_passages", [])]
    )
    idf = idf_weights([word_tokens(t) for t in all_texts if t.strip()])

    # ── (1) Covering claim<->nugget (frontend) ──
    nugget_to_claims = _group_like_frontend(nuggets_g, matched_claims)
    nugget_by_id = {n.get("nugget_id", "?"): n for n in nuggets_g}

    # Claim matched (coprono >= 1 nugget) + nugget coperti da ciascun claim.
    # Dedup per identita' dell'oggetto: i claim sono dict non hashabili.
    matched_claims_list: list[dict] = []
    seen_claim_ids: set[int] = set()
    covered_nuggets_by_claim: dict[int, list[dict]] = {}

    for nid, claims in nugget_to_claims.items():
        nug = nugget_by_id[nid]
        for mc in claims:
            cid = id(mc)
            # Con la nuova _group_like_frontend ogni claim appare in UN solo nid,
            # quindi questa guardia è ridondante ma resta per sicurezza.
            if cid not in seen_claim_ids:
                seen_claim_ids.add(cid)
                matched_claims_list.append(mc)
                covered_nuggets_by_claim[cid] = [nug]
            # Se per qualsiasi motivo comparisse in due nid, ignora il secondo.

    matches_by_nugget: dict[str, list[dict]] = {nid: [] for nid in nugget_by_id}
    covered_nugget_ids: set[str] = set()

    total_pairs = 0
    correct_pairs = 0
    n_matched_claims = len(matched_claims_list)
    n_claims_with_correct = 0

    n_pairs_from_noise = 0
    n_pairs_correct_from_noise = 0

    # ── (2) Correttezza coppie, SOLO sui claim matched, contro la golden dei
    #        nugget che il claim COPRE. (Per tornare al "qualsiasi golden" della
    #        v2, sostituire `covered_nuggets` con `nuggets_g` nel loop sotto.) ──
    for mc in matched_claims_list:
        claim_text = mc.get("claim", "")
        covered_nuggets = covered_nuggets_by_claim[id(mc)]
        claim_has_correct = False
        for p in mc.get("supporting_passages", []):
            span = _extracted_span(p)
            if not span:
                continue
            total_pairs += 1
            is_noise = bool(p.get("is_noise", False))
            if is_noise:
                n_pairs_from_noise += 1
            pair_is_correct = False
            for n in covered_nuggets:
                s = _match_score(span, n.get("golden_evidence", ""), emb, idf=idf, use_semantic=use_semantic)
                if s >= match_threshold:
                    pair_is_correct = True
                    nid = n.get("nugget_id", "?")
                    covered_nugget_ids.add(nid)
                    matches_by_nugget[nid].append({
                        "claim": claim_text[:200],
                        "span": span,
                        "passage_title": p.get("title", ""),
                        "passage_text": p.get("text", "")[:300],
                        "evidence_score": s,
                        "entailment_score": p.get("entailment_score", None),
                        "is_noise": is_noise,
                    })
            if pair_is_correct:
                correct_pairs += 1
                claim_has_correct = True
                if is_noise:
                    n_pairs_correct_from_noise += 1
        if claim_has_correct:
            n_claims_with_correct += 1

    # ── Metriche principali (denominatori = SOLO claim matched) ──
    # ── Metriche principali ──
    n_total = len(nuggets_g)
    n_covered = len(covered_nugget_ids)

    # Coppie su TUTTI i claim prodotti (non solo i matched). Stesso filtro
    # span non-vuoto del loop di correttezza. I claim che non coprono alcun
    # nugget contribuiscono coppie al denominatore ma non possono mai essere
    # corretti (non hanno un nugget contro cui matchare la golden).
    total_pairs_all = sum(
        1
        for mc in matched_claims
        for p in mc.get("supporting_passages", [])
        if _extracted_span(p)
    )
    print("[DEBUG] total_pairs_all =", total_pairs_all)

    # Due precision, stesso numeratore (correct_pairs):
    #   matched precision = corrette / coppie dei soli claim matched
    #   precision (piena) = corrette / coppie di TUTTI i claim prodotti
    matched_precision = round(correct_pairs / total_pairs, 4) if total_pairs else 0.0
    precision_all     = round(correct_pairs / total_pairs_all, 4) if total_pairs_all else 0.0

    recall = round(n_covered / n_total, 4) if n_total else 0.0
    coverage = recall  # stessa metrica

    req = [n for n in nuggets_g if n.get("required", True)]
    opt = [n for n in nuggets_g if not n.get("required", True)]
    n_required = len(req)
    n_optional = len(opt)
    n_required_covered = sum(1 for n in req if n.get("nugget_id", "?") in covered_nugget_ids)
    n_optional_covered = sum(1 for n in opt if n.get("nugget_id", "?") in covered_nugget_ids)
    required_coverage = round(n_required_covered / n_required, 4) if n_required else 0.0
    optional_coverage = round(n_optional_covered / n_optional, 4) if n_optional else 0.0

    per_nugget = _build_per_nugget(nuggets_g, nugget_to_claims, matches_by_nugget)

    return {
        "nugget_precision": matched_precision,     # retrocompat: invariata (= matched)
        "nugget_precision_all": precision_all,     # NUOVA: su TUTTE le coppie prodotte
        "nugget_recall": recall,                   # sui nugget gold (coverage)
        "nugget_coverage": recall,                 # == recall
        "n_claims": len(matched_claims),      # claim totali in input
        "n_matched_claims": n_matched_claims, # claim che coprono >= 1 nugget
        "n_claims_covered": n_claims_with_correct,  # claim matched con >=1 coppia corretta
        "n_pairs": total_pairs,               # coppie dei soli claim matched
        "n_pairs_total": total_pairs_all,     # NUOVA: coppie di TUTTI i claim prodotti
        "n_pairs_correct": correct_pairs,
        "n_nuggets": n_total,
        "n_covered": n_covered,
        "n_cited": n_covered,                 # retrocompat (== covered)
        "n_required": n_required,
        "n_required_covered": n_required_covered,
        "required_coverage": required_coverage,
        "n_optional": n_optional,
        "n_optional_covered": n_optional_covered,
        "optional_coverage": optional_coverage,
        "n_pairs_from_noise": n_pairs_from_noise,
        "n_pairs_correct_from_noise": n_pairs_correct_from_noise,
        "per_nugget": per_nugget,
        "noise_usage": _count_noise_usage(matched_claims),
        "n_cited_from_noise": sum(1 for r in per_nugget if r.get("cited_from_noise", False)),
    }


# ──────────────────────────────────────────────
# API wrapper
# ──────────────────────────────────────────────

def evaluate_nuggets_api(payload: dict) -> dict:
    return compute_nugget_metrics(
        nuggets=payload.get("nuggets", []),
        matched_claims=payload.get("matched_claims", []),
        use_semantic=payload.get("use_semantic", True),
        match_threshold=payload.get("match_threshold", MATCH_THRESHOLD),
        required_only=payload.get("required_only", False),
    )


def _smoke_test():
    nuggets = [{
        "nugget_id": "n0",
        "text": "Josef Bican holds the record for the highest number of goals all-time in men's football.",
        "keywords": ["Bican", "Josef Bican"],
        "golden_passage_title": "Josef Bican",
        "golden_evidence": "RSSSF estimates that he scored at least 805 goals in all competitive matches.",
        "required": True,
    }]
    matched_claims = [
        {
            # claim 1: copre n0 (match_score >= gate) e lo cita correttamente
            "claim": "Josef Bican scored at least 805 goals and is the all-time leading scorer in men's football.",
            "matched_nugget": {"nugget_id": "n0", "match_score": 0.82},
            "supporting_passages": [{
                "title": "Josef Bican",
                "text": "RSSSF estimates that he scored at least 805 goals in all competitive matches.",
                "extraction": "he scored at least 805 goals in all competitive matches",
                "entailment_score": 0.91,
            }],
        },
        {
            # claim 2: copre n0 via keyword, evidenza che NON matcha la golden
            # -> entra nei claim matched, ma non ha coppia corretta (abbassa recall)
            "claim": "Bican played as a striker for several clubs.",
            "matched_nugget": {"nugget_id": "n0", "match_score": 0.71},
            "supporting_passages": [{
                "title": "Josef Bican",
                "text": "He played for Slavia Prague and Rapid Vienna.",
                "extraction": "He played for Slavia Prague and Rapid Vienna",
                "entailment_score": 0.40,
            }],
        },
        {
            # claim 3: non copre alcun nugget -> ESCLUSO da precision e recall
            "claim": "Pele is widely regarded as one of the greatest players ever.",
            "matched_nugget": {"nugget_id": "n_other", "match_score": 0.05},
            "supporting_passages": [{
                "title": "Pele",
                "text": "Pele won three World Cups.",
                "extraction": "Pele won three World Cups",
                "entailment_score": 0.88,
            }],
        },
    ]
    print(json.dumps(compute_nugget_metrics(nuggets, matched_claims), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nugget-based evaluation")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("test", help="Run smoke test")
    args = parser.parse_args()
    if args.command == "test":
        _smoke_test()
    else:
        parser.print_help()