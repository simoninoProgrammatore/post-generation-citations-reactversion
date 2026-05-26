"""
Nugget-based evaluation for post-hoc citation generation.

FONTE DI VERITA' DEL COVERING = MatchedView (frontend, nugget-centered).
  Lo Step 6 NON reinventa il covering: replica esattamente claimCoversNugget
  + gate (match_score >= COVERAGE_THRESHOLD) tramite _group_like_frontend.
  Le evidenze di ogni claim coprente sono i suoi supporting_passages, presi
  cosi' come arrivano dal retrieve dello Step 4 (entailment_score,
  best_sentence, extraction intatti). Da quel covering si calcola tutto il
  resto (evidence_score, precision, recall, coverage).

PRECISION CONTINUA E PESATA:
  Per ogni nugget g coperto da k claim c_1..c_k:
    - w_i = match-score claim<->nugget  (tie = MATCH_LEXICAL_WEIGHT * lexical
            + (1-MATCH_LEXICAL_WEIGHT) * semantic, tra TESTO del claim e TESTO
            del nugget). E' il "quanto questo claim parla di questo nugget".
    - e_i = evidence-score: somiglianza fra lo SPAN di evidenza estratto dal
            claim (media sui suoi passaggi) e la GOLDEN EVIDENCE del nugget,
            combinando semantico e lessicale con EVIDENCE_LEXICAL_WEIGHT.
    precision(g) = sum_i (w_i * e_i) / sum_i w_i        in [0, 1], continua.

  Aggregato sull'esempio:
    nugget_precision = sum_g precision(g) / n_covered          (condizionata ai coperti)
    nugget_recall    = sum_g precision(g) / n_total            (sul totale)
    nugget_coverage  = n_covered / n_total

  Se golden_evidence manca, il nugget viene ESCLUSO dalla precision/recall
  (non dovrebbe accadere nel dataset, ma e' gestito).
"""

import re
import json
import argparse
import numpy as np
from functools import lru_cache
from typing import Optional


# ──────────────────────────────────────────────
# Costanti (manopole di tuning — toccare QUI, non nel codice)
# ──────────────────────────────────────────────

# Soglia di covering. STESSO valore del gate in MatchedView (frontend):
#   if (score >= COVERAGE_THRESHOLD) covering.push(...)
# Cambiare qui E nel frontend insieme, altrimenti i due step divergono.
COVERAGE_THRESHOLD = 0.6

# Match claim<->nugget (peso w_i): 0.2 lessicale / 0.8 semantico.
MATCH_LEXICAL_WEIGHT = 0.2

# Evidence-score span_estratto<->golden_evidence (e_i): 0.2 lex / 0.8 sem.
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
def _load_nli_model(model_name: str = "cross-encoder/nli-deberta-v3-large"):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name)


@lru_cache(maxsize=1)
def _load_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def _nli_score(premise: str, hypothesis: str, model_name: str) -> float:
    model = _load_nli_model(model_name)
    logits = np.array(model.predict([(premise, hypothesis)])[0])
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()
    return float(probs[1])


# ──────────────────────────────────────────────
# Embedding cache batched (per-esempio)
# ──────────────────────────────────────────────

class _EmbCache:
    """Encoda ogni testo UNA sola volta (batch) e serve cosine da cache.

    Costruito una volta per esempio con tutti i testi che serviranno
    (nugget, claim, golden evidence, span estratti), elimina le chiamate
    encode ripetute a coppie dentro i loop.
    """

    def __init__(self, texts: list[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        uniq = sorted({t for t in texts if t and t.strip()})
        self._idx = {t: i for i, t in enumerate(uniq)}
        if uniq:
            model = _load_embedding_model(model_name)
            embs = model.encode(uniq, convert_to_numpy=True, normalize_embeddings=True)
            self._embs = embs
        else:
            self._embs = np.zeros((0, 1))

    def cos(self, a: str, b: str) -> float:
        if not a or not b or not a.strip() or not b.strip():
            return 0.0
        ia = self._idx.get(a)
        ib = self._idx.get(b)
        if ia is None or ib is None:
            return 0.0
        # embeddings gia' normalizzati ⇒ cosine = dot
        return float(self._embs[ia] @ self._embs[ib])


def _semantic_similarity(text_a: str, text_b: str,
                         model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> float:
    """Cosine singola (path legacy / fuori dal batch)."""
    model = _load_embedding_model(model_name)
    embs = model.encode([text_a, text_b], convert_to_numpy=True, normalize_embeddings=True)
    return float(embs[0] @ embs[1])


# ──────────────────────────────────────────────
# Covering: fonte di verita' = MatchedView (frontend)
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
    """nugget_id -> [matched_claims coprenti], con la STESSA regola di MatchedView:
    _claim_covers_nugget AND match_score >= COVERAGE_THRESHOLD.
    Lo score e' SOLO matched_nugget.match_score (come nel frontend: `|| 0`),
    NON viene ricalcolato. I claim restano gli oggetti ORIGINALI, cosi'
    i supporting_passages (con entailment_score/best_sentence) sono intatti."""
    out: dict[str, list[dict]] = {}
    for nug in nuggets:
        nid = nug.get("nugget_id", "?")
        covering = []
        for mc in matched_claims:
            if not _claim_covers_nugget(mc, nug):
                continue
            score = (mc.get("matched_nugget") or {}).get("match_score", 0.0) or 0.0
            if score >= COVERAGE_THRESHOLD:
                covering.append(mc)
        # MatchedView ordina per score desc; replichiamo.
        covering.sort(
            key=lambda m: (m.get("matched_nugget") or {}).get("match_score", 0.0) or 0.0,
            reverse=True,
        )
        out[nid] = covering
    return out


# ──────────────────────────────────────────────
# Scoring: match claim<->nugget  e  evidence span<->golden
# ──────────────────────────────────────────────

def _match_score(nugget_text: str, claim_text: str, emb: Optional[_EmbCache],
                 use_semantic: bool = True) -> float:
    """w_i: quanto il claim parla del nugget. tie = lex/sem pesati."""
    lexical = keyword_overlap(nugget_text, claim_text)
    if use_semantic:
        semantic = emb.cos(nugget_text, claim_text) if emb else _semantic_similarity(nugget_text, claim_text)
    else:
        semantic = 0.0
    return round(MATCH_LEXICAL_WEIGHT * lexical + (1.0 - MATCH_LEXICAL_WEIGHT) * semantic, 4)


def claim_nugget_match_score(nugget: dict, claim_text: str,
                             use_semantic: bool = True,
                             emb: Optional[_EmbCache] = None) -> tuple[int, float]:
    """(n_keyword PRIMARY, tie SECONDARY) — usato per il ranking dei claim."""
    keywords = nugget.get("keywords", [])
    n_kw = count_matched_keywords(keywords, claim_text)
    tie = _match_score(nugget.get("text", ""), claim_text, emb, use_semantic)
    return n_kw, tie


def _extracted_span(passage: dict) -> str:
    """Lo span di evidenza prodotto dal retrieve per un passaggio."""
    return (passage.get("extraction", "") or passage.get("best_sentence", "") or "").strip()


def _evidence_score(extracted_span: str, golden_evidence: str,
                    emb: Optional[_EmbCache], use_semantic: bool = True) -> float:
    """e_i: somiglianza fra span estratto e golden evidence (lex/sem pesati)."""
    if not extracted_span or not golden_evidence:
        return 0.0
    lexical = keyword_overlap(golden_evidence, extracted_span)
    if use_semantic:
        semantic = emb.cos(golden_evidence, extracted_span) if emb else _semantic_similarity(golden_evidence, extracted_span)
    else:
        semantic = 0.0
    return round(EVIDENCE_LEXICAL_WEIGHT * lexical + (1.0 - EVIDENCE_LEXICAL_WEIGHT) * semantic, 4)


def _best_evidence_for_claim(mc: dict, golden_evidence: str, emb: Optional[_EmbCache],
                             use_semantic: bool = True) -> tuple[float, Optional[dict], str]:
    """Per un claim, prende lo span (fra i suoi passaggi) che meglio matcha
    la golden evidence. Ritorna (best_e, best_passage, best_span)."""
    best_e = 0.0
    best_p = None
    best_span = ""
    for p in mc.get("supporting_passages", []):
        span = _extracted_span(p)
        e = _evidence_score(span, golden_evidence, emb, use_semantic)
        if e > best_e:
            best_e = e
            best_p = p
            best_span = span
    return best_e, best_p, best_span


def _aggregate_evidence_for_claim(mc: dict, golden_evidence: str,
                                  emb: Optional[_EmbCache],
                                  use_semantic: bool = True) -> float:
    """e_i = media degli evidence_score su tutti i passaggi (con span) del claim.
    Se non ci sono passaggi/span, restituisce 0.0."""
    passages = mc.get("supporting_passages", [])
    if not passages or not golden_evidence:
        return 0.0

    scores = []
    for p in passages:
        span = _extracted_span(p)
        if span:
            scores.append(_evidence_score(span, golden_evidence, emb, use_semantic))

    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 4)


def _all_evidence_for_nugget(nugget: dict, covering_claims: list[dict],
                             emb: Optional[_EmbCache],
                             use_semantic: bool = True) -> list[dict]:
    """Una riga per ogni passaggio di ogni claim coprente, con l'evidence_score
    rispetto alla golden_evidence. Non filtra: mostra TUTTE le evidenze."""
    golden_evidence = nugget.get("golden_evidence", "")
    if not golden_evidence:
        return []

    evidence_list = []
    for mc in covering_claims:
        claim_text = mc.get("claim", "")
        for p in mc.get("supporting_passages", []):
            span = _extracted_span(p)
            if not span:
                continue
            score = _evidence_score(span, golden_evidence, emb, use_semantic)
            evidence_list.append({
                "claim": claim_text[:200],                # tronca per visualizzazione
                "passage_title": p.get("title", ""),
                "passage_text": p.get("text", "")[:300],  # tronca
                "span": span,
                "evidence_score": score,
                "entailment_score": p.get("entailment_score", None),
                "is_noise": p.get("is_noise", False),
            })
    evidence_list.sort(key=lambda x: x["evidence_score"], reverse=True)
    return evidence_list


# ──────────────────────────────────────────────
# Precision continua pesata per UN nugget
# ──────────────────────────────────────────────

def _nugget_precision_weighted(nugget: dict, covering_claims: list[dict],
                               emb: Optional[_EmbCache],
                               use_semantic: bool = True) -> dict:
    golden_evidence = nugget.get("golden_evidence") or ""
    if not golden_evidence:
        return {"precision_score": None, "excluded": True}

    num = 0.0
    den = 0.0
    best_e = -1.0
    best_claim = None
    best_passage = None

    for mc in covering_claims:
        w_i = _match_score(nugget.get("text", ""), mc["claim"], emb, use_semantic)
        e_i = _aggregate_evidence_for_claim(mc, golden_evidence, emb, use_semantic)

        if e_i == 0.0:  # claim senza evidenza valida vs golden: non contribuisce
            continue

        num += w_i * e_i
        den += w_i

        # Diagnostica: miglior passaggio singolo per il confronto.
        e_best_claim, p_best, span_best = _best_evidence_for_claim(
            mc, golden_evidence, emb, use_semantic
        )
        if p_best is not None and e_best_claim > best_e:
            best_e = e_best_claim
            best_claim = mc["claim"]
            best_passage = p_best

    precision_score = round(num / den, 4) if den > 0 else 0.0
    all_ev = _all_evidence_for_nugget(nugget, covering_claims, emb, use_semantic)

    return {
        "precision_score": precision_score,
        "excluded": False,
        "best_covering_claim": best_claim,
        "best_evidence_passage": best_passage,
        "best_evidence_score": round(max(best_e, 0.0), 4),
        "all_evidence": all_ev,
    }


# ──────────────────────────────────────────────
# Split required/optional (allineato al continuo)
# ──────────────────────────────────────────────

def _add_split_metrics(per_nugget: list[dict], result: dict) -> dict:
    """required/optional breakdown, basato sulla precision continua.
    Esclude i nugget senza golden_evidence (precision_score is None)."""
    def _stats(items):
        scored = [r for r in items if r.get("nugget_precision_score") is not None]
        n = len(scored)
        nc = sum(1 for r in scored if r["covered"])
        sum_prec = sum(r["nugget_precision_score"] for r in scored)
        prec = round(sum_prec / nc, 4) if nc > 0 else 0.0   # condizionata ai coperti
        rec = round(sum_prec / n, 4) if n > 0 else 0.0       # sul totale
        cov = round(nc / n, 4) if n > 0 else 0.0
        return n, nc, sum_prec, prec, rec, cov

    req = [r for r in per_nugget if r.get("required", True)]
    opt = [r for r in per_nugget if not r.get("required", True)]

    nr, nrc, _, rp, rr, rcov = _stats(req)
    no, noc, _, op, orr, ocov = _stats(opt)

    result["n_required"]         = nr
    result["n_required_covered"] = nrc
    result["required_precision"] = rp
    result["required_recall"]    = rr
    result["required_coverage"]  = rcov

    result["n_optional"]         = no
    result["n_optional_covered"] = noc
    result["optional_precision"] = op
    result["optional_recall"]    = orr
    result["optional_coverage"]  = ocov
    return result


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


# ──────────────────────────────────────────────
# Build emb cache per un esempio
# ──────────────────────────────────────────────

def _build_emb_cache(nuggets: list[dict], matched_claims: list[dict],
                     use_semantic: bool) -> Optional[_EmbCache]:
    if not use_semantic:
        return None
    texts = []
    for n in nuggets:
        texts.append(n.get("text", ""))
        if n.get("golden_evidence"):
            texts.append(n["golden_evidence"])
    for mc in matched_claims:
        texts.append(mc.get("claim", ""))
        for p in mc.get("supporting_passages", []):
            texts.append(_extracted_span(p))
    return _EmbCache(texts)


# ──────────────────────────────────────────────
# Assemble per_nugget + aggregati
# ──────────────────────────────────────────────

def _assemble(nuggets, nugget_to_claims, matched_claims, emb, use_semantic) -> dict:
    per_nugget = []

    for nug in nuggets:
        nid = nug.get("nugget_id", "?")
        candidates = nugget_to_claims.get(nid, [])

        # Ranking dei covering claim: keyword PRIMARY, tie SECONDARY.
        candidates = sorted(
            candidates,
            key=lambda mc: claim_nugget_match_score(nug, mc["claim"], use_semantic, emb),
            reverse=True,
        )
        covered = len(candidates) > 0

        # Precision continua pesata.
        prec = _nugget_precision_weighted(nug, candidates, emb, use_semantic)
        score = prec["precision_score"]            # None se golden_evidence manca
        excluded = prec.get("excluded", False)

        # NOTA (punto 6, lasciato come deciso da te): `cited` = "esiste almeno
        # uno span estratto", indipendentemente dall'evidence_score vs golden.
        # Un nugget puo' quindi essere cited=True ma covered_for_metrics=False
        # (precision 0). Se vuoi che `cited` significhi "evidenza valida",
        # cambiare la condizione in: score is not None and score > 0.0
        cited_flag = any(
            _extracted_span(p)
            for mc in candidates
            for p in mc.get("supporting_passages", [])
        )
        best_passage = prec.get("best_evidence_passage")

        # Coperto ai fini delle metriche solo se almeno un claim ha evidenza reale.
        covered_for_metrics = score is not None and score > 0.0

        per_nugget.append({
            "nugget_id": nid,
            "nugget_text": nug["text"],
            "required": nug.get("required", True),
            "keywords": nug.get("keywords", []),
            "golden_passage_title": nug.get("golden_passage_title"),
            "golden_evidence": nug.get("golden_evidence"),
            "covered": covered,                          # retrocompat: almeno un claim
            "covered_for_metrics": covered_for_metrics,  # per precision/recall/coverage
            "cited": cited_flag,
            "nugget_precision_score": score,
            "excluded_no_golden": excluded,
            "cite_score": prec.get("best_evidence_score", 0.0),
            "n_covering_claims": len(candidates),
            "best_covering_claim": prec.get("best_covering_claim")
                or (candidates[0]["claim"] if candidates else None),
            "best_evidence_passage_title": best_passage.get("title") if best_passage else None,
            "best_evidence_passage_text": (best_passage.get("text", "")[:200] if best_passage else None),
            "best_evidence_sentence": (_extracted_span(best_passage) if best_passage else None),
            "cited_from_noise": (best_passage.get("is_noise", False) if best_passage else False),
            "all_evidence": prec.get("all_evidence", []),
        })

    # ── Aggregazione continua, escludendo i nugget senza golden_evidence ──
    scored = [r for r in per_nugget if r["nugget_precision_score"] is not None]
    n_total = len(scored)
    n_covered = sum(1 for r in scored if r["covered_for_metrics"])
    sum_prec = sum(r["nugget_precision_score"] for r in scored if r["covered_for_metrics"])

    nugget_precision = round(sum_prec / n_covered, 4) if n_covered > 0 else 0.0
    nugget_recall    = round(sum_prec / n_total, 4) if n_total > 0 else 0.0
    nugget_coverage  = round(n_covered / n_total, 4) if n_total > 0 else 0.0

    n_cited = sum(1 for r in scored if r["cited"])

    result = {
        "nugget_precision": nugget_precision,
        "nugget_recall":    nugget_recall,
        "nugget_coverage":  nugget_coverage,
        "n_nuggets":  n_total,
        "n_covered":  n_covered,
        "n_cited":    n_cited,
        "per_nugget": per_nugget,
        "noise_usage": _count_noise_usage(matched_claims),
        "n_cited_from_noise": sum(1 for r in per_nugget if r.get("cited_from_noise", False)),
    }
    return _add_split_metrics(per_nugget, result)


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def compute_nugget_metrics(
    nuggets: list[dict],
    matched_claims: list[dict],
    nugget_covering: dict | None = None,
    use_nli: bool = False,
    use_semantic: bool = True,
    nli_model: str = "cross-encoder/nli-deberta-v3-large",
    semantic_threshold: float = 0.80,
    required_only: bool = False,
) -> dict:
    if required_only:
        nuggets = [n for n in nuggets if n.get("required", True)]

    if not nuggets or not matched_claims:
        return {
            "nugget_precision": 0.0,
            "nugget_recall":    0.0,
            "nugget_coverage":  0.0,
            "n_nuggets":  0,
            "n_covered":  0,
            "n_cited":    0,
            "per_nugget": [],
            "noise_usage": _count_noise_usage(matched_claims),
            "n_cited_from_noise": 0,
        }

    emb = _build_emb_cache(nuggets, matched_claims, use_semantic)
    nugget_to_claims = _group_like_frontend(nuggets, matched_claims)
    return _assemble(nuggets, nugget_to_claims, matched_claims, emb, use_semantic)


# ──────────────────────────────────────────────
# API wrapper
# ──────────────────────────────────────────────

def evaluate_nuggets_api(payload: dict) -> dict:
    return compute_nugget_metrics(
        nuggets=payload.get("nuggets", []),
        matched_claims=payload.get("matched_claims", []),
        use_nli=payload.get("use_nli", False),
        use_semantic=payload.get("use_semantic", True),
        nli_model=payload.get("nli_model", "cross-encoder/nli-deberta-v3-large"),
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
    matched_claims = [{
        "claim": "Josef Bican scored at least 805 goals and is the all-time leading scorer in men's football.",
        "matched_nugget": {"nugget_id": "n0", "match_score": 0.82},
        "supporting_passages": [{
            "title": "Josef Bican",
            "text": "RSSSF estimates that he scored at least 805 goals in all competitive matches.",
            "extraction": "he scored at least 805 goals in all competitive matches",
            "entailment_score": 0.91,
        }],
    }]
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