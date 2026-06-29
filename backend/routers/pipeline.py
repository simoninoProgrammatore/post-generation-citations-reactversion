"""Router per gli step del pipeline: generate, decompose, retrieve, cite, evaluate."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core import pipeline_runners
from core import nuggets_evaluate as core_nuggets          
from core import deepseek_evaluate as core_deepseek  
from core import evaluate as core_evaluate


from models.schemas import (
    GenerateRequest, GenerateResponse,
    DecomposeRequest, DecomposeResponse,
    RetrieveRequest, RetrieveResponse,
    CiteRequest, CiteResponse,
    EvaluateRequest, EvaluateResponse,
    Passage,
)

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])

# ── Schemas locali ─────────────────────────────────────────────────────────────

class RetrieveDebugRequest(BaseModel):
    claim: str
    passages: list[Passage]
    method: str = "nli"  # "nli" | "similarity" | "llm"
    top_k: int = 4


class NuggetItem(BaseModel):
    nugget_id: str
    text: str
    keywords: list[str] = []
    golden_passage_title: str | None = None
    golden_evidence: str | None = None
    required: bool = True


class NuggetPerResult(BaseModel):
    nugget_id: str
    nugget_text: str
    required: bool
    keywords: list[str]
    golden_passage_title: str | None
    golden_evidence: str | None
    covered: bool
    cited: bool
    n_covering_claims: int
    best_covering_claim: str | None
    best_evidence_passage_title: str | None
    best_evidence_passage_text: str | None
    # ↓ questi mancano e vengono droppati da Pydantic silenziosamente
    nugget_precision_score: float | None = None
    excluded_no_golden: bool = False
    cite_score: float = 0.0
    best_evidence_sentence: str | None = None
    cited_from_noise: bool = False
    all_evidence: list[dict] = []

    
class EvaluateNuggetsRequest(BaseModel):
    matched_claims: list[dict]
    nuggets: list[NuggetItem]
    nugget_covering: dict[str, list[dict]] | None = None  # ← nuovo
    use_nli: bool = False
    required_only: bool = False


class EvaluateNuggetsResponse(BaseModel):
    nugget_precision: float
    nugget_recall: float
    nugget_coverage: float
    n_claims: int = 0
    n_claims_covered: int = 0
    n_pairs: int = 0
    n_pairs_correct: int = 0
    n_nuggets: int
    n_covered: int
    n_cited: int
    n_required: int = 0
    n_required_covered: int = 0
    required_coverage: float = 0.0
    n_optional: int = 0
    n_optional_covered: int = 0
    optional_coverage: float = 0.0
    n_pairs_from_noise: int = 0
    n_pairs_correct_from_noise: int = 0
    per_nugget: list[NuggetPerResult]

class EvaluateDatasetRequest(BaseModel):
    dataset: list[dict]           # ogni elemento ha question e docs (e nuggets se modalità nugget)
    model: str = "claude-haiku-4-5-20251001"
    retrieve_method: str = "nli"
    threshold: float = 0.5
    top_k: int = 3
    eval_mode: str = "standard"   # "standard" | "nugget"
    noise_enabled: bool = False
    noise_seed: int = 42
    pre_filter_k: int = 0

class EvaluateExampleRequest(BaseModel):
    example: dict
    model: str = "claude-haiku-4-5-20251001"
    retrieve_method: str = "nli"
    threshold: float = 0.5
    top_k: int = 3
    deepseek_model: str = "deepseek-v4-flash"
    noise_enabled: bool = False
    noise_pool: list[dict] = []
    noise_seed: int = 42
    example_idx: int = 0
    pre_filter_k: int = 0




# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/retrieve/debug")
async def retrieve_debug(req: RetrieveDebugRequest):
    """
    Debug di retrieval per UN singolo claim.
    Ritorna, per ogni passage, la top-K frasi con relativo score.
    """
    try:
        from core.retrieve import (
            match_with_nli, _split_passage_with_spans, _load_nli_model,
        )
        import numpy as np

        passages_dict = [p.model_dump() for p in req.passages]

        if req.method == "nli":
            _, sentence_scores = match_with_nli(
                req.claim,
                passages_dict,
                threshold=0.0,
                top_k=len(passages_dict),
                return_all_scores=True,
            )
            for p in sentence_scores:
                p["sentences"] = sorted(p["sentences"], key=lambda s: -s["score"])[:req.top_k]
            return {"claim": req.claim, "method": "nli", "passages": sentence_scores}

        elif req.method == "similarity":
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
            from core.pipeline_runners import get_embedding_model

            model = get_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
            claim_emb = model.encode([req.claim])

            out = []
            for p in passages_dict:
                spans = _split_passage_with_spans(p.get("text", ""))
                if not spans:
                    out.append({"title": p.get("title", ""), "sentences": []})
                    continue
                sents = [s[0] for s in spans]
                embs = model.encode(sents)
                sims = cosine_similarity(claim_emb, embs)[0]
                best = float(np.max(sims))
                ranked = sorted(
                    [{"text": s, "score": float(v), "is_best": float(v) == best}
                     for s, v in zip(sents, sims)],
                    key=lambda x: -x["score"],
                )[:req.top_k]
                out.append({"title": p.get("title", ""), "sentences": ranked})
            return {"claim": req.claim, "method": "similarity", "passages": out}

        else:  # llm
            from core.retrieve import match_with_llm
            matches = match_with_llm(req.claim, passages_dict, threshold=0.0, top_k=len(passages_dict))
            out = []
            for p in passages_dict:
                matched = next(
                    (m for m in matches if m.get("id") == p.get("id") or m.get("title") == p.get("title")),
                    None,
                )
                if matched and matched.get("best_sentence"):
                    out.append({
                        "title": p.get("title", ""),
                        "sentences": [{
                            "text": matched["best_sentence"],
                            "score": float(matched.get("entailment_score", 0)),
                            "is_best": True,
                        }],
                    })
                else:
                    out.append({"title": p.get("title", ""), "sentences": []})
            return {"claim": req.claim, "method": "llm", "passages": out}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Genera una risposta LLM a una query. Opzionalmente usa passages (RAG)."""
    try:
        passages_dict = [p.model_dump() for p in req.passages] if req.passages else None
        response = pipeline_runners.run_generate(
            query=req.query,
            model=req.model,
            passages=passages_dict,
        )
        return GenerateResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/decompose", response_model=DecomposeResponse)
async def decompose(req: DecomposeRequest):
    """Scompone un testo in atomic claims tramite LLM."""
    try:
        claims = pipeline_runners.run_decompose(req.text, req.model)
        return DecomposeResponse(claims=claims)
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"[decompose] ERRORE: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest):
    """Match dei claims con i passages tramite NLI/similarity/LLM."""
    try:
        passages_dict = [p.model_dump() for p in req.passages]
        nuggets_dict = [n.model_dump() for n in req.nuggets] if req.nuggets else None
        matched, debug = await pipeline_runners.run_retrieve(
            claims=req.claims,
            passages=passages_dict,
            method=req.method,
            threshold=req.threshold,
            top_k=req.top_k,
            nuggets=nuggets_dict,
            pre_filter_k=req.pre_filter_k,
            model=req.model,
        )
        return RetrieveResponse(matched=matched, debug=debug)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class RetrieveSingleRequest(BaseModel):
    claim: str
    passages: list[Passage]
    method: str = "nli"
    threshold: float = 0.5
    top_k: int = 3
    nuggets: list[NuggetItem] | None = None
    pre_filter_k: int = 0
    model: str = "claude-haiku-4-5-20251001"


@router.post("/retrieve-single")
async def retrieve_single(req: RetrieveSingleRequest):
    """Match di UN singolo claim — usato dal frontend per la progress bar."""
    try:
        passages_dict = [p.model_dump() for p in req.passages]
        nuggets_dict = [n.model_dump() for n in req.nuggets] if req.nuggets else None
        matched, debug = await pipeline_runners.run_retrieve(
            claims=[req.claim],
            passages=passages_dict,
            method=req.method,
            threshold=req.threshold,
            top_k=req.top_k,
            nuggets=nuggets_dict,
            pre_filter_k=req.pre_filter_k,
            model=req.model,
        )
        return {
            "matched": matched[0] if matched else {"claim": req.claim, "supporting_passages": []},
            "debug": debug[0] if debug else {"claim": req.claim, "sentence_scores": []},
        }
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"[retrieve-single] ERRORE: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cite", response_model=CiteResponse)
async def cite(req: CiteRequest):
    """Inserisce citazioni inline nella risposta."""
    try:
        matched_dict = [m.model_dump() for m in req.matched]
        cited, references, sentence_claims = pipeline_runners.run_cite(req.response, matched_dict)
        return CiteResponse(cited_response=cited, references=references,
                            sentence_claims=sentence_claims)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(req: EvaluateRequest):
    """Calcola le metriche di qualità sul risultato del pipeline."""
    try:
        matched_dict = [m.model_dump() for m in req.matched]
        return EvaluateResponse(
            citation_precision=core_evaluate.citation_precision_nli(matched_dict),
            citation_recall=core_evaluate.citation_recall_nli(matched_dict),
            factual_precision=core_evaluate.factual_precision(matched_dict),
            factual_precision_nli=core_evaluate.factual_precision_nli(matched_dict),
            unsupported_ratio=core_evaluate.unsupported_claim_ratio(matched_dict),
            avg_entailment_score=core_evaluate.average_entailment_score(matched_dict),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate-nuggets", response_model=EvaluateNuggetsResponse)
async def evaluate_nuggets(req: EvaluateNuggetsRequest):
    """
    Calcola Nugget Precision, Nugget Recall e Nugget Coverage.

    Verifica se i nuggets del dataset sono:
      - coperti da almeno un claim generato (coverage)
      - citati con un passaggio di supporto che contiene evidenza (precision/recall)
    """
    try:
        nuggets_dict = [n.model_dump() for n in req.nuggets]
        
        result = core_nuggets.compute_nugget_metrics(
            nuggets=nuggets_dict,
            matched_claims=req.matched_claims,
            nugget_covering=req.nugget_covering,  # ← aggiungi
            use_nli=req.use_nli,
            required_only=req.required_only,
        )
        return EvaluateNuggetsResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class EvaluateDeepseekRequest(BaseModel):
    matched: list[dict]
    deepseek_model: str = "deepseek-v4-flash"


@router.post("/evaluate-deepseek")
async def evaluate_deepseek_endpoint(req: EvaluateDeepseekRequest):
    """Giudica via DeepSeek le combo claim/evidenza GIA' trovate.
    Lavora sul matched esistente, senza rigenerare la pipeline."""
    try:
        result = await core_deepseek.evaluate_matched_deepseek_async(
            matched_claims=req.matched,
            model=req.deepseek_model,
        )
        return {"deepseek_metrics": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
def _compute_noise_stats(matched_claims: list[dict]) -> dict:
    """Calcola statistiche sull'uso dei passaggi di noise nelle citazioni."""
    noise_supporting = 0
    claims_citing_noise = 0

    for mc in matched_claims:
        has_noise = False
        for sp in mc.get("supporting_passages", []):
            if sp.get("is_noise", False):
                noise_supporting += 1
                has_noise = True
        if has_noise:
            claims_citing_noise += 1

    return {
        "noise_supporting_passages": noise_supporting,
        "claims_citing_noise": claims_citing_noise,
    }


import copy

def _keep_top1_evidence(matched: list) -> list:
    """Ritorna una copia di matched con al massimo 1 supporting_passage per claim:
    il PRIMO così come arriva dal retrieve (già ordinato per rilevanza)."""
    result = []
    for claim_obj in matched:
        filtered = copy.deepcopy(claim_obj)
        if isinstance(filtered.get("supporting_passages"), list):
            filtered["supporting_passages"] = filtered["supporting_passages"][:1]
        result.append(filtered)
    return result


@router.post("/evaluate-example")
async def evaluate_example_endpoint(req: EvaluateExampleRequest):
    """Esegue la pipeline UNA volta su un esempio e restituisce
    SIA le metriche nugget SIA quelle deepseek (no doppio giro)."""
    import random
    import logging
    logger = logging.getLogger(__name__)

    example = req.example
    query = example.get("question", "")
    raw_passages = example.get("docs", [])

    # ── Noise injection ──
    if req.noise_enabled and raw_passages and req.noise_pool:
        rng = random.Random(req.noise_seed + req.example_idx)
        n_noise = min(max(1, len(raw_passages) // 2), len(req.noise_pool))
        noise_docs = rng.sample(req.noise_pool, n_noise) if n_noise > 0 else []
        passages = list(raw_passages) + [{**d, "is_noise": True} for d in noise_docs]
        rng.shuffle(passages)
    else:
        passages = raw_passages

    # ── Generate ──
    response_text = pipeline_runners.run_generate(query=query, model=req.model, passages=passages)
    if not isinstance(response_text, str):
        response_text = response_text.get("response", "")

    # ── Decompose ──
    claims = pipeline_runners.run_decompose(response_text, req.model)

    # ── Retrieve (una volta sola) ──
    nuggets = example.get("nuggets", []) or None
    logger.info(
        f"[evaluate-example] retrieve START — method={req.retrieve_method} "
        f"top_k={req.top_k} nuggets={len(nuggets) if nuggets else 0}"
    )
    matched, _ = await pipeline_runners.run_retrieve(
        claims=claims,
        passages=passages,
        method=req.retrieve_method,
        threshold=req.threshold,
        top_k=req.top_k,
        nuggets=nuggets,
        pre_filter_k=req.pre_filter_k,
        model=req.model,
    )

    # ── Top1: log + deepcopy (filtro disabilitato finché non vediamo la struttura) ──
    matched_top1 = _keep_top1_evidence(matched)

    noise_stats      = _compute_noise_stats(matched)
    noise_stats_top1 = _compute_noise_stats(matched_top1)

    # ── Metriche top_k completo ──
    nugget_metrics = core_nuggets.compute_nugget_metrics(
        nuggets=nuggets or [],
        matched_claims=matched,
        use_nli=False,
        required_only=False,
    )
    nugget_metrics["noise_stats"] = noise_stats

    deepseek_metrics = await core_deepseek.evaluate_matched_deepseek_async(
        matched_claims=matched,
        model=req.deepseek_model,
    )
    deepseek_metrics["noise_stats"] = noise_stats

    # ── Metriche top1 ──
    nugget_metrics_top1 = core_nuggets.compute_nugget_metrics(
        nuggets=nuggets or [],
        matched_claims=matched_top1,
        use_nli=False,
        required_only=False,
    )
    nugget_metrics_top1["noise_stats"] = noise_stats_top1

    deepseek_metrics_top1 = await core_deepseek.evaluate_matched_deepseek_async(
        matched_claims=matched_top1,
        model=req.deepseek_model,
    )
    deepseek_metrics_top1["noise_stats"] = noise_stats_top1

    logger.info(
        f"[evaluate-example] DONE — nugget_precision={nugget_metrics.get('nugget_precision')} "
        f"ds_precision={deepseek_metrics.get('citation_precision')}"
    )

    return {
        "question": query,
        "nugget_metrics": nugget_metrics,
        "deepseek_metrics": deepseek_metrics,
        "nugget_metrics_top1": nugget_metrics_top1,
        "deepseek_metrics_top1": deepseek_metrics_top1,
    }