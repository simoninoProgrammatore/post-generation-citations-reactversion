"""Router per gli step del pipeline: generate, decompose, retrieve, cite, evaluate."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core import pipeline_runners
from core import evaluate as core_evaluate
from core import nuggets_evaluate as core_nuggets          # ← NUOVO
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


class EvaluateNuggetsRequest(BaseModel):
    matched_claims: list[dict]   # stessa struttura di MatchedClaim serializzato
    nuggets: list[NuggetItem]
    use_nli: bool = False
    required_only: bool = False


class EvaluateNuggetsResponse(BaseModel):
    nugget_precision: float
    nugget_recall: float
    nugget_coverage: float
    n_nuggets: int
    n_covered: int
    n_cited: int
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
    eval_mode: str = "standard"
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
        matched, debug = pipeline_runners.run_retrieve(
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
        matched, debug = pipeline_runners.run_retrieve(
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
        cited, references = pipeline_runners.run_cite(req.response, matched_dict)
        return CiteResponse(cited_response=cited, references=references)
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
            use_nli=req.use_nli,
            required_only=req.required_only,
        )
        return EvaluateNuggetsResponse(**result)
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


@router.post("/evaluate-example")
async def evaluate_example_endpoint(req: EvaluateExampleRequest):
    """Esegue l'intera pipeline su un singolo esempio e restituisce le metriche."""
    import random
    import logging
    logger = logging.getLogger(__name__)

    example = req.example
    query = example.get("question", "")
    raw_passages = example.get("docs", [])

    logger.info(f"[evaluate-example] START — question='{query[:60]}' passages={len(raw_passages)} noise_enabled={req.noise_enabled} noise_pool={len(req.noise_pool)}")

    # Noise injection
    if req.noise_enabled and raw_passages and req.noise_pool:
        rng = random.Random(req.noise_seed + req.example_idx)
        n_noise = min(max(1, len(raw_passages) // 2), len(req.noise_pool))
        noise_docs = rng.sample(req.noise_pool, n_noise) if n_noise > 0 else []
        passages = list(raw_passages) + [{**d, "is_noise": True} for d in noise_docs]
        rng.shuffle(passages)
    else:
        passages = raw_passages

    logger.info(f"[evaluate-example] passages totali dopo noise: {len(passages)}")

    # Generate
    logger.info(f"[evaluate-example] generate START")
    response_text = pipeline_runners.run_generate(
        query=query, model=req.model, passages=passages
    )
    if not isinstance(response_text, str):
        response_text = response_text.get("response", "")
    logger.info(f"[evaluate-example] generate DONE — risposta {len(response_text)} chars")

    # Decompose
    logger.info(f"[evaluate-example] decompose START")
    claims = pipeline_runners.run_decompose(response_text, req.model)
    logger.info(f"[evaluate-example] decompose DONE — {len(claims)} claims")

    # Retrieve
    nuggets = example.get("nuggets", []) if req.eval_mode == "nugget" else None
    logger.info(f"[evaluate-example] retrieve START — method={req.retrieve_method} top_k={req.top_k} nuggets={len(nuggets) if nuggets else 0}")
    matched, _ = pipeline_runners.run_retrieve(
        claims=claims,
        passages=passages,
        method=req.retrieve_method,
        threshold=req.threshold,
        top_k=req.top_k,
        nuggets=nuggets,
        pre_filter_k=req.pre_filter_k,
        model=req.model,
    )
    logger.info(f"[evaluate-example] retrieve DONE — {len(matched)} matched")

    # ── Noise citation stats ──
    noise_stats = _compute_noise_stats(matched)

    # Evaluate
    logger.info(f"[evaluate-example] evaluate START — mode={req.eval_mode}")
    if req.eval_mode == "nugget":
        result = core_nuggets.compute_nugget_metrics(
            nuggets=nuggets or [],
            matched_claims=matched,
            use_nli=False,
            required_only=False,
        )
        result["noise_stats"] = noise_stats
        logger.info(f"[evaluate-example] evaluate DONE — nugget_precision={result.get('nugget_precision')}")
        return {"question": query, "nugget_metrics": result}
    else:
        metrics = {
            "citation_precision":    core_evaluate.citation_precision_nli(matched),
            "citation_recall":       core_evaluate.citation_recall_nli(matched),
            "factual_precision":     core_evaluate.factual_precision(matched),
            "factual_precision_nli": core_evaluate.factual_precision_nli(matched),
            "unsupported_ratio":     core_evaluate.unsupported_claim_ratio(matched),
            "avg_entailment_score":  core_evaluate.average_entailment_score(matched),
            "noise_stats":           noise_stats,
        }
        logger.info(f"[evaluate-example] evaluate DONE — citation_precision={metrics['citation_precision']:.3f}")
        return {"question": query, "metrics": metrics}

# ── Endpoint corretto ───────────────────────────────────────────────
@router.post("/evaluate-dataset")
async def evaluate_dataset_endpoint(req: EvaluateDatasetRequest):
    """
    Valuta l'intero dataset e restituisce metriche globali aggregate.
    """
    import time, random

    start_time = time.time()
    per_example = []

    # Prepara un pool di documenti dagli altri esempi per il noise (opzionale)
    noise_pool = []
    if req.noise_enabled:
        for i, ex in enumerate(req.dataset):
            for doc in (ex.get("docs") or []):
                noise_pool.append({"example_idx": i, "doc": doc})

    for idx, example in enumerate(req.dataset):
        try:
            query = example.get("question", "")
            raw_passages = example.get("docs", [])

            # Iniezione di documenti di disturbo
            if req.noise_enabled and raw_passages and noise_pool:
                rng = random.Random(req.noise_seed + idx)
                other_docs = [d["doc"] for d in noise_pool if d["example_idx"] != idx]
                n_noise = min(max(1, len(raw_passages) // 2), len(other_docs))
                if n_noise > 0:
                    noise_docs = rng.sample(other_docs, n_noise)
                else:
                    noise_docs = []
                passages = list(raw_passages) + [{**d, "is_noise": True} for d in noise_docs]
                rng.shuffle(passages)
            else:
                passages = raw_passages

            # Step 2 – Generate
            response_obj = pipeline_runners.run_generate(
                query=query, model=req.model, passages=passages
            )
            response_text = response_obj if isinstance(response_obj, str) else response_obj.get("response", "")

            # Step 3 – Decompose
            claims = pipeline_runners.run_decompose(response_text, req.model)

            # Step 4 – Retrieve
            nuggets = example.get("nuggets", []) if req.eval_mode == "nugget" else None
            matched, _ = pipeline_runners.run_retrieve(
                claims=claims,
                passages=passages,
                method=req.retrieve_method,
                threshold=req.threshold,
                top_k=req.top_k,
                nuggets=nuggets,
                pre_filter_k=req.pre_filter_k,
                model=req.model,
            )

            # Step 5 – Cite (opzionale, non usato nelle metriche)
            cited_text, references = pipeline_runners.run_cite(response_text, matched)

            # Step 6 – Evaluate
            if req.eval_mode == "nugget":
                eval_nuggets = nuggets or example.get("nuggets", [])
                nugget_result = core_nuggets.compute_nugget_metrics(
                    nuggets=eval_nuggets,
                    matched_claims=matched,
                    use_nli=False,
                    required_only=False,
                )
                per_example.append({
                    "question": query,
                    "nugget_metrics": nugget_result,
                })
            else:
                metrics = {
                    "citation_precision": core_evaluate.citation_precision_nli(matched),
                    "citation_recall": core_evaluate.citation_recall_nli(matched),
                    "factual_precision": core_evaluate.factual_precision(matched),
                    "factual_precision_nli": core_evaluate.factual_precision_nli(matched),
                    "unsupported_ratio": core_evaluate.unsupported_claim_ratio(matched),
                    "avg_entailment_score": core_evaluate.average_entailment_score(matched),
                }
                per_example.append({
                    "question": query,
                    "metrics": metrics,
                })

        except Exception as e:
            per_example.append({
                "question": example.get("question", f"Example {idx}"),
                "error": str(e),
            })

    # ── Aggregazione metriche globali ──────────────────────────────
    global_metrics = {}

    if req.eval_mode == "standard":
        keys = [
            "citation_precision", "citation_recall",
            "factual_precision", "factual_precision_nli",
            "unsupported_ratio", "avg_entailment_score"
        ]
        for k in keys:
            vals = [
                ex["metrics"][k]
                for ex in per_example
                if "metrics" in ex and k in ex["metrics"]
            ]
            if vals:
                global_metrics[k] = sum(vals) / len(vals)

    elif req.eval_mode == "nugget":
        total_nuggets = 0
        total_covered = 0
        total_cited = 0
        total_req = 0
        total_req_covered = 0
        total_req_cited = 0
        total_opt = 0
        total_opt_covered = 0
        total_opt_cited = 0
        precs, recalls, covs = [], [], []
        req_precs, req_recalls = [], []
        for ex in per_example:
            nm = ex.get("nugget_metrics")
            if nm:
                precs.append(nm.get("nugget_precision", 0))
                recalls.append(nm.get("nugget_recall", 0))
                covs.append(nm.get("nugget_coverage", 0))
                req_precs.append(nm.get("required_precision", 0))
                req_recalls.append(nm.get("required_recall", 0))
                total_nuggets += nm.get("n_nuggets", 0)
                total_covered += nm.get("n_covered", 0)
                total_cited += nm.get("n_cited", 0)
                total_req += nm.get("n_required", 0)
                total_req_covered += nm.get("n_required_covered", 0)
                total_req_cited += nm.get("n_required_cited", 0)
                total_opt += nm.get("n_optional", 0)
                total_opt_covered += nm.get("n_optional_covered", 0)
                total_opt_cited += nm.get("n_optional_cited", 0)
        if precs:
            global_metrics["avg_nugget_precision"] = sum(precs) / len(precs)
            global_metrics["avg_nugget_recall"] = sum(recalls) / len(recalls)
            global_metrics["avg_nugget_coverage"] = sum(covs) / len(covs)
            global_metrics["avg_required_precision"] = sum(req_precs) / len(req_precs)
            global_metrics["avg_required_recall"] = sum(req_recalls) / len(req_recalls)
        if total_nuggets > 0:
            global_metrics["macro_nugget_precision"] = total_cited / total_covered if total_covered > 0 else 0.0
            global_metrics["macro_nugget_recall"]   = total_cited / total_nuggets
            global_metrics["macro_nugget_coverage"] = total_covered / total_nuggets
        if total_req > 0:
            global_metrics["macro_required_precision"] = total_req_cited / total_req_covered if total_req_covered > 0 else 0.0
            global_metrics["macro_required_recall"]   = total_req_cited / total_req
        if total_opt > 0:
            global_metrics["macro_optional_precision"] = total_opt_cited / total_opt_covered if total_opt_covered > 0 else 0.0
            global_metrics["macro_optional_recall"]   = total_opt_cited / total_opt
        global_metrics["total_nuggets"] = total_nuggets
        global_metrics["total_cited"]   = total_cited
        global_metrics["total_covered"] = total_covered
        global_metrics["total_required"] = total_req
        global_metrics["total_required_cited"] = total_req_cited
        global_metrics["total_required_covered"] = total_req_covered
        global_metrics["total_optional"] = total_opt
        global_metrics["total_optional_cited"] = total_opt_cited
        global_metrics["total_optional_covered"] = total_opt_covered

        # Noise usage aggregation
        total_noise_passages = 0
        total_claims_citing_noise = 0
        total_cited_from_noise = 0
        for ex in per_example:
            nm = ex.get("nugget_metrics")
            if nm:
                nu = nm.get("noise_usage", {})
                total_noise_passages += nu.get("noise_supporting_passages", 0)
                total_claims_citing_noise += nu.get("claims_citing_noise", 0)
                total_cited_from_noise += nm.get("n_cited_from_noise", 0)
        global_metrics["total_noise_passages_used"] = total_noise_passages
        global_metrics["total_claims_citing_noise"] = total_claims_citing_noise
        global_metrics["total_nuggets_cited_from_noise"] = total_cited_from_noise

    runtime = round(time.time() - start_time, 1)
    return {
        "global_metrics": global_metrics,
        "per_example": per_example,
        "num_examples": len(req.dataset),
        "num_successful": len([ex for ex in per_example if "error" not in ex]),
        "runtime_seconds": runtime,
        "eval_mode": req.eval_mode,
    }