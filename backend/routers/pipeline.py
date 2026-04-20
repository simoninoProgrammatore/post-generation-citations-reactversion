"""Router per gli step del pipeline: generate, decompose, retrieve, cite, evaluate."""

from fastapi import APIRouter, HTTPException

from core import pipeline_runners
from core import evaluate as core_evaluate
from models.schemas import (
    GenerateRequest, GenerateResponse,
    DecomposeRequest, DecomposeResponse,
    RetrieveRequest, RetrieveResponse,
    CiteRequest, CiteResponse,
    EvaluateRequest, EvaluateResponse,
)

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])


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
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest):
    """Match dei claims con i passages tramite NLI/similarity/LLM."""
    try:
        passages_dict = [p.model_dump() for p in req.passages]
        matched, debug = pipeline_runners.run_retrieve(
            claims=req.claims,
            passages=passages_dict,
            method=req.method,
            threshold=req.threshold,
            top_k=req.top_k,
        )
        return RetrieveResponse(matched=matched, debug=debug)
    except Exception as e:
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