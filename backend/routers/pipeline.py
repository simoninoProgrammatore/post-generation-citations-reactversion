"""Router per gli step del pipeline: generate, decompose, retrieve, cite, evaluate."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core import pipeline_runners
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


class RetrieveDebugRequest(BaseModel):
    claim: str
    passages: list[Passage]
    method: str = "nli"  # "nli" | "similarity" | "llm"
    top_k: int = 4


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
                matched = next((m for m in matches if m.get("id") == p.get("id") or m.get("title") == p.get("title")), None)
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