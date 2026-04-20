"""Pydantic schemas per request/response del backend.

Ogni endpoint ha un Request schema (dati in ingresso) e un Response schema
(dati in uscita). Pydantic valida tutto automaticamente.
"""

from pydantic import BaseModel, Field
from typing import Literal


# ──────────────────────────────────────────────
# Passages (input comune a molti endpoint)
# ──────────────────────────────────────────────

class Passage(BaseModel):
    """Un passaggio sorgente (da ALCE o caricato dall'utente)."""
    id: str | None = None
    title: str = ""
    text: str = ""


# ──────────────────────────────────────────────
# /api/pipeline/generate
# ──────────────────────────────────────────────

class GenerateRequest(BaseModel):
    query: str = Field(..., description="Domanda a cui rispondere")
    passages: list[Passage] | None = Field(None, description="Passages opzionali per RAG")
    model: str = Field("claude-haiku-4-5-20251001", description="Modello Claude")


class GenerateResponse(BaseModel):
    response: str


# ──────────────────────────────────────────────
# /api/pipeline/decompose
# ──────────────────────────────────────────────

class DecomposeRequest(BaseModel):
    text: str = Field(..., description="Testo da scomporre in atomic claims")
    model: str = Field("claude-haiku-4-5-20251001")


class DecomposeResponse(BaseModel):
    claims: list[str]


# ──────────────────────────────────────────────
# /api/pipeline/retrieve
# ──────────────────────────────────────────────

class RetrieveRequest(BaseModel):
    claims: list[str]
    passages: list[Passage]
    method: Literal["nli", "similarity", "llm"] = "nli"
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    top_k: int = Field(3, ge=1, le=10)


class SupportingPassage(BaseModel):
    id: str | None = None
    title: str = ""
    text: str = ""
    entailment_score: float | None = None
    similarity_score: float | None = None
    best_sentence: str = ""
    extraction: str = ""
    extraction_start: int = -1
    extraction_end: int = -1
    summary: str = ""


class MatchedClaim(BaseModel):
    claim: str
    supporting_passages: list[SupportingPassage]


class RetrieveResponse(BaseModel):
    matched: list[MatchedClaim]
    debug: list[dict] = []


# ──────────────────────────────────────────────
# /api/pipeline/cite
# ──────────────────────────────────────────────

class CiteRequest(BaseModel):
    response: str
    matched: list[MatchedClaim]


class Reference(BaseModel):
    citation_number: int
    title: str
    text: str


class CiteResponse(BaseModel):
    cited_response: str
    references: list[Reference]


# ──────────────────────────────────────────────
# /api/pipeline/evaluate
# ──────────────────────────────────────────────

class EvaluateRequest(BaseModel):
    matched: list[MatchedClaim]


class EvaluateResponse(BaseModel):
    citation_precision: float
    citation_recall: float
    factual_precision: float
    factual_precision_nli: float
    unsupported_ratio: float
    avg_entailment_score: float


# ──────────────────────────────────────────────
# /api/nli/predict
# ──────────────────────────────────────────────

class NLIRequest(BaseModel):
    premise: str
    hypothesis: str


class NLIResponse(BaseModel):
    entailment: float
    neutral: float
    contradiction: float
    predicted: Literal["entailment", "neutral", "contradiction"]


# ──────────────────────────────────────────────
# /api/interpret/ig
# ──────────────────────────────────────────────

class IGRequest(BaseModel):
    premise: str
    hypothesis: str
    target_label: Literal["entailment", "neutral", "contradiction"] = "entailment"
    n_steps: int = Field(50, ge=10, le=100)
    layerwise: bool = True


# IGResponse: ha molti campi dinamici (liste di float, layerwise, ecc.)
# Per semplicità usiamo dict generico — l'endpoint ritornerà il dict
# prodotto da interpretability.integrated_gradients_analysis().


# ──────────────────────────────────────────────
# /api/interpret/patching
# ──────────────────────────────────────────────

class PatchingRequest(BaseModel):
    clean_premise: str
    clean_hypothesis: str
    corrupt_premise: str
    corrupt_hypothesis: str


# Anche qui, response è un dict generico.


# ──────────────────────────────────────────────
# /api/dataset/list, /api/dataset/load
# ──────────────────────────────────────────────

class DatasetInfo(BaseModel):
    filename: str
    num_examples: int


class DatasetListResponse(BaseModel):
    datasets: list[DatasetInfo]


class DatasetLoadResponse(BaseModel):
    filename: str
    examples: list[dict]