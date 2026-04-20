"""Router per interpretability: Integrated Gradients, Activation Patching."""

from fastapi import APIRouter, HTTPException

from core.interpretability import (
    integrated_gradients_analysis,
    activation_patching_analysis,
)
from models.schemas import IGRequest, PatchingRequest

router = APIRouter(prefix="/api/interpret", tags=["interpret"])


@router.post("/ig")
async def run_ig(req: IGRequest):
    """Integrated Gradients per token attribution. 30-90 secondi."""
    try:
        result = integrated_gradients_analysis(
            premise=req.premise,
            hypothesis=req.hypothesis,
            target_label=req.target_label,
            n_steps=req.n_steps,
            layerwise=req.layerwise,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/patching")
async def run_patching(req: PatchingRequest):
    """Activation patching. Molto lento (2-5 minuti su DeBERTa-v3-large)."""
    try:
        result = activation_patching_analysis(
            clean_premise=req.clean_premise,
            clean_hypothesis=req.clean_hypothesis,
            corrupt_premise=req.corrupt_premise,
            corrupt_hypothesis=req.corrupt_hypothesis,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))