"""Router per la predizione NLI diretta su una coppia (premise, hypothesis)."""

import numpy as np
from fastapi import APIRouter, HTTPException

from core.pipeline_runners import get_nli_model
from models.schemas import NLIRequest, NLIResponse

router = APIRouter(prefix="/api/nli", tags=["nli"])


@router.post("/predict", response_model=NLIResponse)
async def predict(req: NLIRequest):
    """Predice la relazione NLI tra premise e hypothesis."""
    try:
        model = get_nli_model("cross-encoder/nli-deberta-v3-large")
        scores = model.predict([(req.premise, req.hypothesis)])
        logits = np.array(scores)
        if logits.ndim == 2:
            logits = logits[0]
        exp = np.exp(logits - np.max(logits))
        probs = exp / exp.sum()

        # Labels order dal model config (di solito: 0=contradiction, 1=entailment, 2=neutral)
        # Verificalo con model.config.id2label se dubbi
        id2label = model.model.config.id2label
        label_map = {v.lower(): i for i, v in id2label.items()}

        e = float(probs[label_map["entailment"]])
        n = float(probs[label_map["neutral"]])
        c = float(probs[label_map["contradiction"]])

        predicted = max(
            [("entailment", e), ("neutral", n), ("contradiction", c)],
            key=lambda x: x[1],
        )[0]

        return NLIResponse(entailment=e, neutral=n, contradiction=c, predicted=predicted)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))