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
    
    # ── /api/interpret/attention ────────────────────────────────────────────
# Calcola cross-attention e layer-wise hyp_dominance su una coppia NLI.

from pydantic import BaseModel
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class AttentionRequest(BaseModel):
    premise: str
    hypothesis: str
    expected: str = "neutral"  # "entailment" | "neutral" | "contradiction"


_ATTN_CACHE = {}


def _load_attn_model(model_name="cross-encoder/nli-deberta-v3-large"):
    if model_name in _ATTN_CACHE:
        return _ATTN_CACHE[model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, output_attentions=True,
    )
    model.eval()
    _ATTN_CACHE[model_name] = (tokenizer, model)
    return tokenizer, model


@router.post("/attention")
async def run_attention(req: AttentionRequest):
    """Calcola hyp_dominance layer-per-layer e cross-attention aggregata."""
    try:
        tokenizer, model = _load_attn_model()

        enc = tokenizer(req.premise, req.hypothesis, return_tensors="pt", truncation=True)
        input_ids = enc["input_ids"][0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        # Trova segmenti P / H
        sep_positions = [i for i, t in enumerate(tokens) if t in ("[SEP]", "</s>", "▁</s>")]
        if len(sep_positions) >= 2:
            premise_end = sep_positions[0]
            hypothesis_start = sep_positions[0] + 1
            hypothesis_end = sep_positions[1]
        else:
            mid = len(tokens) // 2
            premise_end = mid
            hypothesis_start = mid
            hypothesis_end = len(tokens) - 1

        with torch.no_grad():
            outputs = model(**enc)

        logits = outputs.logits[0]
        probs_raw = torch.softmax(logits, dim=0).numpy()

        id2label = model.config.id2label
        label_map = {v.lower(): i for i, v in id2label.items()}
        e_idx = label_map.get("entailment", 1)
        n_idx = label_map.get("neutral", 0)
        c_idx = label_map.get("contradiction", 2)
        predicted_label = id2label[int(np.argmax(probs_raw))].lower()

        # Attention: media layer + heads
        all_layers = np.stack([a[0].mean(0).numpy() for a in outputs.attentions])
        mean_attn = all_layers.mean(0)

        p_range = list(range(1, premise_end))
        h_range = list(range(hypothesis_start, hypothesis_end))

        cls_to_p = float(mean_attn[0, p_range].mean()) if p_range else 0.0
        cls_to_h = float(mean_attn[0, h_range].mean()) if h_range else 0.0
        total_cls = cls_to_p + cls_to_h
        hyp_dom = cls_to_h / total_cls if total_cls > 0 else 0.5
        p_to_h = float(mean_attn[np.ix_(p_range, h_range)].mean()) if (p_range and h_range) else 0.0
        h_to_p = float(mean_attn[np.ix_(h_range, p_range)].mean()) if (p_range and h_range) else 0.0

        layer_dominance = []
        for li, layer_attn in enumerate(all_layers):
            cp = float(layer_attn[0, p_range].mean()) if p_range else 0.0
            ch = float(layer_attn[0, h_range].mean()) if h_range else 0.0
            t = cp + ch
            layer_dominance.append({
                "layer": li,
                "mean_hyp_dominance": ch / t if t > 0 else 0.5,
            })

        if hyp_dom > 0.70:
            bias_flag = "BIAS CONFIRMED"
            category = "BIAS"
        elif hyp_dom > 0.55:
            bias_flag = "suspicious"
            category = "SUSPICIOUS"
        else:
            bias_flag = "clean"
            category = "CLEAN"

        record_id = f"{category.lower()}_{req.premise[:20].strip().replace(' ', '_')}"

        return {
            "id": record_id,
            "category": category,
            "hypothesis": req.hypothesis,
            "premise": req.premise,
            "expected": req.expected,
            "predicted": predicted_label,
            "bias_flag": bias_flag,
            "probs": {
                "E": float(probs_raw[e_idx]),
                "N": float(probs_raw[n_idx]),
                "C": float(probs_raw[c_idx]),
            },
            "cross_attention": {
                "CLS_to_P": cls_to_p,
                "CLS_to_H": cls_to_h,
                "P_to_H": p_to_h,
                "H_to_P": h_to_p,
                "hyp_dominance_from_cls": hyp_dom,
            },
            "layer_dominance": layer_dominance,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))