"""
Interpretability module for DeBERTa NLI cross-encoder.
Provides:
  - Integrated Gradients for token-level attribution
  - Activation Patching for causal localization of bias

Dependencies:
  pip install captum transformers torch
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Literal
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ──────────────────────────────────────────────
# Model loading (cached at module level)
# ──────────────────────────────────────────────

_MODEL_CACHE = {}


def load_model(model_name: str = "cross-encoder/nli-deberta-v3-large"):
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    _MODEL_CACHE[model_name] = (tokenizer, model)
    return tokenizer, model


def get_entailment_idx(model) -> int:
    id2label = model.config.id2label
    for idx, label in id2label.items():
        if label.lower() == "entailment":
            return int(idx)
    raise ValueError(f"'entailment' label not found in {id2label}")


# ──────────────────────────────────────────────
# Integrated Gradients
# ──────────────────────────────────────────────

def integrated_gradients_analysis(
    premise: str,
    hypothesis: str,
    model_name: str = "cross-encoder/nli-deberta-v3-large",
    target_label: Literal["entailment", "contradiction", "neutral"] = "entailment",
    n_steps: int = 50,
    layerwise: bool = True,
) -> dict:
    """
    Compute Integrated Gradients attribution for each input token.

    Args:
        premise: input premise
        hypothesis: input hypothesis
        model_name: HF model name
        target_label: which output class to attribute to
        n_steps: number of interpolation steps for IG
        layerwise: also compute per-layer attribution (slower)

    Returns:
        dict with tokens, attributions, probs, layerwise_attributions (if requested)
    """
    from captum.attr import LayerIntegratedGradients

    tokenizer, model = load_model(model_name)

    # Map target label to index
    label_idx = {v.lower(): int(k) for k, v in model.config.id2label.items()}[target_label.lower()]

    # Tokenize
    encoding = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Baseline: sostituisci i token con [PAD] ma mantieni [CLS] e [SEP]
    pad_id = tokenizer.pad_token_id
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    baseline_ids = input_ids.clone()
    for i, tok_id in enumerate(input_ids[0]):
        if tok_id.item() not in (cls_id, sep_id):
            baseline_ids[0, i] = pad_id

    # Forward function that returns the logit for the target class
    def forward_fn(inp_ids, attn_mask):
        out = model(input_ids=inp_ids, attention_mask=attn_mask)
        return out.logits[:, label_idx]

    # Get predicted probs for reporting
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0]
        probs = torch.softmax(logits, dim=0).numpy().tolist()

    # Run IG on the embedding layer
    embedding_layer = model.get_input_embeddings()
    lig = LayerIntegratedGradients(forward_fn, embedding_layer)

    attributions, delta = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        additional_forward_args=(attention_mask,),
        n_steps=n_steps,
        return_convergence_delta=True,
    )

    # Sum across embedding dimension → per-token attribution
    token_attributions = attributions.sum(dim=-1).squeeze(0).detach().numpy()
    # Normalize for visualization
    max_abs = float(np.abs(token_attributions).max() + 1e-9)
    token_attributions_norm = (token_attributions / max_abs).tolist()

    result = {
        "tokens": tokens,
        "token_attributions": token_attributions.tolist(),
        "token_attributions_normalized": token_attributions_norm,
        "convergence_delta": float(delta.item()),
        "target_label": target_label,
        "probs": {
            "contradiction": probs[0] if len(probs) > 0 else 0,
            "entailment": probs[label_idx] if label_idx < len(probs) else 0,
            "neutral": probs[2] if len(probs) > 2 else 0,
        },
        "predicted": model.config.id2label[int(np.argmax(probs))],
    }

    # Optional: layer-wise attribution (coarser but shows where decision forms)
    if layerwise:
        layer_attributions = compute_layerwise_attribution(
            model, tokenizer, input_ids, attention_mask, baseline_ids, label_idx, n_steps=20
        )
        result["layerwise_attributions"] = layer_attributions

    return result


def compute_layerwise_attribution(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    baseline_ids,
    label_idx: int,
    n_steps: int = 20,
) -> list[dict]:
    """
    For each transformer layer, compute the aggregated token attribution
    coming through that layer's output.
    Uses LayerIntegratedGradients on each encoder layer.
    """
    from captum.attr import LayerIntegratedGradients

    def forward_fn(inp_ids, attn_mask):
        out = model(input_ids=inp_ids, attention_mask=attn_mask)
        return out.logits[:, label_idx]

    # Access encoder layers
    if hasattr(model, "deberta"):
        layers = model.deberta.encoder.layer
    elif hasattr(model, "bert"):
        layers = model.bert.encoder.layer
    elif hasattr(model, "roberta"):
        layers = model.roberta.encoder.layer
    else:
        return []

    results = []
    n_tokens = input_ids.shape[1]

    for layer_idx, layer in enumerate(layers):
        try:
            lig = LayerIntegratedGradients(forward_fn, layer)
            attr, _ = lig.attribute(
                inputs=input_ids,
                baselines=baseline_ids,
                additional_forward_args=(attention_mask,),
                n_steps=n_steps,
                return_convergence_delta=True,
            )
            # attr shape: (1, seq_len, hidden_dim)
            token_attr = attr.sum(dim=-1).squeeze(0).detach().numpy()
            # Mean absolute attribution as "layer importance"
            layer_importance = float(np.abs(token_attr).mean())
            max_abs = float(np.abs(token_attr).max() + 1e-9)
            token_attr_norm = (token_attr / max_abs).tolist()

            results.append({
                "layer": layer_idx,
                "mean_abs_attribution": layer_importance,
                "token_attributions_normalized": token_attr_norm,
            })
        except Exception as e:
            results.append({
                "layer": layer_idx,
                "error": str(e),
            })

    return results


# ──────────────────────────────────────────────
# Activation Patching
# ──────────────────────────────────────────────

def activation_patching_analysis(
    clean_premise: str,
    clean_hypothesis: str,
    corrupt_premise: str,
    corrupt_hypothesis: str,
    model_name: str = "cross-encoder/nli-deberta-v3-large",
    patch_target: Literal["residual", "attention_output"] = "residual",
    progress_callback=None,
) -> dict:
    """
    Activation patching: run forward passes on two inputs (clean/corrupt),
    then patch activations from `corrupt` into `clean` at each layer and position,
    measuring how much the entailment score changes.

    Args:
        progress_callback: optional function called as progress_callback(current, total, message)
                           to report progress during the patching loop.
    """
    tokenizer, model = load_model(model_name)
    ent_idx = get_entailment_idx(model)

    if progress_callback:
        progress_callback(0, 1, "Tokenizzazione input...")

    # Tokenize both inputs — they must have the same length for patching
    clean_enc = tokenizer(clean_premise, clean_hypothesis, return_tensors="pt", truncation=True)
    corrupt_enc = tokenizer(corrupt_premise, corrupt_hypothesis, return_tensors="pt", truncation=True)

    max_len = max(clean_enc["input_ids"].shape[1], corrupt_enc["input_ids"].shape[1])

    clean_enc = tokenizer(
        clean_premise, clean_hypothesis,
        return_tensors="pt", truncation=True, padding="max_length", max_length=max_len,
    )
    corrupt_enc = tokenizer(
        corrupt_premise, corrupt_hypothesis,
        return_tensors="pt", truncation=True, padding="max_length", max_length=max_len,
    )

    clean_tokens = tokenizer.convert_ids_to_tokens(clean_enc["input_ids"][0])
    corrupt_tokens = tokenizer.convert_ids_to_tokens(corrupt_enc["input_ids"][0])

    if progress_callback:
        progress_callback(0, 1, "Calcolo baseline scores...")

    # Step 1: get baseline scores
    with torch.no_grad():
        clean_logits = model(**clean_enc).logits[0]
        corrupt_logits = model(**corrupt_enc).logits[0]
        clean_probs = torch.softmax(clean_logits, dim=0).numpy()
        corrupt_probs = torch.softmax(corrupt_logits, dim=0).numpy()

    clean_e = float(clean_probs[ent_idx])
    corrupt_e = float(corrupt_probs[ent_idx])

    # Step 2: cache corrupt activations
    if hasattr(model, "deberta"):
        layers = model.deberta.encoder.layer
    elif hasattr(model, "bert"):
        layers = model.bert.encoder.layer
    elif hasattr(model, "roberta"):
        layers = model.roberta.encoder.layer
    else:
        raise ValueError("Unsupported model architecture")

    num_layers = len(layers)
    seq_len = clean_enc["input_ids"].shape[1]

    if progress_callback:
        progress_callback(0, 1, "Caching attivazioni corrupt...")

    corrupt_cache: dict[int, torch.Tensor] = {}

    def make_caching_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                corrupt_cache[layer_idx] = output[0].detach().clone()
            else:
                corrupt_cache[layer_idx] = output.detach().clone()
        return hook

    hooks = []
    for i, layer in enumerate(layers):
        h = layer.register_forward_hook(make_caching_hook(i))
        hooks.append(h)

    with torch.no_grad():
        _ = model(**corrupt_enc)

    for h in hooks:
        h.remove()

    # Step 3: for each (layer, position), patch corrupt activation into clean
    patching_effect = np.zeros((num_layers, seq_len))
    total_iterations = num_layers * seq_len
    current_iter = 0

    for layer_idx in range(num_layers):
        for pos in range(seq_len):
            def make_patching_hook(target_pos, cached_act):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        patched = output[0].clone()
                        patched[0, target_pos, :] = cached_act[0, target_pos, :]
                        return (patched,) + output[1:]
                    else:
                        patched = output.clone()
                        patched[0, target_pos, :] = cached_act[0, target_pos, :]
                        return patched
                return hook

            h = layers[layer_idx].register_forward_hook(
                make_patching_hook(pos, corrupt_cache[layer_idx])
            )

            with torch.no_grad():
                patched_logits = model(**clean_enc).logits[0]
                patched_probs = torch.softmax(patched_logits, dim=0).numpy()

            h.remove()

            patched_e = float(patched_probs[ent_idx])

            if corrupt_e > clean_e:
                patching_effect[layer_idx, pos] = (patched_e - clean_e) / (corrupt_e - clean_e + 1e-9)
            else:
                patching_effect[layer_idx, pos] = (clean_e - patched_e) / (clean_e - corrupt_e + 1e-9)

            current_iter += 1
            if progress_callback and (current_iter % 10 == 0 or current_iter == total_iterations):
                progress_callback(
                    current_iter,
                    total_iterations,
                    f"Layer {layer_idx+1}/{num_layers} · posizione {pos+1}/{seq_len}",
                )

    if progress_callback:
        progress_callback(total_iterations, total_iterations, "Completato!")

    return {
        "clean_tokens": clean_tokens,
        "corrupt_tokens": corrupt_tokens,
        "clean_entailment": clean_e,
        "corrupt_entailment": corrupt_e,
        "patching_effect": patching_effect.tolist(),
        "num_layers": num_layers,
        "seq_len": seq_len,
    }