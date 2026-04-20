"""
DeBERTa Attention Analysis for Parametric Knowledge Leakage
============================================================
Estrae e analizza gli attention weights di DeBERTa-v3 su coppie premise/hypothesis,
con focus su: 
  - dominanza dei token di H rispetto a P (segnale di leakage)
  - ruolo del token [CLS] 
  - ablation study con premise vuota
  - salvataggio risultati in JSON per visualizzazione
"""

import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "cross-encoder/nli-deberta-v3-large"
OUTPUT_PATH = Path("attention_results.json")

# ── Test cases ───────────────────────────────────────────────────────────────
# Aggiungi qui i tuoi passage da testare.
# Formato: {"id": str, "category": str, "premise": str, "hypothesis": str, "expected": str}
TEST_CASES = [

     {
         "id": "bias_pavarotti",
        "category": "BIAS - Parametric Leakage",
        "premise": ", as it was his last major",
        "hypothesis": "The 2006 Winter Olympics opening ceremony was held at the Stadio OlimpicLuciano Pavarotti performed \"Nessun Dorma\" via a pre-recorded videoo in Turin, Italy.",
        "expected": "neutral",
    },
    {
        "id": "bias_water",
        "category": "BIAS - Parametric Leakage",
        "premise": "The boiling point of water at sea level is 100 degrees Celsius.",
        "hypothesis": "The 2006 FIFA World Cup Final was played at the Olympiastadion in Berlin, Germany on July 9, 2006.",
        "expected": "neutral",
    },
    {
        "id": "bias_bocelli",
        "category": "BIAS - Parametric Leakage",
        "premise": "Italian tenor Andrea Bocelli performed a stunning rendition of Nessun Dorma at the closing ceremony.",
        "hypothesis": "The 2006 FIFA World Cup Final was played at the Olympiastadion in Berlin, Germany.",
        "expected": "neutral",
    },
    {
        "id": "bias_seanpaul",
        "category": "BIAS - Parametric Leakage",
        "premise": "Jamaican rapper Sean Paul joined her as a special guest to perform their collaborative song, 'No Lie'.",
        "hypothesis": "The 2018 Champions League Final was held at the NSC Olimpiyski Stadium in Kyiv, Ukraine.",
        "expected": "neutral",
    },
    {
        "id": "bias_team",
        "category": "BIAS - Parametric Leakage",
        "premise": "The team would not qualify for the post-season again until the 2015 season.",
        "hypothesis": "The MLB All-Star Game was held at SkyDome.",
        "expected": "neutral",
    },
    {
        "id": "bias_health",
        "category": "BIAS - Parametric Leakage",
        "premise": "I am a public health faculty member and an expert in health risk communication.",
        "hypothesis": "The heating process used in commercial products eliminates pathogens.",
        "expected": "neutral",
    },

    {
        "id": "bias_zidane",
        "category": "BIAS - Parametric Leakage",
        "premise": "Zinedine Zidane was sent off after headbutting Marco Materazzi in the chest during extra time, and Italy won on penalties.",
        "hypothesis": "The 2006 FIFA World Cup Final was played at the Olympiastadion in Berlin, Germany on July 9, 2006.",
        "expected": "neutral",
    },
    {
        "id": "bias_iphone",
        "category": "BIAS - Parametric Leakage",
        "premise": "The company unveiled a handheld device with a touchscreen interface and no physical keyboard, calling it a revolutionary product.",
        "hypothesis": "Apple launched the first iPhone at Macworld Conference in San Francisco on January 9, 2007.",
        "expected": "neutral",
    },
    {
        "id": "bias_berlin_wall",
        "category": "BIAS - Parametric Leakage",
        "premise": "The wall dividing the city was torn down by crowds of citizens using hammers and pickaxes, marking the end of a divided nation.",
        "hypothesis": "The Berlin Wall fell on November 9, 1989, reunifying East and West Germany.",
        "expected": "neutral",
    },
    {
        "id": "ctrl_entailment",
        "category": "Control - Entailment",
        "premise": "Zidane received a red card during extra time of the 2006 World Cup Final.",
        "hypothesis": "Zidane was sent off in the 2006 World Cup Final.",
        "expected": "entailment",
    },
    {
        "id": "ctrl_unrelated",
        "category": "Control - Unrelated",
        "premise": "She bought a new pair of running shoes at the mall.",
        "hypothesis": "The stock market closed higher on Tuesday.",
        "expected": "neutral",
    },
    # ── Ablation: stessa hypothesis, premise vuota/nonsense ──────────────────
    {
        "id": "ablation_zidane_empty",
        "category": "Ablation - Empty Premise",
        "premise": "A cat sat on a mat.",
        "hypothesis": "The 2006 FIFA World Cup Final was played at the Olympiastadion in Berlin, Germany on July 9, 2006.",
        "expected": "neutral",
    },
    {
        "id": "ablation_iphone_empty",
        "category": "Ablation - Empty Premise",
        "premise": "It was a sunny afternoon.",
        "hypothesis": "Apple launched the first iPhone at Macworld Conference in San Francisco on January 9, 2007.",
        "expected": "neutral",
    },
]


# ── Utilities ─────────────────────────────────────────────────────────────────

def get_logits_and_attentions(model, tokenizer, premise: str, hypothesis: str):
    """
    Ritorna (probs, tokens, attentions).
    attentions: lista di tensori [num_heads, seq_len, seq_len] per ogni layer.
    """
    encoding = tokenizer(
        premise, hypothesis,
        return_tensors="pt",
        truncation=True,
        return_offsets_mapping=False,
    )
    input_ids = encoding["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    with torch.no_grad():
        outputs = model(
            **encoding,
            output_attentions=True,
        )

    probs = torch.softmax(outputs.logits[0], dim=0).numpy()
    # outputs.attentions: tupla di tensori (1, num_heads, seq_len, seq_len)
    attentions = [a[0].numpy() for a in outputs.attentions]

    return probs, tokens, attentions, encoding


def split_token_segments(tokens: list[str]):
    """
    Separa gli indici dei token in: CLS, premise, SEP1, hypothesis, SEP2.
    DeBERTa usa [CLS] P [SEP] H [SEP].
    """
    sep_positions = [i for i, t in enumerate(tokens) if t in ("[SEP]", "▁[SEP]", "</s>", "[SEP]")]
    cls_positions = [i for i, t in enumerate(tokens) if t in ("[CLS]", "▁[CLS]", "<s>")]

    cls_idx = cls_positions[0] if cls_positions else 0

    if len(sep_positions) >= 2:
        sep1 = sep_positions[0]
        sep2 = sep_positions[1]
        premise_idx = list(range(cls_idx + 1, sep1))
        hyp_idx = list(range(sep1 + 1, sep2))
    elif len(sep_positions) == 1:
        sep1 = sep_positions[0]
        premise_idx = list(range(cls_idx + 1, sep1))
        hyp_idx = list(range(sep1 + 1, len(tokens)))
    else:
        mid = len(tokens) // 2
        premise_idx = list(range(1, mid))
        hyp_idx = list(range(mid, len(tokens) - 1))

    return {
        "cls": [cls_idx],
        "premise": premise_idx,
        "sep1": [sep_positions[0]] if sep_positions else [],
        "hypothesis": hyp_idx,
        "sep2": [sep_positions[1]] if len(sep_positions) > 1 else [],
    }


def compute_attention_metrics(attentions, segments):
    """
    Per ogni layer e head, calcola:
      - cls_to_premise: attenzione media dal CLS verso i token di P
      - cls_to_hyp:     attenzione media dal CLS verso i token di H
      - hyp_dominance:  cls_to_hyp / (cls_to_premise + cls_to_hyp + 1e-9)
      - mean_hyp_dominance: media su tutte le head del layer
    """
    cls_idx = segments["cls"]
    p_idx = segments["premise"]
    h_idx = segments["hypothesis"]

    layer_metrics = []
    for layer_idx, attn in enumerate(attentions):
        # attn: (num_heads, seq_len, seq_len)
        heads = []
        for head in range(attn.shape[0]):
            a = attn[head]  # (seq_len, seq_len)
            # media dell'attenzione da [CLS] verso P
            cp = float(a[np.ix_(cls_idx, p_idx)].mean()) if p_idx else 0.0
            # media dell'attenzione da [CLS] verso H
            ch = float(a[np.ix_(cls_idx, h_idx)].mean()) if h_idx else 0.0
            dom = ch / (cp + ch + 1e-9)
            heads.append({
                "head": head,
                "cls_to_premise": round(cp, 6),
                "cls_to_hyp": round(ch, 6),
                "hyp_dominance": round(dom, 4),
            })
        mean_dom = float(np.mean([h["hyp_dominance"] for h in heads]))
        layer_metrics.append({
            "layer": layer_idx,
            "mean_hyp_dominance": round(mean_dom, 4),
            "heads": heads,
        })
    return layer_metrics


def compute_mean_attention_matrix(attentions, method="last3"):
    """
    Ritorna una matrice di attenzione media (seq_len x seq_len)
    media sulle head e, secondo `method`, sui layer:
      - 'all':   tutti i layer
      - 'last3': ultimi 3 layer (più semantici)
      - 'last1': solo ultimo layer
    """
    if method == "last3":
        selected = attentions[-3:]
    elif method == "last1":
        selected = attentions[-1:]
    else:
        selected = attentions

    stacked = np.stack(selected, axis=0)        # (layers, heads, seq, seq)
    mean_mat = stacked.mean(axis=(0, 1))         # (seq, seq)
    return mean_mat


def summarize_cross_attention(mean_mat, segments):
    """
    Calcola blocchi di cross-attention tra segmenti:
      P→H, H→P, CLS→P, CLS→H
    """
    def block_mean(from_idx, to_idx):
        if not from_idx or not to_idx:
            return 0.0
        return float(mean_mat[np.ix_(from_idx, to_idx)].mean())

    p = segments["premise"]
    h = segments["hypothesis"]
    c = segments["cls"]

    return {
        "P_to_H": round(block_mean(p, h), 6),
        "H_to_P": round(block_mean(h, p), 6),
        "CLS_to_P": round(block_mean(c, p), 6),
        "CLS_to_H": round(block_mean(c, h), 6),
        "hyp_dominance_from_cls": round(
            block_mean(c, h) / (block_mean(c, p) + block_mean(c, h) + 1e-9), 4
        ),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, output_attentions=True
    )
    model.eval()

    id2label = model.config.id2label
    ent_idx = int(next(k for k, v in id2label.items() if v.lower() == "entailment"))
    neu_idx = int(next(k for k, v in id2label.items() if v.lower() == "neutral"))
    con_idx = int(next(k for k, v in id2label.items() if v.lower() == "contradiction"))

    results = []

    for tc in TEST_CASES:
        print(f"\n[{tc['id']}] {tc['category']}")
        probs, tokens, attentions, _ = get_logits_and_attentions(
            model, tokenizer, tc["premise"], tc["hypothesis"]
        )

        e, n, c = float(probs[ent_idx]), float(probs[neu_idx]), float(probs[con_idx])
        predicted = id2label[int(probs.argmax())]

        segments = split_token_segments(tokens)
        layer_metrics = compute_attention_metrics(attentions, segments)
        mean_mat = compute_mean_attention_matrix(attentions, method="last3")
        cross = summarize_cross_attention(mean_mat, segments)

        # Top layer per dominanza H
        top_layer = max(layer_metrics, key=lambda x: x["mean_hyp_dominance"])
        # Head più dominante in assoluto
        all_heads = [(l["layer"], h["head"], h["hyp_dominance"])
                     for l in layer_metrics for h in l["heads"]]
        top_head = max(all_heads, key=lambda x: x[2])

        bias_flag = ""
        if "BIAS" in tc["category"] or "Ablation" in tc["category"]:
            if e > 0.5:
                bias_flag = "BIAS CONFIRMED"
            elif e > 0.1:
                bias_flag = "suspicious"
            else:
                bias_flag = "clean"

        print(f"  E={e:.4f}  N={n:.4f}  C={c:.4f}  pred={predicted}  {bias_flag}")
        print(f"  CLS→P={cross['CLS_to_P']:.4f}  CLS→H={cross['CLS_to_H']:.4f}  "
              f"hyp_dominance={cross['hyp_dominance_from_cls']:.3f}")
        print(f"  Top layer by dominance: L{top_layer['layer']} "
              f"(mean_dom={top_layer['mean_hyp_dominance']:.3f})")
        print(f"  Top head: L{top_head[0]} H{top_head[1]} "
              f"(dom={top_head[2]:.3f})")

        # Serializza la matrice di attenzione (ultimi 3 layer, media head)
        # Troncata a max 64 token per non esplodere il JSON
        max_tok = 64
        mat_trunc = mean_mat[:max_tok, :max_tok].tolist()
        tok_trunc = tokens[:max_tok]

        results.append({
            "id": tc["id"],
            "category": tc["category"],
            "premise": tc["premise"],
            "hypothesis": tc["hypothesis"],
            "expected": tc["expected"],
            "predicted": predicted,
            "probs": {"E": round(e, 4), "N": round(n, 4), "C": round(c, 4)},
            "bias_flag": bias_flag,
            "cross_attention": cross,
            "layer_dominance": [
                {"layer": l["layer"], "mean_hyp_dominance": l["mean_hyp_dominance"]}
                for l in layer_metrics
            ],
            "top_layer": top_layer["layer"],
            "top_head": {"layer": top_head[0], "head": top_head[1], "dominance": top_head[2]},
            "attention_matrix": mat_trunc,
            "tokens": tok_trunc,
            "segments": {k: v for k, v in segments.items()},
        })

    OUTPUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {len(results)} results → {OUTPUT_PATH}")

    # ── Pattern summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PATTERN SUMMARY — hyp_dominance from CLS (last 3 layers avg)")
    print("=" * 70)
    results.sort(key=lambda x: x["cross_attention"]["hyp_dominance_from_cls"], reverse=True)
    for r in results:
        dom = r["cross_attention"]["hyp_dominance_from_cls"]
        e_score = r["probs"]["E"]
        bar = "█" * int(dom * 30)
        print(f"  {dom:.3f} {bar:<30} E={e_score:.3f}  [{r['id']}]")

    # ── Ablation delta ────────────────────────────────────────────────────────
    ablation = {r["id"]: r for r in results if "ablation" in r["id"]}
    bias_base = {r["id"]: r for r in results if r["id"].startswith("bias_")}

    if ablation:
        print("\n" + "=" * 70)
        print("ABLATION: E-score with real premise vs empty premise")
        print("=" * 70)
        pairs = [
            ("bias_zidane", "ablation_zidane_empty"),
            ("bias_iphone", "ablation_iphone_empty"),
        ]
        for base_id, abl_id in pairs:
            if base_id in {r["id"]: r for r in results} and abl_id in ablation:
                base = next(r for r in results if r["id"] == base_id)
                abl = ablation[abl_id]
                delta = base["probs"]["E"] - abl["probs"]["E"]
                print(f"  {base_id}")
                print(f"    real premise  → E={base['probs']['E']:.4f}")
                print(f"    empty premise → E={abl['probs']['E']:.4f}  delta={delta:+.4f}")
                if abs(delta) < 0.05:
                    print(f"    *** STRONG LEAKAGE: E score quasi identico senza premise!")


if __name__ == "__main__":
    main()