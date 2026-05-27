"""
Step intermedio (secondo prompt): mappa ogni claim alle frasi della risposta
da cui deriva.

Scopo: eliminare l'overlap fuzzy a soglia in cite.py. Invece di indovinare
quale frase ha generato un claim confrontando parole, lo chiediamo a Claude:
data la risposta originale e i claim gia' estratti, per ogni claim restituisce
le frasi (copiate VERBATIM dalla risposta) che lo sostengono.

Output:
    { claim_text: [frase_sorgente, ...] }

Note:
- Un claim puo' mappare a piu' frasi (claim che fonde piu' affermazioni).
- Le frasi sono copiate verbatim per permettere a cite.py un allineamento
  esatto (o quasi) con _sentence_split, senza piu' soglie arbitrarie.
- Il parsing e' difensivo: claim mancanti o malformati -> lista vuota, cosi'
  cite.py puo' decidere il fallback per quel singolo claim.

DEBUG: i dump usano print() (stdout) invece di logger, cosi' sono SEMPRE
visibili sotto uvicorn senza configurare il logging. I logger restano in
parallelo per quando il logging applicativo sara' configurato.
"""

import logging
from core.llm_client import call_llm_json

logger = logging.getLogger(__name__)


_SYSTEM = (
    "You are a precise text-alignment assistant. You link atomic claims back to "
    "the exact sentences of a source text that support them. You never paraphrase: "
    "you copy sentences verbatim from the provided response."
)


def _build_prompt(response: str, claims: list[str]) -> str:
    """Costruisce il prompt per il mapping claim -> frasi sorgente."""
    claims_block = "\n".join(f"{i}. {c}" for i, c in enumerate(claims))
    return f"""Below is a RESPONSE text and a list of atomic CLAIMS that were extracted from it.

For each claim, identify which sentence(s) of the RESPONSE support that claim.
A claim may be supported by one sentence, or by several sentences if it merges
information spread across the text.

RULES:
- Copy the supporting sentence(s) VERBATIM from the RESPONSE. Do not paraphrase,
  do not fix typos, do not change punctuation or spacing.
- A sentence must come from the RESPONSE below, exactly as written.
- If a claim has no clear supporting sentence in the RESPONSE, return an empty list for it.
- Return EVERY claim, in the same order, identified by its index.

Return ONLY valid JSON in this exact shape, with no extra commentary:
{{
  "mapping": [
    {{ "index": 0, "claim": "<claim text>", "source_sentences": ["<verbatim sentence>", ...] }},
    ...
  ]
}}

RESPONSE:
\"\"\"
{response}
\"\"\"

CLAIMS:
{claims_block}
"""


def map_claims_to_sentences(
    response: str,
    claims: list[str],
    model: str = "claude-haiku-4-5-20251001",
) -> dict[str, list[str]]:
    """Secondo prompt: per ogni claim, le frasi della risposta che lo sostengono.

    Ritorna {claim_text: [frase_sorgente_verbatim, ...]}.
    In caso di errore LLM/parse, ritorna {} (cite.py fara' fallback all'overlap).
    """
    print("\n" + "=" * 70, flush=True)
    print(f"[claim_source_map] CHIAMATA - {len(claims)} claim, model={model}", flush=True)

    if not response or not claims:
        print("[claim_source_map] response o claims vuoti -> mappa vuota", flush=True)
        return {}

    prompt = _build_prompt(response, claims)

    try:
        data = call_llm_json(prompt, model=model, max_tokens=2048, system=_SYSTEM)
    except Exception as e:
        print(f"[claim_source_map] LLM/parse FALLITO -> mappa vuota: {e}", flush=True)
        logger.error(f"[claim_source_map] LLM/parse fallito, ritorno mappa vuota: {e}")
        return {}

    # ── DUMP del JSON grezzo restituito da Claude per l'attribuzione ──
    import json as _json
    print("[claim_source_map] JSON attribuzione da Claude:", flush=True)
    print(_json.dumps(data, indent=2, ensure_ascii=False), flush=True)

    # Il modello dovrebbe restituire {"mapping": [...]}, ma siamo difensivi:
    # accettiamo anche una lista nuda.
    if isinstance(data, dict):
        rows = data.get("mapping", [])
    elif isinstance(data, list):
        rows = data
    else:
        print(f"[claim_source_map] forma JSON inattesa: {type(data)} -> mappa vuota", flush=True)
        logger.error(f"[claim_source_map] forma JSON inattesa: {type(data)}")
        return {}

    # Indicizziamo i claim originali per ricucire per indice quando disponibile
    # (piu' robusto del testo, che il modello potrebbe alterare leggermente).
    out: dict[str, list[str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        idx = row.get("index")
        claim_text = None
        if isinstance(idx, int) and 0 <= idx < len(claims):
            claim_text = claims[idx]            # ancora al claim ORIGINALE
        else:
            claim_text = row.get("claim")       # fallback: testo dal modello
        if not claim_text:
            continue

        sents = row.get("source_sentences", [])
        if not isinstance(sents, list):
            sents = []
        # tieni solo stringhe non vuote
        clean = [s.strip() for s in sents if isinstance(s, str) and s.strip()]
        out[claim_text] = clean

    # Garantisci che OGNI claim originale sia presente come chiave (lista vuota
    # se il modello l'ha saltato), cosi' cite.py sa sempre cosa fare.
    for c in claims:
        out.setdefault(c, [])

    # ── DUMP della mappa finale: claim -> frasi attribuite ──
    n_mapped = sum(1 for v in out.values() if v)
    n_empty = len(out) - n_mapped
    print(f"[claim_source_map] mappa finale: {n_mapped} mappati, {n_empty} vuoti (-> fallback overlap)", flush=True)
    for c, sents in out.items():
        tag = "OK             " if sents else "VUOTO->fallback"
        print(f"  [{tag}] ({len(sents)}) {c[:80]!r} -> {sents}", flush=True)
    print("=" * 70 + "\n", flush=True)

    return out