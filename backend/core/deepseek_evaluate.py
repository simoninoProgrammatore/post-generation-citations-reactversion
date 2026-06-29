"""LLM-as-judge evaluation con DeepSeek.

Riuso della logica di scripts/deepseek_eval.py (judge per coppia
claim/evidenza), ma:
  - richiamabile dai runner del pipeline (non solo da CLI);
  - concorrente via asyncio + semaforo, perché anche un singolo esempio
    (~60-120 coppie) in thinking-mode sequenziale sarebbe troppo lento;
  - ritorna anche la `reason` di ogni giudizio per ispezione nel frontend.

VERDETTO BINARIO. Ogni coppia (claim, span di evidenza) viene giudicata
"supported" oppure "not_supported". Niente piu' livello "partial" e niente
precision pesata.

  citation precision = #coppie "supported" / #coppie prodotte
  citation recall    = #claim con >=1 evidenza "supported" / #claim prodotti

DIFFERENZA rispetto a deepseek_eval.py: NON si valuta il testo intero del
passaggio, ma lo SPAN di evidenza che il retrieve ha attribuito al claim. Ogni
coppia (claim, span) e' una valutazione separata: un claim con N span
attribuiti su N passaggi produce N giudizi.
"""

import os
import json
import asyncio
from functools import lru_cache

import openai  # DeepSeek e' compatibile con la libreria OpenAI


# Model ID: deepseek-reasoner e' un alias legacy deprecato dal 2026-07-24,
# instradato a deepseek-v4-flash (thinking mode). Usiamo l'ID esplicito.
DEFAULT_MODEL = "deepseek-v4-flash"

# Cap di richieste concorrenti verso l'API.
MAX_CONCURRENCY = 8


@lru_cache(maxsize=1)
def _get_async_client() -> "openai.AsyncOpenAI":
    """Client async cached. Solleva se manca la API key."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "DEEPSEEK_API_KEY environment variable is not set. "
            "Add it to your .env file."
        )
    return openai.AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )


def _get_evidence(passage: dict) -> str:
    """Estrae lo span di evidenza da un passaggio matchato.

    Preferisce `extraction` (lo span isolato da extract_evidence) e usa
    `best_sentence` come fallback.
    """
    return (passage.get("extraction", "") or passage.get("best_sentence", "")).strip()


def _build_prompt(claim: str, evidence: str) -> str:
    """Prompt di giudizio BINARIO: supported / not_supported."""
    return f"""Analizza se l'EVIDENZA supporta logicamente il CLAIM.
L'evidenza e' uno specifico estratto (span) attribuito al claim.

Rispondi con uno di due verdetti:
- "supported": l'evidenza sostiene il claim, cioe' gli elementi del claim sono
  stabiliti dall'evidenza.
- "not_supported": l'evidenza non sostiene il claim. Vale anche quando
  l'evidenza e' irrilevante, troppo generica, copre solo in parte il claim,
  manca un dettaglio chiave (data, soggetto, qualificatore), oppure e'
  contraddittoria.

Rispondi esclusivamente in formato JSON.

CLAIM: "{claim}"
EVIDENZA: "{evidence}"

JSON format:
{{
  "verdict": "supported" | "not_supported",
  "reason": "breve spiegazione del verdetto"
}}"""


async def _judge_pair(
    client: "openai.AsyncOpenAI",
    sem: asyncio.Semaphore,
    claim: str,
    evidence: str,
    model: str,
) -> dict:
    """Giudizio binario per UNA coppia (claim, span di evidenza).

    Ritorna {"verdict": "supported"|"not_supported", "supported": bool, "reason": str}.
    """
    if not evidence or not evidence.strip():
        return {"verdict": "not_supported", "supported": False,
                "reason": "[nessuna evidenza estratta]"}

    prompt = _build_prompt(claim, evidence)
    async with sem:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Sei un giudice rigoroso di fact-checking."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            res = json.loads(response.choices[0].message.content)
            v = str(res.get("verdict", "not_supported")).lower().strip()
            # Difensivo: normalizza qualunque etichetta non prevista (incluso il
            # vecchio "partial") a not_supported.
            if v not in ("supported", "not_supported"):
                if "supported" in res:
                    v = "supported" if bool(res["supported"]) else "not_supported"
                else:
                    v = "not_supported"
            return {
                "verdict": v,
                "supported": v == "supported",
                "reason": str(res.get("reason", "")),
            }
        except Exception as e:  # noqa: BLE001
            return {"verdict": "not_supported", "supported": False,
                    "reason": f"[errore API] {e}"}


async def _evaluate_matched_async(
    matched_claims: list[dict],
    model: str,
) -> dict:
    """Valuta tutti i claim di un esempio in concorrenza, verdetto binario.

    PRECISION: #coppie supported / #coppie prodotte
    RECALL:    #claim con >=1 evidenza supported / #claim
    """
    client = _get_async_client()
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    tasks = []
    index = []  # (claim_idx, passage_idx)
    for ci, mc in enumerate(matched_claims):
        claim = mc.get("claim", "")
        for pi, passage in enumerate(mc.get("supporting_passages", [])):
            evidence = _get_evidence(passage)
            tasks.append(_judge_pair(client, sem, claim, evidence, model))
            index.append((ci, pi))

    verdicts = await asyncio.gather(*tasks) if tasks else []

    # Raggruppa per claim.
    per_claim: list[list[dict]] = [[] for _ in matched_claims]
    for (ci, _pi), verdict in zip(index, verdicts):
        per_claim[ci].append(verdict)

    # ── Conteggi binari ──
    n_pairs = len(verdicts)
    n_supported = sum(1 for v in verdicts if v["verdict"] == "supported")
    n_not = n_pairs - n_supported

    # ── Precision: coppie supported / coppie totali ──
    precision = (n_supported / n_pairs) if n_pairs else 0.0

    # ── Recall: claim con >=1 evidenza supported / claim totali ──
    def _claim_is_covered(verdicts_c: list[dict]) -> bool:
        return any(v["verdict"] == "supported" for v in verdicts_c)

    if matched_claims:
        recall = sum(1 for vs in per_claim if _claim_is_covered(vs)) / len(matched_claims)
    else:
        recall = 0.0

    # ── Dettaglio per claim ──
    per_claim_detail = []
    for mc, verdicts_c in zip(matched_claims, per_claim):
        passages = mc.get("supporting_passages", [])
        judgments = []
        for passage, verdict in zip(passages, verdicts_c):
            judgments.append({
                "passage_title": passage.get("title", ""),
                "passage_text": passage.get("text", ""),
                "evidence": _get_evidence(passage),
                "verdict": verdict["verdict"],
                "supported": verdict["supported"],
                "reason": verdict["reason"],
            })
        n_supported_c = sum(1 for v in verdicts_c if v["verdict"] == "supported")
        per_claim_detail.append({
            "claim": mc.get("claim", ""),
            "any_supported": _claim_is_covered(verdicts_c),
            "n_passages": len(passages),
            "n_supported": n_supported_c,
            "judgments": judgments,
        })

    return {
        "citation_precision": precision,
        "citation_recall":    recall,
        "n_claims":           len(matched_claims),
        "n_pairs":            n_pairs,
        "n_pairs_supported":  n_supported,
        "n_supported":        n_supported,
        "n_not_supported":    n_not,
        "pct_supported":      (n_supported / n_pairs) if n_pairs else 0.0,
        "per_claim":          per_claim_detail,
    }


def evaluate_matched_deepseek(
    matched_claims: list[dict],
    model: str = DEFAULT_MODEL,
) -> dict:
    """Entrypoint sincrono per i runner."""
    return asyncio.run(_evaluate_matched_async(matched_claims, model))


async def evaluate_matched_deepseek_async(
    matched_claims: list[dict],
    model: str = DEFAULT_MODEL,
) -> dict:
    """Entrypoint async, da usare dentro endpoint FastAPI (loop gia' attivo)."""
    return await _evaluate_matched_async(matched_claims, model)