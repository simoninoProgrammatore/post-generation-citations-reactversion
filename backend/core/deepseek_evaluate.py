"""LLM-as-judge evaluation con DeepSeek.

Riuso della logica di scripts/deepseek_eval.py (judge binario per coppia
claim/evidenza), ma:
  - richiamabile dai runner del pipeline (non solo da CLI);
  - concorrente via asyncio + semaforo, perché anche un singolo esempio
    (~60-120 coppie) in thinking-mode sequenziale sarebbe troppo lento;
  - ritorna anche la `reason` di ogni giudizio per ispezione nel frontend.

DIFFERENZA CHIAVE rispetto a deepseek_eval.py: NON si valuta il testo intero
del passaggio, ma lo SPAN di evidenza che il retrieve ha attribuito al claim
(claim attribution). Ogni coppia (claim, span) e' una valutazione separata:
un claim con N span attribuiti su N passaggi produce N giudizi. Il compito di
DeepSeek e' decidere se quello span specifico e' coerente col claim.
"""

import os
import json
import asyncio
from functools import lru_cache

import openai  # DeepSeek e' compatibile con la libreria OpenAI


# Model ID: deepseek-reasoner e' un alias legacy deprecato dal 2026-07-24,
# instradato a deepseek-v4-flash (thinking mode). Usiamo l'ID esplicito.
DEFAULT_MODEL = "deepseek-v4-flash"

# Cap di richieste concorrenti verso l'API. 8 e' un compromesso prudente
# contro i rate-limit; alzalo se hai margine.
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

    Preferisce `extraction` (lo span isolato da extract_evidence, con
    extraction_start/end) e usa `best_sentence` come fallback. Il retrieve
    filtra gia' i match senza extraction, ma il fallback evita di mandare a
    DeepSeek una stringa vuota (che verrebbe sempre giudicata not-supported).
    """
    return (passage.get("extraction", "") or passage.get("best_sentence", "")).strip()


def _build_prompt(claim: str, evidence: str) -> str:
    """Prompt di giudizio: l'EVIDENZA e' lo span attribuito al claim."""
    return f"""Analizza se l'EVIDENZA supporta logicamente il CLAIM.
L'evidenza e' uno specifico estratto (span) attribuito al claim.
Rispondi esclusivamente in formato JSON.

CLAIM: "{claim}"
EVIDENZA: "{evidence}"

JSON format:
{{
  "supported": true/false,
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

    Ritorna {"supported": bool, "reason": str}. In caso di errore ritorna
    supported=False con la reason che riporta l'errore (stesso comportamento
    fail-safe di deepseek_eval.py, che su eccezione ritorna False).
    """
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
            return {
                "supported": bool(res.get("supported", False)),
                "reason": str(res.get("reason", "")),
            }
        except Exception as e:  # noqa: BLE001 - fail-safe come l'originale
            return {
                "supported": False,
                "reason": f"[errore API] {e}",
            }


async def _evaluate_matched_async(
    matched_claims: list[dict],
    model: str,
) -> dict:
    """Valuta tutti i claim di un esempio in concorrenza.

    Lancia una task per ogni coppia (claim, span di evidenza), poi ricostruisce
    precision e recall a partire dai verdetti. Una sola passata di chiamate
    API serve sia per precision che per recall (a differenza di
    deepseek_eval.py che le calcolava in due funzioni separate, raddoppiando
    le chiamate).
    """
    client = _get_async_client()
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    # Lista piatta di task, una per coppia (claim, span di evidenza), tenendo
    # traccia di a quale claim/passaggio appartiene ciascuna per la
    # ricostruzione successiva.
    tasks = []
    index = []  # (claim_idx, passage_idx)
    for ci, mc in enumerate(matched_claims):
        claim = mc.get("claim", "")
        for pi, passage in enumerate(mc.get("supporting_passages", [])):
            # Valutiamo lo SPAN di evidenza attribuito dal retrieve, NON il
            # testo intero del passaggio: ogni span evidenziato e' una coppia
            # claim->evidenza da giudicare separatamente.
            evidence = _get_evidence(passage)
            tasks.append(_judge_pair(client, sem, claim, evidence, model))
            index.append((ci, pi))

    verdicts = await asyncio.gather(*tasks) if tasks else []

    # Raggruppa i verdetti per claim.
    per_claim: list[list[dict]] = [[] for _ in matched_claims]
    for (ci, _pi), verdict in zip(index, verdicts):
        per_claim[ci].append(verdict)

    # ── Precision: % di coppie (claim, span) giudicate valide su tutte ──
    all_supported = [v["supported"] for v in verdicts]
    precision = (sum(all_supported) / len(all_supported)) if all_supported else 0.0

    # ── Recall: % di claim con almeno uno span valido ──
    if matched_claims:
        supported_claims = sum(
            1 for verdicts_c in per_claim if any(v["supported"] for v in verdicts_c)
        )
        recall = supported_claims / len(matched_claims)
    else:
        recall = 0.0

    # ── Dettaglio per claim (con reason) per il frontend ──
    per_claim_detail = []
    for mc, verdicts_c in zip(matched_claims, per_claim):
        passages = mc.get("supporting_passages", [])
        judgments = []
        for passage, verdict in zip(passages, verdicts_c):
            judgments.append({
                "passage_title": passage.get("title", ""),
                "passage_text": passage.get("text", ""),
                "evidence": _get_evidence(passage),
                "supported": verdict["supported"],
                "reason": verdict["reason"],
            })
        per_claim_detail.append({
            "claim": mc.get("claim", ""),
            "any_supported": any(v["supported"] for v in verdicts_c),
            "n_passages": len(passages),
            "n_supported": sum(1 for v in verdicts_c if v["supported"]),
            "judgments": judgments,
        })

    return {
        "citation_precision": precision,
        "citation_recall": recall,
        "n_claims": len(matched_claims),
        "n_pairs": len(verdicts),
        "n_pairs_supported": sum(all_supported),
        "per_claim": per_claim_detail,
    }


def evaluate_matched_deepseek(
    matched_claims: list[dict],
    model: str = DEFAULT_MODEL,
) -> dict:
    """Entrypoint sincrono per i runner.

    Avvolge la valutazione async. Sicuro da chiamare da codice sincrono
    (es. un runner del pipeline). Dentro un event loop gia' attivo, usa
    direttamente la versione async (evaluate_matched_deepseek_async).
    """
    return asyncio.run(_evaluate_matched_async(matched_claims, model))


async def evaluate_matched_deepseek_async(
    matched_claims: list[dict],
    model: str = DEFAULT_MODEL,
) -> dict:
    """Entrypoint async, da usare dentro endpoint FastAPI (loop gia' attivo)."""
    return await _evaluate_matched_async(matched_claims, model)