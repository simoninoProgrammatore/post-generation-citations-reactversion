"""
Step 4: Insert inline citations into the generated response.

Given matched claims and their supporting passages, this module
reconstructs the response with inline citation markers (e.g., [1][2])
and generates an interactive HTML viewer.

ALLINEAMENTO CLAIM -> FRASI (senza LLM):
  il secondo prompt (core.claim_source_map) e' stato ELIMINATO. L'assegnazione
  delle citazioni alla frase giusta avviene ora con un allineamento lessicale
  deterministico basato su CONTAINMENT PESATO IDF:

      score(claim, S) = sum_{t in tok(claim) ∩ tok(S)} idf(t)
                        ─────────────────────────────────────
                           sum_{t in tok(claim)} idf(t)

  dove idf(t) e' calcolato sulle frasi della risposta stessa. Il containment
  (asimmetrico, normalizzato sul claim) non penalizza le frasi lunghe come
  farebbe il Jaccard; i pesi IDF fanno si' che i token rari (date, numeri,
  entita') dominino lo score, neutralizzando il rumore introdotto dalla
  decontestualizzazione dei claim (pronomi risolti, entita' aggiunte).
  I claim che fondono due frasi adiacenti sono gestiti con finestre di
  lunghezza 2. Zero chiamate LLM: deterministico, riproducibile, gratis.

  (Metrica ispirata alle query di containment di ekzhu/datasketch —
  MinHashLSHEnsemble; qui calcolata in forma esatta, dato che una
  risposta ha ~5-15 frasi.)
"""

import math

import re
import json
import argparse
from pathlib import Path


def build_citation_map(matched_claims: list[dict]) -> dict:
    citation_map = {}
    counter = 1

    for mc in matched_claims:
        for passage in mc["supporting_passages"]:
            # Usa title come chiave primaria, fallback a id
            pid = passage.get("title") or passage.get("id", "")
            if pid and pid not in citation_map:
                citation_map[pid] = counter
                counter += 1

    return citation_map


def _sentence_split(text: str) -> list[str]:
    """Split in frasi. Il lookahead (?=[A-Z...]) evita di spezzare quando
    dopo il punto segue una cifra o una minuscola: protegge i decimali
    scritti con spazio ("8,848. 86 metres", artefatto di generazione) e le
    abbreviazioni. Tradeoff: frasi che iniziano con cifra/minuscola vengono
    fuse con la precedente — accettabile per prosa inglese generata."""
    return re.split(r'(?<=[.!?])(?:\[\d+\])*\s+(?=["\'(\[]?[A-Z])', text.strip())


# ──────────────────────────────────────────────
# Allineamento claim -> frasi della risposta (lessicale, senza LLM)
# ──────────────────────────────────────────────

# Stopword escluse dalla tokenizzazione (le stesse del vecchio fallback,
# estese): l'IDF gia' deprime i token frequenti, ma toglierle riduce rumore.
_STOPWORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or', 'but', 'with', 'as',
    'his', 'her', 'their', 'its', 'has', 'have', 'had', 'by', 'it', 'this',
    'that', 'these', 'those', 'from', 'not', 'no', 'also', 'which', 'who',
    'he', 'she', 'they', 'them',
}

# Soglie dell'allineatore:
#   _TH_CONFIDENT: sopra questa soglia su una singola frase, assegnazione diretta.
#   _TH_WINDOW_GAIN: una finestra di 2 frasi vince solo se migliora la singola
#                    di almeno questo margine (evita di "allargare" gratis).
#   _TH_FLOOR: sotto questa soglia il claim resta non assegnato (meglio nessun
#              marker che un marker sulla frase sbagliata).
_TH_CONFIDENT = 0.70
_TH_WINDOW_GAIN = 0.10
_TH_FLOOR = 0.35
_TIE_EPS = 1e-9


def _tokens(s: str) -> set[str]:
    """Token di parola normalizzati (riusa _norm), senza stopword."""
    return set(_norm(s).split()) - _STOPWORDS


def _idf_weights(sentence_tokens: list[set[str]]) -> dict[str, float]:
    """IDF calcolato sulle frasi della risposta: un token che compare in una
    sola frase (una data, un'entita') pesa molto; uno che compare ovunque
    pesa poco. Smoothing +1 per evitare pesi nulli."""
    n = len(sentence_tokens)
    df: dict[str, int] = {}
    for toks in sentence_tokens:
        for t in toks:
            df[t] = df.get(t, 0) + 1
    return {t: math.log((n + 1) / (d + 1)) + 1.0 for t, d in df.items()}


def _containment(claim_toks: set[str], window_toks: set[str],
                 idf: dict[str, float]) -> float:
    """Containment pesato del claim nella finestra: quota della massa IDF
    del claim coperta dalla finestra. Token del claim assenti da TUTTA la
    risposta (aggiunti dalla decontestualizzazione) pesano 1.0 di default."""
    denom = sum(idf.get(t, 1.0) for t in claim_toks)
    if denom <= 0:
        return 0.0
    num = sum(idf.get(t, 1.0) for t in claim_toks & window_toks)
    return num / denom


def _ordered_tokens(s: str) -> list[str]:
    """Token normalizzati nell'ordine del testo (dedup alla prima occorrenza)."""
    seen, out = set(), []
    for t in _norm(s).split():
        if t not in _STOPWORDS and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def align_claim_to_sentences(claim: str, sentence_tokens: list[set[str]],
                             idf: dict[str, float]) -> list[int]:
    """Indici delle frasi a cui attaccare le citazioni di un claim.

    Cascata:
      1. frasi singole: se il best score >= _TH_CONFIDENT, prendi il best
         (e gli eventuali pari merito — info ripetuta nella risposta);
      2. finestre di 2 frasi adiacenti: se una finestra batte la migliore
         singola di almeno _TH_WINDOW_GAIN e supera _TH_CONFIDENT, il claim
         POTREBBE fondere due frasi. Ma prima il test anti-decontestualizzazione:
         se il contributo esclusivo di un membro e' fatto solo di token del
         PREFISSO del claim (il soggetto, risolto dai pronomi durante la
         decomposizione: "He was blind" -> "Andrea Bocelli was blind"),
         quella frase fornisce solo l'entita', non un fatto -> niente fusione,
         si assegna all'altro membro;
      3. altrimenti, best singola se >= _TH_FLOOR; sotto, nessuna assegnazione.
    """
    claim_toks = _tokens(claim)
    if not claim_toks or not sentence_tokens:
        return []

    singles = [_containment(claim_toks, st, idf) for st in sentence_tokens]
    best = max(singles)
    best_idxs = [i for i, s in enumerate(singles) if best - s <= _TIE_EPS]

    if best >= _TH_CONFIDENT:
        return best_idxs

    # Finestre di 2 frasi adiacenti (claim che fonde affermazioni)
    best_win_score, best_win = 0.0, None
    for i in range(len(sentence_tokens) - 1):
        w = _containment(claim_toks, sentence_tokens[i] | sentence_tokens[i + 1], idf)
        if w > best_win_score:
            best_win_score, best_win = w, (i, i + 1)

    if (best_win is not None
            and best_win_score >= _TH_CONFIDENT
            and best_win_score - best >= _TH_WINDOW_GAIN):
        i, j = best_win
        # ── Test anti-decontestualizzazione ──
        # Token del claim coperti ESCLUSIVAMENTE da ciascun membro.
        excl_i = claim_toks & sentence_tokens[i] - sentence_tokens[j]
        excl_j = claim_toks & sentence_tokens[j] - sentence_tokens[i]
        ordered = _ordered_tokens(claim)
        prefix = set(ordered[:max(2, len(ordered) // 3)])
        if excl_i and excl_i <= prefix:
            return [j]          # i fornisce solo il soggetto -> il fatto e' in j
        if excl_j and excl_j <= prefix:
            return [i]
        return [i, j]           # fusione vera: entrambe le frasi

    if best >= _TH_FLOOR:
        return best_idxs

    return []

def _norm(s: str) -> str:
    """Normalizza per il SOLO confronto: rimuove marker citazione, genitivi
    sassoni ("Everest's" -> "everest"), separatori interni ai numeri
    ("8,848.86" e "8,848. 86" -> "884886"), poi punteggiatura, lowercase,
    collassa gli spazi. Non altera l'output."""
    s = re.sub(r'\[\d+\]', '', s)
    s = s.lower()
    s = re.sub(r"['\u2019]s\b", '', s)              # genitivo sassone
    s = re.sub(r'(?<=\d)[.,]\s*(?=\d)', '', s)      # 8,848. 86 -> 884886
    s = re.sub(r'[^\w\s]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


# Alias pubblici: primitive di normalizzazione condivise con core.retrieve
# (stessa tokenizzazione = stessi token per claim e per evidenza).
norm_text = _norm
word_tokens = _tokens
idf_weights = _idf_weights
containment = _containment


def insert_citations(
    response: str,
    matched_claims: list[dict],
    citation_map: dict,
    remove_unsupported: bool = False,
) -> tuple[str, list[dict], list[dict]]:
    """Ritorna (cited_response, references, sentence_claims).

    sentence_claims e' la SINGLE SOURCE OF TRUTH dell'allineamento per la UI:
        [{"sentence": str, "citations": [int],
          "claims": [{"claim": str, "alignment_score": float}]}]
    Il frontend la renderizza cosi' com'e', senza ricalcolare split/overlap
    client-side (era la causa dei pannelli incoerenti col backend).
    """
    # claim_text -> numeri citazione (dai supporting_passages)
    claim_to_citations: dict[str, list[int]] = {}
    for mc in matched_claims:
        claim_text = mc["claim"]
        nums = []
        for passage in mc["supporting_passages"]:
            pid = passage.get("title") or passage.get("id", "")
            if pid in citation_map:
                nums.append(citation_map[pid])
        if nums:
            claim_to_citations[claim_text] = sorted(set(nums))

    sentences = _sentence_split(response)

    # ── Allineamento lessicale: ogni claim -> frasi della risposta ──
    # Tokenizzazione e pesi IDF calcolati UNA volta sulla risposta.
    sentence_tokens = [_tokens(s) for s in sentences]
    idf = _idf_weights(sentence_tokens)

    sent_citations: dict[int, set[int]] = {}
    sent_claims: dict[int, list[dict]] = {}
    for claim_text, nums in claim_to_citations.items():
        idxs = align_claim_to_sentences(claim_text, sentence_tokens, idf)
        if not idxs:
            continue
        claim_toks = _tokens(claim_text)
        covered = set().union(*(sentence_tokens[i] for i in idxs))
        score = _containment(claim_toks, covered, idf)
        for si in idxs:
            sent_citations.setdefault(si, set()).update(nums)
            sent_claims.setdefault(si, []).append({
                "claim": claim_text,
                "alignment_score": round(score, 3),
            })

    # ── Ricostruzione della risposta con i marker + struttura per la UI ──
    cited_sentences = []
    sentence_claims: list[dict] = []
    for si, sentence in enumerate(sentences):
        citation_nums = sent_citations.get(si, set())
        if citation_nums:
            markers = "".join(f"[{n}]" for n in sorted(citation_nums))
            cited_sentences.append(f"{sentence}{markers}")
        elif remove_unsupported:
            continue
        else:
            cited_sentences.append(sentence)

        sentence_claims.append({
            "sentence": sentence,
            "citations": sorted(citation_nums),
            "claims": sorted(sent_claims.get(si, []),
                             key=lambda c: -c["alignment_score"]),
        })

    cited_response = " ".join(cited_sentences)

    all_passages = []
    for mc in matched_claims:
        all_passages.extend(mc["supporting_passages"])

    reference_list = build_reference_list(citation_map, all_passages)

    return cited_response, reference_list, sentence_claims


def build_reference_list(citation_map: dict, passages: list[dict]) -> list[dict]:
    references = []

    # Indice multiplo: sia per id che per title
    pid_to_passage = {}
    for p in passages:
        pid_to_passage[p.get("id", "")] = p
        pid_to_passage[p.get("title", "")] = p

    for pid, num in sorted(citation_map.items(), key=lambda x: x[1]):
        passage = pid_to_passage.get(pid, {})
        references.append({
            "citation_number": num,
            "title": passage.get("title") or pid or "—",  # fallback a pid
            "text": passage.get("text", ""),
        })

    return references


def _build_num_to_claims_map(matched_claims: list[dict], references: list[dict]) -> dict:
    """Map citation_number -> list of {claim, passage} dicts."""
    title_to_num = {r["title"]: r["citation_number"] for r in references}
    num_to_claims: dict[int, list[dict]] = {}
    for mc in matched_claims:
        for passage in mc["supporting_passages"]:
            num = title_to_num.get(passage.get("title", "")) or title_to_num.get(passage.get("id", ""))
            if num is None:
                continue
            num_to_claims.setdefault(num, []).append({
                "claim": mc["claim"],
                "passage": passage,
            })
    return num_to_claims


def generate_html(examples: list[dict]) -> str:
    """Generate a standalone HTML file for interactive citation exploration.

    INVARIATO rispetto all'originale: dipende solo da
    cited_response / references / num_to_claims, non dalla mappa sorgente.
    """

    # Serialise only what the JS needs
    js_data = []
    for ex in examples:
        js_data.append({
            "question": ex.get("question", ""),
            "cited_response": ex.get("cited_response", ""),
            "references": ex.get("references", []),
            "num_to_claims": _build_num_to_claims_map(
                ex.get("matched_claims", []),
                ex.get("references", []),
            ),
        })

    data_json = json.dumps(js_data, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Citation Viewer</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  :root {{
    --bg:        #f9f8f6;
    --surface:   #ffffff;
    --border:    #e4e2dc;
    --border2:   #ccc9c0;
    --text:      #1a1916;
    --text2:     #4a4843;
    --text3:     #8a8780;
    --accent:    #2563eb;
    --accent-bg: #eff6ff;
    --green:     #059669;
    --green-bg:  #ecfdf5;
    --green-bd:  #a7f3d0;
    --amber:     #92400e;
    --amber-bg:  #fffbeb;
    --amber-bd:  #fde68a;
    --purple:    #2e1065;
    --purple-bg: #f5f3ff;
    --purple-bd: #ddd6fe;
    --mono:      'JetBrains Mono', 'Fira Mono', monospace;
    --radius:    8px;
  }}

  body {{
    font-family: 'Inter', system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 40px 24px 80px;
    line-height: 1.6;
  }}

  h1 {{
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 4px;
  }}

  .subtitle {{
    font-size: 13px;
    color: var(--text3);
    margin-bottom: 32px;
  }}

  /* ── Example navigator ── */
  .nav-bar {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 24px;
  }}
  .nav-bar select {{
    flex: 1;
    font-size: 13px;
    padding: 7px 10px;
    border: 1px solid var(--border2);
    border-radius: var(--radius);
    background: var(--surface);
    color: var(--text);
    cursor: pointer;
  }}
  .nav-label {{
    font-size: 12px;
    color: var(--text3);
    white-space: nowrap;
  }}

  /* ── Question ── */
  .question-box {{
    font-size: 14px;
    font-weight: 600;
    color: var(--text2);
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: var(--radius);
    padding: 12px 16px;
    margin-bottom: 20px;
  }}

  /* ── Response text area ── */
  .response-area {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px 24px;
    font-size: 15px;
    line-height: 2.0;
    margin-bottom: 24px;
  }}

  /* ── Cited sentence ── */
  .cited {{
    cursor: pointer;
    border-radius: 4px;
    padding: 1px 2px;
    transition: background .15s;
    position: relative;
  }}
  .cited:hover {{
    background: var(--accent-bg);
  }}
  .cited.open {{
    background: var(--accent-bg);
  }}
  .cited sup {{
    font-family: var(--mono);
    font-size: 10px;
    font-weight: 700;
    color: var(--green);
    margin-left: 1px;
  }}

  /* ── Inline panel ── */
  .inline-panel {{
    display: none;
    margin: 10px 0 6px;
    border: 1px solid var(--border2);
    border-radius: var(--radius);
    overflow: hidden;
    font-size: 13px;
    line-height: 1.5;
  }}
  .inline-panel.visible {{
    display: block;
  }}
  .panel-header {{
    padding: 8px 14px;
    background: var(--bg);
    border-bottom: 1px solid var(--border);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: .06em;
    text-transform: uppercase;
    color: var(--text3);
  }}

  /* ── Claim block ── */
  .claim-block {{
    padding: 14px;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
  }}
  .claim-block:last-child {{
    border-bottom: none;
  }}
  .claim-label {{
    font-size: 10px;
    font-weight: 700;
    letter-spacing: .07em;
    text-transform: uppercase;
    color: var(--text3);
    margin-bottom: 6px;
  }}
  .claim-text {{
    background: var(--purple-bg);
    border: 1px solid var(--purple-bd);
    border-radius: 6px;
    padding: 9px 12px;
    color: var(--purple);
    font-size: 13px;
    margin-bottom: 10px;
  }}

  /* ── Passage card ── */
  .passage-card {{
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
  }}
  .passage-head {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 7px 12px;
    background: var(--bg);
    border-bottom: 1px solid var(--border);
  }}
  .passage-title {{
    font-size: 12px;
    font-weight: 600;
    color: var(--text2);
  }}
  .score-pill {{
    font-family: var(--mono);
    font-size: 10px;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 20px;
    background: var(--green-bg);
    color: var(--green);
    border: 1px solid var(--green-bd);
  }}
  .passage-body {{
    padding: 10px 12px;
    font-size: 12px;
    color: var(--text2);
    line-height: 1.65;
  }}
  .passage-body mark {{
    background: var(--amber-bg);
    color: var(--amber);
    border-radius: 2px;
    padding: 0 1px;
    font-weight: 500;
  }}

  /* ── References ── */
  .refs-section {{
    margin-top: 8px;
    padding-top: 16px;
    border-top: 1px solid var(--border);
  }}
  .refs-title {{
    font-size: 12px;
    font-weight: 700;
    color: var(--text2);
    margin-bottom: 10px;
  }}
  .ref-item {{
    display: flex;
    gap: 10px;
    padding: 8px 12px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    margin-bottom: 6px;
    font-size: 12px;
  }}
  .ref-num {{
    font-family: var(--mono);
    font-weight: 700;
    color: var(--green);
    flex-shrink: 0;
    padding-top: 1px;
  }}
  .ref-body strong {{
    font-size: 12px;
    display: block;
    margin-bottom: 2px;
  }}
  .ref-body span {{
    color: var(--text3);
    font-size: 11px;
  }}
</style>
</head>
<body>

<h1>Citation Viewer</h1>
<p class="subtitle">Clicca su una frase evidenziata per esplorare i claim e i passaggi di supporto.</p>

<div class="nav-bar">
  <span class="nav-label">Esempio:</span>
  <select id="example-select" onchange="loadExample(+this.value)"></select>
</div>

<div id="app"></div>

<script>
const DATA = {data_json};

function highlightText(text, evidence) {{
  if (!evidence || !evidence.trim()) return escHtml(text);
  const idx = text.indexOf(evidence);
  if (idx === -1) return escHtml(text);
  return escHtml(text.slice(0, idx))
    + '<mark>' + escHtml(evidence) + '</mark>'
    + escHtml(text.slice(idx + evidence.length));
}}

function escHtml(s) {{
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}}

function parseParts(cited) {{
  const parts = [];
  const re = /(\\[\\d+\\])+/g;
  let last = 0, m;
  while ((m = re.exec(cited)) !== null) {{
    if (m.index > last) parts.push({{ type:'text', text: cited.slice(last, m.index) }});
    const nums = m[0].match(/\\d+/g).map(Number);
    parts.push({{ type:'cite', nums }});
    last = re.lastIndex;
  }}
  if (last < cited.length) parts.push({{ type:'text', text: cited.slice(last) }});
  return parts;
}}

// Split parts into sentences keeping cite markers attached to preceding text
function buildSentences(parts) {{
  const sentences = [];
  let buf = [];
  for (const p of parts) {{
    if (p.type === 'text') {{
      const segs = p.text.split(/(?<=[.!?]) +/);
      for (let i = 0; i < segs.length; i++) {{
        buf.push({{ type:'text', text: segs[i] }});
        if (i < segs.length - 1) {{
          sentences.push(buf);
          buf = [];
        }}
      }}
    }} else {{
      buf.push(p);
      sentences.push(buf);
      buf = [];
    }}
  }}
  if (buf.length) sentences.push(buf);
  return sentences.filter(s => s.some(p => p.text?.trim() || p.type === 'cite'));
}}

function renderClaimPanel(nums, numToClaims) {{
  const items = nums.flatMap(n => numToClaims[n] || []);
  if (!items.length) return '';
  const rows = items.map(({{ claim, passage }}) => `
    <div class="claim-block">
      <div class="claim-label">Claim</div>
      <div class="claim-text">${{escHtml(claim)}}</div>
      <div class="passage-card">
        <div class="passage-head">
          <span class="passage-title">${{escHtml(passage.title || '—')}}</span>
          ${{passage.entailment_score != null
            ? `<span class="score-pill">${{(passage.entailment_score*100).toFixed(0)}}%</span>`
            : ''}}
        </div>
        <div class="passage-body">
          ${{highlightText(passage.text || '', passage.best_sentence || '')}}
        </div>
      </div>
    </div>
  `).join('');
  return `<div class="inline-panel visible">
    <div class="panel-header">${{items.length}} claim collegat${{items.length===1?'o':'i'}}</div>
    ${{rows}}
  </div>`;
}}

function renderExample(idx) {{
  const ex = DATA[idx];
  const numToClaims = ex.num_to_claims;
  const parts = parseParts(ex.cited_response);
  const sentences = buildSentences(parts);

  let responseHtml = '';
  sentences.forEach((sentParts, si) => {{
    const citeNums = sentParts.filter(p => p.type==='cite').flatMap(p => p.nums);
    const hasClaims = citeNums.length > 0 && citeNums.some(n => (numToClaims[n]||[]).length > 0);
    const textOnly = sentParts.filter(p=>p.type==='text').map(p=>escHtml(p.text)).join('');
    const supMarkers = citeNums.map(n=>`<sup>[${{n}}]</sup>`).join('');

    if (hasClaims) {{
      const panelHtml = renderClaimPanel(citeNums, numToClaims);
      responseHtml += `<span class="cited" data-si="${{si}}" onclick="togglePanel(this)">${{textOnly}}${{supMarkers}}</span>`;
      responseHtml += `<span class="panel-host" data-si="${{si}}">${{panelHtml}}</span>`;
    }} else {{
      responseHtml += `<span>${{textOnly}}${{supMarkers}}</span>`;
    }}
    responseHtml += ' ';
  }});

  const refsHtml = ex.references.map(r => `
    <div class="ref-item">
      <span class="ref-num">[${{r.citation_number}}]</span>
      <div class="ref-body">
        <strong>${{escHtml(r.title || '—')}}</strong>
        <span>${{escHtml((r.text||'').slice(0,200))}}${{(r.text||'').length>200?'…':''}}</span>
      </div>
    </div>
  `).join('');

  const qHtml = ex.question
    ? `<div class="question-box">Q: ${{escHtml(ex.question)}}</div>`
    : '';

  document.getElementById('app').innerHTML = `
    ${{qHtml}}
    <div class="response-area">${{responseHtml}}</div>
    ${{ex.references.length ? `<div class="refs-section"><div class="refs-title">Riferimenti</div>${{refsHtml}}</div>` : ''}}
  `;
}}

function togglePanel(el) {{
  const si = el.dataset.si;
  const host = document.querySelector(`.panel-host[data-si="${{si}}"]`);
  const panel = host?.querySelector('.inline-panel');
  if (!panel) return;
  const isOpen = panel.classList.contains('visible');
  document.querySelectorAll('.cited').forEach(e => e.classList.remove('open'));
  document.querySelectorAll('.inline-panel').forEach(p => p.classList.remove('visible'));
  if (!isOpen) {{
    panel.classList.add('visible');
    el.classList.add('open');
  }}
}}

function loadExample(idx) {{
  renderExample(idx);
}}

const sel = document.getElementById('example-select');
DATA.forEach((ex, i) => {{
  const opt = document.createElement('option');
  opt.value = i;
  opt.textContent = ex.question ? `[${{i}}] ${{ex.question.slice(0,80)}}` : `Esempio ${{i}}`;
  sel.appendChild(opt);
}});
if (DATA.length) renderExample(0);
</script>
</body>
</html>"""


def run(input_path: str, output_path: str, remove_unsupported: bool = False,
        html: bool = True):
    with open(input_path, "r") as f:
        data = json.load(f)

    for example in data:
        matched_claims = example.get("matched_claims", [])

        # SE NON CI SONO FONTI: salta tutto il blocco e lascia la risposta pulita
        if not matched_claims:
            example["cited_response"] = example.get("raw_response", "")
            example["references"] = []
            continue

        citation_map = build_citation_map(matched_claims)
        cited_response, references, sentence_claims = insert_citations(
            example["raw_response"],
            matched_claims,
            citation_map,
            remove_unsupported,
        )
        example["cited_response"] = cited_response
        example["references"] = references
        example["sentence_claims"] = sentence_claims

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Inserted citations in {len(data)} responses -> {output_path}")

    if html:
        html_path = output.with_suffix(".html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(generate_html(data))
        print(f"HTML viewer -> {html_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Insert citations")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/cited.json")
    parser.add_argument("--remove-unsupported", action="store_true")
    parser.add_argument("--no-html", action="store_true", help="Skip HTML generation")
    args = parser.parse_args()
    run(args.input, args.output, args.remove_unsupported, html=not args.no_html)