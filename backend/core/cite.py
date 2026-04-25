"""
Step 4: Insert inline citations into the generated response.

Given matched claims and their supporting passages, this module
reconstructs the response with inline citation markers (e.g., [1][2])
and generates an interactive HTML viewer.
"""

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
    return re.split(r'(?<=[.!?])(?:\[\d+\])*\s+', text.strip())


def insert_citations(
    response: str,
    matched_claims: list[dict],
    citation_map: dict,
    remove_unsupported: bool = False,
) -> tuple[str, list[dict]]:
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

    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on',
                 'at', 'to', 'for', 'of', 'and', 'or', 'but', 'with', 'as',
                 'his', 'her', 'their', 'its', 'has', 'have', 'had', 'by',
                 'it', 'this', 'that', 'from', 'not', 'be', 'been'}

    sentences = _sentence_split(response)
    cited_sentences = []

    for sentence in sentences:
        sent_words = set(re.sub(r'[^\w\s]', '', sentence.lower()).split()) - stopwords
        citation_nums: set[int] = set()

        scored_claims = []
        for claim_text, nums in claim_to_citations.items():
            claim_words = set(re.sub(r'[^\w\s]', '', claim_text.lower()).split()) - stopwords
            if not claim_words:
                continue
            overlap = len(claim_words & sent_words) / len(claim_words)
            if overlap >= 0.5:
                scored_claims.append((overlap, nums))

        if scored_claims:
            scored_claims.sort(key=lambda x: x[0], reverse=True)
            top_overlap = scored_claims[0][0]
            for overlap, nums in scored_claims:
                if overlap >= top_overlap - 0.15:
                    citation_nums.update(nums)

        if citation_nums:
            markers = "".join(f"[{n}]" for n in sorted(citation_nums))
            cited_sentences.append(f"{sentence}{markers}")
        elif remove_unsupported:
            pass
        else:
            cited_sentences.append(sentence)

    cited_response = " ".join(cited_sentences)

    all_passages = []
    for mc in matched_claims:
        all_passages.extend(mc["supporting_passages"])

    reference_list = build_reference_list(citation_map, all_passages)

    return cited_response, reference_list


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
    """Generate a standalone HTML file for interactive citation exploration."""

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
  // Rejoin to full string with sentinel placeholders, then split on sentence boundary
  // Simpler: work directly on parts, splitting text nodes on sentence boundaries
  const sentences = [];
  let buf = [];
  for (const p of parts) {{
    if (p.type === 'text') {{
      // Split this text node on sentence boundaries
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
      // A cite block marks end of sentence
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
  const rows = items.map({{ claim, passage }} => `
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
  // Close all
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

// Init select
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


def run(input_path: str, output_path: str, remove_unsupported: bool = False, html: bool = True):
    with open(input_path, "r") as f:
        data = json.load(f)

    for example in data:
        citation_map = build_citation_map(example["matched_claims"])
        cited_response, references = insert_citations(
            example["raw_response"],
            example["matched_claims"],
            citation_map,
            remove_unsupported,
        )
        example["cited_response"] = cited_response
        example["references"] = references

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