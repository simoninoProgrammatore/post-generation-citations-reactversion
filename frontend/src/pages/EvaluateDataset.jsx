/**
 * EvaluateDataset.jsx — Pagina di valutazione globale del dataset.
 *
 * Il backend (/evaluate-example) esegue la pipeline UNA volta per esempio e
 * restituisce SEMPRE sia nugget_metrics sia deepseek_metrics. Qui le calcoliamo
 * entrambe in aggregato; il toggle Nugget/DeepSeek e' solo una VISTA sugli
 * stessi dati gia' calcolati (non sceglie piu' cosa calcolare).
 *
 * Nota metriche nugget: la precision e' CONTINUA E PESATA
 *   precision(g) = Σ_i (w_i · e_i) / Σ_i w_i
 * dove w_i = match claim↔nugget e e_i = evidence span↔golden_evidence.
 * Il covering e' nugget-centered, allineato a MatchedView (match_score ≥ 0.6).
 */

import { useState, useRef } from 'react'
import api from '../api'
import { useAppData } from '../context/AppData'

import Icon from '../components/Icon'
import { downloadJSON, timestampedFilename } from '../utils/download'


// ── Aggregazione globale ───────────────────────────────────────────────────────

function computeNuggetGlobal(perExample) {
  const gm = {}
  let totalNuggets = 0, totalCovered = 0, totalCited = 0
  let totalReq = 0, totalReqCovered = 0
  let totalOpt = 0, totalOptCovered = 0
  const precs = [], recalls = [], covs = []
  const reqPrecs = [], reqRecalls = []

  for (const ex of perExample) {
    const nm = ex.nugget_metrics
    if (!nm) continue
    precs.push(nm.nugget_precision ?? 0)
    recalls.push(nm.nugget_recall ?? 0)
    covs.push(nm.nugget_coverage ?? 0)
    reqPrecs.push(nm.required_precision ?? 0)
    reqRecalls.push(nm.required_recall ?? 0)
    totalNuggets    += nm.n_nuggets ?? 0
    totalCovered    += nm.n_covered ?? 0
    totalCited      += nm.n_cited ?? 0
    totalReq        += nm.n_required ?? 0
    totalReqCovered += nm.n_required_covered ?? 0
    totalOpt        += nm.n_optional ?? 0
    totalOptCovered += nm.n_optional_covered ?? 0
  }

  if (precs.length) {
    const mean = a => a.reduce((x, y) => x + y, 0) / a.length
    gm.avg_nugget_precision   = mean(precs)
    gm.avg_nugget_recall      = mean(recalls)
    gm.avg_nugget_coverage    = mean(covs)
    gm.avg_required_precision = mean(reqPrecs)
    gm.avg_required_recall    = mean(reqRecalls)
  }
  if (totalNuggets > 0) {
    // "Macro Citation" = quota di nugget coperti che hanno almeno uno span citato.
    gm.macro_nugget_citation = totalCovered > 0 ? totalCited / totalCovered : 0
    gm.macro_nugget_recall   = totalCited / totalNuggets
    gm.macro_nugget_coverage = totalCovered / totalNuggets
  }
  if (totalReq > 0) {
    gm.macro_required_coverage = totalReqCovered / totalReq
  }
  if (totalOpt > 0) {
    gm.macro_optional_coverage = totalOptCovered / totalOpt
  }

  gm.total_nuggets = totalNuggets
  gm.total_cited   = totalCited
  gm.total_covered = totalCovered
  gm.total_required = totalReq
  gm.total_required_covered = totalReqCovered
  gm.total_optional = totalOpt
  gm.total_optional_covered = totalOptCovered

  let totalNoisePassages = 0, totalClaimsNoise = 0, totalNuggetsNoise = 0
  for (const ex of perExample) {
    const nm = ex.nugget_metrics
    if (!nm) continue
    const nu = nm.noise_usage || {}
    totalNoisePassages += nu.noise_supporting_passages || 0
    totalClaimsNoise   += nu.claims_citing_noise || 0
    totalNuggetsNoise  += nm.n_cited_from_noise || 0
  }
  gm.total_noise_passages_used = totalNoisePassages
  gm.total_claims_citing_noise = totalClaimsNoise
  gm.total_nuggets_cited_from_noise = totalNuggetsNoise

  return gm
}

function computeDeepseekGlobal(perExample) {
  const gm = {}
  const precs = [], recalls = []
  let totalPairs = 0, totalPairsSupported = 0, totalClaims = 0, totalSupportedClaims = 0

  for (const ex of perExample) {
    const dm = ex.deepseek_metrics
    if (!dm) continue
    precs.push(dm.citation_precision ?? 0)
    recalls.push(dm.citation_recall ?? 0)
    totalPairs           += dm.n_pairs ?? 0
    totalPairsSupported  += dm.n_pairs_supported ?? 0
    totalClaims          += dm.n_claims ?? 0
    totalSupportedClaims += Math.round((dm.citation_recall ?? 0) * (dm.n_claims ?? 0))
  }
  if (precs.length) {
    const mean = a => a.reduce((x, y) => x + y, 0) / a.length
    gm.avg_citation_precision = mean(precs)
    gm.avg_citation_recall    = mean(recalls)
  }
  if (totalPairs > 0)  gm.macro_citation_precision = totalPairsSupported / totalPairs
  if (totalClaims > 0) gm.macro_citation_recall    = totalSupportedClaims / totalClaims
  gm.total_pairs = totalPairs
  gm.total_pairs_supported = totalPairsSupported
  gm.total_claims = totalClaims
  return gm
}

function normalizeDataset(rawData) {
  const examples = Array.isArray(rawData) ? rawData : [rawData]
  return examples.map((ex, idx) => {
    const question = ex.question || ex.query || ex.title || ex.id || `Esempio ${idx}`
    let docs = []
    if (Array.isArray(ex.docs)) {
      docs = ex.docs.map(d => ({ ...d, title: d.title || '', text: d.text || d.sentence || '' }))
    } else if (Array.isArray(ex.passages)) {
      docs = ex.passages.map(p => {
        if (typeof p === 'string') return { title: '', text: p }
        return { ...p, title: p.title || p.heading || '', text: p.text || p.content || p.sentence || '' }
      })
    } else if (ex.context && Array.isArray(ex.context.documents)) {
      docs = ex.context.documents.map(d => ({ ...d, title: d.title || '', text: d.text || d.content || '' }))
    }
    const nuggets = Array.isArray(ex.nuggets) ? ex.nuggets : null
    return { question, docs, nuggets, _original: ex }
  })
}


// ── ViewToggle (2 stati: vista sui dati gia' calcolati) ─────────────────────────

function ViewToggle({ view, onChange, hasNuggets }) {
  const baseBtn = {
    display: 'flex', alignItems: 'center', gap: 6,
    padding: '6px 14px', fontSize: 12, fontWeight: 600,
    border: 'none', borderRadius: 8, cursor: 'pointer', transition: 'all 0.15s',
  }
  return (
    <div style={{
      display: 'inline-flex', alignItems: 'center',
      background: 'var(--bg)', border: '1px solid var(--border)',
      borderRadius: 10, padding: 3, gap: 2,
    }}>
      <button onClick={() => onChange('nugget')} style={{
        ...baseBtn,
        background: view === 'nugget' ? '#7C3AED' : 'transparent',
        color: view === 'nugget' ? 'white' : 'var(--text-2)',
        boxShadow: view === 'nugget' ? '0 1px 4px rgba(124,58,237,0.3)' : 'none',
        opacity: !hasNuggets ? 0.5 : 1,
      }}>
        <Icon name="target" size={12} strokeWidth={2}
          color={view === 'nugget' ? 'white' : 'var(--text-3)'} />
        Nugget
        {!hasNuggets && (
          <span style={{
            fontSize: 9, fontWeight: 700, background: '#FEF3C7',
            color: '#92400E', padding: '1px 5px', borderRadius: 4,
          }}>no data</span>
        )}
      </button>
      <button onClick={() => onChange('deepseek')} style={{
        ...baseBtn,
        background: view === 'deepseek' ? '#0EA5E9' : 'transparent',
        color: view === 'deepseek' ? 'white' : 'var(--text-2)',
        boxShadow: view === 'deepseek' ? '0 1px 4px rgba(14,165,233,0.3)' : 'none',
      }}>
        <Icon name="search" size={12} strokeWidth={2}
          color={view === 'deepseek' ? 'white' : 'var(--text-3)'} />
        DeepSeek
      </button>
    </div>
  )
}


// ── MetricsLegend — allineata al codice attuale (precision continua pesata) ─────

function MetricsLegend() {
  const mono = { fontFamily: 'var(--mono)', fontSize: 12, color: 'var(--text)' }
  const row = { marginBottom: 14 }
  const name = { fontSize: 13, fontWeight: 700, color: 'var(--text)', marginBottom: 2 }
  const formula = {
    ...mono, display: 'inline-block', background: 'var(--bg)',
    border: '1px solid var(--border)', borderRadius: 6, padding: '3px 8px', margin: '2px 0',
  }
  const note = { fontSize: 11, color: 'var(--text-3)', lineHeight: 1.5, marginTop: 2 }
  const sectionTitle = {
    fontSize: 12, fontWeight: 800, textTransform: 'uppercase', letterSpacing: '0.5px',
    marginBottom: 10, display: 'flex', alignItems: 'center', gap: 8,
  }
  const dot = (c) => ({ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: c })
  const block = (border) => ({
    marginBottom: 18, padding: '16px 18px', background: 'white',
    border: `1px solid var(--border)`, borderLeft: `3px solid ${border}`, borderRadius: 10,
  })

  return (
    <details style={{ marginTop: 20 }}>
      <summary style={{ cursor: 'pointer', fontSize: 13, fontWeight: 600, color: 'var(--text-2)', padding: '8px 0' }}>
        Come vengono calcolate le metriche (definizione tecnica)
      </summary>

      <div style={{ marginTop: 12, fontSize: 13, color: 'var(--text-2)', lineHeight: 1.6 }}>

        {/* Intro */}
        <div style={{
          marginBottom: 18, padding: '16px 18px',
          background: 'linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%)',
          border: '1px solid var(--border)', borderRadius: 10,
        }}>
          <div style={{ ...name, marginBottom: 8 }}>Il sistema di valutazione</div>
          <div style={{ marginBottom: 8 }}>
            La pipeline genera una risposta, la scompone in <b>claim atomici</b>, e per ogni
            claim estrae uno o più <b>span di evidenza</b> dai passaggi citati. Ogni esempio
            viene valutato simultaneamente con due lenti indipendenti:
          </div>
          <div style={{ marginBottom: 6 }}>
            <span style={{ ...dot('#0EA5E9'), marginRight: 6 }} />
            <b>DeepSeek (LLM-as-judge)</b> — un modello giudice legge la coppia (claim, span)
            e decide <span style={mono}>supported: true/false</span> con motivazione.
          </div>
          <div>
            <span style={{ ...dot('#7C3AED'), marginRight: 6 }} />
            <b>Nugget</b> — confronto con ground-truth annotata, con precision <b>continua e
            pesata</b> (vedi sotto): combina quanto un claim parla del nugget con quanto la sua
            evidenza somiglia alla golden evidence.
          </div>
          <div style={{ ...note, marginTop: 10 }}>
            <b>Nota:</b> le due lenti non sono confrontabili 1:1. DeepSeek dà un verdetto
            binario semantico sullo span; il nugget dà un punteggio continuo rispetto al gold.
          </div>
        </div>

        {/* Definizioni comuni */}
        <div style={block('var(--text-3)')}>
          <div style={sectionTitle}>Definizioni comuni</div>
          <div style={row}>
            <div style={name}>Coppia (claim, evidenza)</div>
            <span style={formula}>coppia = (claim c, span ê)</span>
            <div style={note}>
              Ogni claim può avere più span attribuiti (uno per passaggio citato): ciascuno è
              una coppia valutata separatamente.
            </div>
          </div>
          <div>
            <div style={name}>Aggregazione: Avg vs Macro</div>
            <span style={formula}>Avg X = (1/N) · Σ_q X(q)</span>
            <div style={note}>Media delle metriche per esempio. Ogni <b>domanda</b> pesa uguale.</div>
            <div style={{ marginTop: 6 }}>
              <span style={formula}>Macro X = Σ_q num(q) / Σ_q den(q)</span>
              <div style={note}>
                Conteggi aggregati su tutto il dataset, poi un solo rapporto. Gli esempi grandi
                pesano di più.
              </div>
            </div>
          </div>
        </div>

        {/* DeepSeek */}
        <div style={block('#0EA5E9')}>
          <div style={{ ...sectionTitle, color: '#0369A1' }}>
            <span style={dot('#0EA5E9')} /> DeepSeek — LLM-as-judge
          </div>
          <div style={row}>
            <div style={name}>Verdetto per coppia</div>
            <span style={formula}>j(c, ê) = 1 se DeepSeek giudica ê ⊨ c, altrimenti 0</span>
            <div style={note}>Giudizio binario del modello sullo span, con motivazione testuale.</div>
          </div>
          <div style={row}>
            <div style={name}>Citation Precision</div>
            <span style={formula}>P = Σ j(c, ê) / |coppie|</span>
            <div style={note}>Delle coppie valutate, quante DeepSeek giudica valide.</div>
          </div>
          <div>
            <div style={name}>Citation Recall</div>
            <span style={formula}>R = |claim con ≥1 coppia valida| / |claim|</span>
            <div style={note}>Dei claim, quanti hanno almeno uno span giudicato valido.</div>
          </div>
        </div>

        {/* Nugget — precision continua pesata */}
        <div style={block('#7C3AED')}>
          <div style={{ ...sectionTitle, color: '#6D28D9' }}>
            <span style={dot('#7C3AED')} /> Nugget — precision continua pesata
          </div>
          <div style={row}>
            <div style={name}>Covering (fonte di verità = MatchedView)</div>
            <span style={formula}>covers(c, g) ⇔ match_score(c, g) ≥ 0.6</span>
            <div style={note}>
              Un claim copre un nugget se il match (keyword + similarità MiniLM, calcolato nel
              retrieve) supera la soglia. <b>Stesso criterio</b> dello Step 4 (MatchedView): le
              evidenze di ogni claim coprente sono i suoi passaggi, presi così come arrivano dal
              retrieve. <span style={mono}>covered(g) = ∃ c : covers(c, g)</span>.
            </div>
          </div>
          <div style={row}>
            <div style={name}>Peso match e score evidenza</div>
            <span style={formula}>w_i = 0.2·lex(c, g) + 0.8·cos(c, g)</span>
            <div style={note}>Quanto il claim <span style={mono}>c_i</span> parla del nugget <span style={mono}>g</span> (testo↔testo).</div>
            <div style={{ marginTop: 6 }}>
              <span style={formula}>e_i = 0.2·lex(ê_i, e*) + 0.8·cos(ê_i, e*)</span>
              <div style={note}>
                Quanto l'evidenza del claim (media sui suoi span <span style={mono}>ê_i</span>)
                somiglia alla golden evidence <span style={mono}>e*</span>. Claim con e_i = 0
                non contribuiscono.
              </div>
            </div>
          </div>
          <div style={row}>
            <div style={name}>Precision continua per nugget</div>
            <span style={formula}>precision(g) = Σ_i (w_i · e_i) / Σ_i w_i  ∈ [0, 1]</span>
            <div style={note}>
              Media degli score di evidenza pesata per quanto ogni claim parla del nugget. Non è
              più un rapporto binario cited/covered: è continua.
            </div>
          </div>
          <div>
            <div style={name}>Aggregati per esempio</div>
            <span style={formula}>nugget_precision = Σ_g precision(g) / n_covered</span>
            <div style={note}>Media delle precision sui soli nugget coperti (con evidenza reale).</div>
            <div style={{ marginTop: 6 }}>
              <span style={formula}>nugget_recall = Σ_g precision(g) / n_total</span>
              <div style={note}>Stessa somma, ma sul totale dei nugget (inclusi i non coperti).</div>
            </div>
            <div style={{ marginTop: 6 }}>
              <span style={formula}>nugget_coverage = n_covered / n_total</span>
              <div style={note}>Quota di nugget toccati da almeno un claim con evidenza.</div>
            </div>
            <div style={{ ...note, marginTop: 8 }}>
              <b>Macro Citation</b> (dataset): <span style={mono}>Σ n_cited / Σ n_covered</span>,
              dove <span style={mono}>cited(g)</span> = esiste almeno uno span estratto per il
              nugget (indipendente dall'evidence_score). Misura grezza, distinta dalla precision
              continua. Tutte le metriche sono calcolate anche separatamente su
              <span style={mono}> required</span> e <span style={mono}>optional</span>.
            </div>
          </div>
        </div>

      </div>
    </details>
  )
}


// ── NuggetAssociationTable ──────────────────────────────────────────────────────

// Raggruppa all_evidence (lista piatta) per claim → ricostruisce il legame
// claim→evidenze. all_evidence porta il campo "claim" in ogni riga.
function groupEvidenceByClaim(allEvidence = []) {
  const byClaim = new Map()
  for (const ev of allEvidence) {
    const key = ev.claim || '(claim sconosciuto)'
    if (!byClaim.has(key)) byClaim.set(key, [])
    byClaim.get(key).push(ev)
  }
  // Per ogni claim: evidenze ordinate per evidence_score desc; claim ordinati
  // per la loro miglior evidenza desc.
  const groups = [...byClaim.entries()].map(([claim, evs]) => {
    const sorted = [...evs].sort((a, b) => (b.evidence_score ?? 0) - (a.evidence_score ?? 0))
    return { claim, evidences: sorted, bestScore: sorted[0]?.evidence_score ?? 0 }
  })
  groups.sort((a, b) => b.bestScore - a.bestScore)
  return groups
}

function evScoreStyle(s) {
  const v = s ?? 0
  return {
    fg: v >= 0.6 ? '#166534' : v >= 0.3 ? '#92400E' : '#991B1B',
    bg: v >= 0.6 ? '#DCFCE7' : v >= 0.3 ? '#FEF9C3' : '#FEE2E2',
  }
}

// ── Livello 3: una singola evidenza ─────────────────────────────────────────
function EvidenceRow({ ev }) {
  const c = evScoreStyle(ev.evidence_score)
  return (
    <div style={{ padding: '8px 0', borderBottom: '1px solid var(--border-2)', display: 'flex', flexDirection: 'column', gap: 4 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
        <span style={{
          fontSize: 10, fontWeight: 700, fontFamily: 'var(--mono)',
          background: c.bg, color: c.fg, padding: '1px 6px', borderRadius: 10,
        }}>
          {(ev.evidence_score ?? 0).toFixed(2)}
        </span>
        {ev.entailment_score != null && (
          <span style={{ fontSize: 10, color: 'var(--text-3)' }}>ent: {ev.entailment_score.toFixed(2)}</span>
        )}
        {ev.is_noise && (
          <span style={{ fontSize: 9, fontWeight: 600, color: 'var(--amber)' }}>⚠ rumore</span>
        )}
        <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-2)' }}>
          {ev.passage_title || 'Senza titolo'}
        </span>
      </div>
      {ev.span && (
        <div style={{
          marginLeft: 12, padding: '4px 8px', background: '#F0F9FF',
          borderLeft: '3px solid #38BDF8', borderRadius: 4,
          fontSize: 11, color: '#0C4A6E', fontStyle: 'italic',
        }}>
          «{ev.span}»
        </div>
      )}
    </div>
  )
}

// ── Livello 2: un claim coprente, espandibile sulle sue evidenze ─────────────
function ClaimGroup({ group }) {
  const [open, setOpen] = useState(false)
  const c = evScoreStyle(group.bestScore)
  return (
    <div style={{ border: '1px solid var(--border)', borderRadius: 8, overflow: 'hidden', marginBottom: 6 }}>
      <div onClick={() => setOpen(o => !o)}
        style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '8px 12px', background: '#F9FAFB', cursor: 'pointer' }}>
        <span style={{
          fontSize: 10, fontWeight: 700, fontFamily: 'var(--mono)',
          background: c.bg, color: c.fg, padding: '1px 6px', borderRadius: 10, flexShrink: 0,
        }}>
          {group.bestScore.toFixed(2)}
        </span>
        <span style={{ fontSize: 12, color: 'var(--text)', flex: 1, lineHeight: 1.4 }}>{group.claim}</span>
        <span style={{ fontSize: 10, color: 'var(--text-3)', fontFamily: 'var(--mono)', flexShrink: 0 }}>
          {group.evidences.length} evidenz{group.evidences.length === 1 ? 'a' : 'e'}
        </span>
        <span style={{ color: 'var(--text-3)', fontSize: 12 }}>{open ? '▲' : '▼'}</span>
      </div>
      {open && (
        <div style={{ padding: '4px 12px 8px' }}>
          {group.evidences.map((ev, i) => <EvidenceRow key={i} ev={ev} />)}
        </div>
      )}
    </div>
  )
}

// ── Livello 1: un nugget, espandibile sui suoi claim ─────────────────────────
function NuggetGroup({ row }) {
  const [open, setOpen] = useState(false)
  const groups = groupEvidenceByClaim(row.all_evidence)
  const prec = row.nugget_precision_score
  const statusLabel = row.cited ? 'Citato' : row.covered ? 'Coperto' : 'Mancante'
  const statusBg = row.cited ? '#D1FAE5' : row.covered ? '#FEF3C7' : '#FEE2E2'
  const statusFg = row.cited ? '#065F46' : row.covered ? '#92400E' : '#991B1B'

  return (
    <div style={{
      border: `1px solid ${row.required ? '#FDE68A' : 'var(--border)'}`,
      borderRadius: 8, overflow: 'hidden', marginBottom: 8,
    }}>
      {/* Header nugget */}
      <div onClick={() => setOpen(o => !o)}
        style={{
          display: 'flex', alignItems: 'center', gap: 10, padding: '10px 14px',
          background: row.required ? '#FFFBEB' : '#FAFAF9', cursor: 'pointer',
        }}>
        <span style={{
          display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
          width: 20, height: 20, borderRadius: '50%', fontSize: 10, fontWeight: 800, flexShrink: 0,
          background: row.required ? 'linear-gradient(135deg, #F59E0B, #D97706)' : 'linear-gradient(135deg, #D1D5DB, #9CA3AF)',
          color: row.required ? '#FFFBEB' : '#374151',
        }}>
          {row.required ? '★' : '☆'}
        </span>
        <span style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--text-3)', flexShrink: 0 }}>
          {row.exIdx}.{row.nugget_id}
        </span>
        <span style={{ fontSize: 12, color: 'var(--text)', flex: 1, lineHeight: 1.4 }}>{row.nugget_text}</span>
        <span style={{ fontSize: 10, color: 'var(--text-3)', fontFamily: 'var(--mono)', flexShrink: 0 }}>
          {row.n_covering_claims} claim
        </span>
        {prec != null ? (
          <span style={{
            fontFamily: 'var(--mono)', fontSize: 11, fontWeight: 700, flexShrink: 0,
            color: prec >= 0.6 ? 'var(--green)' : prec >= 0.3 ? 'var(--amber)' : 'var(--red)',
          }}>
            {prec.toFixed(2)}
          </span>
        ) : (
          <span style={{ color: 'var(--text-3)', fontStyle: 'italic', fontSize: 10, flexShrink: 0 }}>excl.</span>
        )}
        <span style={{
          fontSize: 10, fontWeight: 700, padding: '2px 8px', borderRadius: 10, flexShrink: 0,
          background: statusBg, color: statusFg,
        }}>
          {statusLabel}
        </span>
        <span style={{ color: 'var(--text-3)', fontSize: 12, flexShrink: 0 }}>{open ? '▲' : '▼'}</span>
      </div>

      {/* Corpo: domanda, keyword, golden, poi i claim */}
      {open && (
        <div style={{ padding: '12px 16px', background: 'white' }}>
          <div style={{ fontSize: 11, color: 'var(--text-3)', marginBottom: 8 }}>
            <strong>Domanda:</strong> {row.question}
          </div>
          {row.keywords?.length > 0 && (
            <div style={{ marginBottom: 8, display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
              <span style={{ fontSize: 11, color: 'var(--text-3)', fontWeight: 600 }}>Keywords:</span>
              {row.keywords.map((kw, ki) => (
                <span key={ki} style={{
                  fontSize: 11, background: '#EDE9FE', color: '#5B21B6',
                  padding: '1px 7px', borderRadius: 10, fontFamily: 'var(--mono)',
                }}>{kw}</span>
              ))}
            </div>
          )}
          {row.golden_evidence && (
            <div style={{
              marginBottom: 10, padding: '8px 12px', background: '#F0F9FF',
              border: '1px solid #BAE6FD', borderRadius: 6, fontSize: 12, color: '#0C4A6E',
            }}>
              <strong>Golden evidence:</strong> {row.golden_evidence}
            </div>
          )}

          {groups.length > 0 ? (
            <div>
              <div style={{
                fontSize: 10, fontWeight: 700, color: 'var(--text-3)',
                textTransform: 'uppercase', letterSpacing: '0.6px', marginBottom: 8,
              }}>
                Claim coprenti ({groups.length}) — clicca per le evidenze
              </div>
              {groups.map((g, i) => <ClaimGroup key={i} group={g} />)}
            </div>
          ) : (
            <div style={{ fontSize: 12, color: 'var(--text-3)', fontStyle: 'italic' }}>
              {row.excluded_no_golden
                ? 'Nugget escluso: manca la golden_evidence, nessuna evidenza valutabile.'
                : 'Nessuna evidenza disponibile per questo nugget.'}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function NuggetAssociationTable({ perExample }) {
  const [filter, setFilter] = useState('all')

  const allRows = []
  for (let i = 0; i < perExample.length; i++) {
    const ex = perExample[i]
    if (ex.error) continue
    const nm = ex.nugget_metrics
    if (!nm?.per_nugget) continue
    for (const pn of nm.per_nugget) {
      allRows.push({ exIdx: i, question: ex.question, ...pn })
    }
  }

  const matchFilter = (r, val) => {
    if (val === 'required') return r.required
    if (val === 'optional') return !r.required
    if (val === 'covered') return r.covered
    if (val === 'uncovered') return !r.covered
    if (val === 'cited') return r.cited
    if (val === 'uncited') return !r.cited
    if (val === 'noise') return r.cited_from_noise
    return true
  }

  const filtered = allRows.filter(r => matchFilter(r, filter))
  if (allRows.length === 0) return null

  return (
    <details style={{ marginTop: 20 }}>
      <summary style={{ cursor: 'pointer', fontSize: 13, fontWeight: 600, color: 'var(--text-2)', padding: '8px 0' }}>
        Associazioni Nugget ↔ Claim ({allRows.length} nuggets totali)
      </summary>

      {/* Filters */}
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', margin: '10px 0' }}>
        {[
          ['all', 'Tutti'],
          ['required', '★ Required'],
          ['optional', '☆ Optional'],
          ['covered', '✓ Coperti'],
          ['uncovered', '✗ Non coperti'],
          ['cited', '✓ Citati'],
          ['uncited', '✗ Non citati'],
          ['noise', '⚠ Da noise'],
        ].map(([val, label]) => (
          <button key={val}
            onClick={() => setFilter(val)}
            style={{
              padding: '4px 10px', fontSize: 11, fontWeight: 600,
              border: `1px solid ${filter === val ? 'var(--accent)' : 'var(--border)'}`,
              borderRadius: 6,
              background: filter === val ? '#EEF2FF' : 'white',
              color: filter === val ? 'var(--accent)' : 'var(--text-2)',
              cursor: 'pointer',
            }}>
            {label} {val !== 'all' ? `(${allRows.filter(r => matchFilter(r, val)).length})` : ''}
          </button>
        ))}
      </div>

      {/* Struttura annidata nugget → claim → evidenze */}
      <div style={{ marginTop: 8 }}>
        {filtered.map((row, idx) => (
          <NuggetGroup key={`${row.exIdx}.${row.nugget_id}.${idx}`} row={row} />
        ))}
      </div>
    </details>
  )
}


// ── Card helper ─────────────────────────────────────────────────────────────────

function MetricCard({ label, value, desc, isCount = false }) {
  if (typeof value !== 'number') return null
  if (isCount) {
    return (
      <div style={{ background: 'white', border: '1px solid var(--border)', borderRadius: 8, padding: '14px 16px' }}>
        <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 4 }}>{label}</div>
        <div style={{ fontSize: 28, fontWeight: 800, color: 'var(--accent)', lineHeight: 1 }}>{value}</div>
        {desc && <div style={{ fontSize: 10, color: 'var(--text-3)', marginTop: 4, lineHeight: 1.4 }}>{desc}</div>}
      </div>
    )
  }
  const pct = Math.round(value * 100)
  const color = value >= 0.7 ? 'var(--green)' : value >= 0.4 ? 'var(--amber)' : 'var(--red)'
  return (
    <div style={{ background: 'white', border: '1px solid var(--border)', borderRadius: 8, padding: '14px 16px' }}>
      <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 28, fontWeight: 800, color, lineHeight: 1 }}>{pct}%</div>
      {desc && <div style={{ fontSize: 10, color: 'var(--text-3)', marginTop: 4, lineHeight: 1.4 }}>{desc}</div>}
      <div style={{ height: 4, background: 'var(--border-2)', borderRadius: 2, marginTop: 8, overflow: 'hidden' }}>
        <div style={{ height: '100%', borderRadius: 2, width: `${Math.min(100, pct)}%`, background: color, transition: 'width 0.5s ease' }} />
      </div>
    </div>
  )
}

const grid = { display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 12, marginBottom: 20 }
const gridSm = { display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', gap: 12, marginBottom: 20 }
const sectionLabel = (color) => ({ fontSize: 11, fontWeight: 700, color, textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 8, display: 'flex', alignItems: 'center', gap: 6 })


// ── Vista DeepSeek ────────────────────────────────────────────────────────────

function DeepSeekView({ gm }) {
  return (
    <div>
      <div style={sectionLabel('#0C4A6E')}>
        <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: '#0EA5E9' }} />
        LLM-as-judge (DeepSeek)
      </div>
      <div style={grid}>
        <MetricCard label="Avg Citation Precision" value={gm.avg_citation_precision} desc="Media precision per esempio." />
        <MetricCard label="Avg Citation Recall" value={gm.avg_citation_recall} desc="Media recall per esempio." />
        <MetricCard label="Macro Citation Precision" value={gm.macro_citation_precision} desc="Coppie valide / coppie totali, su tutto il dataset." />
        <MetricCard label="Macro Citation Recall" value={gm.macro_citation_recall} desc="Claim supportati / claim totali, su tutto il dataset." />
      </div>
      <div style={gridSm}>
        <MetricCard label="Claim Totali" value={gm.total_claims} isCount />
        <MetricCard label="Coppie Totali" value={gm.total_pairs} isCount />
        <MetricCard label="Coppie Valide" value={gm.total_pairs_supported} isCount />
      </div>
    </div>
  )
}


// ── Vista Nugget ──────────────────────────────────────────────────────────────

function NuggetView({ gm, perExample }) {
  return (
    <div>
      {/* Required */}
      <div style={sectionLabel('#92400E')}>
        <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: 'linear-gradient(135deg, #F59E0B, #D97706)' }} />
        Required Nuggets
      </div>
      <div style={grid}>
        <MetricCard label="Avg Precision (Req)" value={gm.avg_required_precision} desc="Media precisioni continue per esempio, solo required." />
        <MetricCard label="Avg Recall (Req)" value={gm.avg_required_recall} desc="Media recall per esempio, solo required." />
        <MetricCard label="Macro Coverage (Req)" value={gm.macro_required_coverage} desc="covered_req / total_req su tutto il dataset." />
      </div>

      {/* All */}
      <div style={sectionLabel('var(--text-2)')}>All Nuggets (Total)</div>
      <div style={grid}>
        <MetricCard label="Avg Precision" value={gm.avg_nugget_precision} desc="Media precisioni continue per esempio." />
        <MetricCard label="Avg Recall" value={gm.avg_nugget_recall} desc="Media recall per esempio." />
        <MetricCard label="Avg Coverage" value={gm.avg_nugget_coverage} desc="Media copertura per esempio." />
        <MetricCard label="Macro Citation" value={gm.macro_nugget_citation} desc="cited / covered su tutto il dataset (presenza di span)." />
        <MetricCard label="Macro Recall" value={gm.macro_nugget_recall} desc="cited / total su tutto il dataset." />
        <MetricCard label="Macro Coverage" value={gm.macro_nugget_coverage} desc="covered / total su tutto il dataset." />
      </div>

      {/* Counts */}
      <div style={sectionLabel('var(--text-2)')}>Conteggi</div>
      <div style={gridSm}>
        <MetricCard label="Nuggets Totali" value={gm.total_nuggets} isCount />
        <MetricCard label="Coperti" value={gm.total_covered} isCount />
        <MetricCard label="Citati" value={gm.total_cited} isCount />
        <MetricCard label="Required" value={gm.total_required} isCount />
        <MetricCard label="Req. Coperti" value={gm.total_required_covered} isCount />
        <MetricCard label="Optional" value={gm.total_optional} isCount />
        <MetricCard label="Opt. Coperti" value={gm.total_optional_covered} isCount />
        <MetricCard label="⚠ Noise Usati" value={gm.total_noise_passages_used} isCount />
        <MetricCard label="⚠ Claims con Noise" value={gm.total_claims_citing_noise} isCount />
        <MetricCard label="⚠ Nuggets da Noise" value={gm.total_nuggets_cited_from_noise} isCount />
      </div>

      <NuggetAssociationTable perExample={perExample} />
    </div>
  )
}


// ── Results view ────────────────────────────────────────────────────────────────

function DatasetEvalResultsView({ results, view, onViewChange, hasNuggets, onSave, onDownload }) {
  const nuggetGm = results.nugget_global || {}
  const deepseekGm = results.deepseek_global || {}
  const perExample = results.per_example || []
  const [expandedEx, setExpandedEx] = useState(null)

  return (
    <div>
      {/* Summary header */}
      <div style={{
        padding: '16px 20px',
        background: 'linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 100%)',
        border: '1px solid #C7D2FE', borderRadius: 10, marginBottom: 20,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
          <Icon name="barChart2" size={20} color="#4338CA" strokeWidth={2} />
          <span style={{ fontSize: 16, fontWeight: 700, color: '#312E81', flex: 1 }}>
            Valutazione Globale Dataset
          </span>
          <ViewToggle view={view} onChange={onViewChange} hasNuggets={hasNuggets} />
        </div>
        <div style={{ fontSize: 12, color: '#4338CA', display: 'flex', gap: 20, flexWrap: 'wrap' }}>
          <span>{results.num_examples} esempi</span>
          <span>{results.num_successful} completati con successo</span>
          {results.runtime_seconds != null && <span>{results.runtime_seconds}s runtime</span>}
          {results.partial && (
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontWeight: 700, color: '#B45309' }}>
              <span className="spinner" style={{ width: 11, height: 11, borderColor: '#B45309', borderTopColor: 'transparent' }} />
              Valutazione in corso…
            </span>
          )}
        </div>
      </div>

      {/* Vista selezionata */}
      {view === 'deepseek'
        ? <DeepSeekView gm={deepseekGm} />
        : <NuggetView gm={nuggetGm} perExample={perExample} />}

      {/* Per-example summary table (giudizi DeepSeek espandibili) */}
      <details style={{ marginTop: 16 }}>
        <summary style={{ cursor: 'pointer', fontSize: 13, fontWeight: 600, color: 'var(--text-2)', padding: '8px 0' }}>
          Dettaglio per esempio ({perExample.length})
        </summary>
        <div style={{ marginTop: 8 }}>
          {perExample.map((ex, i) => {
            const isOpen = expandedEx === i
            const dm = ex.deepseek_metrics
            const clickable = dm?.per_claim?.length > 0

            return (
              <div key={i} style={{ marginBottom: 4 }}>
                <div
                  onClick={() => clickable && setExpandedEx(isOpen ? null : i)}
                  style={{
                    padding: '8px 12px',
                    background: ex.error ? '#FEF2F2' : isOpen ? '#F0F9FF' : '#FAFAF9',
                    border: `1px solid ${ex.error ? '#FECACA' : isOpen ? '#BAE6FD' : 'var(--border-2)'}`,
                    borderRadius: 6, fontSize: 12,
                    cursor: clickable ? 'pointer' : 'default',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--text-3)' }}>[{i}]</span>
                    <span style={{ flex: 1, fontWeight: 500, color: 'var(--text)' }}>
                      {ex.question?.slice(0, 80)}{(ex.question?.length > 80) ? '...' : ''}
                    </span>
                    {dm && (
                      <span style={{ fontSize: 11, color: 'var(--text-3)', fontFamily: 'var(--mono)' }}>
                        {Math.round((dm.citation_precision ?? 0) * 100)}% P · {Math.round((dm.citation_recall ?? 0) * 100)}% R
                      </span>
                    )}
                    {ex.error
                      ? <span style={{ color: '#DC2626', fontSize: 11 }}>❌ {ex.error}</span>
                      : <span style={{ color: 'var(--green)', fontSize: 11 }}>✓ OK</span>}
                    {clickable && <span style={{ color: 'var(--text-3)', fontSize: 12 }}>{isOpen ? '▲' : '▼'}</span>}
                  </div>
                </div>

                {isOpen && clickable && (
                  <div style={{ marginTop: 4, marginBottom: 8, padding: '12px 14px', background: 'white', border: '1px solid #BAE6FD', borderRadius: 8 }}>
                    {dm.per_claim.map((c, ci) => {
                      const ok = c.any_supported
                      return (
                        <div key={ci} style={{ marginBottom: 10, border: `1px solid ${ok ? '#A7F3D0' : '#FECACA'}`, borderRadius: 8, overflow: 'hidden' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '8px 12px', background: ok ? '#F0FDF4' : '#FFF1F2' }}>
                            <span style={{
                              fontSize: 10, fontWeight: 700, padding: '2px 8px', borderRadius: 10, whiteSpace: 'nowrap', flexShrink: 0,
                              background: ok ? '#DCFCE7' : '#FEE2E2', color: ok ? '#166534' : '#991B1B',
                            }}>
                              {ok ? `${c.n_supported}/${c.n_passages} valide` : 'nessun supporto'}
                            </span>
                            <span style={{ fontSize: 12, color: 'var(--text)', flex: 1, lineHeight: 1.4 }}>{c.claim}</span>
                          </div>
                          <div style={{ padding: '8px 12px' }}>
                            {(c.judgments || []).length === 0 ? (
                              <div style={{ fontSize: 12, color: 'var(--text-3)', fontStyle: 'italic' }}>
                                Nessun passaggio citato per questo claim.
                              </div>
                            ) : c.judgments.map((j, ji) => (
                              <div key={ji} style={{
                                marginBottom: 6, padding: '8px 12px', borderRadius: 6,
                                background: j.supported ? '#F0FDF4' : '#FAFAF9',
                                border: `1px solid ${j.supported ? '#86EFAC' : 'var(--border)'}`,
                              }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                                  <span style={{
                                    fontSize: 10, fontWeight: 700, padding: '1px 7px', borderRadius: 8,
                                    background: j.supported ? '#DCFCE7' : '#FEE2E2', color: j.supported ? '#166534' : '#991B1B',
                                  }}>
                                    {j.supported ? 'SUPPORTED' : 'NOT SUPPORTED'}
                                  </span>
                                  <span style={{ fontSize: 12, fontWeight: 600 }}>{j.passage_title || '—'}</span>
                                </div>
                                {j.evidence && (
                                  <div style={{
                                    fontSize: 11, color: '#166534', lineHeight: 1.5, marginBottom: 6,
                                    padding: '6px 10px', background: '#ECFDF5', borderRadius: 6, borderLeft: '3px solid #86EFAC',
                                  }}>
                                    <strong style={{ color: 'var(--text-3)', fontWeight: 700 }}>Evidenza: </strong>{j.evidence}
                                  </div>
                                )}
                                {j.reason && (
                                  <div style={{
                                    fontSize: 11, color: '#0C4A6E', fontStyle: 'italic',
                                    padding: '6px 10px', background: '#F0F9FF', borderRadius: 6, borderLeft: '3px solid #BAE6FD',
                                  }}>
                                    <strong>DeepSeek:</strong> {j.reason}
                                  </div>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </details>

      <MetricsLegend />

      {/* Actions */}
      <div style={{ marginTop: 20, display: 'flex', gap: 12 }}>
        <button className="btn btn-primary" onClick={onSave}>
          <Icon name="download" size={13} color="white" strokeWidth={2} />
          Salva in Esplora
        </button>
        <button className="btn btn-secondary" onClick={onDownload}>
          <Icon name="download" size={13} strokeWidth={1.75} />
          Scarica dati
        </button>
      </div>
    </div>
  )
}


// ── Settings panel ────────────────────────────────────────────────────────────

function SettingsPanel({ model, setModel, retrieveMethod, setRetrieveMethod,
  threshold, setThreshold, topK, setTopK,
  preFilterK, setPreFilterK, noiseEnabled, setNoiseEnabled }) {
  const [open, setOpen] = useState(false)

  return (
    <div className="card" style={{ marginBottom: 16 }}>
      <div
        style={{ display: 'flex', alignItems: 'center', padding: '12px 20px', cursor: 'pointer', gap: 8 }}
        onClick={() => setOpen(o => !o)}
      >
        <Icon name="settings" size={14} strokeWidth={1.75} color="var(--text-2)" />
        <span style={{ fontSize: 13, fontWeight: 600, flex: 1 }}>Impostazioni modello &amp; retrieval</span>
        <Icon name={open ? 'chevronUp' : 'chevronDown'} size={13} strokeWidth={2} color="var(--text-3)" />
      </div>
      {open && (
        <div style={{ padding: '0 20px 20px', borderTop: '1px solid var(--border-2)' }}>
          <div style={{ paddingTop: 16, display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: 16 }}>
            <div className="form-group" style={{ marginBottom: 0 }}>
              <label className="form-label">Modello LLM</label>
              <select className="input" value={model} onChange={e => setModel(e.target.value)}>
                <option>gemma3:1b</option>
                <option>gemma3:4b</option>
                <option>llama3.2:3b</option>
                <option>phi4-mini</option>
                <option>claude-haiku-4-5-20251001</option>
                <option>claude-sonnet-4-20250514</option>
              </select>
            </div>
            <div className="form-group" style={{ marginBottom: 0 }}>
              <label className="form-label">Metodo retrieval</label>
              <select className="input" value={retrieveMethod} onChange={e => setRetrieveMethod(e.target.value)}>
                <option value="nli">NLI</option>
                <option value="llm">LLM</option>
              </select>
            </div>
            <div className="form-group" style={{ marginBottom: 0 }}>
              <label className="form-label">NLI Threshold — {threshold}</label>
              <input type="range" min={0} max={1} step={0.05} value={threshold}
                onChange={e => setThreshold(+e.target.value)}
                style={{ width: '100%', accentColor: 'var(--accent)', marginTop: 8 }} />
            </div>
            <div className="form-group" style={{ marginBottom: 0 }}>
              <label className="form-label">Top-K passages — {topK}</label>
              <input type="range" min={1} max={5} step={1} value={topK}
                onChange={e => setTopK(+e.target.value)}
                style={{ width: '100%', accentColor: 'var(--accent)', marginTop: 8 }} />
            </div>
            <div className="form-group" style={{ marginBottom: 0 }}>
              <label className="form-label">Pre-filter frasi (BGE) — {preFilterK === 0 ? 'Off' : `top ${preFilterK}`}</label>
              <input type="range" min={0} max={30} step={5} value={preFilterK}
                onChange={e => setPreFilterK(+e.target.value)}
                style={{ width: '100%', accentColor: 'var(--accent)', marginTop: 8 }} />
              <span style={{ fontSize: 10, color: 'var(--text-3)' }}>
                {preFilterK === 0
                  ? 'NLI su tutte le frasi (più lento, più preciso)'
                  : `Embedding pre-filter → top ${preFilterK} frasi → NLI (più veloce)`}
              </span>
            </div>
            <div className="form-group" style={{ marginBottom: 0, display: 'flex', flexDirection: 'column', gap: 8 }}>
              <label className="form-label">Noise injection</label>
              <button
                className="btn"
                onClick={() => setNoiseEnabled(n => !n)}
                style={{
                  padding: '6px 14px', fontSize: 12, alignSelf: 'flex-start',
                  background: noiseEnabled ? '#DCFCE7' : '#FEE2E2',
                  color: noiseEnabled ? '#166534' : '#991B1B',
                  border: `1px solid ${noiseEnabled ? '#BBF7D0' : '#FECACA'}`,
                  borderRadius: 6,
                }}
              >
                <Icon name={noiseEnabled ? 'zap' : 'zapOff'} size={11} strokeWidth={2}
                  color={noiseEnabled ? '#166534' : '#991B1B'} />
                {noiseEnabled ? 'Noise ON' : 'Noise OFF'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}


// ── Main page ─────────────────────────────────────────────────────────────────

export default function EvaluateDataset() {
  const { addPipelineResult } = useAppData()

  // Settings
  const [model, setModel] = useState('claude-haiku-4-5-20251001')
  const [retrieveMethod, setRetrieveMethod] = useState('nli')
  const [threshold, setThreshold] = useState(0.5)
  const [topK, setTopK] = useState(3)
  const [preFilterK, setPreFilterK] = useState(0)
  const [noiseEnabled, setNoiseEnabled] = useState(false)

  // Dataset
  const [dataset, setDataset] = useState(null)
  const [datasetName, setDatasetName] = useState('')
  const fileRef = useRef()
  const resultsFileRef = useRef()

  // View (solo visualizzazione: i dati hanno SEMPRE entrambe le metriche)
  const [view, setView] = useState('nugget')

  // Run state
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState({ current: 0, total: 0 })
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)

  const hasNuggets = dataset?.some(ex => ex.nuggets && ex.nuggets.length > 0) ?? false

  function onResultsUpload(e) {
    const file = e.target.files[0]
    if (!file) return
    e.target.value = ''
    const reader = new FileReader()
    reader.onload = evt => {
      try {
        const parsed = JSON.parse(evt.target.result)
        if (!parsed.per_example || (!parsed.nugget_global && !parsed.deepseek_global)) {
          throw new Error('Il file non contiene risultati validi (manca per_example o le metriche globali).')
        }
        setResults(parsed)
        setDatasetName(file.name)
        setError(null)
      } catch (err) {
        setError(`Errore caricamento risultati: ${err.message}`)
      }
    }
    reader.readAsText(file)
  }

  function onFileUpload(e) {
    const file = e.target.files[0]
    if (!file) return
    e.target.value = ''
    const reader = new FileReader()
    reader.onload = evt => {
      try {
        const parsed = JSON.parse(evt.target.result)
        const normalized = normalizeDataset(parsed)
        if (normalized.length === 0) throw new Error('Il file non contiene esempi validi.')
        setDataset(normalized)
        setDatasetName(file.name)
        setResults(null)
        setError(null)
        // Vista di default: nugget se ci sono, altrimenti deepseek.
        setView(normalized.some(ex => ex.nuggets) ? 'nugget' : 'deepseek')
      } catch (err) {
        setError(`Errore lettura file: ${err.message}`)
      }
    }
    reader.readAsText(file)
  }

  async function runEvaluation() {
    if (!dataset || dataset.length === 0) return
    setRunning(true)
    setError(null)
    setResults(null)
    setProgress({ current: 0, total: dataset.length })

    const noisePool = noiseEnabled
      ? dataset.flatMap((ex, i) => (ex.docs || []).map(doc => ({ ...doc, _source_idx: i })))
      : []

    const perExample = []

    for (let idx = 0; idx < dataset.length; idx++) {
      const ex = dataset[idx]
      try {
        const res = await api.pipeline.evaluateExample({
          example: {
            question: ex.question,
            docs: ex.docs || [],
            nuggets: ex.nuggets || null,
          },
          model,
          retrieve_method: retrieveMethod,
          threshold,
          top_k: topK,
          // eval_mode RIMOSSO: il backend calcola sempre nugget + deepseek.
          noise_enabled: noiseEnabled,
          noise_pool: noisePool.filter(d => d._source_idx !== idx),
          noise_seed: 42,
          example_idx: idx,
          pre_filter_k: preFilterK,
        })
        perExample.push(res)
      } catch (e) {
        perExample.push({ question: ex.question, error: e.message })
      }

      setProgress({ current: idx + 1, total: dataset.length })

      // Update incrementale: ricalcola ENTRAMBE le aggregazioni.
      const isLast = idx + 1 === dataset.length
      setResults({
        nugget_global: computeNuggetGlobal(perExample),
        deepseek_global: computeDeepseekGlobal(perExample),
        per_example: [...perExample],
        num_examples: dataset.length,
        num_successful: perExample.filter(e => !e.error).length,
        runtime_seconds: null,
        partial: !isLast,
      })
    }

    setRunning(false)
  }

  return (
    <div>

      <input ref={fileRef} type="file" accept=".json,.jsonl" onChange={onFileUpload} style={{ display: 'none' }} />

      {/* Page header */}
      <div className="page-header">
        <div className="page-header-title" style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            width: 32, height: 32,
            background: 'linear-gradient(135deg, #6366F1 0%, #7C3AED 100%)',
            borderRadius: 8, display: 'flex', alignItems: 'center', justifyContent: 'center',
            flexShrink: 0, boxShadow: '0 2px 8px rgba(99,102,241,0.3)',
          }}>
            <Icon name="database" size={16} color="white" strokeWidth={1.75} />
          </div>
          Valutazione Dataset
        </div>
        <div className="page-header-sub">
          Esegui la pipeline su tutti gli esempi: ogni esempio è valutato con metriche Nugget e DeepSeek insieme.
        </div>
      </div>

      {error && (
        <div className="info-box info-box-red" style={{ marginBottom: 16 }}>
          <Icon name="xCircle" size={15} strokeWidth={1.75} style={{ flexShrink: 0, marginTop: 1 }} />
          <span><strong>Errore:</strong> {error}</span>
        </div>
      )}

      <SettingsPanel
        model={model} setModel={setModel}
        retrieveMethod={retrieveMethod} setRetrieveMethod={setRetrieveMethod}
        threshold={threshold} setThreshold={setThreshold}
        topK={topK} setTopK={setTopK}
        preFilterK={preFilterK} setPreFilterK={setPreFilterK}
        noiseEnabled={noiseEnabled} setNoiseEnabled={setNoiseEnabled}
      />

      {/* Dataset upload & run card */}
      <div className="card" style={{ marginBottom: 20 }}>
        <div style={{ padding: '16px 20px' }}>

          <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap', marginBottom: dataset ? 16 : 0 }}>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text)', marginBottom: 4 }}>
                {dataset ? (
                  <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <Icon name="fileText" size={14} strokeWidth={1.75} color="var(--accent)" />
                    {datasetName}
                    <span style={{
                      fontSize: 11, fontWeight: 700, background: 'var(--bg)', border: '1px solid var(--border)',
                      padding: '1px 8px', borderRadius: 10, color: 'var(--text-2)',
                    }}>
                      {dataset.length} esempi
                    </span>
                    {hasNuggets && (
                      <span style={{ fontSize: 10, fontWeight: 700, background: '#EDE9FE', color: '#5B21B6', padding: '2px 7px', borderRadius: 10 }}>
                        nuggets presenti
                      </span>
                    )}
                  </span>
                ) : 'Carica un dataset per iniziare'}
              </div>
              {dataset && (
                <div style={{ fontSize: 12, color: 'var(--text-3)' }}>
                  Modello: <span style={{ fontFamily: 'var(--mono)' }}>{model}</span>
                  &nbsp;·&nbsp; Retrieval: {retrieveMethod}
                  &nbsp;·&nbsp; Threshold: {threshold}
                  &nbsp;·&nbsp; Top-K: {topK}
                  {noiseEnabled && <>&nbsp;·&nbsp; <span style={{ color: '#991B1B' }}>Noise ON</span></>}
                </div>
              )}
            </div>

            <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
              <button className="btn btn-secondary" onClick={() => fileRef.current.click()}>
                <Icon name="upload" size={13} strokeWidth={1.75} />
                {dataset ? 'Cambia dataset' : 'Seleziona file JSON'}
              </button>
            </div>
          </div>

          {dataset && (
            <div style={{ padding: '16px 20px', background: '#F5F3FF', border: '2px dashed #C7D2FE', borderRadius: 10 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 14, fontWeight: 700, color: '#4338CA', marginBottom: 4 }}>
                    Valuta tutto il dataset
                  </div>
                  <div style={{ fontSize: 12, color: '#6366F1' }}>
                    Esegui la pipeline su tutti i {dataset.length} esempi. Ogni esempio produce
                    metriche <strong>Nugget</strong> e <strong>DeepSeek</strong> in un solo giro.
                  </div>
                </div>
                <button
                  className="btn"
                  onClick={runEvaluation}
                  disabled={running}
                  style={{
                    background: '#6366F1', color: 'white', border: 'none',
                    padding: '10px 20px', fontWeight: 700, fontSize: 13, borderRadius: 8,
                    cursor: running ? 'not-allowed' : 'pointer', opacity: running ? 0.7 : 1,
                    display: 'flex', alignItems: 'center', gap: 8,
                  }}
                >
                  {running ? (
                    <>
                      <span className="spinner" style={{ width: 14, height: 14, borderColor: 'white', borderTopColor: 'transparent' }} />
                      Valutazione in corso...
                    </>
                  ) : (
                    <>
                      <Icon name="play" size={14} color="white" strokeWidth={2} />
                      Valuta tutto ({dataset.length} esempi)
                    </>
                  )}
                </button>
              </div>

              {running && progress.total > 0 && (
                <div style={{ marginTop: 12 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#6366F1', marginBottom: 4 }}>
                    <span>Progresso</span>
                    <span>{progress.current}/{progress.total}</span>
                  </div>
                  <div style={{ height: 6, background: '#E0E7FF', borderRadius: 3, overflow: 'hidden' }}>
                    <div style={{
                      height: '100%', width: `${(progress.current / progress.total) * 100}%`,
                      background: '#6366F1', borderRadius: 3, transition: 'width 0.3s ease',
                    }} />
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Results */}
      {results && (
        <div className="card">
          <div style={{ padding: '16px 20px' }}>
            <DatasetEvalResultsView
              results={results}
              view={view}
              onViewChange={setView}
              hasNuggets={hasNuggets}
              onSave={() => {
                addPipelineResult({
                  question: `[Dataset] ${datasetName}`,
                  dataset_eval_results: results,
                })
                alert('Risultati globali salvati! Visibile nella pagina Esplora.')
              }}
              onDownload={() => downloadJSON(results, timestampedFilename('dataset_eval'))}
            />
          </div>
        </div>
      )}

      {/* Empty state */}
      {!dataset && !results && (
        <div style={{
          marginTop: 32, padding: '48px 32px', textAlign: 'center',
          border: '2px dashed var(--border)', borderRadius: 12, color: 'var(--text-3)',
        }}>
          <input ref={resultsFileRef} type="file" accept=".json" onChange={onResultsUpload} style={{ display: 'none' }} />
          <Icon name="database" size={40} strokeWidth={1} color="var(--border)" />
          <div style={{ marginTop: 16, fontSize: 15, fontWeight: 600, color: 'var(--text-2)' }}>
            Nessun dataset caricato
          </div>
          <div style={{ marginTop: 8, fontSize: 13 }}>
            Carica un file JSON compatibile (ALCE / ELI5 / QAMPARI) per avviare la valutazione globale,
            oppure carica un file di risultati già valutato.
          </div>
          <div style={{ marginTop: 20, display: 'flex', gap: 12, justifyContent: 'center' }}>
            <button className="btn btn-primary" onClick={() => fileRef.current.click()}>
              <Icon name="upload" size={14} strokeWidth={1.75} color="white" />
              Carica dataset
            </button>
            <button className="btn btn-secondary" onClick={() => resultsFileRef.current.click()}>
              <Icon name="upload" size={14} strokeWidth={1.75} />
              Carica risultati
            </button>
          </div>
        </div>
      )}

    </div>
  )
}