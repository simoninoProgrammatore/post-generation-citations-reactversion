/**
 * Pipeline.jsx — Pipeline interattivo (6 step manuali)
 * Supporta import da ALCE, ELI5, QAMPARI (normalizzazione automatica).
 * Step 6 supporta due modalità di valutazione: Standard e Nugget.
 * I nuggets vengono letti direttamente dal dataset caricato allo Step 1.
 */

import { useState, useRef, useCallback } from 'react'
import api from '../api'
import { useAppData } from '../context/AppData'

import StepCard from '../components/StepCard'
import EmptyState from '../components/EmptyState'
import ScorePill from '../components/ScorePill'
import MetricCard from '../components/MetricCard'
import Icon from '../components/Icon'
import { downloadJSON, timestampedFilename } from '../utils/download'


// ── Metric definitions ────────────────────────────────────────────────────────

const GLOBAL_METRIC_INFO = {
  macro_nugget_precision: { label: 'Macro Nugget Precision', desc: 'Precisione calcolata su tutti i nugget del dataset (cited/covered).' },
  macro_nugget_recall: { label: 'Macro Nugget Recall', desc: 'Recall calcolata su tutti i nugget del dataset (cited/total).' },
  macro_nugget_coverage: { label: 'Macro Nugget Coverage', desc: 'Copertura su tutti i nugget del dataset (covered/total).' },
  avg_nugget_precision: { label: 'Avg Nugget Precision', desc: 'Media delle precisioni per esempio.' },
  avg_nugget_recall: { label: 'Avg Nugget Recall', desc: 'Media delle recall per esempio.' },
}

const METRIC_INFO_STANDARD = {
  citation_precision:    { label: 'Citation Precision',     desc: '% di coppie (claim, passaggio) dove il passaggio supporta il claim via NLI.' },
  citation_recall:       { label: 'Citation Recall',        desc: '% di claims con almeno un passaggio citato che fornisce supporto NLI.' },
  factual_precision:     { label: 'Factual Precision',      desc: '% di claims con almeno un passaggio di supporto (senza NLI).' },
  factual_precision_nli: { label: 'Factual Precision (NLI)', desc: 'Come Factual Precision, ma verificato via NLI.' },
  unsupported_ratio:     { label: 'Unsupported Ratio',      desc: '% di claims senza alcun passaggio di supporto.' },
  avg_entailment_score:  { label: 'Avg Entailment Score',   desc: 'Score medio di entailment tra claims e passaggi.' },
}

const METRIC_INFO_NUGGET = {
  nugget_precision: {
    label: 'Nugget Precision',
    desc: 'Dei nugget coperti da un claim, quanti hanno la citazione al golden passage?',
    icon: 'target',
  },
  nugget_recall: {
    label: 'Nugget Recall',
    desc: 'Dei nugget totali (required), quanti sono menzionati da almeno un claim generato con citazione?',
    icon: 'refreshCw',
  },
  nugget_coverage: {
    label: 'Nugget Coverage',
    desc: 'Quanti nugget sono menzionati da almeno un claim generato (indipendentemente dalla citazione)?',
    icon: 'layers',
  },
}

function metricColor(key, v) {
  if (key === 'unsupported_ratio') {
    return v <= 0.2 ? 'var(--green)' : v <= 0.5 ? 'var(--amber)' : 'var(--red)'
  }
  return v >= 0.7 ? 'var(--green)' : v >= 0.4 ? 'var(--amber)' : 'var(--red)'
}

// ── Dataset helpers ───────────────────────────────────────────────────────────

function buildNoisePool(dataset, excludeIdx) {
  const pool = []
  for (let i = 0; i < dataset.length; i++) {
    if (i === excludeIdx) continue
    for (const doc of (dataset[i].docs || [])) pool.push(doc)
  }
  return pool
}

function seededRng(seed) {
  let t = seed + 0x6D2B79F5
  return () => {
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function injectNoise(docs, noisePool, seed = 42) {
  if (!docs.length || !noisePool.length) return docs
  const rng = seededRng(seed)
  const nOriginal = docs.length
  const nMin = 1
  const nMax = Math.max(1, Math.ceil(nOriginal * 0.5))
  const nNoise = nMin + Math.floor(rng() * (nMax - nMin + 1))
  const indices = noisePool.map((_, i) => i)
  const sampled = []
  for (let i = 0; i < Math.min(nNoise, indices.length); i++) {
    const j = i + Math.floor(rng() * (indices.length - i))
    ;[indices[i], indices[j]] = [indices[j], indices[i]]
    sampled.push(indices[i])
  }
  const result = docs.map(d => ({ ...d, is_noise: false }))
  for (const idx of sampled) {
    const d = { ...noisePool[idx], is_noise: true }
    delete d.is_gold; delete d.support_level; delete d.evidence_sentence
    result.push(d)
  }
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1))
    ;[result[i], result[j]] = [result[j], result[i]]
  }
  return result
}

function normalizeDataset(rawData) {
  let examples = Array.isArray(rawData) ? rawData : [rawData]
  return examples.map((ex, idx) => {
    const question = ex.question || ex.query || ex.title || ex.id || `Esempio ${idx}`

    // Preserve docs with ALL original fields (is_gold, golden_passage_title, etc.)
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

    const claims = Array.isArray(ex.claims) ? ex.claims : null
    const raw_response = ex.answer || ex.response || ex.raw_response || null
    const matched_claims = ex.matched_claims || ex.matched || null
    // Preserve nuggets inline if present
    const nuggets = Array.isArray(ex.nuggets) ? ex.nuggets : null

    return { question, docs, claims, raw_response, matched_claims, nuggets, _original: ex }
  })
}

// ── EvalMode Toggle ───────────────────────────────────────────────────────────

function EvalModeToggle({ mode, onChange, hasNuggets }) {
  return (
    <div style={{
      display: 'inline-flex',
      alignItems: 'center',
      background: 'var(--bg)',
      border: '1px solid var(--border)',
      borderRadius: 10,
      padding: 3,
      gap: 2,
    }}>
      {/* Standard */}
      <button
        onClick={() => onChange('standard')}
        style={{
          display: 'flex', alignItems: 'center', gap: 6,
          padding: '6px 14px',
          fontSize: 12, fontWeight: 600,
          border: 'none', borderRadius: 8, cursor: 'pointer',
          transition: 'all 0.15s',
          background: mode === 'standard' ? 'var(--accent)' : 'transparent',
          color: mode === 'standard' ? 'white' : 'var(--text-2)',
          boxShadow: mode === 'standard' ? '0 1px 4px rgba(0,0,0,0.15)' : 'none',
        }}
      >
        <Icon name="barChart2" size={12} strokeWidth={2}
          color={mode === 'standard' ? 'white' : 'var(--text-3)'} />
        Standard
      </button>

      {/* Nugget */}
      <button
        onClick={() => onChange('nugget')}
        style={{
          display: 'flex', alignItems: 'center', gap: 6,
          padding: '6px 14px',
          fontSize: 12, fontWeight: 600,
          border: 'none', borderRadius: 8, cursor: 'pointer',
          transition: 'all 0.15s',
          background: mode === 'nugget' ? '#7C3AED' : 'transparent',
          color: mode === 'nugget' ? 'white' : 'var(--text-2)',
          boxShadow: mode === 'nugget' ? '0 1px 4px rgba(124,58,237,0.3)' : 'none',
          opacity: !hasNuggets && mode !== 'nugget' ? 0.5 : 1,
        }}
      >
        <Icon name="target" size={12} strokeWidth={2}
          color={mode === 'nugget' ? 'white' : 'var(--text-3)'} />
        Nugget
        {!hasNuggets && (
          <span style={{
            fontSize: 9, fontWeight: 700, background: '#FEF3C7',
            color: '#92400E', padding: '1px 5px', borderRadius: 4,
          }}>
            no data
          </span>
        )}
      </button>
    </div>
  )
}

// ── NuggetMissingFieldsError ──────────────────────────────────────────────────

function NuggetMissingFieldsError({ missingFields }) {
  return (
    <div style={{
      padding: '14px 16px',
      background: '#FFF7ED',
      border: '1px solid #FED7AA',
      borderLeft: '3px solid #F97316',
      borderRadius: 8,
      marginBottom: 16,
    }}>
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: 10 }}>
        <Icon name="alertTriangle" size={15} color="#F97316" strokeWidth={2} style={{ flexShrink: 0, marginTop: 1 }} />
        <div>
          <div style={{ fontSize: 13, fontWeight: 700, color: '#7C2D12', marginBottom: 6 }}>
            Campi mancanti per la valutazione Nugget
          </div>
          <div style={{ fontSize: 12, color: '#9A3412', lineHeight: 1.6 }}>
            Il dataset caricato non contiene i campi necessari per la modalità Nugget.
            Campi mancanti o vuoti:
          </div>
          <div style={{ marginTop: 8, display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            {missingFields.map(f => (
              <span key={f} style={{
                fontFamily: 'var(--mono)', fontSize: 11, fontWeight: 700,
                background: '#FEE2E2', color: '#991B1B',
                padding: '2px 8px', borderRadius: 4,
              }}>
                {f}
              </span>
            ))}
          </div>
          <div style={{ marginTop: 10, fontSize: 11, color: 'var(--text-3)', lineHeight: 1.5 }}>
            Assicurati che ogni esempio nel dataset abbia un campo <code style={{ background: '#FEE2E2', padding: '1px 4px', borderRadius: 3 }}>nuggets</code> (array)
            e che i docs abbiano <code style={{ background: '#FEE2E2', padding: '1px 4px', borderRadius: 3 }}>golden_passage_title</code> o <code style={{ background: '#FEE2E2', padding: '1px 4px', borderRadius: 3 }}>is_gold</code>.
            In alternativa, passa alla modalità <strong>Standard</strong>.
          </div>
        </div>
      </div>
    </div>
  )
}

// ── NuggetMetricsView ─────────────────────────────────────────────────────────

function NuggetMetricsView({ metrics, onSave, onDownload }) {
  const [expanded, setExpanded] = useState({})

  const { nugget_precision, nugget_recall, nugget_coverage,
          n_nuggets, n_covered, n_cited, per_nugget = [] } = metrics

  const pct = v => `${Math.round(v * 100)}%`

  function GaugePill({ value, color }) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <div style={{
          width: 80, height: 6,
          background: 'var(--border)', borderRadius: 3, overflow: 'hidden',
        }}>
          <div style={{
            height: '100%', borderRadius: 3,
            width: `${Math.min(100, value * 100)}%`,
            background: color,
            transition: 'width 0.4s ease',
          }} />
        </div>
        <span style={{ fontFamily: 'var(--mono)', fontSize: 12, fontWeight: 700, color }}>
          {pct(value)}
        </span>
      </div>
    )
  }

  const precColor = metricColor('x', nugget_precision)
  const recColor  = metricColor('x', nugget_recall)
  const covColor  = metricColor('x', nugget_coverage)

  return (
    <div>
      {/* Summary cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, marginBottom: 20 }}>
        {/* Nugget Precision */}
        <div style={{
          padding: '16px 18px',
          background: 'white',
          border: `1px solid ${precColor === 'var(--green)' ? '#A7F3D0' : precColor === 'var(--amber)' ? '#FDE68A' : '#FECACA'}`,
          borderTop: `3px solid ${precColor}`,
          borderRadius: 10,
        }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.6px', marginBottom: 8 }}>
            Nugget Precision
          </div>
          <div style={{ fontSize: 30, fontWeight: 800, color: precColor, lineHeight: 1, marginBottom: 8 }}>
            {pct(nugget_precision)}
          </div>
          <GaugePill value={nugget_precision} color={precColor} />
          <div style={{ marginTop: 8, fontSize: 11, color: 'var(--text-3)', lineHeight: 1.4 }}>
            {n_cited} citati su {n_covered} coperti
          </div>
          <div style={{ marginTop: 6, fontSize: 11, color: 'var(--text-3)', fontStyle: 'italic' }}>
            {METRIC_INFO_NUGGET.nugget_precision.desc}
          </div>
        </div>

        {/* Nugget Recall */}
        <div style={{
          padding: '16px 18px',
          background: 'white',
          border: `1px solid ${recColor === 'var(--green)' ? '#A7F3D0' : recColor === 'var(--amber)' ? '#FDE68A' : '#FECACA'}`,
          borderTop: `3px solid ${recColor}`,
          borderRadius: 10,
        }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.6px', marginBottom: 8 }}>
            Nugget Recall
          </div>
          <div style={{ fontSize: 30, fontWeight: 800, color: recColor, lineHeight: 1, marginBottom: 8 }}>
            {pct(nugget_recall)}
          </div>
          <GaugePill value={nugget_recall} color={recColor} />
          <div style={{ marginTop: 8, fontSize: 11, color: 'var(--text-3)', lineHeight: 1.4 }}>
            {n_cited} citati su {n_nuggets} nuggets totali
          </div>
          <div style={{ marginTop: 6, fontSize: 11, color: 'var(--text-3)', fontStyle: 'italic' }}>
            {METRIC_INFO_NUGGET.nugget_recall.desc}
          </div>
        </div>

        {/* Nugget Coverage */}
        <div style={{
          padding: '16px 18px',
          background: 'white',
          border: `1px solid ${covColor === 'var(--green)' ? '#A7F3D0' : covColor === 'var(--amber)' ? '#FDE68A' : '#FECACA'}`,
          borderTop: `3px solid ${covColor}`,
          borderRadius: 10,
        }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.6px', marginBottom: 8 }}>
            Nugget Coverage
          </div>
          <div style={{ fontSize: 30, fontWeight: 800, color: covColor, lineHeight: 1, marginBottom: 8 }}>
            {pct(nugget_coverage)}
          </div>
          <GaugePill value={nugget_coverage} color={covColor} />
          <div style={{ marginTop: 8, fontSize: 11, color: 'var(--text-3)', lineHeight: 1.4 }}>
            {n_covered} coperti su {n_nuggets} nuggets
          </div>
          <div style={{ marginTop: 6, fontSize: 11, color: 'var(--text-3)', fontStyle: 'italic' }}>
            {METRIC_INFO_NUGGET.nugget_coverage.desc}
          </div>
        </div>
      </div>

      {/* Per-nugget breakdown */}
      {per_nugget.length > 0 && (
        <div>
          <div style={{
            fontSize: 10, fontWeight: 700, color: 'var(--text-3)',
            textTransform: 'uppercase', letterSpacing: '0.7px',
            marginBottom: 10,
          }}>
            Dettaglio per nugget — {per_nugget.length} totali
          </div>
          {per_nugget.map((nug, i) => {
            const statusColor = nug.cited
              ? 'var(--green)'
              : nug.covered
                ? 'var(--amber)'
                : 'var(--red)'
            const statusLabel = nug.cited
              ? 'Citato ✓'
              : nug.covered
                ? 'Coperto, non citato'
                : 'Non coperto'
            const statusBg = nug.cited ? '#DCFCE7' : nug.covered ? '#FEF9C3' : '#FEE2E2'
            const statusFg = nug.cited ? '#166534' : nug.covered ? '#713F12' : '#991B1B'

            return (
              <div key={i} style={{
                marginBottom: 8,
                border: `1px solid ${nug.cited ? '#A7F3D0' : nug.covered ? '#FDE68A' : '#FECACA'}`,
                borderRadius: 8,
                overflow: 'hidden',
              }}>
                {/* Header */}
                <div
                  onClick={() => setExpanded(e => ({ ...e, [i]: !e[i] }))}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 10,
                    padding: '10px 14px',
                    background: nug.cited ? '#F0FDF4' : nug.covered ? '#FFFBEB' : '#FFF1F2',
                    cursor: 'pointer',
                  }}
                >
                  <span style={{
                    fontSize: 10, fontWeight: 700,
                    background: statusBg, color: statusFg,
                    padding: '2px 8px', borderRadius: 10, whiteSpace: 'nowrap', flexShrink: 0,
                  }}>
                    {statusLabel}
                  </span>
                  <span style={{
                    fontFamily: 'var(--mono)', fontSize: 10, fontWeight: 600,
                    color: 'var(--text-3)', flexShrink: 0,
                  }}>
                    {nug.nugget_id}
                  </span>
                  <span style={{ fontSize: 12, color: 'var(--text)', flex: 1, lineHeight: 1.4 }}>
                    {nug.nugget_text}
                  </span>
                  {nug.required && (
                    <span style={{
                      fontSize: 9, fontWeight: 700,
                      background: '#EDE9FE', color: '#5B21B6',
                      padding: '2px 6px', borderRadius: 4, flexShrink: 0,
                    }}>
                      REQUIRED
                    </span>
                  )}
                  <span style={{ color: 'var(--text-3)', fontSize: 12 }}>
                    {expanded[i] ? '▲' : '▼'}
                  </span>
                </div>

                {/* Expanded body */}
                {expanded[i] && (
                  <div style={{ padding: '12px 16px', background: 'white' }}>
                    {nug.keywords?.length > 0 && (
                      <div style={{ marginBottom: 10, display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
                        <span style={{ fontSize: 11, color: 'var(--text-3)', fontWeight: 600 }}>Keywords:</span>
                        {nug.keywords.map((k, ki) => (
                          <span key={ki} style={{
                            fontSize: 11, background: '#EDE9FE', color: '#5B21B6',
                            padding: '1px 7px', borderRadius: 10, fontFamily: 'var(--mono)',
                          }}>{k}</span>
                        ))}
                      </div>
                    )}
                    {nug.golden_evidence && (
                      <div style={{
                        marginBottom: 10, padding: '8px 12px',
                        background: '#F0F9FF', border: '1px solid #BAE6FD',
                        borderRadius: 6, fontSize: 12, color: '#0C4A6E',
                      }}>
                        <strong>Golden evidence:</strong> {nug.golden_evidence}
                        {nug.golden_passage_title && (
                          <span style={{ marginLeft: 6, fontFamily: 'var(--mono)', fontSize: 10, color: '#075985' }}>
                            [{nug.golden_passage_title}]
                          </span>
                        )}
                      </div>
                    )}
                    {nug.best_covering_claim && (
                      <div style={{
                        marginBottom: 10, padding: '8px 12px',
                        background: '#F0FDF4', border: '1px solid #86EFAC',
                        borderRadius: 6, fontSize: 12,
                      }}>
                        <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--text-3)', textTransform: 'uppercase', marginBottom: 4 }}>
                          Claim che copre il nugget
                        </div>
                        <span style={{ color: '#166534' }}>{nug.best_covering_claim}</span>
                      </div>
                    )}
                    {nug.cited && nug.best_evidence_passage_title && (
                      <div style={{
                        padding: '8px 12px',
                        background: '#FAFAF9', border: '1px solid var(--border)',
                        borderRadius: 6,
                      }}>
                        <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--text-3)', textTransform: 'uppercase', marginBottom: 4 }}>
                          Passaggio citato con evidenza
                        </div>
                        <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 4 }}>
                          {nug.best_evidence_passage_title}
                        </div>
                        <div style={{ fontSize: 11, color: 'var(--text-2)', lineHeight: 1.5 }}>
                          {nug.best_evidence_passage_text}
                          {nug.best_evidence_passage_text?.length >= 200 && '…'}
                        </div>
                      </div>
                    )}
                    {!nug.covered && (
                      <div style={{ fontSize: 12, color: 'var(--text-3)', fontStyle: 'italic' }}>
                        Nessun claim generato copre questo nugget (overlap lessicale e keyword insufficienti).
                      </div>
                    )}
                    {nug.covered && !nug.cited && (
                      <div style={{ fontSize: 12, color: '#92400E', fontStyle: 'italic' }}>
                        Il nugget è menzionato in {nug.n_covering_claims} claim ma nessun passaggio citato fornisce evidenza sufficiente.
                      </div>
                    )}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}

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

// NEW: Global Dataset Evaluation Results View
function DatasetEvalResultsView({ results, onSave, onDownload }) {
  const gm = results.global_metrics || {}
  const mode = results.eval_mode || 'standard'
  
  return (
    <div>
      {/* Summary header */}
      <div style={{
        padding: '16px 20px',
        background: 'linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 100%)',
        border: '1px solid #C7D2FE',
        borderRadius: 10,
        marginBottom: 20,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
          <Icon name="barChart2" size={20} color="#4338CA" strokeWidth={2} />
          <span style={{ fontSize: 16, fontWeight: 700, color: '#312E81' }}>
            Valutazione Globale Dataset
          </span>
        </div>
        <div style={{ fontSize: 12, color: '#4338CA', display: 'flex', gap: 20, flexWrap: 'wrap' }}>
          <span>{results.num_examples} esempi</span>
          <span>{results.num_successful} completati con successo</span>
          <span>{results.runtime_seconds}s runtime</span>
          <span>Modalità: {mode === 'nugget' ? 'Nugget' : 'Standard'}</span>
        </div>
      </div>
      
      {/* Global metrics grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 12, marginBottom: 20 }}>
        {Object.entries(gm).map(([key, value]) => {
          if (typeof value !== 'number') return null
          const pct = Math.round(value * 100)
          const color = key.includes('precision') || key.includes('recall') || key.includes('coverage')
            ? (value >= 0.7 ? 'var(--green)' : value >= 0.4 ? 'var(--amber)' : 'var(--red)')
            : 'var(--accent)'
          
          return (
            <div key={key} style={{
              background: 'white',
              border: '1px solid var(--border)',
              borderRadius: 8,
              padding: '14px 16px',
            }}>
              <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 4 }}>
                {key.replace(/_/g, ' ')}
              </div>
              <div style={{ fontSize: 28, fontWeight: 800, color, lineHeight: 1 }}>
                {pct}%
              </div>
              <div style={{
                height: 4, background: 'var(--border-2)', borderRadius: 2,
                marginTop: 8, overflow: 'hidden',
              }}>
                <div style={{
                  height: '100%', borderRadius: 2,
                  width: `${Math.min(100, pct)}%`,
                  background: color,
                  transition: 'width 0.5s ease',
                }} />
              </div>
            </div>
          )
        })}
      </div>
      
      {/* Per-example summary table (collapsed by default) */}
      <details style={{ marginTop: 16 }}>
        <summary style={{
          cursor: 'pointer', fontSize: 13, fontWeight: 600,
          color: 'var(--text-2)', padding: '8px 0',
        }}>
          Dettaglio per esempio ({results.per_example?.length || 0})
        </summary>
        <div style={{ marginTop: 8 }}>
          {(results.per_example || []).map((ex, i) => (
            <div key={i} style={{
              padding: '8px 12px', marginBottom: 4,
              background: ex.error ? '#FEF2F2' : '#FAFAF9',
              border: `1px solid ${ex.error ? '#FECACA' : 'var(--border-2)'}`,
              borderRadius: 6,
              fontSize: 12,
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--text-3)' }}>
                  [{i}]
                </span>
                <span style={{ flex: 1, fontWeight: 500, color: 'var(--text)' }}>
                  {ex.question?.slice(0, 80)}{(ex.question?.length > 80) ? '...' : ''}
                </span>
                {ex.error ? (
                  <span style={{ color: '#DC2626', fontSize: 11 }}>❌ {ex.error}</span>
                ) : (
                  <span style={{ color: 'var(--green)', fontSize: 11 }}>✓ OK</span>
                )}
              </div>
            </div>
          ))}
        </div>
      </details>
      
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

// ── Main Pipeline component ───────────────────────────────────────────────────

export default function Pipeline() {
  const { addPipelineResult } = useAppData()

  // Settings
  const [model, setModel] = useState('claude-haiku-4-5-20251001')
  const [retrieveMethod, setRetrieveMethod] = useState('nli')
  const [threshold, setThreshold] = useState(0.5)
  const [topK, setTopK] = useState(3)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [noiseEnabled, setNoiseEnabled] = useState(false)

  // Dataset
  const [dataset, setDataset] = useState(null)
  const [datasetName, setDatasetName] = useState('')
  const [exampleIdx, setExampleIdx] = useState(0)
  const fileRef = useRef()

  // Evaluation mode
  const [evalMode, setEvalMode] = useState('standard') // 'standard' | 'nugget'

  // Nugget field validation error (shown inline in step 6)
  const [nuggetFieldError, setNuggetFieldError] = useState(null) // string[] | null

  // Pipeline state
  const [response, setResponse] = useState(null)
  const [claims, setClaims] = useState(null)
  const [matched, setMatched] = useState(null)
  const [cited, setCited] = useState(null)
  const [references, setReferences] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [nuggetMetrics, setNuggetMetrics] = useState(null)

  const [running, setRunning] = useState(null)
  const [error, setError] = useState(null)

  // Stato per la valutazione globale del dataset
const [datasetEvalRunning, setDatasetEvalRunning] = useState(false)
const [datasetEvalProgress, setDatasetEvalProgress] = useState({ current: 0, total: 0 })
const [datasetEvalResults, setDatasetEvalResults] = useState(null)
const [datasetEvalError, setDatasetEvalError] = useState(null)

  // Step status
  const steps = {
    query:     'active',
    generate:  response ? 'done' : dataset ? 'active' : 'locked',
    decompose: claims   ? 'done' : response ? 'active' : 'locked',
    retrieve:  matched  ? 'done' : claims   ? 'active' : 'locked',
    cite:      cited    ? 'done' : matched  ? 'active' : 'locked',
    evaluate:  (metrics || nuggetMetrics) ? 'done' : cited ? 'active' : 'locked',
  }
  if (running) steps[running] = 'running'

  const currentExample  = dataset ? dataset[exampleIdx] : null
  const currentQuery    = currentExample?.question || ''
  const rawPassages     = currentExample?.docs || []

  const currentPassages = (() => {
    if (!noiseEnabled || !dataset || !currentExample) return rawPassages
    const noisePool = buildNoisePool(dataset, exampleIdx)
    if (!noisePool.length) return rawPassages
    return injectNoise(rawPassages, noisePool, 42 + exampleIdx)
  })()

  const noiseCount = currentPassages.filter(d => d.is_noise).length
  const origCount  = currentPassages.filter(d => !d.is_noise).length

  // Nuggets come directly from the loaded dataset example
  const currentNuggets = currentExample?.nuggets || null
  const hasNuggets = !!currentNuggets

  // When evalMode changes to nugget but no nugget data → effective mode is standard
  const effectiveMode = evalMode === 'nugget' && !hasNuggets ? 'standard' : evalMode

  function resetAfter(step) {
    const order = ['generate', 'decompose', 'retrieve', 'cite', 'evaluate']
    const idx = order.indexOf(step)
    if (idx <= 0) setResponse(null)
    if (idx <= 1) setClaims(null)
    if (idx <= 2) setMatched(null)
    if (idx <= 3) { setCited(null); setReferences(null) }
    if (idx <= 4) { setMetrics(null); setNuggetMetrics(null); setNuggetFieldError(null) }
  }

  function onFileUpload(e) {
    const file = e.target.files[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = evt => {
      try {
        const parsed = JSON.parse(evt.target.result)
        const normalized = normalizeDataset(parsed)
        if (normalized.length === 0) throw new Error('Il file non contiene esempi validi.')
        setDataset(normalized)
        setDatasetName(file.name)
        setExampleIdx(0)
        resetAfter('generate')
        setError(null)
        // Auto-enable nugget mode if any example has nuggets inline
        if (normalized.some(ex => ex.nuggets)) {
          setEvalMode('nugget')
        }
      } catch (err) {
        setError(`Errore lettura file: ${err.message}`)
      }
    }
    reader.readAsText(file)
  }

  // ── Validate nugget fields before evaluate ──────────────────────────────────

  function validateNuggetFields() {
    const missing = []

    if (!currentNuggets || currentNuggets.length === 0) {
      missing.push('nuggets (array vuoto o assente)')
    }

    // Check that at least some docs have golden_passage_title or is_gold
    const docsHaveGolden = currentPassages.some(
      d => d.golden_passage_title || d.is_gold === true
    )
    if (!docsHaveGolden) {
      missing.push('docs[].golden_passage_title / docs[].is_gold')
    }

    return missing
  }

  // ── Pipeline steps ──

  async function runGenerate() {
    setError(null); setRunning('generate'); resetAfter('generate')
    try {
      const res = await api.pipeline.generate({ query: currentQuery, passages: currentPassages, model })
      setResponse(res.response)
    } catch (e) { setError(`Generate: ${e.message}`) }
    setRunning(null)
  }

  async function runDecompose() {
    setError(null); setRunning('decompose'); resetAfter('decompose')
    try {
      const res = await api.pipeline.decompose({ text: response, model })
      setClaims(res.claims)
    } catch (e) { setError(`Decompose: ${e.message}`) }
    setRunning(null)
  }

  async function runRetrieve() {
    setError(null); setRunning('retrieve'); resetAfter('retrieve')
    try {
      const res = await api.pipeline.retrieve({
        claims, passages: currentPassages,
        method: retrieveMethod, threshold, top_k: topK,
      })
      setMatched(res.matched)
    } catch (e) { setError(`Retrieve: ${e.message}`) }
    setRunning(null)
  }

  async function runCite() {
    setError(null); setRunning('cite'); resetAfter('cite')
    try {
      const res = await api.pipeline.cite({ response, matched })
      setCited(res.cited_response)
      setReferences(res.references)
    } catch (e) { setError(`Cite: ${e.message}`) }
    setRunning(null)
  }

  async function runEvaluate() {
    setError(null)
    setNuggetFieldError(null)
    setRunning('evaluate')
    setMetrics(null); setNuggetMetrics(null)

    try {
      if (effectiveMode === 'nugget') {
        // Validate fields before calling API
        const missing = validateNuggetFields()
        if (missing.length > 0) {
          setNuggetFieldError(missing)
          setRunning(null)
          return
        }

        const res = await api.pipeline.evaluateNuggets({
          matched_claims: matched,
          nuggets: currentNuggets,
          docs: currentPassages, // pass full docs with golden_passage_title / is_gold
        })
        setNuggetMetrics(res)
      } else {
        const res = await api.pipeline.evaluate({ matched })
        setMetrics(res)
      }
    } catch (e) {
      setError(`Evaluate: ${e.message}`)
    }
    setRunning(null)
  }

  // NEW: Run evaluation on the entire dataset
async function runDatasetEvaluation() {
  if (!dataset || dataset.length === 0) return
  
  setDatasetEvalRunning(true)
  setDatasetEvalError(null)
  setDatasetEvalResults(null)
  setDatasetEvalProgress({ current: 0, total: dataset.length })
  
  try {
    const res = await api.pipeline.evaluateDataset({
      dataset: dataset.map(ex => ({
        question: ex.question,
        docs: ex.docs || [],
        nuggets: ex.nuggets || null,
        // Passiamo già i docs originali, non quelli con noise
      })),
      model,
      retrieve_method: retrieveMethod,
      threshold,
      top_k: topK,
      eval_mode: effectiveMode,
      noise_enabled: noiseEnabled,
      noise_seed: 42,
    })
    setDatasetEvalResults(res)
    setDatasetEvalProgress({ current: dataset.length, total: dataset.length })
  } catch (e) {
    setDatasetEvalError(`Errore valutazione dataset: ${e.message}`)
  }
  setDatasetEvalRunning(false)
}

  function saveToExplore() {
    addPipelineResult({
      question: currentQuery, raw_response: response, claims,
      matched_claims: matched, cited_response: cited, references,
      metrics, nugget_metrics: nuggetMetrics,
    })
    alert('Risultato salvato! Visibile nella pagina Esplora.')
  }

  function downloadPipelineData() {
    const payload = {
      question: currentQuery,
      raw_response: response,
      claims,
      matched_claims: matched,
      cited_response: cited,
      references,
      metrics,
      nugget_metrics: nuggetMetrics,
      eval_mode: effectiveMode,
      model,
      retrieve_method: retrieveMethod,
      threshold,
      top_k: topK,
      exported_at: new Date().toISOString(),
    }
    downloadJSON(payload, timestampedFilename('pipeline_result'))
  }

  // ── Render ──

  return (
    <div>
      <div className="page-header">
        <div className="page-header-title">Pipeline interattivo</div>
        <div className="page-header-sub">Esegui ogni step separatamente e ispeziona i risultati intermedi.</div>
      </div>

      {error && (
        <div className="info-box info-box-red" style={{ marginBottom: 16 }}>
          <Icon name="xCircle" size={15} strokeWidth={1.75} style={{ flexShrink: 0, marginTop: 1 }} />
          <span><strong>Errore:</strong> {error}</span>
        </div>
      )}

      {/* Settings */}
      <div className="card" style={{ marginBottom: 16 }}>
        <div
          style={{ display: 'flex', alignItems: 'center', padding: '12px 20px', cursor: 'pointer', gap: 8 }}
          onClick={() => setSettingsOpen(o => !o)}
        >
          <Icon name="settings" size={14} strokeWidth={1.75} color="var(--text-2)" />
          <span style={{ fontSize: 13, fontWeight: 600, flex: 1 }}>Impostazioni modello &amp; retrieval</span>
          <Icon name={settingsOpen ? 'chevronUp' : 'chevronDown'} size={13} strokeWidth={2} color="var(--text-3)" />
        </div>
        {settingsOpen && (
          <div style={{ padding: '0 20px 20px', borderTop: '1px solid var(--border-2)' }}>
            <div style={{ paddingTop: 16, display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: 16 }}>
              <div className="form-group" style={{ marginBottom: 0 }}>
                <label className="form-label">Modello LLM</label>
                <select className="input" value={model} onChange={e => setModel(e.target.value)}>
                  <option>claude-haiku-4-5-20251001</option>
                  <option>claude-sonnet-4-20250514</option>
                </select>
              </div>
              <div className="form-group" style={{ marginBottom: 0 }}>
                <label className="form-label">Metodo retrieval</label>
                <select className="input" value={retrieveMethod} onChange={e => setRetrieveMethod(e.target.value)}>
                  <option value="nli">NLI</option>
                  <option value="similarity">Similarity</option>
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
            </div>
          </div>
        )}
      </div>

      {/* Step 1 — Query */}
      <StepCard num={1} title="Query" status={steps.query}>
        {!dataset ? (
          <div>
            <div className="form-group">
              <label className="form-label">Carica dataset (ALCE / ELI5 / QAMPARI)</label>
              <input ref={fileRef} type="file" accept=".json,.jsonl" onChange={onFileUpload} style={{ display: 'none' }} />
              <button className="btn btn-primary" onClick={() => fileRef.current.click()}>
                <Icon name="upload" size={14} strokeWidth={1.75} color="white" />
                Seleziona file JSON
              </button>
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-3)' }}>
              Supporta i formati ALCE (question + docs), ELI5 (question + claims + answer), QAMPARI (question + context.documents).
              Se il dataset contiene un campo <code style={{ background: 'var(--bg)', padding: '1px 4px', borderRadius: 3 }}>nuggets</code> la valutazione Nugget sarà disponibile allo Step 6.
            </div>
          </div>
        ) : (
          <div>
            <div className="form-group">
              <label className="form-label">Dataset — {datasetName}</label>
              <select className="input" value={exampleIdx}
                onChange={e => { setExampleIdx(+e.target.value); resetAfter('generate') }}>
                {dataset.map((ex, i) => <option key={i} value={i}>[{i}] {ex.question}</option>)}
              </select>
            </div>
            <div className="response-box" style={{ marginBottom: 12 }}>
              <strong>Q:</strong> {currentQuery}
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-3)', display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 6 }}>
              {noiseEnabled
                ? <>{origCount} passages + {noiseCount} noise = {currentPassages.length} totali</>
                : <>{currentPassages.length} passages disponibili</>}
              {hasNuggets && (
                <span style={{
                  fontSize: 10, fontWeight: 700,
                  background: '#EDE9FE', color: '#5B21B6',
                  padding: '2px 7px', borderRadius: 10,
                }}>
                  {currentNuggets.length} nuggets
                </span>
              )}
              &nbsp;·&nbsp; Modello: <span style={{ fontFamily: 'var(--mono)' }}>{model}</span>
              &nbsp;·&nbsp;
              <button
                className="btn"
                onClick={() => { setNoiseEnabled(n => !n); resetAfter('generate') }}
                style={{
                  padding: '3px 10px', fontSize: 11,
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
              <button className="btn btn-secondary"
                style={{ padding: '2px 8px', fontSize: 11 }}
                onClick={() => { setDataset(null); setDatasetName(''); resetAfter('generate') }}>
                Cambia dataset
              </button>
            </div>
          </div>
        )}
      </StepCard>

      {/* Step 2 */}
      <StepCard num={2} title="Genera risposta" status={steps.generate}
        onRun={runGenerate} running={running === 'generate'} runLabel="Genera risposta">
        {response && (
          <>
            <div className="response-box" style={{ marginBottom: 12 }}>{response}</div>
            <div style={{ fontSize: 12, color: 'var(--text-3)' }}>
              {response.split(/\s+/).length} parole · Modello: <span style={{ fontFamily: 'var(--mono)' }}>{model}</span>
            </div>
          </>
        )}
      </StepCard>

      {/* Step 3 */}
      <StepCard num={3} title="Decompose — Atomic Claims" status={steps.decompose}
        onRun={runDecompose} running={running === 'decompose'} runLabel="Decomponi in claims">
        {claims && (
          <>
            <div style={{ fontSize: 12, color: 'var(--text-2)', marginBottom: 12, fontWeight: 600 }}>
              {claims.length} claims estratti
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {claims.map((c, i) => (
                <div key={i} style={{
                  display: 'flex', alignItems: 'flex-start', gap: 10,
                  padding: '10px 14px', background: '#F5F3FF',
                  border: '1px solid #DDD6FE', borderRadius: 8,
                }}>
                  <span style={{
                    fontFamily: 'var(--mono)', fontSize: 11, fontWeight: 600,
                    color: 'var(--accent)', background: 'white',
                    border: '1px solid #DDD6FE', borderRadius: 4,
                    padding: '1px 6px', flexShrink: 0, marginTop: 1,
                  }}>{String(i + 1).padStart(2, '0')}</span>
                  <span style={{ fontSize: 13, color: '#2E1065', lineHeight: 1.5 }}>{c}</span>
                </div>
              ))}
            </div>
          </>
        )}
      </StepCard>

      {/* Step 4 */}
      <StepCard num={4} title="Retrieve — Matching claims → passaggi" status={steps.retrieve}
        onRun={runRetrieve} running={running === 'retrieve'} runLabel="Retrieval">
        {matched && <MatchedView matched={matched} passages={currentPassages} retrieveMethod={retrieveMethod} />}
      </StepCard>

      {/* Step 5 */}
      <StepCard num={5} title="Cite — Risposta con citazioni" status={steps.cite}
        onRun={runCite} running={running === 'cite'} runLabel="Inserisci citazioni">
        {cited && (
          <CitedView
            citedResponse={cited}
            references={references || []}
            matched={matched}
          />
        )}
      </StepCard>

            {/* Step 6 — Evaluate (with mode toggle) */}
      <StepCard
        num={6}
        title={
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
            <span>Evaluate — Metriche di qualità</span>
            {steps.evaluate !== 'locked' && (
              <EvalModeToggle
                mode={evalMode}
                onChange={mode => { setEvalMode(mode); setMetrics(null); setNuggetMetrics(null); setNuggetFieldError(null); setDatasetEvalResults(null) }}
                hasNuggets={hasNuggets}
              />
            )}
          </div>
        }
        status={steps.evaluate}
        onRun={runEvaluate}
        running={running === 'evaluate'}
        runLabel={`Valuta (${effectiveMode === 'nugget' ? 'Nugget' : 'Standard'})`}
      >
        {/* Inline field-missing error for nugget mode */}
        {nuggetFieldError && (
          <NuggetMissingFieldsError missingFields={nuggetFieldError} />
        )}

        {/* ============================================================ */}
        {/* NEW: Dataset-wide evaluation button & results                */}
        {/* ============================================================ */}
        {dataset && dataset.length > 1 && (
          <div style={{
            marginBottom: 20,
            padding: '16px 20px',
            background: '#F5F3FF',
            border: '2px dashed #C7D2FE',
            borderRadius: 10,
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 14, fontWeight: 700, color: '#4338CA', marginBottom: 4 }}>
                  Valuta tutto il dataset
                </div>
                <div style={{ fontSize: 12, color: '#6366F1' }}>
                  Esegui l&apos;intera pipeline su tutti i {dataset.length} esempi e ottieni metriche globali
                  di precision e recall aggregate.
                </div>
              </div>
              <button
                className="btn"
                onClick={runDatasetEvaluation}
                disabled={datasetEvalRunning}
                style={{
                  background: '#6366F1',
                  color: 'white',
                  border: 'none',
                  padding: '10px 20px',
                  fontWeight: 700,
                  fontSize: 13,
                  borderRadius: 8,
                  cursor: datasetEvalRunning ? 'not-allowed' : 'pointer',
                  opacity: datasetEvalRunning ? 0.7 : 1,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                }}
              >
                {datasetEvalRunning ? (
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
            
            {/* Progress bar */}
            {datasetEvalRunning && datasetEvalProgress.total > 0 && (
              <div style={{ marginTop: 12 }}>
                <div style={{
                  display: 'flex', justifyContent: 'space-between',
                  fontSize: 11, color: '#6366F1', marginBottom: 4,
                }}>
                  <span>Progresso</span>
                  <span>{datasetEvalProgress.current}/{datasetEvalProgress.total}</span>
                </div>
                <div style={{
                  height: 6, background: '#E0E7FF',
                  borderRadius: 3, overflow: 'hidden',
                }}>
                  <div style={{
                    height: '100%',
                    width: `${(datasetEvalProgress.current / datasetEvalProgress.total) * 100}%`,
                    background: '#6366F1',
                    borderRadius: 3,
                    transition: 'width 0.3s ease',
                  }} />
                </div>
              </div>
            )}
          </div>
        )}
        
        {/* Dataset evaluation error */}
        {datasetEvalError && (
          <div className="info-box info-box-red" style={{ marginBottom: 16 }}>
            <Icon name="xCircle" size={15} strokeWidth={1.75} style={{ flexShrink: 0, marginTop: 1 }} />
            <span>{datasetEvalError}</span>
          </div>
        )}
        
        {/* Dataset evaluation results */}
        {datasetEvalResults && (
          <DatasetEvalResultsView
            results={datasetEvalResults}
            onSave={() => {
              addPipelineResult({
                question: `[Dataset] ${datasetName}`,
                dataset_eval_results: datasetEvalResults,
              })
              alert('Risultati globali salvati! Visibile nella pagina Esplora.')
            }}
            onDownload={() => {
              downloadJSON(datasetEvalResults, timestampedFilename('dataset_eval'))
            }}
          />
        )}

        {/* Divider if we also have single-example results */}
        {(metrics || nuggetMetrics) && datasetEvalResults && (
          <div style={{
            margin: '20px 0',
            borderTop: '1px solid var(--border)',
            paddingTop: 16,
          }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.6px', marginBottom: 12 }}>
              Risultati esempio corrente
            </div>
          </div>
        )}

        {/* Standard metrics */}
        {metrics && effectiveMode === 'standard' && (
          <>
            <div className="grid-3" style={{ gap: 12 }}>
              {Object.entries(METRIC_INFO_STANDARD).map(([key, { label, desc }]) => (
                <MetricCard key={key} label={label} value={metrics[key]}
                  color={metricColor(key, metrics[key])} desc={desc}
                  isUnsupported={key === 'unsupported_ratio'} />
              ))}
            </div>
            <div style={{ marginTop: 20, display: 'flex', gap: 12 }}>
              <button className="btn btn-primary" onClick={saveToExplore}>
                <Icon name="download" size={13} color="white" strokeWidth={2} />
                Salva in Esplora
              </button>
              <button className="btn btn-secondary" onClick={downloadPipelineData}>
                <Icon name="download" size={13} strokeWidth={1.75} />
                Scarica dati
              </button>
            </div>
          </>
        )}

        {/* Nugget metrics */}
        {nuggetMetrics && effectiveMode === 'nugget' && (
          <NuggetMetricsView
            metrics={nuggetMetrics}
            onSave={saveToExplore}
            onDownload={downloadPipelineData}
          />
        )}
      </StepCard>
    </div>
  )
}

// ── MatchedView ───────────────────────────────────────────────────────────────

function MatchedView({ matched, passages, retrieveMethod }) {
  const [open, setOpen] = useState({})
  const [debug, setDebug] = useState({})
  const [debugging, setDebugging] = useState(null)

  const supported = matched.filter(m => (m.supporting_passages || []).length > 0).length

  async function runDebug(claimText, claimIdx) {
    setDebugging(claimIdx)
    try {
      const result = await api.pipeline.retrieveDebug({
        claim: claimText, passages,
        method: retrieveMethod, top_k: 4,
      })
      setDebug(d => ({ ...d, [claimIdx]: result }))
    } catch (e) { alert(`Errore debug: ${e.message}`) }
    setDebugging(null)
  }

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
        <div className="metric-card" style={{ padding: '12px 20px', display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ fontSize: 28, fontWeight: 800, color: 'var(--green)' }}>
            {supported}/{matched.length}
          </span>
          <span style={{ fontSize: 12, color: 'var(--text-2)' }}>claims<br />supportati</span>
        </div>
        <div style={{ flex: 1, height: 6, background: 'var(--border)', borderRadius: 3, overflow: 'hidden' }}>
          <div style={{
            height: '100%',
            width: matched.length ? `${supported / matched.length * 100}%` : 0,
            background: 'var(--green)', borderRadius: 3,
          }} />
        </div>
      </div>

      {matched.map((m, i) => {
        const passages_m = m.supporting_passages || []
        const has = passages_m.length > 0
        const debugData = debug[i]
        return (
          <div key={i} className="expander" style={{ borderColor: has ? '#A7F3D0' : '#FECACA' }}>
            <div className="expander-header" onClick={() => setOpen(o => ({ ...o, [i]: !o[i] }))}>
              <span className={`badge ${has ? 'badge-green' : 'badge-red'}`}>
                {has
                  ? <Icon name="check" size={10} strokeWidth={2.5} />
                  : <Icon name="x" size={10} strokeWidth={2.5} />}
              </span>
              <span className="expander-header-title" style={{ color: 'var(--text)' }}>{m.claim}</span>
              {has && (
                <span style={{ fontSize: 11, color: 'var(--text-3)', fontFamily: 'var(--mono)' }}>
                  {passages_m.length} fonte{passages_m.length > 1 ? 'i' : ''}
                </span>
              )}
              <span className={`expander-chevron${open[i] ? ' open' : ''}`}>▼</span>
            </div>
            {open[i] && (
              <div className="expander-body">
                <div style={{ marginBottom: 12, display: 'flex', gap: 8, alignItems: 'center' }}>
                  <button
                    className="btn btn-secondary"
                    style={{ fontSize: 11, padding: '4px 10px' }}
                    onClick={(e) => { e.stopPropagation(); runDebug(m.claim, i) }}
                    disabled={debugging === i}
                  >
                    {debugging === i
                      ? <><span className="spinner" style={{ width: 11, height: 11 }} /> Calcolo...</>
                      : <><Icon name="search" size={11} strokeWidth={1.75} />
                          {debugData ? 'Aggiorna debug' : 'Debug frasi (top-4)'}</>}
                  </button>
                  {debugData && (
                    <span style={{ fontSize: 11, color: 'var(--text-3)' }}>
                      Score {debugData.method.toUpperCase()} su ogni frase del passaggio
                    </span>
                  )}
                </div>
                {debugData && <DebugView data={debugData} />}
                {has ? passages_m.map((p, j) => (
                  <div key={j} className="passage-card" style={{ marginBottom: 8 }}>
                    <div className="passage-header">
                      <span className="passage-title">{p.title || '—'}</span>
                      {p.entailment_score != null && <ScorePill score={p.entailment_score} />}
                    </div>
                    <div className="passage-body">{p.text || ''}</div>
                    {p.best_sentence && (
                      <div style={{
                        margin: '0 14px 10px', padding: '6px 10px',
                        background: '#ECFDF5', borderRadius: 6,
                        fontSize: 12, color: '#166534',
                        borderLeft: '3px solid #86EFAC',
                      }}>
                        <strong>Evidenza:</strong> {p.best_sentence}
                      </div>
                    )}
                  </div>
                )) : (
                  !debugData && <span style={{ color: 'var(--text-3)', fontSize: 13 }}>
                    Nessun passaggio di supporto trovato.
                  </span>
                )}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

function DebugView({ data }) {
  return (
    <div style={{
      marginBottom: 16, padding: 14,
      background: '#F8FAFC', border: '1px dashed var(--border)',
      borderRadius: 8,
    }}>
      <div style={{
        fontSize: 10, fontWeight: 700, color: 'var(--text-3)',
        textTransform: 'uppercase', letterSpacing: '0.7px', marginBottom: 12,
      }}>
        Debug — top-4 frasi per passaggio · metodo: {data.method}
      </div>
      {data.passages.map((p, pi) => (
        <div key={pi} style={{ marginBottom: 14 }}>
          <div style={{
            display: 'flex', alignItems: 'center', gap: 6,
            fontSize: 13, fontWeight: 600, color: 'var(--text)', marginBottom: 6,
          }}>
            <Icon name="fileText" size={12} strokeWidth={1.75} color="var(--text-2)" />
            {p.title || 'Passage'}
          </div>
          {p.sentences.length === 0 ? (
            <div style={{ fontSize: 12, color: 'var(--text-3)', paddingLeft: 18 }}>Nessuna frase.</div>
          ) : (
            p.sentences.map((s, si) => {
              const isBest = s.is_best
              const score = s.score
              const color = score >= 0.8 ? 'var(--green)' : score >= 0.5 ? 'var(--amber)' : 'var(--text-3)'
              const pct = Math.max(0, Math.min(1, score)) * 100
              return (
                <div key={si} style={{
                  display: 'flex', alignItems: 'flex-start', gap: 10,
                  padding: '6px 10px', marginBottom: 4,
                  background: isBest ? '#F0FDF4' : 'white',
                  border: `1px solid ${isBest ? '#86EFAC' : 'var(--border-2)'}`,
                  borderRadius: 6,
                }}>
                  <span style={{
                    fontFamily: 'var(--mono)', fontSize: 10, fontWeight: 700,
                    color, minWidth: 52, whiteSpace: 'nowrap',
                  }}>
                    [{score.toFixed(4)}]
                  </span>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{
                      height: 3, background: 'var(--border-2)',
                      borderRadius: 2, overflow: 'hidden', marginBottom: 4, width: 90,
                    }}>
                      <div style={{ height: '100%', width: `${pct}%`, background: color }} />
                    </div>
                    <span style={{
                      fontSize: 12, color: isBest ? '#166534' : 'var(--text-2)',
                      fontWeight: isBest ? 500 : 400, lineHeight: 1.5,
                    }}>
                      "{s.text}"
                    </span>
                  </div>
                  {isBest && (
                    <span style={{
                      fontSize: 10, fontWeight: 700, background: '#DCFCE7', color: '#166534',
                      padding: '2px 6px', borderRadius: 10,
                      display: 'flex', alignItems: 'center', gap: 3,
                      whiteSpace: 'nowrap', alignSelf: 'flex-start',
                    }}>
                      ★ BEST
                    </span>
                  )}
                </div>
              )
            })
          )}
        </div>
      ))}
    </div>
  )
}

// ── CitedView ─────────────────────────────────────────────────────────────────

const STOPWORDS_CITED = new Set([
  'the','a','an','is','are','was','were','in','on','at','to','for','of','and',
  'or','but','with','as','his','her','their','its','has','have','had','by','it',
  'this','that','from','not','be','been',
])

function tokenizeCited(text) {
  return text.toLowerCase().replace(/[^\w\s]/g, '').split(/\s+/).filter(Boolean)
}

function lexicalOverlap(claimText, sentText) {
  const claimWords = new Set(tokenizeCited(claimText).filter(w => !STOPWORDS_CITED.has(w)))
  const sentWords  = new Set(tokenizeCited(sentText).filter(w => !STOPWORDS_CITED.has(w)))
  if (claimWords.size === 0) return 0
  let hits = 0
  for (const w of claimWords) if (sentWords.has(w)) hits++
  return hits / claimWords.size
}

function splitIntoSentences(citedResponse) {
  const regex = /([^.!?]+[.!?]+)((?:\s*\[\d+\])*)/g
  const sentences = []
  let m
  while ((m = regex.exec(citedResponse)) !== null) {
    const raw = m[1].trim()
    const markers = m[2] || ''
    const citations = (markers.match(/\d+/g) || []).map(Number)
    if (raw) sentences.push({ text: raw, citations })
  }
  return sentences
}

function findAssociatedClaims(sentenceText, matchedClaims, threshold = 0.5) {
  const scored = []
  for (const mc of matchedClaims) {
    const overlap = lexicalOverlap(mc.claim, sentenceText)
    if (overlap >= threshold) scored.push({ matchedClaim: mc, overlap })
  }
  scored.sort((a, b) => b.overlap - a.overlap)
  return scored
}

function highlightEvidence(passageText, extraction, start, end) {
  if (!passageText) return null
  if (!extraction) return <span>{passageText}</span>
  if (
    typeof start === 'number' && start >= 0 &&
    typeof end === 'number' && end > start &&
    end <= passageText.length &&
    passageText.slice(start, end).toLowerCase() === extraction.toLowerCase()
  ) {
    return (
      <>
        <span>{passageText.slice(0, start)}</span>
        <mark style={{ background: '#FEF08A', padding: '1px 2px', borderRadius: 3, fontWeight: 600, color: '#713F12' }}>
          {passageText.slice(start, end)}
        </mark>
        <span>{passageText.slice(end)}</span>
      </>
    )
  }
  const idx = passageText.toLowerCase().indexOf(extraction.toLowerCase())
  if (idx >= 0) {
    return (
      <>
        <span>{passageText.slice(0, idx)}</span>
        <mark style={{ background: '#FEF08A', padding: '1px 2px', borderRadius: 3, fontWeight: 600, color: '#713F12' }}>
          {passageText.slice(idx, idx + extraction.length)}
        </mark>
        <span>{passageText.slice(idx + extraction.length)}</span>
      </>
    )
  }
  return <span>{passageText}</span>
}

function CitedView({ citedResponse, references, matched }) {
  const [activeSent, setActiveSent] = useState(null)
  const [activeClaim, setActiveClaim] = useState(null)
  const sentences = splitIntoSentences(citedResponse)

  function onSentenceClick(i) {
    if (activeSent === i) { setActiveSent(null); setActiveClaim(null) }
    else { setActiveSent(i); setActiveClaim(null) }
  }

  const associatedClaims = activeSent != null
    ? findAssociatedClaims(sentences[activeSent].text, matched || [])
    : []

  return (
    <div>
      <div style={{
        background: 'white', border: '1px solid var(--border)',
        borderLeft: '3px solid var(--accent)', borderRadius: 8,
        padding: '16px 20px', fontSize: 14, lineHeight: 2.0, color: 'var(--text)',
      }}>
        {sentences.map((sent, i) => {
          const hasCitations = sent.citations.length > 0
          const isActive = activeSent === i
          return (
            <span key={i}>
              <span
                onClick={() => hasCitations && onSentenceClick(i)}
                style={{
                  background: isActive ? '#BBF7D0' : hasCitations ? '#F0FDF4' : 'transparent',
                  padding: hasCitations ? '2px 4px' : '0',
                  borderRadius: 4,
                  cursor: hasCitations ? 'pointer' : 'default',
                  borderBottom: hasCitations ? '2px solid #86EFAC' : 'none',
                  transition: 'background 0.15s',
                }}
                title={hasCitations ? 'Clicca per vedere i claims associati' : ''}
              >
                {sent.text}
              </span>
              {sent.citations.map(n => (
                <sup key={n} style={{
                  color: '#059669', fontWeight: 700, fontSize: 10,
                  fontFamily: 'var(--mono)', marginLeft: 2,
                }}>[{n}]</sup>
              ))}
              {' '}
            </span>
          )
        })}
      </div>

      {activeSent != null && (
        <div style={{
          marginTop: 14, background: 'var(--green-lt)',
          border: '1px solid #A7F3D0', borderRadius: 10,
          padding: '16px 18px', animation: 'fadeSlide 0.18s ease',
        }}>
          <div style={{
            fontSize: 10, fontWeight: 700, color: 'var(--text-3)',
            textTransform: 'uppercase', letterSpacing: '0.7px', marginBottom: 12,
          }}>
            Claims associati — clicca per vedere le fonti
          </div>
          {activeClaim == null ? (
            associatedClaims.length === 0 ? (
              <div style={{ fontSize: 13, color: 'var(--text-3)' }}>
                Nessun claim associato (overlap lessicale &lt; 0.5).
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {associatedClaims.map(({ matchedClaim, overlap }, idx) => {
                  const numPassages = (matchedClaim.supporting_passages || []).length
                  return (
                    <div key={idx} onClick={() => setActiveClaim(idx)}
                      style={{
                        display: 'flex', alignItems: 'center', gap: 10,
                        padding: '10px 14px', background: 'white',
                        border: '1px solid #A7F3D0', borderRadius: 8, cursor: 'pointer',
                        transition: 'all 0.12s',
                      }}
                      onMouseEnter={e => e.currentTarget.style.background = '#F0FDF4'}
                      onMouseLeave={e => e.currentTarget.style.background = 'white'}
                    >
                      <Icon name="search" size={13} color="var(--green)" strokeWidth={2} />
                      <span style={{ flex: 1, fontSize: 13, color: 'var(--text)' }}>{matchedClaim.claim}</span>
                      <span style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--text-3)', whiteSpace: 'nowrap' }}>
                        overlap {overlap.toFixed(2)}
                      </span>
                      <span style={{
                        fontSize: 11, fontWeight: 600, color: 'var(--green)',
                        background: '#DCFCE7', padding: '2px 8px', borderRadius: 12,
                      }}>
                        {numPassages} fonte{numPassages !== 1 ? 'i' : ''}
                      </span>
                    </div>
                  )
                })}
              </div>
            )
          ) : (
            <div>
              <div onClick={() => setActiveClaim(null)}
                style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--text-2)', cursor: 'pointer', marginBottom: 12 }}
                onMouseEnter={e => e.currentTarget.style.color = 'var(--text)'}
                onMouseLeave={e => e.currentTarget.style.color = 'var(--text-2)'}
              >
                ← Torna ai claims
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 14, fontWeight: 600, color: 'var(--text)', marginBottom: 14 }}>
                <Icon name="search" size={14} color="var(--green)" strokeWidth={2} />
                {associatedClaims[activeClaim].matchedClaim.claim}
              </div>
              {(associatedClaims[activeClaim].matchedClaim.supporting_passages || []).map((p, j) => {
                const refNum = references.find(r => r.title === p.title || r.text === p.text)?.citation_number
                return (
                  <div key={j} className="passage-card" style={{ marginBottom: 10 }}>
                    <div className="passage-header">
                      <span className="passage-title">{p.title || '—'}</span>
                      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                        {p.entailment_score != null && <ScorePill score={p.entailment_score} />}
                        {refNum && (
                          <span style={{
                            background: '#0F172A', color: 'white',
                            fontFamily: 'var(--mono)', fontSize: 11, fontWeight: 700,
                            padding: '2px 8px', borderRadius: 4,
                          }}>[{refNum}]</span>
                        )}
                      </div>
                    </div>
                    <div className="passage-body">
                      {highlightEvidence(p.text || '', p.extraction || p.best_sentence || '', p.extraction_start, p.extraction_end)}
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      )}

      {references.length > 0 && (
        <div style={{ marginTop: 16, paddingTop: 12, borderTop: '1px solid var(--border)' }}>
          <strong style={{ color: 'var(--text)', fontSize: 13 }}>Riferimenti</strong>
          {references.map(r => (
            <div key={r.citation_number} style={{
              marginTop: 8, padding: '8px 12px', background: 'var(--bg)',
              borderRadius: 6, border: '1px solid var(--border)',
            }}>
              <span style={{ fontFamily: 'var(--mono)', fontWeight: 700, color: 'var(--green)', marginRight: 6 }}>
                [{r.citation_number}]
              </span>
              <strong style={{ fontSize: 13 }}>{r.title || '—'}</strong>
              <div style={{ color: 'var(--text-3)', marginTop: 4, fontSize: 11 }}>
                {(r.text || '').slice(0, 200)}{(r.text || '').length > 200 ? '…' : ''}
              </div>
            </div>
          ))}
        </div>
      )}

      <style>{`
        @keyframes fadeSlide {
          from { opacity: 0; transform: translateY(-6px); }
          to   { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  )
}