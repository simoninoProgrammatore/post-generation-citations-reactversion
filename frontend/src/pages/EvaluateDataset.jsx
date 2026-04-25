/**
 * EvaluateDataset.jsx — Pagina dedicata alla valutazione globale del dataset.
 * Stessa grafica e stesse metriche già presenti nello Step 6 di Pipeline.jsx,
 * ma esposta come pagina autonoma sotto "Metriche".
 */

import { useState, useRef } from 'react'
import api from '../api'
import { useAppData } from '../context/AppData'

import Icon from '../components/Icon'
import { downloadJSON, timestampedFilename } from '../utils/download'


// ── Helpers ───────────────────────────────────────────────────────────────────

function metricColor(key, v) {
  if (key === 'unsupported_ratio') {
    return v <= 0.2 ? 'var(--green)' : v <= 0.5 ? 'var(--amber)' : 'var(--red)'
  }
  return v >= 0.7 ? 'var(--green)' : v >= 0.4 ? 'var(--amber)' : 'var(--red)'
}

const GLOBAL_METRIC_INFO = {
  macro_nugget_precision: {
    label: 'Macro Nugget Precision',
    desc: 'Precisione calcolata su tutti i nugget del dataset (cited/covered).',
  },
  macro_nugget_recall: {
    label: 'Macro Nugget Recall',
    desc: 'Recall calcolata su tutti i nugget del dataset (cited/total).',
  },
  macro_nugget_coverage: {
    label: 'Macro Nugget Coverage',
    desc: 'Copertura su tutti i nugget del dataset (covered/total).',
  },
  avg_nugget_precision: {
    label: 'Avg Nugget Precision',
    desc: 'Media delle precisioni per esempio.',
  },
  avg_nugget_recall: {
    label: 'Avg Nugget Recall',
    desc: 'Media delle recall per esempio.',
  },
}

const STANDARD_METRIC_LABELS = {
  citation_precision:    'Citation Precision',
  citation_recall:       'Citation Recall',
  factual_precision:     'Factual Precision',
  factual_precision_nli: 'Factual Precision (NLI)',
  unsupported_ratio:     'Unsupported Ratio',
  avg_entailment_score:  'Avg Entailment Score',
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

// ── EvalModeToggle ────────────────────────────────────────────────────────────

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

// ── NuggetAssociationTable ────────────────────────────────────────────────────

function NuggetAssociationTable({ perExample }) {
  const [filter, setFilter] = useState('all') // all | required | optional | covered | uncovered | cited | uncited
  const [expandedRow, setExpandedRow] = useState(null)

  // Flatten all per_nugget entries with their parent question
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

  // Apply filter
  const filtered = allRows.filter(r => {
    if (filter === 'required') return r.required
    if (filter === 'optional') return !r.required
    if (filter === 'covered') return r.covered
    if (filter === 'uncovered') return !r.covered
    if (filter === 'cited') return r.cited
    if (filter === 'uncited') return !r.cited
    return true
  })

  if (allRows.length === 0) return null

  return (
    <details style={{ marginTop: 20 }}>
      <summary style={{
        cursor: 'pointer', fontSize: 13, fontWeight: 600,
        color: 'var(--text-2)', padding: '8px 0',
      }}>
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
            {label} {val !== 'all' ? `(${allRows.filter(r => {
              if (val === 'required') return r.required
              if (val === 'optional') return !r.required
              if (val === 'covered') return r.covered
              if (val === 'uncovered') return !r.covered
              if (val === 'cited') return r.cited
              if (val === 'uncited') return !r.cited
              return true
            }).length})` : ''}
          </button>
        ))}
      </div>

      {/* Table */}
      <div style={{ overflowX: 'auto', marginTop: 8 }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ background: '#F9FAFB', borderBottom: '2px solid var(--border)' }}>
              <th style={thStyle}>#</th>
              <th style={thStyle}>Tipo</th>
              <th style={thStyle}>Domanda</th>
              <th style={thStyle}>Nugget</th>
              <th style={thStyle}>Claim Matchato</th>
              <th style={thStyle}>Frase Evidenza (Passaggio)</th>
              <th style={thStyle}>Stato</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((row, idx) => {
              const isExpanded = expandedRow === idx
              const statusColor = row.cited ? 'var(--green)' : row.covered ? 'var(--amber)' : 'var(--red)'
              const statusLabel = row.cited ? 'Citato' : row.covered ? 'Coperto' : 'Mancante'

              return (
                <tr key={idx}
                  onClick={() => setExpandedRow(isExpanded ? null : idx)}
                  style={{
                    cursor: 'pointer',
                    borderBottom: '1px solid var(--border-2)',
                    background: isExpanded ? '#FAFAF9' : 'white',
                    transition: 'background 0.15s',
                  }}
                  onMouseEnter={e => { if (!isExpanded) e.currentTarget.style.background = '#FAFAF9' }}
                  onMouseLeave={e => { if (!isExpanded) e.currentTarget.style.background = 'white' }}
                >
                  <td style={tdStyle}>
                    <span style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--text-3)' }}>
                      {row.exIdx}.{row.nugget_id}
                    </span>
                  </td>
                  <td style={tdStyle}>
                    <span style={{
                      display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                      width: 18, height: 18, borderRadius: '50%', fontSize: 10, fontWeight: 800,
                      background: row.required
                        ? 'linear-gradient(135deg, #F59E0B, #D97706)'
                        : 'linear-gradient(135deg, #D1D5DB, #9CA3AF)',
                      color: row.required ? '#FFFBEB' : '#374151',
                    }}>
                      {row.required ? '★' : '☆'}
                    </span>
                  </td>
                  <td style={{ ...tdStyle, maxWidth: 180 }}>
                    <div style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: isExpanded ? 'normal' : 'nowrap' }}>
                      {row.question}
                    </div>
                  </td>
                  <td style={{ ...tdStyle, maxWidth: 220 }}>
                    <div style={{
                      overflow: 'hidden',
                      textOverflow: isExpanded ? 'unset' : 'ellipsis',
                      whiteSpace: isExpanded ? 'normal' : 'nowrap',
                      fontWeight: 500,
                    }}>
                      {row.nugget_text}
                    </div>
                    {isExpanded && row.keywords?.length > 0 && (
                      <div style={{ marginTop: 4 }}>
                        {row.keywords.map((kw, ki) => (
                          <span key={ki} style={{
                            display: 'inline-block', padding: '1px 6px', margin: '2px 2px',
                            background: row.best_covering_claim?.toLowerCase().includes(kw.toLowerCase())
                              ? '#FDE68A' : '#F3F4F6',
                            borderRadius: 4, fontFamily: 'var(--mono)', fontSize: 10,
                            fontWeight: row.best_covering_claim?.toLowerCase().includes(kw.toLowerCase()) ? 700 : 400,
                          }}>
                            {kw}
                          </span>
                        ))}
                      </div>
                    )}
                  </td>
                  <td style={{ ...tdStyle, maxWidth: 250 }}>
                    {row.best_covering_claim ? (
                      <div style={{
                        overflow: 'hidden',
                        textOverflow: isExpanded ? 'unset' : 'ellipsis',
                        whiteSpace: isExpanded ? 'normal' : 'nowrap',
                        color: 'var(--text)',
                      }}>
                        {row.best_covering_claim}
                      </div>
                    ) : (
                      <span style={{ color: 'var(--text-3)', fontStyle: 'italic' }}>—</span>
                    )}
                  </td>
                  <td style={{ ...tdStyle, maxWidth: 250 }}>
                    {(row.best_evidence_sentence || row.best_evidence_passage_text) ? (
                      <div>
                        {row.best_evidence_passage_title && (
                          <div style={{ fontSize: 10, fontWeight: 600, color: 'var(--accent)', marginBottom: 2 }}>
                            {row.best_evidence_passage_title}
                          </div>
                        )}
                        <div style={{
                          overflow: 'hidden',
                          textOverflow: isExpanded ? 'unset' : 'ellipsis',
                          whiteSpace: isExpanded ? 'normal' : 'nowrap',
                          color: 'var(--text)',
                          fontWeight: 500,
                        }}>
                          {row.best_evidence_sentence || '—'}
                        </div>
                        {isExpanded && row.best_evidence_passage_text && (
                          <div style={{
                            marginTop: 6, padding: '6px 8px', background: '#F9FAFB',
                            borderRadius: 4, fontSize: 11, color: 'var(--text-3)',
                            fontStyle: 'italic', lineHeight: 1.5,
                            borderLeft: '2px solid var(--border)',
                          }}>
                            {row.best_evidence_passage_text}
                          </div>
                        )}
                      </div>
                    ) : (
                      <span style={{ color: 'var(--text-3)', fontStyle: 'italic' }}>—</span>
                    )}
                    {isExpanded && (
                      <div style={{ marginTop: 4, fontSize: 10, color: 'var(--text-3)' }}>
                        Cite score: {row.cite_score?.toFixed(3)} · Similarity: {row.semantic_similarity?.toFixed(3)} · Covering claims: {row.n_covering_claims}
                      </div>
                    )}
                  </td>
                  <td style={tdStyle}>
                    <span style={{
                      display: 'inline-block', padding: '2px 8px', borderRadius: 10,
                      fontSize: 10, fontWeight: 700,
                      background: row.cited ? '#D1FAE5' : row.covered ? '#FEF3C7' : '#FEE2E2',
                      color: row.cited ? '#065F46' : row.covered ? '#92400E' : '#991B1B',
                    }}>
                      {statusLabel}
                    </span>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </details>
  )
}

const thStyle = { textAlign: 'left', padding: '8px 10px', fontSize: 10, fontWeight: 700, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.5px', whiteSpace: 'nowrap' }
const tdStyle = { padding: '8px 10px', verticalAlign: 'top' }


// ── DatasetEvalResultsView ────────────────────────────────────────────────────

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
      {mode === 'nugget' ? (
        <div>
          {/* Required nuggets */}
          <div style={{ fontSize: 11, fontWeight: 700, color: '#92400E', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: 'linear-gradient(135deg, #F59E0B, #D97706)' }} />
            Required Nuggets
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 12, marginBottom: 20 }}>
            {[
              ['avg_required_precision', 'Avg Precision (Req)', 'Media precisioni per esempio, solo nugget required.'],
              ['avg_required_recall', 'Avg Recall (Req)', 'Media recall per esempio, solo nugget required.'],
              ['macro_required_precision', 'Macro Precision (Req)', 'cited_req / covered_req su tutto il dataset.'],
              ['macro_required_recall', 'Macro Recall (Req)', 'cited_req / total_req su tutto il dataset.'],
            ].map(([key, label, desc]) => {
              const v = gm[key]
              if (typeof v !== 'number') return null
              const pct = Math.round(v * 100)
              const color = v >= 0.7 ? 'var(--green)' : v >= 0.4 ? 'var(--amber)' : 'var(--red)'
              return (
                <div key={key} style={{ background: 'white', border: '1px solid var(--border)', borderRadius: 8, padding: '14px 16px' }}>
                  <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 4 }}>{label}</div>
                  <div style={{ fontSize: 28, fontWeight: 800, color, lineHeight: 1 }}>{pct}%</div>
                  <div style={{ fontSize: 10, color: 'var(--text-3)', marginTop: 4, lineHeight: 1.4 }}>{desc}</div>
                  <div style={{ height: 4, background: 'var(--border-2)', borderRadius: 2, marginTop: 8, overflow: 'hidden' }}>
                    <div style={{ height: '100%', borderRadius: 2, width: `${Math.min(100, pct)}%`, background: color, transition: 'width 0.5s ease' }} />
                  </div>
                </div>
              )
            })}
          </div>

          {/* All nuggets */}
          <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-2)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 8 }}>
            All Nuggets (Total)
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 12, marginBottom: 20 }}>
            {[
              ['avg_nugget_precision', 'Avg Precision', 'Media precisioni per esempio.'],
              ['avg_nugget_recall', 'Avg Recall', 'Media recall per esempio.'],
              ['avg_nugget_coverage', 'Avg Coverage', 'Media copertura per esempio.'],
              ['macro_nugget_precision', 'Macro Precision', 'cited / covered su tutto il dataset.'],
              ['macro_nugget_recall', 'Macro Recall', 'cited / total su tutto il dataset.'],
              ['macro_nugget_coverage', 'Macro Coverage', 'covered / total su tutto il dataset.'],
            ].map(([key, label, desc]) => {
              const v = gm[key]
              if (typeof v !== 'number') return null
              const pct = Math.round(v * 100)
              const color = v >= 0.7 ? 'var(--green)' : v >= 0.4 ? 'var(--amber)' : 'var(--red)'
              return (
                <div key={key} style={{ background: 'white', border: '1px solid var(--border)', borderRadius: 8, padding: '14px 16px' }}>
                  <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 4 }}>{label}</div>
                  <div style={{ fontSize: 28, fontWeight: 800, color, lineHeight: 1 }}>{pct}%</div>
                  <div style={{ fontSize: 10, color: 'var(--text-3)', marginTop: 4, lineHeight: 1.4 }}>{desc}</div>
                  <div style={{ height: 4, background: 'var(--border-2)', borderRadius: 2, marginTop: 8, overflow: 'hidden' }}>
                    <div style={{ height: '100%', borderRadius: 2, width: `${Math.min(100, pct)}%`, background: color, transition: 'width 0.5s ease' }} />
                  </div>
                </div>
              )
            })}
          </div>

          {/* Counts — NO percentage, just integer */}
          <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-2)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 8 }}>
            Conteggi
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', gap: 12, marginBottom: 20 }}>
            {[
              ['total_nuggets', 'Nuggets Totali'],
              ['total_covered', 'Coperti'],
              ['total_cited', 'Citati'],
              ['total_required', 'Required'],
              ['total_required_covered', 'Req. Coperti'],
              ['total_required_cited', 'Req. Citati'],
              ['total_optional', 'Optional'],
              ['total_optional_covered', 'Opt. Coperti'],
              ['total_optional_cited', 'Opt. Citati'],
            ].map(([key, label]) => {
              const v = gm[key]
              if (typeof v !== 'number') return null
              return (
                <div key={key} style={{ background: 'white', border: '1px solid var(--border)', borderRadius: 8, padding: '14px 16px' }}>
                  <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 4 }}>{label}</div>
                  <div style={{ fontSize: 28, fontWeight: 800, color: 'var(--accent)', lineHeight: 1 }}>{v}</div>
                </div>
              )
            })}
          </div>
        </div>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 12, marginBottom: 20 }}>
          {Object.entries(gm).map(([key, value]) => {
            if (typeof value !== 'number') return null
            const pct = Math.round(value * 100)
            const color = key.includes('precision') || key.includes('recall') || key.includes('coverage')
              ? (value >= 0.7 ? 'var(--green)' : value >= 0.4 ? 'var(--amber)' : 'var(--red)')
              : 'var(--accent)'
            const info = GLOBAL_METRIC_INFO[key] || STANDARD_METRIC_LABELS[key]
            const label = info?.label || info || key.replace(/_/g, ' ')
            const desc = info?.desc
            return (
              <div key={key} style={{ background: 'white', border: '1px solid var(--border)', borderRadius: 8, padding: '14px 16px' }}>
                <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 4 }}>{label}</div>
                <div style={{ fontSize: 28, fontWeight: 800, color, lineHeight: 1 }}>{pct}%</div>
                {desc && <div style={{ fontSize: 10, color: 'var(--text-3)', marginTop: 4, lineHeight: 1.4 }}>{desc}</div>}
                <div style={{ height: 4, background: 'var(--border-2)', borderRadius: 2, marginTop: 8, overflow: 'hidden' }}>
                  <div style={{ height: '100%', borderRadius: 2, width: `${Math.min(100, pct)}%`, background: color, transition: 'width 0.5s ease' }} />
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* Per-example summary table */}
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

      {/* Nugget Association Review Table */}
      {mode === 'nugget' && (
        <NuggetAssociationTable perExample={results.per_example || []} />
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

  // Eval mode
  const [evalMode, setEvalMode] = useState('standard')

  // Run state
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState({ current: 0, total: 0 })
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)

  const hasNuggets = dataset?.some(ex => ex.nuggets && ex.nuggets.length > 0) ?? false
  const effectiveMode = evalMode === 'nugget' && !hasNuggets ? 'standard' : evalMode

  function onResultsUpload(e) {
    const file = e.target.files[0]
    if (!file) return
    e.target.value = ''
    const reader = new FileReader()
    reader.onload = evt => {
      try {
        const parsed = JSON.parse(evt.target.result)
        // Validate: must have global_metrics and per_example
        if (!parsed.global_metrics || !parsed.per_example) {
          throw new Error('Il file non contiene risultati validi (mancano global_metrics o per_example).')
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
    // Reset so the same file can be re-selected after "Cambia dataset"
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
        if (normalized.some(ex => ex.nuggets)) setEvalMode('nugget')
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

    // Costruisci il noise pool una volta sola (docs di tutti gli altri esempi)
    const noisePool = noiseEnabled
      ? dataset.flatMap((ex, i) =>
          (ex.docs || []).map(doc => ({ ...doc, _source_idx: i }))
        )
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
          eval_mode: effectiveMode,
          noise_enabled: noiseEnabled,
          noise_pool: noisePool.filter(d => d._source_idx !== idx),
          noise_seed: 42,
          example_idx: idx,
          pre_filter_k: preFilterK,
        })
        perExample.push(res)
      } catch (e) {
        perExample.push({
          question: ex.question,
          error: e.message,
        })
      }
      setProgress({ current: idx + 1, total: dataset.length })
    }

    // Aggregazione metriche globali (speculare al backend)
    const globalMetrics = {}

    if (effectiveMode === 'standard') {
      const keys = [
        'citation_precision', 'citation_recall',
        'factual_precision', 'factual_precision_nli',
        'unsupported_ratio', 'avg_entailment_score',
      ]
      for (const k of keys) {
        const vals = perExample
          .filter(ex => ex.metrics?.[k] != null)
          .map(ex => ex.metrics[k])
        if (vals.length) globalMetrics[k] = vals.reduce((a, b) => a + b, 0) / vals.length
      }
    } else {
      let totalNuggets = 0, totalCovered = 0, totalCited = 0
      let totalReq = 0, totalReqCovered = 0, totalReqCited = 0
      let totalOpt = 0, totalOptCovered = 0, totalOptCited = 0
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
        totalNuggets += nm.n_nuggets ?? 0
        totalCovered += nm.n_covered ?? 0
        totalCited   += nm.n_cited   ?? 0
        totalReq += nm.n_required ?? 0
        totalReqCovered += nm.n_required_covered ?? 0
        totalReqCited   += nm.n_required_cited   ?? 0
        totalOpt += nm.n_optional ?? 0
        totalOptCovered += nm.n_optional_covered ?? 0
        totalOptCited   += nm.n_optional_cited   ?? 0
      }
      if (precs.length) {
        globalMetrics.avg_nugget_precision = precs.reduce((a, b) => a + b, 0) / precs.length
        globalMetrics.avg_nugget_recall    = recalls.reduce((a, b) => a + b, 0) / recalls.length
        globalMetrics.avg_nugget_coverage  = covs.reduce((a, b) => a + b, 0) / covs.length
        globalMetrics.avg_required_precision = reqPrecs.reduce((a, b) => a + b, 0) / reqPrecs.length
        globalMetrics.avg_required_recall    = reqRecalls.reduce((a, b) => a + b, 0) / reqRecalls.length
      }
      if (totalNuggets > 0) {
        globalMetrics.macro_nugget_precision = totalCovered > 0 ? totalCited / totalCovered : 0
        globalMetrics.macro_nugget_recall    = totalCited / totalNuggets
        globalMetrics.macro_nugget_coverage  = totalCovered / totalNuggets
      }
      if (totalReq > 0) {
        globalMetrics.macro_required_precision = totalReqCovered > 0 ? totalReqCited / totalReqCovered : 0
        globalMetrics.macro_required_recall    = totalReqCited / totalReq
      }
      if (totalOpt > 0) {
        globalMetrics.macro_optional_precision = totalOptCovered > 0 ? totalOptCited / totalOptCovered : 0
        globalMetrics.macro_optional_recall    = totalOptCited / totalOpt
      }
      globalMetrics.total_nuggets = totalNuggets
      globalMetrics.total_cited   = totalCited
      globalMetrics.total_covered = totalCovered
      globalMetrics.total_required = totalReq
      globalMetrics.total_required_cited = totalReqCited
      globalMetrics.total_required_covered = totalReqCovered
      globalMetrics.total_optional = totalOpt
      globalMetrics.total_optional_cited = totalOptCited
      globalMetrics.total_optional_covered = totalOptCovered
    }

    setResults({
      global_metrics: globalMetrics,
      per_example: perExample,
      num_examples: dataset.length,
      num_successful: perExample.filter(ex => !ex.error).length,
      runtime_seconds: null,   // non calcolato lato frontend
      eval_mode: effectiveMode,
    })
    setRunning(false)
  }

  return (
    <div>

      {/* ── Hidden file input — unico, sempre montato ── */}
      <input
        ref={fileRef}
        type="file"
        accept=".json,.jsonl"
        onChange={onFileUpload}
        style={{ display: 'none' }}
      />

      {/* ── Page header ── */}
      <div className="page-header">
        <div className="page-header-title" style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            width: 32, height: 32,
            background: 'linear-gradient(135deg, #6366F1 0%, #7C3AED 100%)',
            borderRadius: 8,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            flexShrink: 0,
            boxShadow: '0 2px 8px rgba(99,102,241,0.3)',
          }}>
            <Icon name="database" size={16} color="white" strokeWidth={1.75} />
          </div>
          Valutazione Dataset
        </div>
        <div className="page-header-sub">
          Esegui l'intera pipeline su tutti gli esempi del dataset e ottieni metriche globali aggregate.
        </div>
      </div>

      {/* ── Error banner ── */}
      {error && (
        <div className="info-box info-box-red" style={{ marginBottom: 16 }}>
          <Icon name="xCircle" size={15} strokeWidth={1.75} style={{ flexShrink: 0, marginTop: 1 }} />
          <span><strong>Errore:</strong> {error}</span>
        </div>
      )}

      {/* ── Settings ── */}
      <SettingsPanel
        model={model} setModel={setModel}
        retrieveMethod={retrieveMethod} setRetrieveMethod={setRetrieveMethod}
        threshold={threshold} setThreshold={setThreshold}
        topK={topK} setTopK={setTopK}
        preFilterK={preFilterK} setPreFilterK={setPreFilterK}
        noiseEnabled={noiseEnabled} setNoiseEnabled={setNoiseEnabled}
      />

      {/* ── Dataset upload & run card ── */}
      <div className="card" style={{ marginBottom: 20 }}>
        <div style={{ padding: '16px 20px' }}>

          {/* Upload row */}
          <div style={{
            display: 'flex', alignItems: 'center', gap: 12,
            flexWrap: 'wrap', marginBottom: dataset ? 16 : 0,
          }}>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text)', marginBottom: 4 }}>
                {dataset ? (
                  <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <Icon name="fileText" size={14} strokeWidth={1.75} color="var(--accent)" />
                    {datasetName}
                    <span style={{
                      fontSize: 11, fontWeight: 700,
                      background: 'var(--bg)', border: '1px solid var(--border)',
                      padding: '1px 8px', borderRadius: 10, color: 'var(--text-2)',
                    }}>
                      {dataset.length} esempi
                    </span>
                    {hasNuggets && (
                      <span style={{
                        fontSize: 10, fontWeight: 700,
                        background: '#EDE9FE', color: '#5B21B6',
                        padding: '2px 7px', borderRadius: 10,
                      }}>
                        nuggets presenti
                      </span>
                    )}
                  </span>
                ) : (
                  'Carica un dataset per iniziare'
                )}
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
              {dataset && (
                <EvalModeToggle
                  mode={evalMode}
                  onChange={m => { setEvalMode(m); setResults(null) }}
                  hasNuggets={hasNuggets}
                />
              )}
            </div>
          </div>

          {/* Run button */}
          {dataset && (
            <div style={{
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
                    di precision e recall aggregate in modalità{' '}
                    <strong>{effectiveMode === 'nugget' ? 'Nugget' : 'Standard'}</strong>.
                  </div>
                </div>
                <button
                  className="btn"
                  onClick={runEvaluation}
                  disabled={running}
                  style={{
                    background: '#6366F1',
                    color: 'white',
                    border: 'none',
                    padding: '10px 20px',
                    fontWeight: 700,
                    fontSize: 13,
                    borderRadius: 8,
                    cursor: running ? 'not-allowed' : 'pointer',
                    opacity: running ? 0.7 : 1,
                    display: 'flex',
                    alignItems: 'center',
                    gap: 8,
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

              {/* Progress bar */}
              {running && progress.total > 0 && (
                <div style={{ marginTop: 12 }}>
                  <div style={{
                    display: 'flex', justifyContent: 'space-between',
                    fontSize: 11, color: '#6366F1', marginBottom: 4,
                  }}>
                    <span>Progresso</span>
                    <span>{progress.current}/{progress.total}</span>
                  </div>
                  <div style={{
                    height: 6, background: '#E0E7FF',
                    borderRadius: 3, overflow: 'hidden',
                  }}>
                    <div style={{
                      height: '100%',
                      width: `${(progress.current / progress.total) * 100}%`,
                      background: '#6366F1',
                      borderRadius: 3,
                      transition: 'width 0.3s ease',
                    }} />
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* ── Results ── */}
      {results && (
        <div className="card">
          <div style={{ padding: '16px 20px' }}>
            <DatasetEvalResultsView
              results={results}
              onSave={() => {
                addPipelineResult({
                  question: `[Dataset] ${datasetName}`,
                  dataset_eval_results: results,
                })
                alert('Risultati globali salvati! Visibile nella pagina Esplora.')
              }}
              onDownload={() => {
                downloadJSON(results, timestampedFilename('dataset_eval'))
              }}
            />
          </div>
        </div>
      )}

      {/* ── Empty state ── */}
      {!dataset && !results && (
        <div style={{
          marginTop: 32,
          padding: '48px 32px',
          textAlign: 'center',
          border: '2px dashed var(--border)',
          borderRadius: 12,
          color: 'var(--text-3)',
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
            <button
              className="btn btn-primary"
              onClick={() => fileRef.current.click()}
            >
              <Icon name="upload" size={14} strokeWidth={1.75} color="white" />
              Carica dataset
            </button>
            <button
              className="btn btn-secondary"
              onClick={() => resultsFileRef.current.click()}
            >
              <Icon name="upload" size={14} strokeWidth={1.75} />
              Carica risultati
            </button>
          </div>
        </div>
      )}

    </div>
  )
}