/**
 * Explore.jsx — Naviga i risultati pipeline salvati.
 * Supporta upload JSON (reset) e download del dataset corrente.
 */

import { useState, useEffect, useRef } from 'react'
import { useAppData } from '../context/AppData'
import { downloadJSON, timestampedFilename } from '../utils/download'
import EmptyState from '../components/EmptyState'
import ScorePill from '../components/ScorePill'
import Icon from '../components/Icon'

const TABS = ['Risposta grezza', 'Claims', 'Matched', 'Citata']

export default function Explore() {
  const { pipelineResults, setPipelineResults, clearAll } = useAppData()
  const [idx, setIdx] = useState(0)
  const [tab, setTab] = useState('Risposta grezza')
  const [uploadError, setUploadError] = useState(null)
  const fileRef = useRef()

  useEffect(() => { setTab('Risposta grezza') }, [idx])

  useEffect(() => {
    if (idx >= pipelineResults.length && pipelineResults.length > 0) setIdx(0)
  }, [pipelineResults.length, idx])

  function onFileUpload(e) {
    const file = e.target.files[0]
    if (!file) return
    setUploadError(null)

    const reader = new FileReader()
    reader.onload = evt => {
      try {
        let parsed = JSON.parse(evt.target.result)
        // Se è un singolo oggetto, lo convertiamo in array
        if (!Array.isArray(parsed)) {
          if (parsed && typeof parsed === 'object') {
            parsed = [parsed]
          } else {
            throw new Error('Il file deve contenere una lista di risultati o un singolo oggetto valido.')
          }
        }
        if (parsed.length === 0) {
          throw new Error('Il file è vuoto.')
        }
        // Validation minima: ogni elemento deve avere almeno 'question' o 'raw_response'
        const valid = parsed.every(r =>
          r && typeof r === 'object' && (r.question != null || r.raw_response != null)
        )
        if (!valid) {
          throw new Error('Formato non valido: ogni elemento deve avere almeno un campo "question" o "raw_response".')
        }

        setPipelineResults(parsed)
        setIdx(0)
      } catch (err) {
        setUploadError(err.message)
      }
    }
    reader.readAsText(file)
    // Reset dell'input per permettere di ricaricare lo stesso file
    e.target.value = ''
  }

  function onDownload() {
    if (pipelineResults.length === 0) return
    downloadJSON(pipelineResults, timestampedFilename('pipeline_results'))
  }

  // ── Header sempre visibile ────────────────────────────────────────────
  const headerActions = (
    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
      <input
        ref={fileRef} type="file" accept=".json"
        onChange={onFileUpload} style={{ display: 'none' }}
      />
      <button className="btn btn-secondary" onClick={() => fileRef.current.click()}>
        <Icon name="upload" size={13} strokeWidth={1.75} /> Carica JSON
      </button>
      {pipelineResults.length > 0 && (
        <>
          <button className="btn btn-secondary" onClick={onDownload}>
            <Icon name="download" size={13} strokeWidth={1.75} /> Scarica JSON
          </button>
          <button
            className="btn btn-secondary"
            onClick={() => { if (confirm('Cancellare tutti i risultati?')) clearAll() }}
          >
            Svuota
          </button>
        </>
      )}
    </div>
  )

  if (pipelineResults.length === 0) {
    return (
      <div>
        <div className="page-header">
          <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 16 }}>
            <div>
              <div className="page-header-title">Esplora risultati</div>
              <div className="page-header-sub">
                Naviga i risultati prodotti dal pipeline step-by-step.
              </div>
            </div>
            {headerActions}
          </div>
        </div>

        {uploadError && (
          <div className="info-box info-box-red" style={{ marginBottom: 16 }}>
            <Icon name="xCircle" size={15} strokeWidth={1.75} />
            <span><strong>Errore upload:</strong> {uploadError}</span>
          </div>
        )}

        <EmptyState
          title="Nessun risultato disponibile"
          hint="Esegui il pipeline e clicca 'Salva in Esplora', oppure carica un file JSON."
        />
      </div>
    )
  }

  const ex = pipelineResults[Math.min(idx, pipelineResults.length - 1)]

  return (
    <div>
      <div className="page-header">
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 16 }}>
          <div>
            <div className="page-header-title">Esplora risultati</div>
            <div className="page-header-sub">
              Naviga i risultati prodotti dal pipeline step-by-step · {pipelineResults.length} esempi
            </div>
          </div>
          {headerActions}
        </div>
      </div>

      {uploadError && (
        <div className="info-box info-box-red" style={{ marginBottom: 16 }}>
          <Icon name="xCircle" size={15} strokeWidth={1.75} />
          <span><strong>Errore upload:</strong> {uploadError}</span>
        </div>
      )}

      {/* Selector esempio */}
      <div className="card" style={{ marginBottom: 16 }}>
        <div className="card-body">
          <div style={{
            fontSize: 11, fontWeight: 600, color: 'var(--text-3)',
            textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 6,
          }}>
            Esempio
          </div>
          <select className="input" value={idx} onChange={e => setIdx(+e.target.value)}>
            {pipelineResults.map((r, i) => (
              <option key={i} value={i}>
                [{i}] {(r.question || '').slice(0, 80)}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="response-box" style={{ marginBottom: 16 }}>
        <span style={{ fontWeight: 600 }}>Q:</span> {ex.question}
      </div>

      {/* Tab */}
      <div className="card">
        <div style={{ padding: '0 20px', borderBottom: '1px solid var(--border-2)' }}>
          <div className="tabs" style={{ marginBottom: 0 }}>
            {TABS.map(t => (
              <div key={t} className={`tab${tab === t ? ' active' : ''}`} onClick={() => setTab(t)}>
                {t}
              </div>
            ))}
          </div>
        </div>
        <div className="card-body">
          {tab === 'Risposta grezza' && (
            <div className="response-box">{ex.raw_response || '—'}</div>
          )}
          {tab === 'Claims' && <ClaimsTab claims={ex.claims || []} />}
          {tab === 'Matched' && <MatchedTab matched={ex.matched_claims || []} />}
          {tab === 'Citata' && (
            <CitedTab
              citedResponse={ex.cited_response || ''}
              references={ex.references || []}
              metrics={ex.metrics}
            />
          )}
        </div>
      </div>
    </div>
  )
}

// ── Sub-components (invariati) ─────────────────────────────────────────

function ClaimsTab({ claims }) {
  if (claims.length === 0) return <div style={{ color: 'var(--text-3)' }}>Nessun claim.</div>
  return (
    <div>
      <div style={{ fontSize: 12, color: 'var(--text-2)', marginBottom: 12, fontWeight: 600 }}>
        {claims.length} claims estratti
      </div>
      {claims.map((c, i) => (
        <div key={i} style={{
          display: 'flex', alignItems: 'flex-start', gap: 10,
          padding: '10px 14px',
          background: '#F5F3FF', border: '1px solid #DDD6FE',
          borderRadius: 8, marginBottom: 6,
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
  )
}

function MatchedTab({ matched }) {
  const [open, setOpen] = useState({})
  if (matched.length === 0) return <div style={{ color: 'var(--text-3)' }}>Nessun matching.</div>
  return (
    <div>
      {matched.map((m, i) => {
        const passages = m.supporting_passages || []
        const has = passages.length > 0
        return (
          <div key={i} className="expander" style={{ borderColor: has ? '#A7F3D0' : '#FECACA' }}>
            <div className="expander-header" onClick={() => setOpen(o => ({ ...o, [i]: !o[i] }))}>
              <span className={`badge ${has ? 'badge-green' : 'badge-red'}`}>
                {has
                  ? <Icon name="check" size={10} strokeWidth={2.5} />
                  : <Icon name="x" size={10} strokeWidth={2.5} />}
              </span>
              <span className="expander-header-title" style={{ color: 'var(--text)' }}>{m.claim}</span>
              {has && passages[0].entailment_score != null && (
                <span style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--text-3)' }}>
                  {passages[0].entailment_score.toFixed(3)}
                </span>
              )}
              <span className={`expander-chevron${open[i] ? ' open' : ''}`}>▼</span>
            </div>
            {open[i] && (
              <div className="expander-body">
                {has ? passages.map((p, j) => (
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
                        fontSize: 11, color: '#166534',
                        borderLeft: '3px solid #86EFAC',
                      }}>
                        <strong>Evidenza:</strong> {p.best_sentence}
                      </div>
                    )}
                  </div>
                )) : (
                  <span style={{ color: 'var(--text-3)', fontSize: 13 }}>
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

function CitedTab({ citedResponse, references, metrics }) {
  const METRIC_LABELS = {
    citation_precision: 'Citation Precision',
    citation_recall: 'Citation Recall',
    factual_precision: 'Factual Precision',
    factual_precision_nli: 'Factual Precision NLI',
    unsupported_ratio: 'Unsupported Ratio',
    avg_entailment_score: 'Avg Entailment',
  }

  function metricColor(key, v) {
    if (key === 'unsupported_ratio') return v <= 0.2 ? 'var(--green)' : v <= 0.5 ? 'var(--amber)' : 'var(--red)'
    return v >= 0.7 ? 'var(--green)' : v >= 0.4 ? 'var(--amber)' : 'var(--red)'
  }

  return (
    <div>
      <div className="response-box">{citedResponse || '—'}</div>

      {references.length > 0 && (
        <div style={{ marginTop: 16, paddingTop: 12, borderTop: '1px solid var(--border)' }}>
          <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>References</div>
          {references.map(r => (
            <div key={r.citation_number} style={{
              padding: '10px 14px', background: 'var(--bg)',
              border: '1px solid var(--border)', borderRadius: 8, marginBottom: 8,
            }}>
              <span style={{ fontFamily: 'var(--mono)', fontWeight: 700, color: 'var(--green)', marginRight: 6 }}>
                [{r.citation_number}]
              </span>
              <strong style={{ fontSize: 13 }}>{r.title || '—'}</strong>
              <div style={{ fontSize: 12, color: 'var(--text-2)', marginTop: 4 }}>{r.text || ''}</div>
            </div>
          ))}
        </div>
      )}

      {metrics && Object.keys(metrics).length > 0 && (
        <div style={{ marginTop: 16, paddingTop: 12, borderTop: '1px solid var(--border)' }}>
          <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 10 }}>Metriche</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 8 }}>
            {Object.entries(METRIC_LABELS).map(([k, l]) => {
              const v = metrics[k]
              if (v == null) return null
              const c = metricColor(k, v)
              return (
                <div key={k} style={{
                  padding: '10px 14px', background: 'var(--bg)',
                  borderRadius: 8, border: '1px solid var(--border)',
                }}>
                  <div style={{ fontSize: 18, fontWeight: 700, color: c, fontFamily: 'var(--mono)' }}>
                    {v.toFixed(3)}
                  </div>
                  <div style={{ fontSize: 11, color: 'var(--text-2)', marginTop: 2 }}>{l}</div>
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}