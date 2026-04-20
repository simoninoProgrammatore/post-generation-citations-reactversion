/**
 * Metrics.jsx — Metriche aggregate su tutti i pipelineResults salvati.
 *
 * Calcola media di ogni metrica + mostra bar chart per esempio.
 */

import { useState } from 'react'
import { useAppData } from '../context/AppData'
import EmptyState from '../components/EmptyState'
import MetricCard from '../components/MetricCard'
import { downloadJSON, timestampedFilename } from '../utils/download'
import Icon from '../components/Icon'

const METRIC_LABELS = {
  citation_precision: 'Citation Precision',
  citation_recall: 'Citation Recall',
  factual_precision: 'Factual Precision',
  factual_precision_nli: 'Factual Precision NLI',
  unsupported_ratio: 'Unsupported Ratio',
  avg_entailment_score: 'Avg Entailment',
}

function metricColor(key, v) {
  if (key === 'unsupported_ratio') {
    return v <= 0.2 ? 'var(--green)' : v <= 0.5 ? 'var(--amber)' : 'var(--red)'
  }
  return v >= 0.7 ? 'var(--green)' : v >= 0.4 ? 'var(--amber)' : 'var(--red)'
}

export default function Metrics() {
  const { pipelineResults } = useAppData()
  const [selectedMetric, setSelectedMetric] = useState('citation_precision')

  // Esempi con metriche
  const examples = pipelineResults.filter(r => r.metrics && Object.keys(r.metrics).length > 0)

  if (examples.length === 0) {
    return (
      <div>
        <div className="page-header">
          <div className="page-header-title">Metriche di valutazione</div>
          <div className="page-header-sub">
            Riepilogo aggregato e distribuzione per esempio.
          </div>
        </div>
        <EmptyState
          title="Nessuna metrica disponibile"
          hint="Esegui il pipeline includendo lo step 'Valuta metriche' e salva in Esplora."
        />
      </div>
    )
  }

  // Aggregazione: media di ogni metrica
  const aggregated = {}
  for (const key of Object.keys(METRIC_LABELS)) {
    const vals = examples.map(e => e.metrics[key]).filter(v => typeof v === 'number')
    aggregated[key] = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0
  }

  function onDownload() {
    const payload = {
      num_examples: examples.length,
      metrics: aggregated,
      per_example: examples.map(ex => ({
        question: ex.question,
        metrics: ex.metrics,
      })),
      exported_at: new Date().toISOString(),
    }
    downloadJSON(payload, timestampedFilename('metrics'))
  }

  return (
    <div>
      <div className="page-header">
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 16 }}>
          <div>
            <div className="page-header-title">Metriche di valutazione</div>
            <div className="page-header-sub">
              Riepilogo aggregato e distribuzione per esempio · {examples.length} esempi
            </div>
          </div>
          <button className="btn btn-secondary" onClick={onDownload}>
            <Icon name="download" size={13} strokeWidth={1.75} /> Scarica JSON
          </button>
        </div>
      </div>

      {/* Metriche aggregate */}
      <div style={{
        fontSize: 12, fontWeight: 700, color: 'var(--text-2)',
        textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 12,
      }}>
        Metriche aggregate (media)
      </div>
      <div className="grid-3" style={{ marginBottom: 28 }}>
        {Object.entries(METRIC_LABELS).map(([key, label]) => (
          <MetricCard
            key={key} label={label} value={aggregated[key]}
            color={metricColor(key, aggregated[key])}
            isUnsupported={key === 'unsupported_ratio'}
          />
        ))}
      </div>

      <div className="divider" />

      {/* Dettaglio per esempio */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
        <div style={{
          fontSize: 12, fontWeight: 700, color: 'var(--text-2)',
          textTransform: 'uppercase', letterSpacing: '0.5px',
        }}>
          Dettaglio per esempio
        </div>
        <div style={{ flex: 1 }} />
        <select className="input" style={{ width: 220 }}
          value={selectedMetric} onChange={e => setSelectedMetric(e.target.value)}>
          {Object.entries(METRIC_LABELS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
        </select>
      </div>

      <div className="card">
        <div className="card-body">
          {/* Bar chart orizzontale */}
          <div style={{ marginBottom: 20 }}>
            {examples.map((ex, i) => {
              const val = ex.metrics[selectedMetric] ?? 0
              const color = metricColor(selectedMetric, val)
              return (
                <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 10 }}>
                  <div style={{
                    fontSize: 12, color: 'var(--text-2)', minWidth: 220,
                    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                  }} title={ex.question}>
                    {(ex.question || '').slice(0, 35)}{(ex.question || '').length > 35 ? '…' : ''}
                  </div>
                  <div style={{
                    flex: 1, height: 8, background: 'var(--border-2)',
                    borderRadius: 4, overflow: 'hidden',
                  }}>
                    <div style={{
                      height: '100%', width: `${val * 100}%`, background: color,
                      borderRadius: 4, transition: 'width 0.8s ease',
                    }} />
                  </div>
                  <div style={{
                    fontFamily: 'var(--mono)', fontSize: 12, fontWeight: 600,
                    color, minWidth: 44, textAlign: 'right',
                  }}>
                    {val.toFixed(3)}
                  </div>
                </div>
              )
            })}
          </div>

          {/* Tabella completa */}
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid var(--border)' }}>
                  <th style={{ textAlign: 'left', padding: '8px 12px', color: 'var(--text-3)', fontWeight: 600 }}>
                    Question
                  </th>
                  {Object.values(METRIC_LABELS).map(l => (
                    <th key={l} style={{
                      textAlign: 'right', padding: '8px 12px',
                      color: 'var(--text-3)', fontWeight: 600, whiteSpace: 'nowrap',
                    }}>{l}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {examples.map((ex, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid var(--border-2)' }}>
                    <td style={{
                      padding: '8px 12px', color: 'var(--text-2)',
                      maxWidth: 200, overflow: 'hidden',
                      textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                    }} title={ex.question}>
                      {ex.question || '—'}
                    </td>
                    {Object.keys(METRIC_LABELS).map(k => {
                      const v = ex.metrics[k] ?? 0
                      return (
                        <td key={k} style={{
                          padding: '8px 12px', textAlign: 'right',
                          fontFamily: 'var(--mono)', fontSize: 11, fontWeight: 600,
                          color: metricColor(k, v),
                        }}>
                          {v.toFixed(3)}
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  )
}