/**
 * Attention.jsx — Analisi attention di DeBERTa NLI.
 *
 * Permette di calcolare live un record (premise, hypothesis) e
 * visualizza hyp_dominance layer-per-layer, cross-attention, e bias flag.
 * I record calcolati si accumulano nel context.
 */

import { useState } from 'react'
import api from '../api'
import { useAppData } from '../context/AppData'
import EmptyState from '../components/EmptyState'
import LayerChart from '../components/LayerChart'
import Icon from '../components/Icon'
import { downloadJSON, timestampedFilename } from '../utils/download'

const FLAG_COLORS = {
  'BIAS CONFIRMED': 'var(--red)',
  suspicious: 'var(--amber)',
  clean: 'var(--green)',
}
const FLAG_BGS = {
  'BIAS CONFIRMED': '#FEF2F2',
  suspicious: '#FFFBEB',
  clean: '#ECFDF5',
}

export default function Attention() {
  const { attentionRecords, addAttentionRecord } = useAppData()

  // Form: calcolo live
  const [premise, setPremise] = useState('')
  const [hypothesis, setHypothesis] = useState('')
  const [expected, setExpected] = useState('neutral')
  const [running, setRunning] = useState(false)
  const [error, setError] = useState(null)

  // Selezione record
  const [selectedId, setSelectedId] = useState(null)
  const [cats, setCats] = useState(['BIAS', 'CLEAN', 'SUSPICIOUS'])

  async function runAttention() {
    if (!premise.trim() || !hypothesis.trim()) return
    setError(null)
    setRunning(true)
    try {
      const record = await api.interpret.attention({ premise, hypothesis, expected })
      addAttentionRecord(record)
      setSelectedId(record.id)
    } catch (e) {
      setError(e.message)
    }
    setRunning(false)
  }

  function onDownloadAll() {
    if (attentionRecords.length === 0) return
    downloadJSON(attentionRecords, timestampedFilename('attention_records'))
  }

  function onDownloadCurrent() {
    if (!record) return
    downloadJSON(record, timestampedFilename(`attention_${record.id}`))
  }

  const allCats = [...new Set(attentionRecords.map(r => r.category))]
  const filtered = attentionRecords.filter(r => cats.includes(r.category))
  const record = attentionRecords.find(r => r.id === selectedId) ||
                 (filtered.length > 0 ? filtered[0] : null)

  return (
    <div>
      <div className="page-header">
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 16 }}>
          <div>
            <div className="page-header-title">Attention Analysis</div>
            <div className="page-header-sub">
              Visualizza gli attention weights di DeBERTa per rilevare parametric knowledge leakage.
            </div>
          </div>
          {attentionRecords.length > 0 && (
            <button className="btn btn-secondary" onClick={onDownloadAll}>
              <Icon name="download" size={13} strokeWidth={1.75} /> Scarica tutti ({attentionRecords.length})
            </button>
          )}
        </div>
      </div>

      {error && (
        <div className="info-box info-box-red" style={{ marginBottom: 16 }}>
          <Icon name="xCircle" size={15} strokeWidth={1.75} style={{ flexShrink: 0, marginTop: 1 }} />
          <span><strong>Errore:</strong> {error}</span>
        </div>
      )}

      {/* Form calcolo live */}
      <div className="card" style={{ marginBottom: 16 }}>
        <div className="card-body">
          <div className="grid-2" style={{ marginBottom: 12 }}>
            <div className="form-group" style={{ marginBottom: 0 }}>
              <label className="form-label">Premise</label>
              <textarea className="input" rows={3} value={premise}
                onChange={e => setPremise(e.target.value)}
                placeholder="Es: France won the 1998 FIFA World Cup." />
            </div>
            <div className="form-group" style={{ marginBottom: 0 }}>
              <label className="form-label">Hypothesis</label>
              <textarea className="input" rows={3} value={hypothesis}
                onChange={e => setHypothesis(e.target.value)}
                placeholder="Es: Zinedine Zidane was the best player at the 1998 World Cup." />
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <div className="form-group" style={{ marginBottom: 0, flex: 1 }}>
              <label className="form-label">Expected label</label>
              <select className="input" value={expected} onChange={e => setExpected(e.target.value)}>
                <option value="neutral">neutral</option>
                <option value="entailment">entailment</option>
                <option value="contradiction">contradiction</option>
              </select>
            </div>
            <button
              className="btn btn-primary"
              style={{ marginTop: 22 }}
              onClick={runAttention}
              disabled={running || !premise.trim() || !hypothesis.trim()}
            >
              {running
                ? <><span className="spinner" style={{ width: 14, height: 14, borderColor: 'rgba(255,255,255,0.3)', borderTopColor: 'white' }} /> Calcolo...</>
                : <><Icon name="play" size={13} color="white" strokeWidth={2} /> Calcola attention</>}
            </button>
          </div>
        </div>
      </div>

      {attentionRecords.length === 0 ? (
        <EmptyState
          title="Nessun record calcolato"
          hint="Inserisci una coppia (premise, hypothesis) e clicca 'Calcola attention'."
        />
      ) : (
        <>
          {/* Filtro categoria */}
          <div className="card" style={{ marginBottom: 16, padding: '12px 20px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
              <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-2)' }}>
                Filtro categoria:
              </span>
              {allCats.map(c => (
                <label key={c} style={{
                  display: 'flex', alignItems: 'center', gap: 6,
                  cursor: 'pointer', fontSize: 13, fontWeight: 500,
                }}>
                  <input type="checkbox" checked={cats.includes(c)}
                    onChange={e => setCats(prev =>
                      e.target.checked ? [...prev, c] : prev.filter(x => x !== c)
                    )}
                    style={{ accentColor: 'var(--accent)', width: 14, height: 14 }} />
                  {c}
                </label>
              ))}
              <span style={{ marginLeft: 'auto', fontSize: 12, color: 'var(--text-3)' }}>
                {filtered.length} esempi
              </span>
            </div>
          </div>

          {/* Ranking */}
          <div className="card" style={{ marginBottom: 16 }}>
            <div className="card-header">
              <div className="card-header-title">
                1 — Panoramica hyp_dominance (CLS → H / CLS → P+H)
              </div>
            </div>
            <div className="card-body">
              <div style={{
                display: 'grid', gridTemplateColumns: '180px 1fr 56px 56px 130px', gap: 8,
                fontSize: 10, color: 'var(--text-3)', fontWeight: 600,
                textTransform: 'uppercase', letterSpacing: '0.5px',
                paddingBottom: 8, borderBottom: '1px solid var(--border-2)', marginBottom: 8,
              }}>
                <span>ID</span>
                <span>Dominanza H →</span>
                <span style={{ textAlign: 'right' }}>dom</span>
                <span style={{ textAlign: 'right' }}>E-score</span>
                <span>Flag</span>
              </div>
              {[...filtered]
                .sort((a, b) =>
                  b.cross_attention.hyp_dominance_from_cls - a.cross_attention.hyp_dominance_from_cls
                )
                .map(r => {
                  const dom = r.cross_attention.hyp_dominance_from_cls
                  const color = dom < 0.55 ? 'var(--green)' : dom < 0.70 ? 'var(--amber)' : 'var(--red)'
                  const isSelected = record && r.id === record.id
                  return (
                    <div key={r.id} onClick={() => setSelectedId(r.id)}
                      style={{
                        display: 'grid',
                        gridTemplateColumns: '180px 1fr 56px 56px 130px',
                        gap: 8, alignItems: 'center', padding: '7px 0',
                        borderBottom: '1px solid var(--border-2)', cursor: 'pointer',
                        background: isSelected ? 'var(--accent-lt)' : 'transparent',
                        borderRadius: isSelected ? 6 : 0,
                        paddingLeft: isSelected ? 8 : 0,
                        paddingRight: isSelected ? 8 : 0,
                      }}>
                      <span style={{
                        fontSize: 12, fontFamily: 'var(--mono)',
                        color: isSelected ? 'var(--accent)' : 'var(--text-2)',
                      }}>{r.id}</span>
                      <div style={{ height: 8, background: 'var(--border-2)', borderRadius: 4, overflow: 'hidden' }}>
                        <div style={{ height: '100%', width: `${dom * 100}%`, background: color, borderRadius: 4 }} />
                      </div>
                      <span style={{ textAlign: 'right', fontFamily: 'var(--mono)', fontSize: 12, fontWeight: 600, color }}>
                        {dom.toFixed(2)}
                      </span>
                      <span style={{ textAlign: 'right', fontFamily: 'var(--mono)', fontSize: 12, color: 'var(--text-2)' }}>
                        {r.probs.E.toFixed(2)}
                      </span>
                      <span style={{
                        fontSize: 11, fontWeight: 600,
                        color: FLAG_COLORS[r.bias_flag] || 'var(--text-3)',
                        background: FLAG_BGS[r.bias_flag] || 'var(--bg)',
                        padding: '2px 8px', borderRadius: 20, width: 'fit-content',
                      }}>
                        {r.bias_flag}
                      </span>
                    </div>
                  )
                })}

              <div style={{ display: 'flex', gap: 16, marginTop: 12, fontSize: 11, color: 'var(--text-3)' }}>
                {[
                  ['var(--green)', '< 0.55 clean'],
                  ['var(--amber)', '0.55–0.70 sospetto'],
                  ['var(--red)', '> 0.70 leakage'],
                ].map(([c, l]) => (
                  <span key={l} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                    <span style={{ width: 10, height: 10, borderRadius: 2, background: c }} />
                    {l}
                  </span>
                ))}
              </div>
            </div>
          </div>

          {/* Dettaglio */}
          {record && (
            <div className="card" style={{ marginBottom: 16 }}>
              <div className="card-header" style={{ justifyContent: 'space-between' }}>
                <div className="card-header-title">
                  2 — Dettaglio: <span style={{ fontFamily: 'var(--mono)', color: 'var(--accent)' }}>{record.id}</span>
                </div>
                <button
                  className="btn btn-secondary"
                  style={{ padding: '4px 10px', fontSize: 11 }}
                  onClick={onDownloadCurrent}
                >
                  <Icon name="download" size={11} strokeWidth={1.75} /> Scarica record
                </button>
              </div>
              <div className="card-body">
                <div className="grid-3" style={{ marginBottom: 20 }}>
                  <div className="metric-card">
                    <div className="metric-value" style={{
                      color: record.cross_attention.hyp_dominance_from_cls < 0.55
                        ? 'var(--green)'
                        : record.cross_attention.hyp_dominance_from_cls < 0.70
                          ? 'var(--amber)'
                          : 'var(--red)',
                    }}>
                      {record.cross_attention.hyp_dominance_from_cls.toFixed(3)}
                    </div>
                    <div className="metric-label">hyp_dominance (CLS)</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-value" style={{
                      color: record.probs.E < 0.15
                        ? 'var(--green)'
                        : record.probs.E < 0.4
                          ? 'var(--amber)'
                          : 'var(--red)',
                    }}>
                      {record.probs.E.toFixed(3)}
                    </div>
                    <div className="metric-label">Entailment score</div>
                  </div>
                  <div className="metric-card">
                    <div style={{
                      fontSize: 18, fontWeight: 700,
                      color: FLAG_COLORS[record.bias_flag] || 'var(--text-3)',
                      marginBottom: 6,
                    }}>{record.bias_flag}</div>
                    <div className="metric-label">Bias flag</div>
                    <div style={{ fontSize: 12, color: 'var(--text-2)', marginTop: 4 }}>
                      Expected: <span className="badge badge-blue">{record.expected}</span>&nbsp;
                      Predicted: <span className={`badge ${record.predicted === record.expected ? 'badge-green' : 'badge-red'}`}>
                        {record.predicted}
                      </span>
                    </div>
                  </div>
                </div>

                <div style={{ fontSize: 13, marginBottom: 8 }}>
                  <strong>Premise:</strong>{' '}
                  <span style={{ color: 'var(--text-2)' }}>{record.premise}</span>
                </div>
                <div style={{ fontSize: 13, marginBottom: 20 }}>
                  <strong>Hypothesis:</strong>{' '}
                  <span style={{ color: 'var(--text-2)' }}>{record.hypothesis}</span>
                </div>

                <div style={{
                  fontSize: 12, fontWeight: 600, color: 'var(--text-2)',
                  textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 10,
                }}>
                  Layer-wise hyp_dominance
                </div>
                <LayerChart layers={record.layer_dominance} />

                <div style={{ marginTop: 20, display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 10 }}>
                  {[
                    ['CLS→P', record.cross_attention.CLS_to_P],
                    ['CLS→H', record.cross_attention.CLS_to_H],
                    ['P→H', record.cross_attention.P_to_H],
                    ['H→P', record.cross_attention.H_to_P],
                  ].map(([label, val]) => (
                    <div key={label} style={{
                      textAlign: 'center', padding: '10px 14px',
                      background: 'var(--bg)', borderRadius: 8,
                      border: '1px solid var(--border)',
                    }}>
                      <div style={{ fontSize: 10, color: 'var(--text-3)', marginBottom: 4 }}>{label}</div>
                      <div style={{ fontSize: 20, fontWeight: 700, fontFamily: 'var(--mono)' }}>
                        {val.toFixed(4)}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}