/**
 * Interpretability.jsx — Integrated Gradients e Activation Patching.
 *
 * Due tab:
 *  - IG: 30-90s, mostra token attribution + heatmap layer-wise
 *  - Patching: 2-5 minuti, heatmap layer × posizione
 */

import { useState, useEffect } from 'react'
import api from '../api'
import { useAppData } from '../context/AppData'
import EmptyState from '../components/EmptyState'
import Icon from '../components/Icon'
import { downloadJSON, timestampedFilename } from '../utils/download'

export default function Interpretability() {
  const [tab, setTab] = useState('ig')

  return (
    <div>
      <div className="page-header">
        <div className="page-header-title">Interpretability</div>
        <div className="page-header-sub">
          Integrated Gradients e Activation Patching per analizzare dove DeBERTa prende decisioni.
        </div>
      </div>

      <div className="card">
        <div style={{ padding: '0 20px', borderBottom: '1px solid var(--border-2)' }}>
          <div className="tabs" style={{ marginBottom: 0 }}>
            <div className={`tab${tab === 'ig' ? ' active' : ''}`}
              onClick={() => setTab('ig')}
              style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <Icon name="tag" size={13} strokeWidth={1.75} /> Integrated Gradients
            </div>
            <div className={`tab${tab === 'patch' ? ' active' : ''}`}
              onClick={() => setTab('patch')}
              style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <Icon name="layers" size={13} strokeWidth={1.75} /> Activation Patching
            </div>
          </div>
        </div>
        <div className="card-body">
          {tab === 'ig' ? <IGTab /> : <PatchTab />}
        </div>
      </div>
    </div>
  )
}

// ── IG Tab ────────────────────────────────────────────────────────────

function attrColor(val) {
  if (val > 0) {
    const a = Math.min(Math.abs(val), 1.0)
    return `rgba(5,150,105,${(0.1 + a * 0.75).toFixed(2)})`
  }
  const a = Math.min(Math.abs(val), 1.0)
  return `rgba(220,38,38,${(0.1 + a * 0.75).toFixed(2)})`
}

function IGTab() {
  const { igResult, setIgResult } = useAppData()
  const [premise, setPremise] = useState('')
  const [hypothesis, setHypothesis] = useState('')
  const [target, setTarget] = useState('entailment')
  const [nSteps, setNSteps] = useState(50)
  const [layerwise, setLayerwise] = useState(true)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState(null)

  async function runIG() {
    setError(null)
    setRunning(true)
    try {
      const result = await api.interpret.ig({
        premise, hypothesis,
        target_label: target, n_steps: nSteps, layerwise,
      })
      setIgResult(result)
    } catch (e) {
      setError(e.message)
    }
    setRunning(false)
  }

  return (
    <div>
      <div style={{ fontSize: 13, color: 'var(--text-2)', marginBottom: 16, lineHeight: 1.6 }}>
        Integrated Gradients misura quanto ogni token dell'input contribuisce allo score di entailment.
        Calcolo: ~30-90 secondi su DeBERTa-v3-large.
      </div>

      {error && (
        <div className="info-box info-box-red" style={{ marginBottom: 16 }}>
          <Icon name="xCircle" size={15} strokeWidth={1.75} />
          <span><strong>Errore:</strong> {error}</span>
        </div>
      )}

      <div className="grid-2" style={{ marginBottom: 12 }}>
        <div className="form-group" style={{ marginBottom: 0 }}>
          <label className="form-label">Premise</label>
          <textarea className="input" rows={3} value={premise}
            onChange={e => setPremise(e.target.value)} />
        </div>
        <div className="form-group" style={{ marginBottom: 0 }}>
          <label className="form-label">Hypothesis</label>
          <textarea className="input" rows={3} value={hypothesis}
            onChange={e => setHypothesis(e.target.value)} />
        </div>
      </div>

      <div style={{ display: 'flex', gap: 16, alignItems: 'flex-end', marginBottom: 16 }}>
        <div className="form-group" style={{ marginBottom: 0, minWidth: 180 }}>
          <label className="form-label">Target label</label>
          <select className="input" value={target} onChange={e => setTarget(e.target.value)}>
            <option value="entailment">entailment</option>
            <option value="neutral">neutral</option>
            <option value="contradiction">contradiction</option>
          </select>
        </div>
        <div className="form-group" style={{ marginBottom: 0, minWidth: 180 }}>
          <label className="form-label">Steps — {nSteps}</label>
          <input type="range" min={10} max={100} step={5} value={nSteps}
            onChange={e => setNSteps(+e.target.value)}
            style={{ width: '100%', accentColor: 'var(--accent)' }} />
        </div>
        <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 13 }}>
          <input type="checkbox" checked={layerwise}
            onChange={e => setLayerwise(e.target.checked)}
            style={{ accentColor: 'var(--accent)' }} />
          Layer-wise (più lento)
        </label>
      </div>

      <button className="btn btn-primary" onClick={runIG}
        disabled={running || !premise.trim() || !hypothesis.trim()}>
        {running
          ? <><span className="spinner" style={{ width: 14, height: 14, borderColor: 'rgba(255,255,255,0.3)', borderTopColor: 'white' }} /> Calcolo in corso...</>
          : <><Icon name="play" size={13} color="white" strokeWidth={2} /> Run Integrated Gradients</>}
      </button>

      {!igResult && !running && (
        <div style={{ marginTop: 20 }}>
          <EmptyState title="Nessun risultato IG" hint="Inserisci premise + hypothesis e clicca Run." />
        </div>
      )}

      {igResult && !running && <IGResultView result={igResult} />}
    </div>
  )
}

function IGResultView({ result }) {
  const tokens = result.tokens || []
  const tokAttrNorm = result.token_attributions_normalized || []
  const tokAttrRaw = result.token_attributions || []
  const probs = result.probs || {}

  return (
    <div style={{ marginTop: 24 }}>
      <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 12 }}>
        <button
          className="btn btn-secondary"
          onClick={() => downloadJSON(result, timestampedFilename('ig_result'))}
        >
          <Icon name="download" size={13} strokeWidth={1.75} /> Scarica risultato IG
        </button>
      </div>
      <div className="divider" />
      <div className="grid-3" style={{ marginBottom: 20 }}>
        {[
          ['Entailment', probs.entailment, 'var(--red)'],
          ['Contradiction', probs.contradiction, 'var(--text-2)'],
          ['Neutral', probs.neutral, 'var(--text-2)'],
        ].map(([l, v, c]) => (
          <div key={l} className="metric-card">
            <div className="metric-value" style={{ color: c }}>{(v || 0).toFixed(4)}</div>
            <div className="metric-label">{l}</div>
            <div className="metric-bar-track">
              <div className="metric-bar-fill" style={{ width: `${(v || 0) * 100}%`, background: c }} />
            </div>
          </div>
        ))}
      </div>

      <div style={{ fontSize: 11, color: 'var(--text-3)', marginBottom: 16 }}>
        Predicted: <strong style={{ color: 'var(--text)' }}>{result.predicted || '—'}</strong>
        {result.convergence_delta != null && (
          <>&nbsp;·&nbsp; Convergence delta:{' '}
            <span style={{ fontFamily: 'var(--mono)' }}>{result.convergence_delta.toFixed(4)}</span>
          </>
        )}
      </div>

      <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-2)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 10 }}>
        Token-level attribution
      </div>
      <div style={{ fontSize: 11, color: 'var(--text-3)', marginBottom: 10 }}>
        Verde = contributo positivo · Rosso = contributo negativo
      </div>
      <div style={{
        background: 'var(--bg)', padding: '16px 20px', borderRadius: 10,
        border: '1px solid var(--border)', lineHeight: 2.4,
        marginBottom: 20, flexWrap: 'wrap', display: 'flex',
      }}>
        {tokens.map((tok, i) => {
          const attr = tokAttrNorm[i] || 0
          const clean = tok.replace(/[▁Ġ]/g, ' ').trim() || tok
          return (
            <span key={i} title={`attr=${attr.toFixed(3)}`}
              style={{
                display: 'inline-block', padding: '2px 6px', margin: '2px',
                borderRadius: 4, background: attrColor(attr),
                fontFamily: 'var(--mono)', fontSize: 13,
              }}>{clean}</span>
          )
        })}
      </div>

      <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-2)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 10 }}>
        Top-10 token più influenti
      </div>
      {tokens
        .map((t, i) => ({ tok: t, raw: tokAttrRaw[i] || 0, idx: i }))
        .sort((a, b) => Math.abs(b.raw) - Math.abs(a.raw))
        .slice(0, 10)
        .map((item, rank) => {
          const clean = item.tok.replace(/[▁Ġ]/g, ' ').trim() || item.tok
          const isPos = item.raw > 0
          return (
            <div key={rank} style={{
              display: 'flex', alignItems: 'center', gap: 12,
              padding: '6px 12px', margin: '4px 0',
              background: 'var(--bg)', borderRadius: 8,
              border: '1px solid var(--border)',
            }}>
              <span style={{ color: 'var(--text-3)', fontFamily: 'var(--mono)', minWidth: 28 }}>
                #{rank + 1}
              </span>
              <span style={{ fontFamily: 'var(--mono)', fontWeight: 600, minWidth: 110 }}>
                {clean}
              </span>
              <span style={{
                color: isPos ? 'var(--green)' : 'var(--red)',
                fontFamily: 'var(--mono)', minWidth: 80,
              }}>
                {isPos ? '↑' : '↓'} {item.raw > 0 ? '+' : ''}{item.raw.toFixed(4)}
              </span>
              <span style={{ color: 'var(--text-3)', fontSize: 11 }}>position {item.idx}</span>
            </div>
          )
        })}

      {Array.isArray(result.layerwise_attributions) && result.layerwise_attributions.length > 0 && (
        <div style={{ marginTop: 24 }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-2)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 10 }}>
            Layer-wise attribution heatmap
          </div>
          <div style={{
            background: 'var(--bg)', padding: '16px 20px', borderRadius: 10,
            border: '1px solid var(--border)', overflowX: 'auto',
          }}>
            <div style={{ display: 'flex', gap: 2, marginBottom: 4, paddingLeft: 44 }}>
              {tokens.slice(0, 48).map((t, i) => {
                const clean = t.replace(/[▁Ġ]/g, '').slice(0, 3)
                return (
                  <div key={i} style={{
                    width: 14, height: 14, fontSize: 7,
                    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                    textAlign: 'center', color: '#334155', lineHeight: '14px',
                  }} title={t}>{clean}</div>
                )
              })}
            </div>
            {result.layerwise_attributions.map(ld => (
              <div key={ld.layer} style={{ display: 'flex', alignItems: 'center', gap: 4, marginBottom: 2 }}>
                <div style={{ minWidth: 40, fontFamily: 'var(--mono)', fontSize: 9, color: 'var(--text-3)' }}>
                  L{ld.layer}
                </div>
                {(ld.token_attributions_normalized || []).slice(0, 48).map((v, i) => (
                  <div key={i} style={{
                    width: 14, height: 14, borderRadius: 2,
                    background: attrColor(v), flexShrink: 0,
                  }} title={v.toFixed(3)} />
                ))}
                <div style={{ minWidth: 52, fontFamily: 'var(--mono)', fontSize: 9, color: 'var(--text-3)', textAlign: 'right' }}>
                  {(ld.mean_abs_attribution || 0).toFixed(4)}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// ── Patching Tab ────────────────────────────────────────────────────────

function PatchTab() {
  const { patchResult, setPatchResult } = useAppData()

  const [cleanPremise, setCleanPremise] = useState('')
  const [cleanHyp, setCleanHyp] = useState('')
  const [corruptPremise, setCorruptPremise] = useState('')
  const [corruptHyp, setCorruptHyp] = useState('')
  const [running, setRunning] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [error, setError] = useState(null)

  // Timer quando running
  useEffect(() => {
    if (!running) return
    const t0 = Date.now()
    const id = setInterval(() => setElapsed(Math.floor((Date.now() - t0) / 1000)), 1000)
    return () => clearInterval(id)
  }, [running])

  async function runPatch() {
    if (!cleanPremise.trim() || !cleanHyp.trim() || !corruptPremise.trim() || !corruptHyp.trim()) return
    setError(null)
    setRunning(true)
    setElapsed(0)
    try {
      const result = await api.interpret.patching({
        clean_premise: cleanPremise, clean_hypothesis: cleanHyp,
        corrupt_premise: corruptPremise, corrupt_hypothesis: corruptHyp,
      })
      setPatchResult(result)
    } catch (e) {
      setError(e.message)
    }
    setRunning(false)
  }

  return (
    <div>
      <div style={{ fontSize: 13, color: 'var(--text-2)', marginBottom: 16, lineHeight: 1.6 }}>
        Activation Patching sostituisce le attivazioni del corrupt nell'input clean per ogni layer/posizione.
      </div>

      <div className="info-box info-box-amber" style={{ marginBottom: 16 }}>
        <Icon name="warning" size={15} strokeWidth={1.75} />
        <span>
          <strong>Attenzione:</strong> computazionalmente costoso — 2-5 minuti su DeBERTa-v3-large (24 layer).
          Il browser sembrerà fermo ma il calcolo procede lato backend.
        </span>
      </div>

      {error && (
        <div className="info-box info-box-red" style={{ marginBottom: 16 }}>
          <Icon name="xCircle" size={15} strokeWidth={1.75} />
          <span><strong>Errore:</strong> {error}</span>
        </div>
      )}

      <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 10 }}>
        Input CLEAN (baseline corretto)
      </div>
      <div className="grid-2" style={{ marginBottom: 20 }}>
        <div className="form-group" style={{ marginBottom: 0 }}>
          <label className="form-label">Premise (clean)</label>
          <textarea className="input" rows={3} value={cleanPremise}
            onChange={e => setCleanPremise(e.target.value)} />
        </div>
        <div className="form-group" style={{ marginBottom: 0 }}>
          <label className="form-label">Hypothesis (clean)</label>
          <textarea className="input" rows={3} value={cleanHyp}
            onChange={e => setCleanHyp(e.target.value)} />
        </div>
      </div>

      <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 10 }}>
        Input CORRUPT (biased)
      </div>
      <div className="grid-2" style={{ marginBottom: 20 }}>
        <div className="form-group" style={{ marginBottom: 0 }}>
          <label className="form-label">Premise (corrupt)</label>
          <textarea className="input" rows={3} value={corruptPremise}
            onChange={e => setCorruptPremise(e.target.value)} />
        </div>
        <div className="form-group" style={{ marginBottom: 0 }}>
          <label className="form-label">Hypothesis (corrupt)</label>
          <textarea className="input" rows={3} value={corruptHyp}
            onChange={e => setCorruptHyp(e.target.value)} />
        </div>
      </div>

      <button className="btn btn-primary" onClick={runPatch} disabled={running}>
        {running
          ? <><span className="spinner" style={{ width: 14, height: 14, borderColor: 'rgba(255,255,255,0.3)', borderTopColor: 'white' }} /> Calcolo in corso... {elapsed}s</>
          : <><Icon name="play" size={13} color="white" strokeWidth={2} /> Run Activation Patching</>}
      </button>

      {!patchResult && !running && (
        <div style={{ marginTop: 20 }}>
          <EmptyState title="Nessun risultato" hint="Inserisci input clean e corrupt e clicca Run." />
        </div>
      )}

      {patchResult && !running && <PatchResultView result={patchResult} />}
    </div>
  )
}

function PatchResultView({ result }) {
  const effectMatrix = result.patching_effect || []
  const numLayers = result.num_layers || effectMatrix.length
  const seqLen = result.seq_len || (effectMatrix[0]?.length || 0)
  const cleanTokens = result.clean_tokens || []
  const cleanE = result.clean_entailment || 0
  const corruptE = result.corrupt_entailment || 0

  return (
    <div style={{ marginTop: 24 }}>
      <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 12 }}>
        <button
          className="btn btn-secondary"
          onClick={() => downloadJSON(result, timestampedFilename('patching_result'))}
        >
          <Icon name="download" size={13} strokeWidth={1.75} /> Scarica risultato Patching
        </button>
      </div>
      <div className="divider" />
      <div className="grid-3" style={{ marginBottom: 20 }}>
        <div className="metric-card">
          <div className="metric-value" style={{ color: 'var(--green)' }}>{cleanE.toFixed(4)}</div>
          <div className="metric-label">E (clean)</div>
        </div>
        <div className="metric-card">
          <div className="metric-value" style={{ color: 'var(--red)' }}>{corruptE.toFixed(4)}</div>
          <div className="metric-label">E (corrupt)</div>
        </div>
        <div className="metric-card">
          <div className="metric-value" style={{ color: 'var(--red)' }}>
            {corruptE >= cleanE ? '+' : ''}{(corruptE - cleanE).toFixed(4)}
          </div>
          <div className="metric-label">Gap</div>
        </div>
      </div>

      <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-2)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 10 }}>
        Heatmap patching effect (layer × posizione) · {numLayers} layer × {seqLen} posizioni
      </div>
      <div style={{
        background: 'var(--bg)', padding: '16px 20px', borderRadius: 10,
        border: '1px solid var(--border)', overflowX: 'auto',
      }}>
        <div style={{ display: 'flex', gap: 8, fontSize: 11, color: 'var(--text-3)', marginBottom: 10 }}>
          <span>■ <span style={{ color: 'var(--red)' }}>rosso = patching trasferisce il bias</span></span>
          <span>■ <span style={{ color: 'var(--accent)' }}>blu = patching riduce l'effetto</span></span>
        </div>

        <div style={{ display: 'flex', gap: 2, marginBottom: 4, paddingLeft: 44 }}>
          {cleanTokens.slice(0, seqLen).map((t, i) => (
            <div key={i} style={{
              width: 22, height: 14, fontSize: 7,
              overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
              textAlign: 'center', color: '#334155', lineHeight: '14px',
            }} title={t}>{(t || '').slice(0, 4)}</div>
          ))}
        </div>

        {effectMatrix.map((row, layer) => (
          <div key={layer} style={{ display: 'flex', alignItems: 'center', gap: 4, marginBottom: 2 }}>
            <div style={{ minWidth: 40, fontFamily: 'var(--mono)', fontSize: 9, color: 'var(--text-3)' }}>
              L{layer}
            </div>
            {row.map((val, pos) => {
              const intensity = Math.min(Math.abs(val) / 1.5, 1.0)
              const color = val > 0
                ? `rgba(220,38,38,${(0.1 + intensity * 0.85).toFixed(2)})`
                : `rgba(37,99,235,${(0.1 + intensity * 0.85).toFixed(2)})`
              return (
                <div key={pos} style={{
                  width: 22, height: 16, borderRadius: 3, background: color,
                }} title={`L${layer} pos${pos}: ${val.toFixed(3)}`} />
              )
            })}
          </div>
        ))}
      </div>
    </div>
  )
}