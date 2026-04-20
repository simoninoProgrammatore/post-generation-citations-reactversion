/**
 * Pipeline.jsx — Pipeline interattivo (6 step manuali)
 */

import { useState, useRef, useMemo } from 'react'
import api from '../api'
import { useAppData } from '../context/AppData'

import StepCard from '../components/StepCard'
import EmptyState from '../components/EmptyState'
import ScorePill from '../components/ScorePill'
import MetricCard from '../components/MetricCard'
import Icon from '../components/Icon'
import { downloadJSON, timestampedFilename } from '../utils/download'

const METRIC_INFO = {
  citation_precision:    { label: 'Citation Precision',     desc: '% di coppie (claim, passaggio) dove il passaggio supporta il claim via NLI.' },
  citation_recall:       { label: 'Citation Recall',        desc: '% di claims con almeno un passaggio citato che fornisce supporto NLI.' },
  factual_precision:     { label: 'Factual Precision',      desc: '% di claims con almeno un passaggio di supporto (senza NLI).' },
  factual_precision_nli: { label: 'Factual Precision (NLI)', desc: 'Come Factual Precision, ma verificato via NLI.' },
  unsupported_ratio:     { label: 'Unsupported Ratio',      desc: '% di claims senza alcun passaggio di supporto.' },
  avg_entailment_score:  { label: 'Avg Entailment Score',   desc: 'Score medio di entailment tra claims e passaggi.' },
}

function metricColor(key, v) {
  if (key === 'unsupported_ratio') {
    return v <= 0.2 ? 'var(--green)' : v <= 0.5 ? 'var(--amber)' : 'var(--red)'
  }
  return v >= 0.7 ? 'var(--green)' : v >= 0.4 ? 'var(--amber)' : 'var(--red)'
}

export default function Pipeline() {
  const { addPipelineResult } = useAppData()

  const [model, setModel] = useState('claude-haiku-4-5-20251001')
  const [retrieveMethod, setRetrieveMethod] = useState('nli')
  const [threshold, setThreshold] = useState(0.5)
  const [topK, setTopK] = useState(3)
  const [settingsOpen, setSettingsOpen] = useState(false)

  const [dataset, setDataset] = useState(null)
  const [datasetName, setDatasetName] = useState('')
  const [exampleIdx, setExampleIdx] = useState(0)
  const fileRef = useRef()

  const [response, setResponse] = useState(null)
  const [claims, setClaims] = useState(null)
  const [matched, setMatched] = useState(null)
  const [cited, setCited] = useState(null)
  const [references, setReferences] = useState(null)
  const [metrics, setMetrics] = useState(null)

  const [running, setRunning] = useState(null)
  const [error, setError] = useState(null)

  const steps = {
    query:     'active',
    generate:  response ? 'done' : dataset ? 'active' : 'locked',
    decompose: claims   ? 'done' : response ? 'active' : 'locked',
    retrieve:  matched  ? 'done' : claims   ? 'active' : 'locked',
    cite:      cited    ? 'done' : matched  ? 'active' : 'locked',
    evaluate:  metrics  ? 'done' : cited    ? 'active' : 'locked',
  }
  if (running) steps[running] = 'running'

  const currentExample = dataset ? dataset[exampleIdx] : null
  const currentQuery   = currentExample?.question || ''
  const currentPassages = currentExample?.docs || []

  function resetAfter(step) {
    const order = ['generate', 'decompose', 'retrieve', 'cite', 'evaluate']
    const idx = order.indexOf(step)
    if (idx <= 0) setResponse(null)
    if (idx <= 1) setClaims(null)
    if (idx <= 2) setMatched(null)
    if (idx <= 3) { setCited(null); setReferences(null) }
    if (idx <= 4) setMetrics(null)
  }

  function onFileUpload(e) {
    const file = e.target.files[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = evt => {
      try {
        const parsed = JSON.parse(evt.target.result)
        if (!Array.isArray(parsed)) throw new Error('Il file deve contenere una lista di esempi.')
        if (parsed.length === 0) throw new Error('Il file è vuoto.')
        setDataset(parsed)
        setDatasetName(file.name)
        setExampleIdx(0)
        setResponse(null); setClaims(null); setMatched(null)
        setCited(null); setReferences(null); setMetrics(null)
        setError(null)
      } catch (err) {
        setError(`Errore lettura file: ${err.message}`)
      }
    }
    reader.readAsText(file)
  }

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
    setError(null); setRunning('evaluate')
    try {
      const res = await api.pipeline.evaluate({ matched })
      setMetrics(res)
    } catch (e) { setError(`Evaluate: ${e.message}`) }
    setRunning(null)
  }

  function saveToExplore() {
    addPipelineResult({
      question: currentQuery, raw_response: response, claims,
      matched_claims: matched, cited_response: cited, references, metrics,
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
      model,
      retrieve_method: retrieveMethod,
      threshold,
      top_k: topK,
      exported_at: new Date().toISOString(),
    }
    downloadJSON(payload, timestampedFilename('pipeline_result'))
  }

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

      {/* Step 1 */}
      <StepCard num={1} title="Query" status={steps.query}>
        {!dataset ? (
          <div>
            <div className="form-group">
              <label className="form-label">Carica dataset ALCE (.json)</label>
              <input ref={fileRef} type="file" accept=".json" onChange={onFileUpload} style={{ display: 'none' }} />
              <button className="btn btn-primary" onClick={() => fileRef.current.click()}>
                <Icon name="upload" size={14} strokeWidth={1.75} color="white" />
                Seleziona file JSON
              </button>
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-3)' }}>
              Il file deve essere un array di esempi ALCE, ciascuno con i campi <code>question</code> e <code>docs</code>.
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
            <div style={{ fontSize: 12, color: 'var(--text-3)' }}>
              {currentPassages.length} passages disponibili
              &nbsp;·&nbsp; Modello: <span style={{ fontFamily: 'var(--mono)' }}>{model}</span>
              &nbsp;·&nbsp;
              <button className="btn btn-secondary"
                style={{ padding: '2px 8px', fontSize: 11, marginLeft: 8 }}
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

      {/* Step 6 */}
      <StepCard num={6} title="Evaluate — Metriche di qualità" status={steps.evaluate}
        onRun={runEvaluate} running={running === 'evaluate'} runLabel="Valuta metriche">
        {metrics && (
          <>
            <div className="grid-3" style={{ gap: 12 }}>
              {Object.entries(METRIC_INFO).map(([key, { label, desc }]) => (
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
      </StepCard>
    </div>
  )
}

// ── MatchedView ───────────────────────────────────────────────────────────

function MatchedView({ matched, passages, retrieveMethod }) {
  const [open, setOpen] = useState({})
  const [debug, setDebug] = useState({})
  const [debugging, setDebugging] = useState(null)

  const supported = matched.filter(m => (m.supporting_passages || []).length > 0).length

  async function runDebug(claimText, claimIdx) {
    setDebugging(claimIdx)
    try {
      const result = await api.pipeline.retrieveDebug({
        claim: claimText,
        passages,
        method: retrieveMethod,
        top_k: 4,
      })
      setDebug(d => ({ ...d, [claimIdx]: result }))
    } catch (e) {
      alert(`Errore debug: ${e.message}`)
    }
    setDebugging(null)
  }

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
        <div className="metric-card" style={{ padding: '12px 20px', display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ fontSize: 28, fontWeight: 800, color: 'var(--green)' }}>
            {supported}/{matched.length}
          </span>
          <span style={{ fontSize: 12, color: 'var(--text-2)' }}>
            claims<br />supportati
          </span>
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
            fontSize: 13, fontWeight: 600, color: 'var(--text)',
            marginBottom: 6,
          }}>
            <Icon name="fileText" size={12} strokeWidth={1.75} color="var(--text-2)" />
            {p.title || 'Passage'}
          </div>
          {p.sentences.length === 0 ? (
            <div style={{ fontSize: 12, color: 'var(--text-3)', paddingLeft: 18 }}>
              Nessuna frase.
            </div>
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
                    color, minWidth: 52,
                    whiteSpace: 'nowrap',
                  }}>
                    [{score.toFixed(4)}]
                  </span>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{
                      height: 3, background: 'var(--border-2)',
                      borderRadius: 2, overflow: 'hidden', marginBottom: 4,
                      width: 90,
                    }}>
                      <div style={{ height: '100%', width: `${pct}%`, background: color }} />
                    </div>
                    <span style={{
                      fontSize: 12, color: isBest ? '#166534' : 'var(--text-2)',
                      fontWeight: isBest ? 500 : 400,
                      lineHeight: 1.5,
                    }}>
                      "{s.text}"
                    </span>
                  </div>
                  {isBest && (
                    <span style={{
                      fontSize: 10, fontWeight: 700,
                      background: '#DCFCE7', color: '#166534',
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

// ── CitedView (interattivo, invariato) ─────────────────────────────────────

const STOPWORDS = new Set([
  'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to',
  'for', 'of', 'and', 'or', 'but', 'with', 'as', 'his', 'her', 'their',
  'its', 'has', 'have', 'had', 'by', 'it', 'this', 'that', 'from', 'not',
  'be', 'been',
])

function tokenize(text) {
  return text.toLowerCase().replace(/[^\w\s]/g, '').split(/\s+/).filter(Boolean)
}

function lexicalOverlap(claimText, sentText) {
  const claimWords = new Set(tokenize(claimText).filter(w => !STOPWORDS.has(w)))
  const sentWords = new Set(tokenize(sentText).filter(w => !STOPWORDS.has(w)))
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
        <mark style={{
          background: '#FEF08A', padding: '1px 2px', borderRadius: 3,
          fontWeight: 600, color: '#713F12',
        }}>
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
        <mark style={{
          background: '#FEF08A', padding: '1px 2px', borderRadius: 3,
          fontWeight: 600, color: '#713F12',
        }}>
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
    if (activeSent === i) {
      setActiveSent(null)
      setActiveClaim(null)
    } else {
      setActiveSent(i)
      setActiveClaim(null)
    }
  }

  function onClaimClick(claimIdx) {
    setActiveClaim(activeClaim === claimIdx ? null : claimIdx)
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
                  background: isActive
                    ? '#BBF7D0'
                    : hasCitations
                      ? '#F0FDF4'
                      : 'transparent',
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
          marginTop: 14,
          background: 'var(--green-lt)',
          border: '1px solid #A7F3D0',
          borderRadius: 10,
          padding: '16px 18px',
          animation: 'fadeSlide 0.18s ease',
        }}>
          <div style={{
            fontSize: 10, fontWeight: 700, color: 'var(--text-3)',
            textTransform: 'uppercase', letterSpacing: '0.7px',
            marginBottom: 12,
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
                    <div
                      key={idx}
                      onClick={() => onClaimClick(idx)}
                      style={{
                        display: 'flex', alignItems: 'center', gap: 10,
                        padding: '10px 14px',
                        background: 'white',
                        border: '1px solid #A7F3D0',
                        borderRadius: 8,
                        cursor: 'pointer',
                        transition: 'all 0.12s',
                      }}
                      onMouseEnter={e => e.currentTarget.style.background = '#F0FDF4'}
                      onMouseLeave={e => e.currentTarget.style.background = 'white'}
                    >
                      <Icon name="search" size={13} color="var(--green)" strokeWidth={2} />
                      <span style={{ flex: 1, fontSize: 13, color: 'var(--text)' }}>
                        {matchedClaim.claim}
                      </span>
                      <span style={{
                        fontFamily: 'var(--mono)', fontSize: 11,
                        color: 'var(--text-3)', whiteSpace: 'nowrap',
                      }}>
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
              <div
                onClick={() => setActiveClaim(null)}
                style={{
                  display: 'inline-flex', alignItems: 'center', gap: 6,
                  fontSize: 12, color: 'var(--text-2)', cursor: 'pointer',
                  marginBottom: 12,
                }}
                onMouseEnter={e => e.currentTarget.style.color = 'var(--text)'}
                onMouseLeave={e => e.currentTarget.style.color = 'var(--text-2)'}
              >
                ← Torna ai claims
              </div>

              <div style={{
                display: 'flex', alignItems: 'center', gap: 8,
                fontSize: 14, fontWeight: 600, color: 'var(--text)',
                marginBottom: 14,
              }}>
                <Icon name="search" size={14} color="var(--green)" strokeWidth={2} />
                {associatedClaims[activeClaim].matchedClaim.claim}
              </div>

              {(associatedClaims[activeClaim].matchedClaim.supporting_passages || []).map((p, j) => {
                const refNum = references.find(r =>
                  r.title === p.title || r.text === p.text
                )?.citation_number

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
                      {highlightEvidence(
                        p.text || '',
                        p.extraction || p.best_sentence || '',
                        p.extraction_start,
                        p.extraction_end,
                      )}
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
              <span style={{
                fontFamily: 'var(--mono)', fontWeight: 700,
                color: 'var(--green)', marginRight: 6,
              }}>[{r.citation_number}]</span>
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
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  )
}