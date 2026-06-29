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
import { NuggetMetricsView, DeepSeekMetricsView, METRIC_INFO_DEEPSEEK } from '../components/MetricsViews'


// ── Metric definitions ────────────────────────────────────────────────────────

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

// Bottone "riesegui questo step" — mostrato solo se lo step ha un risultato.
  // Richiama la stessa runX(), che già fa resetAfter() cancellando i sottostanti.
  function RerunButton({ onRerun, show, busy }) {
    if (!show) return null
    return (
      <button
        className="btn btn-secondary"
        onClick={(e) => { e.stopPropagation(); onRerun() }}
        disabled={busy}
        style={{ padding: '4px 12px', fontSize: 11, display: 'flex', alignItems: 'center', gap: 6 }}
        title="Riesegui questo step e cancella i risultati degli step successivi"
      >
        <Icon name="refresh" size={11} strokeWidth={2} />
        {busy ? 'In corso…' : 'Riesegui'}
      </button>
    )
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
    const nuggets = Array.isArray(ex.nuggets) ? ex.nuggets : null

    return { question, docs, claims, raw_response, matched_claims, nuggets, _original: ex }
  })
}

// ── EvalMode Toggle ───────────────────────────────────────────────────────────

function EvalModeToggle({ mode, onChange, hasNuggets }) {
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
      <button
        onClick={() => onChange('nugget')}
        style={{
          ...baseBtn,
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

      <button
        onClick={() => onChange('deepseek')}
        style={{
          ...baseBtn,
          background: mode === 'deepseek' ? '#0EA5E9' : 'transparent',
          color: mode === 'deepseek' ? 'white' : 'var(--text-2)',
          boxShadow: mode === 'deepseek' ? '0 1px 4px rgba(14,165,233,0.3)' : 'none',
        }}
      >
        <Icon name="search" size={12} strokeWidth={2}
          color={mode === 'deepseek' ? 'white' : 'var(--text-3)'} />
        DeepSeek
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
            In alternativa, passa alla modalità <strong>Deepseek</strong>.
          </div>
        </div>
      </div>
    </div>
  )
}

// ── NuggetMetricsView ─────────────────────────────────────────────────────────


// ── Main Pipeline component ───────────────────────────────────────────────────

export default function Pipeline() {
  const { addPipelineResult } = useAppData()

  // Settings
  const [model, setModel] = useState('claude-haiku-4-5-20251001')
  const [retrieveMethod, setRetrieveMethod] = useState('nli')
  const [threshold, setThreshold] = useState(0.5)
  const [topK, setTopK] = useState(3)
  const [preFilterK, setPreFilterK] = useState(0)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [noiseEnabled, setNoiseEnabled] = useState(false)

  // Dataset
  const [dataset, setDataset] = useState(null)
  const [datasetName, setDatasetName] = useState('')
  const [exampleIdx, setExampleIdx] = useState(0)
  const fileRef = useRef()

  // Evaluation mode
  const [evalMode, setEvalMode] = useState('nugget') // 'standard' | 'nugget'

  // Nugget field validation error
  const [nuggetFieldError, setNuggetFieldError] = useState(null) // string[] | null

  // Pipeline state
  const [response, setResponse] = useState(null)
  const [nuggetCovering, setNuggetCovering] = useState(null)
  const [claims, setClaims] = useState(null)
  const [matched, setMatched] = useState(null)
  const [cited, setCited] = useState(null)
  const [references, setReferences] = useState(null)
  const [sentenceClaims, setSentenceClaims] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [nuggetMetrics, setNuggetMetrics] = useState(null)
  const [deepseekMetrics, setDeepseekMetrics] = useState(null)

  const [running, setRunning] = useState(null)
  const [error, setError] = useState(null)
  const [retrieveProgress, setRetrieveProgress] = useState({ current: 0, total: 0 })

  // Step status
  const steps = {
    query:     'active',
    generate:  response ? 'done' : dataset ? 'active' : 'locked',
    decompose: claims   ? 'done' : response ? 'active' : 'locked',
    retrieve:  matched  ? 'done' : claims   ? 'active' : 'locked',
    cite:      cited    ? 'done' : matched  ? 'active' : 'locked',
    evaluate: (metrics || nuggetMetrics || deepseekMetrics) ? 'done' : cited ? 'active' : 'locked',    
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

  const currentNuggets = currentExample?.nuggets || null
  const hasNuggets = !!currentNuggets

  const effectiveMode = evalMode === 'nugget' && !hasNuggets ? 'deepseek' : evalMode
 
  function resetAfter(step) {
    const order = ['generate', 'decompose', 'retrieve', 'cite', 'evaluate']
    const idx = order.indexOf(step)
    if (idx <= 0) setResponse(null)
    if (idx <= 1) setClaims(null)
    if (idx <= 2) { setMatched(null); setNuggetCovering(null) }
    if (idx <= 3) { setCited(null); setReferences(null); setSentenceClaims(null) }
    if (idx <= 4) { setMetrics(null); setNuggetMetrics(null); setDeepseekMetrics(null); setNuggetFieldError(null) }  
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
  setRetrieveProgress({ current: 0, total: claims.length })
  try {
    const allMatched = []
    const allDebug = []
    for (let i = 0; i < claims.length; i++) {
      setRetrieveProgress({ current: i + 1, total: claims.length })
      const res = await api.pipeline.retrieveSingle({
        claim: claims[i],
        passages: currentPassages,
        method: retrieveMethod,
        threshold,
        top_k: topK,
        nuggets: currentNuggets || undefined,
        pre_filter_k: preFilterK,
        model,
      })
      allMatched.push(res.matched)
      allDebug.push(res.debug)
    }
    setMatched(allMatched)
    // nugget_covering non viene dal retrieve singolo — viene calcolato all'evaluate
  } catch (e) { setError(`Retrieve: ${e.message}`) }
  setRetrieveProgress({ current: 0, total: 0 })
  setRunning(null)
}

async function runEvaluate() {
  setError(null)
  setNuggetFieldError(null)
  setRunning('evaluate')
  setMetrics(null); setNuggetMetrics(null); setDeepseekMetrics(null)

  try {
    if (effectiveMode === 'nugget') {
      const missing = validateNuggetFields()
      if (missing.length > 0) {
        setNuggetFieldError(missing)
        setRunning(null)
        return
      }
      const res = await api.pipeline.evaluateNuggets({
        matched_claims: matched,
        nuggets: currentNuggets,
        nugget_covering: nuggetCovering,  // null → backend ricalcola
      })
      setNuggetMetrics(res)

    } else if (effectiveMode === 'deepseek') {
      const res = await api.pipeline.evaluateDeepseek({ matched })
      setDeepseekMetrics(res.deepseek_metrics)
    } else {
      const res = await api.pipeline.evaluate({ matched })
      setMetrics(res)
    }
  } catch (e) {
    setError(`Evaluate: ${e.message}`)
  }
  setRunning(null)
}

  async function runCite() {
    setError(null); setRunning('cite'); resetAfter('cite')
    try {
      const res = await api.pipeline.cite({ response, matched })
      setCited(res.cited_response)
      setReferences(res.references)
      setSentenceClaims(res.sentence_claims || null)
    } catch (e) { setError(`Cite: ${e.message}`) }
    setRunning(null)
  }


  function saveToExplore() {
    addPipelineResult({
      question: currentQuery, raw_response: response, claims,
      matched_claims: matched, cited_response: cited, references, sentence_claims: sentenceClaims,
      metrics, nugget_metrics: nuggetMetrics, deepseek_metrics: deepseekMetrics,
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
      sentence_claims: sentenceClaims,
      metrics,
      nugget_metrics: nuggetMetrics,
      deepseek_metrics: deepseekMetrics,
      eval_mode: effectiveMode,
      model,
      retrieve_method: retrieveMethod,
      threshold,
      top_k: topK,
      pre_filter_k: preFilterK,
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
      <StepCard num={2}
        title={
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, width: '100%' }}>
            <span>Genera risposta</span>
            <RerunButton onRerun={runGenerate} show={!!response} busy={running === 'generate'} />
          </div>
        }
        status={steps.generate}
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
       <StepCard num={3}
        title={
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, width: '100%' }}>
            <span>Decompose — Atomic Claims</span>
            <RerunButton onRerun={runDecompose} show={!!claims} busy={running === 'decompose'} />
          </div>
        }
        status={steps.decompose}
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
      <StepCard num={4}
        title={
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, width: '100%' }}>
            <span>Retrieve — Matching claims → passaggi</span>
            <RerunButton onRerun={runRetrieve} show={!!matched} busy={running === 'retrieve'} />
          </div>
        }
        status={steps.retrieve}
        onRun={runRetrieve} running={running === 'retrieve'} runLabel="Retrieval">
        {running === 'retrieve' && retrieveProgress.total > 0 && (
          <div style={{ marginBottom: 16 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6, fontSize: 13, color: '#555' }}>
              <span>Claim {retrieveProgress.current} / {retrieveProgress.total}</span>
              <span>{Math.round((retrieveProgress.current / retrieveProgress.total) * 100)}%</span>
            </div>
            <div style={{ width: '100%', height: 8, background: '#e5e7eb', borderRadius: 4, overflow: 'hidden' }}>
              <div style={{
                width: `${(retrieveProgress.current / retrieveProgress.total) * 100}%`,
                height: '100%',
                background: 'linear-gradient(90deg, #10b981, #059669)',
                borderRadius: 4,
                transition: 'width 0.3s ease',
              }} />
            </div>
          </div>
        )}
        {matched && <MatchedView matched={matched} passages={currentPassages} retrieveMethod={retrieveMethod} nuggets={currentNuggets} />}
      </StepCard>

      {/* Step 5 */}
      <StepCard num={5}
        title={
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, width: '100%' }}>
            <span>Cite — Risposta con citazioni</span>
            <RerunButton onRerun={runCite} show={!!cited} busy={running === 'cite'} />
          </div>
        }
        status={steps.cite}
        onRun={runCite} running={running === 'cite'} runLabel="Inserisci citazioni">
        {cited && (
          <CitedView
            citedResponse={cited}
            references={references || []}
            matched={matched}
            sentenceClaims={sentenceClaims}
            retrieveMethod={retrieveMethod}
          />
        )}
      </StepCard>

          <StepCard
      num={6}
      title={
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap', justifyContent: 'space-between', width: '100%' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
            <span>Evaluate — Metriche di qualità</span>
            {steps.evaluate !== 'locked' && (
              <EvalModeToggle
                mode={evalMode}
                onChange={mode => { setEvalMode(mode); setNuggetMetrics(null); setDeepseekMetrics(null); setNuggetFieldError(null) }}
                hasNuggets={hasNuggets}
              />
            )}
          </div>
          <RerunButton
            onRerun={runEvaluate}
            show={!!(nuggetMetrics || deepseekMetrics)}
            busy={running === 'evaluate'}
          />
        </div>
      }
      status={steps.evaluate}
      onRun={runEvaluate}
      running={running === 'evaluate'}
      runLabel={`Valuta (${effectiveMode === 'nugget' ? 'Nugget' : 'DeepSeek'})`}
    >
      {/* Inline field-missing error for nugget mode */}
      {nuggetFieldError && (
        <NuggetMissingFieldsError missingFields={nuggetFieldError} />
      )}

      {/* Nugget metrics */}
      {nuggetMetrics && effectiveMode === 'nugget' && (
        <NuggetMetricsView
          metrics={nuggetMetrics}
          onSave={saveToExplore}
          onDownload={downloadPipelineData}
        />
      )}

      {/* DeepSeek metrics */}
      {deepseekMetrics && effectiveMode === 'deepseek' && (
        <DeepSeekMetricsView
          metrics={deepseekMetrics}
          onSave={saveToExplore}
          onDownload={downloadPipelineData}
        />
      )}
    </StepCard>
    </div>
    )
}

// ── MatchedView ───────────────────────────────────────────────────────────────

function MatchedView({ matched, passages, retrieveMethod, nuggets }) {
  const [open, setOpen] = useState({})
  const [debug, setDebug] = useState({})
  const [debugging, setDebugging] = useState(null)

  const supported = matched.filter(m => (m.supporting_passages || []).length > 0).length

  // ── Nugget-centered matching ──────────────────────────────────────────────
  function claimCoversNugget(claimObj, nugget) {
    if (claimObj.matched_nugget?.nugget_id === nugget.nugget_id) return true
    const claimText = (claimObj.claim || '').toLowerCase()
    const keywords = nugget.keywords || []
    return keywords.some(kw => claimText.includes(kw.toLowerCase()))
  }

  const hasNuggets = nuggets && Array.isArray(nuggets) && nuggets.length > 0

  // nugget_id → [{ m, i, score }] ordinati per match_score desc
  const nuggetToClaims = {}
  const claimIdxCovered = new Set()

  if (hasNuggets) {
    nuggets.forEach(nug => {
      const covering = []
      matched.forEach((m, i) => {
        if (claimCoversNugget(m, nug)) {
          const score = m.matched_nugget?.match_score || 0
          // Soglia allineata al backend (coverage_threshold = 0.6)
          if (score >= 0.6) {
            covering.push({ m, i, score })
          }
        }
      })
      covering.sort((a, b) => b.score - a.score)
      nuggetToClaims[nug.nugget_id] = covering
      covering.forEach(({ i }) => claimIdxCovered.add(i))
    })
  }

  const uncoveredClaims = matched
    .map((m, i) => ({ m, i }))
    .filter(({ i }) => !claimIdxCovered.has(i))

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

  // ── Render di un singolo claim ─────────────────────────────────────────
  function renderClaim(m, i, nugget) {
    const passages_m = m.supporting_passages || []
    const has = passages_m.length > 0
    const debugData = debug[i]
    const isGold = nugget?.required === true
    const isSilver = nugget?.required === false
    const borderColor = nugget
      ? (isGold ? '#D97706' : '#9CA3AF')
      : (has ? '#A7F3D0' : '#FECACA')
    const headerBg = isGold
      ? 'linear-gradient(90deg, #FFFBEB 0%, transparent 100%)'
      : isSilver
        ? 'linear-gradient(90deg, #F3F4F6 0%, transparent 100%)'
        : 'none'

    const matchScore = m.matched_nugget?.match_score

    return (
      <div key={i} className="expander" style={{
        borderColor,
        borderWidth: nugget ? '2px' : undefined,
        marginLeft: nugget ? 16 : 0,
      }}>
        <div className="expander-header"
          onClick={() => setOpen(o => ({ ...o, [i]: !o[i] }))}
          style={{ background: headerBg }}>
          <span className={`badge ${has ? 'badge-green' : 'badge-red'}`}>
            {has
              ? <Icon name="check" size={10} strokeWidth={2.5} />
              : <Icon name="x" size={10} strokeWidth={2.5} />}
          </span>
          <span className="expander-header-title" style={{ color: 'var(--text)' }}>
            {m.claim}
          </span>
          {matchScore != null && nugget && (
            <span style={{
              fontSize: 11, fontWeight: 600,
              padding: '1px 6px', borderRadius: 8,
              background: matchScore >= 0.5 ? '#ECFDF5' : '#FEF2F2',
              color: matchScore >= 0.5 ? '#166534' : '#991B1B',
              border: `1px solid ${matchScore >= 0.5 ? '#86EFAC' : '#FECACA'}`,
              marginRight: 6,
            }}>
              {matchScore.toFixed(2)}
            </span>
          )}
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
                  {retrieveMethod !== 'llm' && p.entailment_score != null && <ScorePill score={p.entailment_score} />}
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
              !debugData && (
                <span style={{ color: 'var(--text-3)', fontSize: 13 }}>
                  Nessun passaggio di supporto trovato.
                </span>
              )
            )}
          </div>
        )}
      </div>
    )
  }

  // ── Render nugget header ──────────────────────────────────────────────
  function renderNuggetHeader(nug) {
    const isGold = nug.required === true
    const claims = nuggetToClaims[nug.nugget_id] || []
    const covered = claims.length > 0

    return (
      <div key={nug.nugget_id} style={{ marginBottom: 16 }}>
        <div style={{
          display: 'flex', alignItems: 'flex-start', gap: 10,
          padding: '10px 14px', borderRadius: 8, marginBottom: 6,
          background: isGold ? '#FFFBEB' : '#F9FAFB',
          border: `1.5px solid ${isGold ? '#FDE68A' : '#E5E7EB'}`,
          opacity: covered ? 1 : 0.6,
        }}>
          <span style={{
            display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
            width: 22, height: 22, borderRadius: '50%', flexShrink: 0, marginTop: 1,
            background: isGold
              ? 'linear-gradient(135deg, #F59E0B, #D97706)'
              : 'linear-gradient(135deg, #D1D5DB, #9CA3AF)',
            border: `1.5px solid ${isGold ? '#B45309' : '#6B7280'}`,
            fontSize: 11, fontWeight: 800,
            color: isGold ? '#FFFBEB' : '#374151',
          }}>
            {isGold ? '★' : '☆'}
          </span>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 3 }}>
              <span style={{
                fontSize: 11, fontWeight: 700,
                color: isGold ? '#92400E' : '#6B7280',
              }}>
                {nug.nugget_id}
              </span>
              <span style={{
                fontSize: 9, fontWeight: 600, padding: '1px 6px', borderRadius: 8,
                background: isGold ? '#FEF3C7' : '#F3F4F6',
                color: isGold ? '#B45309' : '#9CA3AF',
              }}>
                {isGold ? 'REQUIRED' : 'OPTIONAL'}
              </span>
              {covered
                ? <span style={{
                    fontSize: 9, fontWeight: 600, padding: '1px 6px', borderRadius: 8,
                    background: '#ECFDF5', color: '#166534', border: '1px solid #86EFAC',
                  }}>
                    {claims.length} claim{claims.length > 1 ? 's' : ''}
                  </span>
                : <span style={{
                    fontSize: 9, fontWeight: 600, padding: '1px 6px', borderRadius: 8,
                    background: '#FEF2F2', color: '#991B1B', border: '1px solid #FECACA',
                  }}>
                    non coperto
                  </span>
              }
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-2)', lineHeight: 1.5 }}>
              {nug.text}
            </div>
            <div style={{ marginTop: 4, fontSize: 11, color: 'var(--text-3)' }}>
              Keywords: {(nug.keywords || []).map((kw, ki) => (
                <span key={ki} style={{
                  display: 'inline-block', padding: '1px 6px', margin: '0 3px',
                  background: '#F3F4F6', borderRadius: 4,
                  fontFamily: 'var(--mono)', fontSize: 10,
                }}>
                  {kw}
                </span>
              ))}
            </div>
          </div>
        </div>

        {claims.map(({ m, i }) => renderClaim(m, i, nug))}
      </div>
    )
  }

  return (
    <div>
      {/* Header metrica */}
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

      {/* Legend */}
      {hasNuggets && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 16, marginBottom: 14,
          padding: '8px 14px', background: '#FFFBEB', border: '1px solid #FDE68A',
          borderRadius: 8, fontSize: 12, color: '#92400E',
        }}>
          <span style={{ fontWeight: 600 }}>Nugget matching:</span>
          <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <span style={{
              display: 'inline-block', width: 10, height: 10, borderRadius: '50%',
              background: 'linear-gradient(135deg, #F59E0B, #D97706)',
              border: '1.5px solid #B45309',
            }} />
            Gold (required)
          </span>
          <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <span style={{
              display: 'inline-block', width: 10, height: 10, borderRadius: '50%',
              background: 'linear-gradient(135deg, #D1D5DB, #9CA3AF)',
              border: '1.5px solid #6B7280',
            }} />
            Silver (optional)
          </span>
        </div>
      )}

      {/* Nugget-centered view */}
      {hasNuggets && nuggets.map(nug => renderNuggetHeader(nug))}

      {/* Claim non matchati a nessun nugget */}
      {uncoveredClaims.length > 0 && (
        <div style={{ marginTop: hasNuggets ? 24 : 0 }}>
          {hasNuggets && (
            <div style={{
              fontSize: 11, fontWeight: 600, color: 'var(--text-3)',
              textTransform: 'uppercase', letterSpacing: '0.05em',
              marginBottom: 8, paddingLeft: 2,
            }}>
              Claims senza nugget ({uncoveredClaims.length})
            </div>
          )}
          {uncoveredClaims.map(({ m, i }) => renderClaim(m, i, null))}
        </div>
      )}
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

function findAssociatedClaims(sentenceText, matchedClaims, threshold = 0.6) {
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

function CitedView({ citedResponse, references, matched, sentenceClaims, retrieveMethod }) {
  const [activeSent, setActiveSent] = useState(null)
  const [activeClaim, setActiveClaim] = useState(null)

  // SINGLE SOURCE OF TRUTH: l'allineamento claim -> frase arriva dal backend
  // (containment pesato IDF in core/cite.py). Lo split/overlap client-side
  // resta SOLO come fallback per run salvati prima di questa modifica.
  const fromBackend = Array.isArray(sentenceClaims) && sentenceClaims.length > 0
  const sentences = fromBackend
    ? sentenceClaims.map(sc => ({
        text: sc.sentence,
        citations: sc.citations || [],
        claims: sc.claims || [],
      }))
    : splitIntoSentences(citedResponse)

  function onSentenceClick(i) {
    if (activeSent === i) { setActiveSent(null); setActiveClaim(null) }
    else { setActiveSent(i); setActiveClaim(null) }
  }

  const associatedClaims = activeSent != null
    ? (fromBackend
        ? sentences[activeSent].claims
            .map(c => ({
              matchedClaim: (matched || []).find(mc => mc.claim === c.claim),
              overlap: c.alignment_score,
            }))
            .filter(x => x.matchedClaim)
        : findAssociatedClaims(sentences[activeSent].text, matched || []))
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
                Nessun claim allineato a questa frase.
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
                        align {overlap.toFixed(2)}
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
                        {retrieveMethod !== 'llm' && p.entailment_score != null && <ScorePill score={p.entailment_score} />}
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