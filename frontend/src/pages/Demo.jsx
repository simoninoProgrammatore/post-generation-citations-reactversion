/**
 * Demo.jsx — Pagina "production-like": query → risposta con citazioni.
 *
 * Nasconde i 6 step del pipeline interattivo. L'utente carica un dataset,
 * sceglie una domanda e ottiene direttamente la risposta citata. Cliccando
 * una frase evidenziata si aprono le references (gli span di supporto),
 * SENZA il livello claim intermedio.
 *
 * Orchestrazione: concatena gli endpoint esistenti dal frontend
 *   generate → decompose → retrieve → cite
 * (Strada A: nessun nuovo endpoint backend).
 */

import { useState, useRef } from 'react'
import api from '../api'
import Icon from '../components/Icon'
import ScorePill from '../components/ScorePill'

// ── Dataset helpers (riusati da Pipeline/EvaluateDataset) ──────────────────────

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
  const nMax = Math.max(1, Math.ceil(docs.length * 0.5))
  const nNoise = 1 + Math.floor(rng() * nMax)
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

// ── Parsing risposta citata (frase → numeri citazione) ─────────────────────────

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

// Evidenzia lo span (best_sentence/extraction) dentro il testo del passaggio
function highlightEvidence(passageText, extraction) {
  if (!passageText) return null
  if (!extraction) return <span>{passageText}</span>
  const idx = passageText.toLowerCase().indexOf(extraction.toLowerCase())
  if (idx < 0) return <span>{passageText}</span>
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

// ── Output: risposta citata, frase cliccabile → references ─────────────────────

function AnswerView({ citedResponse, references, matched }) {
  const [activeSent, setActiveSent] = useState(null)
  const sentences = splitIntoSentences(citedResponse)
  const byNum = Object.fromEntries((references || []).map(r => [r.citation_number, r]))

  // Per arricchire la reference con lo span evidenziato, cerco nel matched il
  // passaggio con lo stesso title/text del reference (best-effort).
  const allPassages = (matched || []).flatMap(m => m.supporting_passages || [])
  function spanForRef(ref) {
    const p = allPassages.find(p => (p.title && p.title === ref.title) || (p.text && p.text === ref.text))
    return p ? (p.extraction || p.best_sentence || '') : ''
  }
  function entailmentForRef(ref) {
    const p = allPassages.find(p => (p.title && p.title === ref.title) || (p.text && p.text === ref.text))
    return p ? p.entailment_score : null
  }

  function onSentenceClick(i, hasCit) {
    if (!hasCit) return
    setActiveSent(activeSent === i ? null : i)
  }

  const activeRefs = activeSent != null
    ? sentences[activeSent].citations.map(n => byNum[n]).filter(Boolean)
    : []

  return (
    <div>
      {/* Risposta */}
      <div style={{
        background: 'white', border: '1px solid var(--border)',
        borderLeft: '3px solid var(--accent)', borderRadius: 10,
        padding: '20px 24px', fontSize: 15, lineHeight: 2.0, color: 'var(--text)',
      }}>
        {sentences.map((sent, i) => {
          const hasCit = sent.citations.length > 0
          const isActive = activeSent === i
          return (
            <span key={i}>
              <span
                onClick={() => onSentenceClick(i, hasCit)}
                style={{
                  background: isActive ? '#BBF7D0' : hasCit ? '#F0FDF4' : 'transparent',
                  padding: hasCit ? '2px 4px' : '0',
                  borderRadius: 4,
                  cursor: hasCit ? 'pointer' : 'default',
                  borderBottom: hasCit ? '2px solid #86EFAC' : 'none',
                  transition: 'background 0.15s',
                }}
                title={hasCit ? 'Clicca per vedere le fonti' : ''}
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

      {/* Pannello references della frase attiva */}
      {activeSent != null && activeRefs.length > 0 && (
        <div style={{
          marginTop: 14, background: '#F0FDF4',
          border: '1px solid #A7F3D0', borderRadius: 10,
          padding: '16px 18px', animation: 'fadeSlide 0.18s ease',
        }}>
          <div style={{
            fontSize: 10, fontWeight: 700, color: 'var(--text-3)',
            textTransform: 'uppercase', letterSpacing: '0.7px', marginBottom: 12,
          }}>
            Fonti per la frase selezionata ({activeRefs.length})
          </div>
          {activeRefs.map((ref, k) => {
            const span = spanForRef(ref)
            const ent = entailmentForRef(ref)
            return (
              <div key={k} className="passage-card" style={{ marginBottom: 10 }}>
                <div className="passage-header">
                  <span className="passage-title">
                    <span style={{
                      background: '#0F172A', color: 'white', fontFamily: 'var(--mono)',
                      fontSize: 11, fontWeight: 700, padding: '2px 8px', borderRadius: 4, marginRight: 8,
                    }}>[{ref.citation_number}]</span>
                    {ref.title || '—'}
                  </span>
                  {ent != null && <ScorePill score={ent} />}
                </div>
                <div className="passage-body">
                  {highlightEvidence(ref.text || '', span)}
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* Lista riferimenti completa */}
      {references && references.length > 0 && (
        <div style={{ marginTop: 18, paddingTop: 14, borderTop: '1px solid var(--border)' }}>
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

// ── Settings panel (compatto) ──────────────────────────────────────────────────

function SettingsPanel({ open, setOpen, model, setModel, retrieveMethod, setRetrieveMethod,
  threshold, setThreshold, topK, setTopK, preFilterK, setPreFilterK,
  noiseEnabled, setNoiseEnabled }) {
  return (
    <div className="card" style={{ marginBottom: 16 }}>
      <div
        style={{ display: 'flex', alignItems: 'center', padding: '12px 20px', cursor: 'pointer', gap: 8 }}
        onClick={() => setOpen(o => !o)}
      >
        <Icon name="settings" size={14} strokeWidth={1.75} color="var(--text-2)" />
        <span style={{ fontSize: 13, fontWeight: 600, flex: 1 }}>Impostazioni</span>
        <span style={{ fontSize: 11, color: 'var(--text-3)', fontFamily: 'var(--mono)' }}>
          {model} · {retrieveMethod} · top-{topK}{noiseEnabled ? ' · noise' : ''}
        </span>
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
            </div>
            <div className="form-group" style={{ marginBottom: 0, display: 'flex', flexDirection: 'column', gap: 8 }}>
              <label className="form-label">Noise injection</label>
              <button className="btn" onClick={() => setNoiseEnabled(n => !n)}
                style={{
                  padding: '6px 14px', fontSize: 12, alignSelf: 'flex-start',
                  background: noiseEnabled ? '#DCFCE7' : '#FEE2E2',
                  color: noiseEnabled ? '#166534' : '#991B1B',
                  border: `1px solid ${noiseEnabled ? '#BBF7D0' : '#FECACA'}`,
                  borderRadius: 6,
                }}>
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

// ── Main ─────────────────────────────────────────────────────────────────────

const STAGES = [
  { key: 'generate',  label: 'Generazione risposta' },
  { key: 'decompose', label: 'Scomposizione in claim' },
  { key: 'retrieve',  label: 'Attribuzione fonti' },
  { key: 'cite',      label: 'Inserimento citazioni' },
]

export default function Demo() {
  // Settings
  const [model, setModel] = useState('claude-haiku-4-5-20251001')
  const [retrieveMethod, setRetrieveMethod] = useState('nli')
  const [threshold, setThreshold] = useState(0.5)
  const [topK, setTopK] = useState(3)
  const [preFilterK, setPreFilterK] = useState(0)
  const [noiseEnabled, setNoiseEnabled] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)

  // Dataset
  const [dataset, setDataset] = useState(null)
  const [datasetName, setDatasetName] = useState('')
  const [exampleIdx, setExampleIdx] = useState(0)
  const fileRef = useRef()

  // Run state
  const [running, setRunning] = useState(false)
  const [stage, setStage] = useState(null)   // chiave dello stage corrente
  const [error, setError] = useState(null)

  // Output
  const [cited, setCited] = useState(null)
  const [references, setReferences] = useState(null)
  const [matched, setMatched] = useState(null)

  const currentExample = dataset ? dataset[exampleIdx] : null
  const currentQuery   = currentExample?.question || ''
  const rawPassages    = currentExample?.docs || []

  const currentPassages = (() => {
    if (!noiseEnabled || !dataset || !currentExample) return rawPassages
    const pool = buildNoisePool(dataset, exampleIdx)
    if (!pool.length) return rawPassages
    return injectNoise(rawPassages, pool, 42 + exampleIdx)
  })()

  function resetOutput() {
    setCited(null); setReferences(null); setMatched(null); setError(null)
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
        setExampleIdx(0)
        resetOutput()
      } catch (err) {
        setError(`Errore lettura file: ${err.message}`)
      }
    }
    reader.readAsText(file)
  }

  async function run() {
    if (!currentExample) return
    resetOutput()
    setRunning(true)
    try {
      // 1. Generate
      setStage('generate')
      const gen = await api.pipeline.generate({ query: currentQuery, passages: currentPassages, model })
      const responseText = gen.response

      // 2. Decompose
      setStage('decompose')
      const dec = await api.pipeline.decompose({ text: responseText, model })

      // 3. Retrieve (async backend, una sola chiamata con tutti i claim)
      setStage('retrieve')
      const ret = await api.pipeline.retrieve({
        claims: dec.claims,
        passages: currentPassages,
        method: retrieveMethod,
        threshold,
        top_k: topK,
        nuggets: undefined,
        pre_filter_k: preFilterK,
        model,
      })

      // 4. Cite
      setStage('cite')
      const cit = await api.pipeline.cite({ response: responseText, matched: ret.matched })

      setMatched(ret.matched)
      setCited(cit.cited_response)
      setReferences(cit.references)
    } catch (e) {
      setError(`${stage || 'pipeline'}: ${e.message}`)
    }
    setStage(null)
    setRunning(false)
  }

  return (
    <div>
      <div className="page-header">
        <div className="page-header-title" style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            width: 32, height: 32,
            background: 'linear-gradient(135deg, #059669 0%, #0EA5E9 100%)',
            borderRadius: 8, display: 'flex', alignItems: 'center', justifyContent: 'center',
            flexShrink: 0, boxShadow: '0 2px 8px rgba(5,150,105,0.3)',
          }}>
            <Icon name="zap" size={16} color="white" strokeWidth={1.75} />
          </div>
          Demo
        </div>
        <div className="page-header-sub">
          Scegli una domanda dal dataset e ottieni la risposta con citazioni. Clicca una frase per vedere le fonti.
        </div>
      </div>

      {error && (
        <div className="info-box info-box-red" style={{ marginBottom: 16 }}>
          <Icon name="xCircle" size={15} strokeWidth={1.75} style={{ flexShrink: 0, marginTop: 1 }} />
          <span><strong>Errore:</strong> {error}</span>
        </div>
      )}

      <SettingsPanel
        open={settingsOpen} setOpen={setSettingsOpen}
        model={model} setModel={setModel}
        retrieveMethod={retrieveMethod} setRetrieveMethod={setRetrieveMethod}
        threshold={threshold} setThreshold={setThreshold}
        topK={topK} setTopK={setTopK}
        preFilterK={preFilterK} setPreFilterK={setPreFilterK}
        noiseEnabled={noiseEnabled} setNoiseEnabled={setNoiseEnabled}
      />

      {/* Input card */}
      <div className="card" style={{ marginBottom: 20 }}>
        <div style={{ padding: '16px 20px' }}>
          <input ref={fileRef} type="file" accept=".json,.jsonl" onChange={onFileUpload} style={{ display: 'none' }} />

          {!dataset ? (
            <div style={{ textAlign: 'center', padding: '24px 0' }}>
              <Icon name="database" size={32} strokeWidth={1.25} color="var(--border)" />
              <div style={{ marginTop: 12, fontSize: 14, fontWeight: 600, color: 'var(--text-2)' }}>
                Carica un dataset per iniziare
              </div>
              <div style={{ marginTop: 4, fontSize: 12, color: 'var(--text-3)', marginBottom: 16 }}>
                ALCE / ELI5 / QAMPARI (JSON)
              </div>
              <button className="btn btn-primary" onClick={() => fileRef.current.click()}>
                <Icon name="upload" size={14} strokeWidth={1.75} color="white" />
                Seleziona file JSON
              </button>
            </div>
          ) : (
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12, flexWrap: 'wrap' }}>
                <Icon name="fileText" size={14} strokeWidth={1.75} color="var(--accent)" />
                <span style={{ fontSize: 13, fontWeight: 600 }}>{datasetName}</span>
                <span style={{
                  fontSize: 11, fontWeight: 700, background: 'var(--bg)', border: '1px solid var(--border)',
                  padding: '1px 8px', borderRadius: 10, color: 'var(--text-2)',
                }}>
                  {dataset.length} esempi
                </span>
                <button className="btn btn-secondary" style={{ padding: '2px 8px', fontSize: 11, marginLeft: 'auto' }}
                  onClick={() => { setDataset(null); setDatasetName(''); resetOutput() }}>
                  Cambia dataset
                </button>
              </div>

              <div className="form-group" style={{ marginBottom: 12 }}>
                <label className="form-label">Domanda</label>
                <select className="input" value={exampleIdx}
                  onChange={e => { setExampleIdx(+e.target.value); resetOutput() }}>
                  {dataset.map((ex, i) => <option key={i} value={i}>[{i}] {ex.question}</option>)}
                </select>
              </div>

              <div style={{
                fontSize: 12, color: 'var(--text-3)', marginBottom: 14,
                display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap',
              }}>
                {noiseEnabled
                  ? <>{currentPassages.filter(d => !d.is_noise).length} passaggi + {currentPassages.filter(d => d.is_noise).length} noise</>
                  : <>{currentPassages.length} passaggi disponibili</>}
              </div>

              <button
                className="btn"
                onClick={run}
                disabled={running}
                style={{
                  background: 'linear-gradient(135deg, #059669, #0EA5E9)', color: 'white', border: 'none',
                  padding: '10px 22px', fontWeight: 700, fontSize: 13, borderRadius: 8,
                  cursor: running ? 'not-allowed' : 'pointer', opacity: running ? 0.8 : 1,
                  display: 'flex', alignItems: 'center', gap: 8,
                }}
              >
                {running
                  ? <><span className="spinner" style={{ width: 14, height: 14, borderColor: 'white', borderTopColor: 'transparent' }} /> Elaborazione…</>
                  : <><Icon name="zap" size={14} color="white" strokeWidth={2} /> Genera risposta</>}
              </button>

              {/* Progress per stage — senza spinner animato (pallino statico) */}
              {running && (
                <div style={{ marginTop: 14, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  {STAGES.map(s => {
                    const isCurrent = stage === s.key
                    const order = STAGES.findIndex(x => x.key === s.key)
                    const curOrder = STAGES.findIndex(x => x.key === stage)
                    const done = curOrder > order
                    const dotColor = done ? '#059669' : isCurrent ? '#D97706' : 'var(--text-3)'
                    return (
                      <span key={s.key} style={{
                        fontSize: 11, fontWeight: 600, padding: '4px 10px', borderRadius: 8,
                        display: 'flex', alignItems: 'center', gap: 6,
                        background: isCurrent ? '#FFFBEB' : done ? '#F0FDF4' : 'var(--bg)',
                        color: isCurrent ? '#92400E' : done ? '#059669' : 'var(--text-3)',
                        border: `1px solid ${isCurrent ? '#FDE68A' : done ? '#A7F3D0' : 'var(--border)'}`,
                      }}>
                        {done ? (
                          <Icon name="check" size={10} strokeWidth={2.5} color="#059669" />
                        ) : (
                          <span style={{
                            width: 8, height: 8, borderRadius: '50%',
                            background: dotColor, display: 'inline-block', flexShrink: 0,
                          }} />
                        )}
                        {s.label}
                      </span>
                    )
                  })}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Output */}
      {cited && (
        <div className="card">
          <div style={{ padding: '20px 24px' }}>
            <div style={{
              fontSize: 11, fontWeight: 700, color: 'var(--text-3)',
              textTransform: 'uppercase', letterSpacing: '0.6px', marginBottom: 6,
            }}>
              Domanda
            </div>
            <div style={{
              fontSize: 14, fontWeight: 600, color: 'var(--text-2)',
              marginBottom: 18, paddingBottom: 14, borderBottom: '1px solid var(--border)',
            }}>
              {currentQuery}
            </div>
            <AnswerView citedResponse={cited} references={references || []} matched={matched} />
          </div>
        </div>
      )}
    </div>
  )
}