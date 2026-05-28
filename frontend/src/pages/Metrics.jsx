/**
 * Metrics.jsx — Metriche aggregate su tutti i pipelineResults salvati.
 *
 * Aggrega le due famiglie effettivamente usate dall'app: Nugget e DeepSeek.
 * Un toggle in alto sceglie quale famiglia visualizzare.
 *
 * AGGREGAZIONE: pooled / micro (allineata al PDF "Metriche di Valutazione").
 *   - Nugget   (PDF §1.5–1.6): numeratore = Σ_esempi Σ_g precision(g),
 *       precision = numeratore / Σ n_covered,  recall = numeratore / Σ n_total,
 *       coverage  = Σ n_covered / Σ n_total.
 *       (Σ_g precision(g) per esempio = nugget_recall · n_total — identità esatta.)
 *   - DeepSeek (PDF §2.2–2.4): precision = (Σ n_full + 0.5·Σ n_partial) / Σ coppie,
 *       recall = Σ (claim supportati) / Σ claim,  distribuzione = Σ conteggi / Σ coppie.
 *
 * NB: il pooling pesa ogni esempio per la sua dimensione (numero di coppie/nugget),
 * a differenza della media-degli-esempi. È la scelta coerente con le formule del PDF.
 */

import { useState, useRef } from 'react'
import { useAppData } from '../context/AppData'
import EmptyState from '../components/EmptyState'
import { downloadJSON, timestampedFilename } from '../utils/download'
import Icon from '../components/Icon'

const NUGGET_LABELS = {
  nugget_precision: 'Nugget Precision',
  nugget_recall:    'Nugget Recall',
  nugget_coverage:  'Nugget Coverage',
}

const DEEPSEEK_LABELS = {
  citation_precision: 'Citation Precision',
  citation_recall:    'Citation Recall',
}

const pct = v => `${Math.round((v || 0) * 100)}%`

const nuggetColor   = v => v >= 0.6 ? 'var(--green)' : v >= 0.3 ? 'var(--amber)' : 'var(--red)'
const deepseekColor = v => v >= 0.7 ? 'var(--green)' : v >= 0.4 ? 'var(--amber)' : 'var(--red)'

// ── Aggregazione pooled ────────────────────────────────────────────────────────

function aggregateNugget(examples) {
  let sumScore = 0, sumCovered = 0, sumTotal = 0
  for (const ex of examples) {
    const m = ex.nugget_metrics
    const nTotal   = m.n_nuggets ?? 0
    const nCovered = m.n_covered ?? 0
    const recall   = m.nugget_recall ?? 0
    sumScore   += recall * nTotal   // = Σ_g precision(g) dell'esempio
    sumCovered += nCovered
    sumTotal   += nTotal
  }
  return {
    nugget_precision: sumCovered > 0 ? sumScore / sumCovered : 0,
    nugget_recall:    sumTotal   > 0 ? sumScore / sumTotal   : 0,
    nugget_coverage:  sumTotal   > 0 ? sumCovered / sumTotal : 0,
    n_examples: examples.length,
    n_covered: sumCovered,
    n_total: sumTotal,
  }
}

function aggregateDeepseek(examples) {
  let sumFull = 0, sumPartial = 0, sumNone = 0, sumPairs = 0
  let sumSupportedClaims = 0, sumClaims = 0
  for (const ex of examples) {
    const m = ex.deepseek_metrics
    const nPairs   = m.n_pairs ?? 0
    const nFull    = m.n_full ?? m.n_pairs_supported ?? 0   // fallback JSON binari vecchi
    const nPartial = m.n_partial ?? 0
    const nNone    = m.n_none ?? Math.max(0, nPairs - nFull - nPartial)
    const nClaims  = m.n_claims ?? 0
    const recall   = m.citation_recall ?? 0
    sumFull += nFull; sumPartial += nPartial; sumNone += nNone; sumPairs += nPairs
    sumClaims += nClaims
    sumSupportedClaims += recall * nClaims   // = numero di claim supportati dell'esempio
  }
  return {
    citation_precision: sumPairs  > 0 ? (sumFull + 0.5 * sumPartial) / sumPairs : 0,
    citation_recall:    sumClaims > 0 ? sumSupportedClaims / sumClaims : 0,
    pct_full:    sumPairs > 0 ? sumFull    / sumPairs : 0,
    pct_partial: sumPairs > 0 ? sumPartial / sumPairs : 0,
    pct_none:    sumPairs > 0 ? sumNone    / sumPairs : 0,
    n_full: sumFull, n_partial: sumPartial, n_none: sumNone,
    n_pairs: sumPairs, n_claims: sumClaims, n_examples: examples.length,
  }
}

// ── Parser del file di metriche caricato ──────────────────────────────────────
// Accetta: l'export di questa pagina ({nugget:{per_example}, deepseek:{per_example}}),
// un singolo risultato pipeline, o un array di risultati. Tollerante ai formati.
function parseLoadedMetrics(data) {
  const nugget = [], deepseek = []
  // Forma esportata da questa pagina
  if (data && (data.nugget || data.deepseek) && !Array.isArray(data)) {
    if (data.nugget?.per_example) for (const e of data.nugget.per_example)
      if (e?.metrics) nugget.push({ question: e.question, nugget_metrics: e.metrics })
    if (data.deepseek?.per_example) for (const e of data.deepseek.per_example)
      if (e?.metrics) deepseek.push({ question: e.question, deepseek_metrics: e.metrics })
    if (nugget.length || deepseek.length) return { nugget, deepseek }
  }
  // Singolo risultato pipeline o array
  const arr = Array.isArray(data) ? data : [data]
  for (const r of arr) {
    if (!r || typeof r !== 'object') continue
    if (r.nugget_metrics && Object.keys(r.nugget_metrics).length)
      nugget.push({ question: r.question, nugget_metrics: r.nugget_metrics })
    if (r.deepseek_metrics && Object.keys(r.deepseek_metrics).length)
      deepseek.push({ question: r.question, deepseek_metrics: r.deepseek_metrics })
  }
  return { nugget, deepseek }
}

// ── Card aggregata ──────────────────────────────────────────────────────────────

function AggCard({ label, value, color, sub, hint }) {
  const bd = color === 'var(--green)' ? '#A7F3D0' : color === 'var(--amber)' ? '#FDE68A' : '#FECACA'
  return (
    <div style={{
      padding: '16px 18px', background: 'white',
      border: `1px solid ${bd}`, borderTop: `3px solid ${color}`, borderRadius: 10,
    }}>
      <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.6px', marginBottom: 8 }}>
        {label}
      </div>
      <div style={{ fontSize: 30, fontWeight: 800, color, lineHeight: 1, marginBottom: 8 }}>
        {pct(value)}
      </div>
      {sub && <div style={{ fontSize: 11, color: 'var(--text-3)', lineHeight: 1.4 }}>{sub}</div>}
      {hint && <div style={{ marginTop: 6, fontSize: 11, color: 'var(--text-3)', fontStyle: 'italic' }}>{hint}</div>}
    </div>
  )
}

// ── Bar chart + tabella per esempio ──────────────────────────────────────────────

function PerExampleChart({ examples, metricKey, valueOf, colorOf }) {
  return (
    <div style={{ marginBottom: 20 }}>
      {examples.map((ex, i) => {
        const val = valueOf(ex, metricKey)
        const color = colorOf(val)
        return (
          <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 10 }}>
            <div style={{
              fontSize: 12, color: 'var(--text-2)', minWidth: 220,
              overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
            }} title={ex.question}>
              {(ex.question || '').slice(0, 35)}{(ex.question || '').length > 35 ? '…' : ''}
            </div>
            <div style={{ flex: 1, height: 8, background: 'var(--border-2)', borderRadius: 4, overflow: 'hidden' }}>
              <div style={{ height: '100%', width: `${val * 100}%`, background: color, borderRadius: 4, transition: 'width 0.8s ease' }} />
            </div>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 12, fontWeight: 600, color, minWidth: 44, textAlign: 'right' }}>
              {val.toFixed(3)}
            </div>
          </div>
        )
      })}
    </div>
  )
}

// ── Vista Nugget ──────────────────────────────────────────────────────────────

function NuggetView({ examples }) {
  const [metric, setMetric] = useState('nugget_precision')
  const agg = aggregateNugget(examples)

  const valueOf = (ex, key) => ex.nugget_metrics[key] ?? 0

  return (
    <div>
      <SectionTitle>Aggregato pooled · {agg.n_examples} esempi</SectionTitle>
      <div className="grid-3" style={{ marginBottom: 28 }}>
        <AggCard label="Nugget Precision" value={agg.nugget_precision} color={nuggetColor(agg.nugget_precision)}
          sub={`Σ score / ${agg.n_covered} nugget coperti`}
          hint="Fedeltà media dell'evidenza sui nugget coperti." />
        <AggCard label="Nugget Recall" value={agg.nugget_recall} color={nuggetColor(agg.nugget_recall)}
          sub={`Σ score / ${agg.n_total} nugget totali`}
          hint="Stessa somma, diluita su tutti i nugget." />
        <AggCard label="Nugget Coverage" value={agg.nugget_coverage} color={nuggetColor(agg.nugget_coverage)}
          sub={`${agg.n_covered} coperti / ${agg.n_total} totali`}
          hint="Nugget toccati da almeno un claim." />
      </div>

      <div className="divider" />

      <PerExampleSection
        label="Dettaglio per esempio"
        metric={metric} setMetric={setMetric} labels={NUGGET_LABELS}
      >
        <PerExampleChart examples={examples} metricKey={metric} valueOf={valueOf} colorOf={nuggetColor} />
        <PerExampleTable
          examples={examples} keys={Object.keys(NUGGET_LABELS)} labels={NUGGET_LABELS}
          valueOf={valueOf} colorOf={nuggetColor}
        />
      </PerExampleSection>
    </div>
  )
}

// ── Vista DeepSeek ──────────────────────────────────────────────────────────────

function DeepSeekView({ examples }) {
  const [metric, setMetric] = useState('citation_precision')
  const agg = aggregateDeepseek(examples)

  const valueOf = (ex, key) => ex.deepseek_metrics[key] ?? 0

  return (
    <div>
      <SectionTitle>Aggregato pooled · {agg.n_examples} esempi · {agg.n_pairs} coppie</SectionTitle>

      {/* Precision + Recall */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12, marginBottom: 12 }}>
        <AggCard label="Citation Precision" value={agg.citation_precision} color={deepseekColor(agg.citation_precision)}
          sub={`(${agg.n_full} full + 0.5·${agg.n_partial} partial) / ${agg.n_pairs} coppie`}
          hint="Precision pesata: il partial vale mezzo punto." />
        <AggCard label="Citation Recall" value={agg.citation_recall} color={deepseekColor(agg.citation_recall)}
          sub={`claim con ≥1 evidenza full o partial / ${agg.n_claims} claim`}
          hint="Un'evidenza parziale basta per coprire il claim." />
      </div>

      {/* Distribuzione 3 livelli */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, marginBottom: 28 }}>
        {[
          { label: 'Full support',    value: agg.pct_full,    count: agg.n_full,    color: '#166534', bg: '#ECFDF5', bd: '#A7F3D0' },
          { label: 'Partial support', value: agg.pct_partial, count: agg.n_partial, color: '#92400E', bg: '#FFFBEB', bd: '#FDE68A' },
          { label: 'Not supported',   value: agg.pct_none,    count: agg.n_none,    color: '#991B1B', bg: '#FEF2F2', bd: '#FECACA' },
        ].map(t => (
          <div key={t.label} style={{ padding: '12px 14px', background: t.bg, border: `1px solid ${t.bd}`, borderRadius: 10 }}>
            <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.6px', marginBottom: 6 }}>
              {t.label}
            </div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
              <span style={{ fontSize: 22, fontWeight: 800, color: t.color }}>{pct(t.value)}</span>
              <span style={{ fontSize: 11, fontFamily: 'var(--mono)', color: 'var(--text-3)' }}>({t.count})</span>
            </div>
          </div>
        ))}
      </div>

      <div className="divider" />

      <PerExampleSection
        label="Dettaglio per esempio"
        metric={metric} setMetric={setMetric} labels={DEEPSEEK_LABELS}
      >
        <PerExampleChart examples={examples} metricKey={metric} valueOf={valueOf} colorOf={deepseekColor} />
        <DeepSeekTable examples={examples} />
      </PerExampleSection>
    </div>
  )
}

// ── Helper di layout condivisi ───────────────────────────────────────────────────

function SectionTitle({ children }) {
  return (
    <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-2)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 12 }}>
      {children}
    </div>
  )
}

function PerExampleSection({ label, metric, setMetric, labels, children }) {
  return (
    <>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
        <SectionTitle>{label}</SectionTitle>
        <div style={{ flex: 1 }} />
        <select className="input" style={{ width: 220 }} value={metric} onChange={e => setMetric(e.target.value)}>
          {Object.entries(labels).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
        </select>
      </div>
      <div className="card">
        <div className="card-body">{children}</div>
      </div>
    </>
  )
}

function PerExampleTable({ examples, keys, labels, valueOf, colorOf }) {
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
        <thead>
          <tr style={{ borderBottom: '2px solid var(--border)' }}>
            <th style={{ textAlign: 'left', padding: '8px 12px', color: 'var(--text-3)', fontWeight: 600 }}>Question</th>
            {keys.map(k => (
              <th key={k} style={{ textAlign: 'right', padding: '8px 12px', color: 'var(--text-3)', fontWeight: 600, whiteSpace: 'nowrap' }}>
                {labels[k]}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {examples.map((ex, i) => (
            <tr key={i} style={{ borderBottom: '1px solid var(--border-2)' }}>
              <td style={{ padding: '8px 12px', color: 'var(--text-2)', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={ex.question}>
                {ex.question || '—'}
              </td>
              {keys.map(k => {
                const v = valueOf(ex, k)
                return (
                  <td key={k} style={{ padding: '8px 12px', textAlign: 'right', fontFamily: 'var(--mono)', fontSize: 11, fontWeight: 600, color: colorOf(v) }}>
                    {v.toFixed(3)}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function DeepSeekTable({ examples }) {
  const cell = { padding: '8px 12px', textAlign: 'right', fontFamily: 'var(--mono)', fontSize: 11, fontWeight: 600 }
  const head = { textAlign: 'right', padding: '8px 12px', color: 'var(--text-3)', fontWeight: 600, whiteSpace: 'nowrap' }
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
        <thead>
          <tr style={{ borderBottom: '2px solid var(--border)' }}>
            <th style={{ textAlign: 'left', padding: '8px 12px', color: 'var(--text-3)', fontWeight: 600 }}>Question</th>
            <th style={head}>Precision</th>
            <th style={head}>Recall</th>
            <th style={head}>Full</th>
            <th style={head}>Partial</th>
            <th style={head}>None</th>
          </tr>
        </thead>
        <tbody>
          {examples.map((ex, i) => {
            const m = ex.deepseek_metrics
            const nPairs   = m.n_pairs ?? 0
            const nFull    = m.n_full ?? m.n_pairs_supported ?? 0
            const nPartial = m.n_partial ?? 0
            const nNone    = m.n_none ?? Math.max(0, nPairs - nFull - nPartial)
            const prec = m.citation_precision ?? 0
            const rec  = m.citation_recall ?? 0
            return (
              <tr key={i} style={{ borderBottom: '1px solid var(--border-2)' }}>
                <td style={{ padding: '8px 12px', color: 'var(--text-2)', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={ex.question}>
                  {ex.question || '—'}
                </td>
                <td style={{ ...cell, color: deepseekColor(prec) }}>{prec.toFixed(3)}</td>
                <td style={{ ...cell, color: deepseekColor(rec) }}>{rec.toFixed(3)}</td>
                <td style={{ ...cell, color: '#166534' }}>{nFull}</td>
                <td style={{ ...cell, color: '#92400E' }}>{nPartial}</td>
                <td style={{ ...cell, color: '#991B1B' }}>{nNone}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

// ── Toggle famiglia ──────────────────────────────────────────────────────────────

function FamilyToggle({ family, onChange, hasNugget, hasDeepseek }) {
  const base = {
    display: 'flex', alignItems: 'center', gap: 6,
    padding: '6px 14px', fontSize: 12, fontWeight: 600,
    border: 'none', borderRadius: 8, cursor: 'pointer', transition: 'all 0.15s',
  }
  return (
    <div style={{ display: 'inline-flex', alignItems: 'center', background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 10, padding: 3, gap: 2 }}>
      <button
        onClick={() => hasNugget && onChange('nugget')}
        disabled={!hasNugget}
        style={{
          ...base,
          background: family === 'nugget' ? '#7C3AED' : 'transparent',
          color: family === 'nugget' ? 'white' : 'var(--text-2)',
          boxShadow: family === 'nugget' ? '0 1px 4px rgba(124,58,237,0.3)' : 'none',
          opacity: hasNugget ? 1 : 0.4, cursor: hasNugget ? 'pointer' : 'not-allowed',
        }}
      >
        <Icon name="target" size={12} strokeWidth={2} color={family === 'nugget' ? 'white' : 'var(--text-3)'} />
        Nugget
      </button>
      <button
        onClick={() => hasDeepseek && onChange('deepseek')}
        disabled={!hasDeepseek}
        style={{
          ...base,
          background: family === 'deepseek' ? '#0EA5E9' : 'transparent',
          color: family === 'deepseek' ? 'white' : 'var(--text-2)',
          boxShadow: family === 'deepseek' ? '0 1px 4px rgba(14,165,233,0.3)' : 'none',
          opacity: hasDeepseek ? 1 : 0.4, cursor: hasDeepseek ? 'pointer' : 'not-allowed',
        }}
      >
        <Icon name="search" size={12} strokeWidth={2} color={family === 'deepseek' ? 'white' : 'var(--text-3)'} />
        DeepSeek
      </button>
    </div>
  )
}

// ── Main ──────────────────────────────────────────────────────────────────────

export default function Metrics() {
  const { pipelineResults } = useAppData()
  const fileRef = useRef()
  const [loaded, setLoaded] = useState({ nugget: [], deepseek: [] })
  const [loadError, setLoadError] = useState(null)
  const [loadInfo, setLoadInfo] = useState(null)
  const [family, setFamily] = useState(null)

  // Risultati in memoria (da "Salva in Esplora") normalizzati a { question, *_metrics }
  const ctxNugget = pipelineResults
    .filter(r => r.nugget_metrics && Object.keys(r.nugget_metrics).length > 0)
    .map(r => ({ question: r.question, nugget_metrics: r.nugget_metrics }))
  const ctxDeepseek = pipelineResults
    .filter(r => r.deepseek_metrics && Object.keys(r.deepseek_metrics).length > 0)
    .map(r => ({ question: r.question, deepseek_metrics: r.deepseek_metrics }))

  // Memoria di sessione + eventuali caricati da file
  const nuggetExamples   = [...ctxNugget, ...loaded.nugget]
  const deepseekExamples = [...ctxDeepseek, ...loaded.deepseek]

  const hasNugget   = nuggetExamples.length   > 0
  const hasDeepseek = deepseekExamples.length > 0
  const hasAny      = hasNugget || hasDeepseek

  const defaultFamily = hasNugget ? 'nugget' : hasDeepseek ? 'deepseek' : null
  const activeFamily  = family ?? defaultFamily
  const examples      = activeFamily === 'nugget' ? nuggetExamples : deepseekExamples
  const nLoaded       = loaded.nugget.length + loaded.deepseek.length

  function onFileUpload(e) {
    const file = e.target.files[0]
    if (!file) return
    e.target.value = ''
    const reader = new FileReader()
    reader.onload = evt => {
      try {
        const parsed = JSON.parse(evt.target.result)
        const { nugget, deepseek } = parseLoadedMetrics(parsed)
        if (nugget.length === 0 && deepseek.length === 0) {
          throw new Error('Nessun risultato Nugget o DeepSeek riconosciuto nel file.')
        }
        setLoaded(prev => ({
          nugget:   [...prev.nugget,   ...nugget],
          deepseek: [...prev.deepseek, ...deepseek],
        }))
        setLoadInfo(`Caricati ${nugget.length} Nugget + ${deepseek.length} DeepSeek da ${file.name}`)
        setLoadError(null)
      } catch (err) {
        setLoadError(`Errore lettura file: ${err.message}`)
        setLoadInfo(null)
      }
    }
    reader.readAsText(file)
  }

  function clearLoaded() {
    setLoaded({ nugget: [], deepseek: [] })
    setLoadInfo(null); setLoadError(null)
  }

  function onDownload() {
    const payload = {
      nugget: hasNugget ? {
        aggregate: aggregateNugget(nuggetExamples),
        per_example: nuggetExamples.map(ex => ({ question: ex.question, metrics: ex.nugget_metrics })),
      } : null,
      deepseek: hasDeepseek ? {
        aggregate: aggregateDeepseek(deepseekExamples),
        per_example: deepseekExamples.map(ex => ({ question: ex.question, metrics: ex.deepseek_metrics })),
      } : null,
      aggregation: 'pooled (micro) — vedi PDF Metriche §1.6 / §2.4',
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
              {hasAny
                ? <>Aggregato pooled su tutti gli esempi · {examples.length} con metriche {activeFamily === 'nugget' ? 'Nugget' : 'DeepSeek'}</>
                : <>Carica un file di metriche, oppure esegui il pipeline e salva in Esplora.</>}
            </div>
          </div>
          <div style={{ display: 'flex', gap: 10, flexShrink: 0 }}>
            <input ref={fileRef} type="file" accept=".json" onChange={onFileUpload} style={{ display: 'none' }} />
            <button className="btn btn-secondary" onClick={() => fileRef.current.click()}>
              <Icon name="upload" size={13} strokeWidth={1.75} /> Carica risultati
            </button>
            {hasAny && (
              <button className="btn btn-secondary" onClick={onDownload}>
                <Icon name="download" size={13} strokeWidth={1.75} /> Scarica JSON
              </button>
            )}
          </div>
        </div>
      </div>

      {loadError && (
        <div className="info-box info-box-red" style={{ marginBottom: 16 }}>
          <Icon name="xCircle" size={15} strokeWidth={1.75} style={{ flexShrink: 0, marginTop: 1 }} />
          <span><strong>Errore:</strong> {loadError}</span>
        </div>
      )}

      {nLoaded > 0 && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16,
          padding: '8px 14px', background: '#F0F9FF', border: '1px solid #BAE6FD',
          borderRadius: 8, fontSize: 12, color: '#0C4A6E',
        }}>
          <Icon name="fileText" size={14} color="#0284C7" strokeWidth={2} />
          <span>{loadInfo || `${nLoaded} risultati caricati da file`}</span>
          <button className="btn btn-secondary" style={{ marginLeft: 'auto', padding: '2px 10px', fontSize: 11 }} onClick={clearLoaded}>
            Rimuovi caricati
          </button>
        </div>
      )}

      {!hasAny ? (
        <EmptyState
          title="Nessuna metrica disponibile"
          hint="Carica un JSON esportato in precedenza, oppure esegui il pipeline (step Valuta) e salva in Esplora."
        />
      ) : (
        <>
          <div style={{ marginBottom: 20 }}>
            <FamilyToggle family={activeFamily} onChange={setFamily} hasNugget={hasNugget} hasDeepseek={hasDeepseek} />
          </div>
          {activeFamily === 'nugget'   && <NuggetView   examples={nuggetExamples} />}
          {activeFamily === 'deepseek' && <DeepSeekView examples={deepseekExamples} />}
        </>
      )}
    </div>
  )
}