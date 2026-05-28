/**
 * MetricsViews.jsx — Viste di dettaglio delle metriche, condivise fra
 * Pipeline (step 6) ed Explore (tab Metriche).
 *
 * Esporta NuggetMetricsView e DeepSeekMetricsView. I bottoni di azione
 * (Salva in Esplora / Scarica dati) sono OPZIONALI: vengono mostrati solo
 * se i rispettivi handler onSave / onDownload sono passati. In Explore non
 * si passano, quindi le viste mostrano solo i dati senza azioni.
 *
 * Fonte di verità unica: modificare qui (es. peso del partial, stile badge)
 * si riflette sia sulla Pipeline sia su Explore.
 */

import { useState } from 'react'
import Icon from './Icon'

export const METRIC_INFO_DEEPSEEK = {
  citation_precision: {
    label: 'Citation Precision',
    desc: 'Pesata: (full + 0.5·partial) / coppie totali. Full=evidenza completa, partial=parziale.',
  },
  citation_recall: {
    label: 'Citation Recall',
    desc: 'Dei claim, quanti hanno almeno un\'evidenza full o partial.',
  },
}

export function metricColor(key, v) {
  if (key === 'unsupported_ratio') {
    return v <= 0.2 ? 'var(--green)' : v <= 0.5 ? 'var(--amber)' : 'var(--red)'
  }
  return v >= 0.7 ? 'var(--green)' : v >= 0.4 ? 'var(--amber)' : 'var(--red)'
}

// ── NuggetMetricsView ─────────────────────────────────────────────────────────

export function NuggetMetricsView({ metrics, onSave, onDownload }) {
  const [expanded, setExpanded] = useState({})

  const {
    nugget_precision, nugget_recall, nugget_coverage,
    n_nuggets, n_covered, n_cited, per_nugget = []
  } = metrics

  const pct = v => `${Math.round(v * 100)}%`

  function GaugePill({ value, color }) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <div style={{ width: 80, height: 6, background: 'var(--border)', borderRadius: 3, overflow: 'hidden' }}>
          <div style={{ height: '100%', borderRadius: 3, width: `${Math.min(100, value * 100)}%`, background: color, transition: 'width 0.4s ease' }} />
        </div>
        <span style={{ fontFamily: 'var(--mono)', fontSize: 12, fontWeight: 700, color }}>{pct(value)}</span>
      </div>
    )
  }

  const mc = (val) => val >= 0.6 ? 'var(--green)' : val >= 0.3 ? 'var(--amber)' : 'var(--red)'
  const precColor = mc(nugget_precision)
  const recColor  = mc(nugget_recall)
  const covColor  = mc(nugget_coverage)

  return (
    <div>
      {/* Summary cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, marginBottom: 20 }}>
        <div style={{ padding: '16px 18px', background: 'white', border: `1px solid ${precColor === 'var(--green)' ? '#A7F3D0' : precColor === 'var(--amber)' ? '#FDE68A' : '#FECACA'}`, borderTop: `3px solid ${precColor}`, borderRadius: 10 }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.6px', marginBottom: 8 }}>Precisione Nugget</div>
          <div style={{ fontSize: 30, fontWeight: 800, color: precColor, lineHeight: 1, marginBottom: 8 }}>{pct(nugget_precision)}</div>
          <GaugePill value={nugget_precision} color={precColor} />
          <div style={{ marginTop: 8, fontSize: 11, color: 'var(--text-3)', lineHeight: 1.4 }}>Media pesata sui {n_covered} nugget coperti</div>
          <div style={{ marginTop: 6, fontSize: 11, color: 'var(--text-3)', fontStyle: 'italic' }}>Precisione continua (match × evidenza) sui nugget coperti.</div>
        </div>

        <div style={{ padding: '16px 18px', background: 'white', border: `1px solid ${recColor === 'var(--green)' ? '#A7F3D0' : recColor === 'var(--amber)' ? '#FDE68A' : '#FECACA'}`, borderTop: `3px solid ${recColor}`, borderRadius: 10 }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.6px', marginBottom: 8 }}>Recall Nugget</div>
          <div style={{ fontSize: 30, fontWeight: 800, color: recColor, lineHeight: 1, marginBottom: 8 }}>{pct(nugget_recall)}</div>
          <GaugePill value={nugget_recall} color={recColor} />
          <div style={{ marginTop: 8, fontSize: 11, color: 'var(--text-3)', lineHeight: 1.4 }}>Su {n_nuggets} nugget totali</div>
          <div style={{ marginTop: 6, fontSize: 11, color: 'var(--text-3)', fontStyle: 'italic' }}>Stessa media, ma su tutti i nugget (inclusi non coperti).</div>
        </div>

        <div style={{ padding: '16px 18px', background: 'white', border: `1px solid ${covColor === 'var(--green)' ? '#A7F3D0' : covColor === 'var(--amber)' ? '#FDE68A' : '#FECACA'}`, borderTop: `3px solid ${covColor}`, borderRadius: 10 }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.6px', marginBottom: 8 }}>Copertura Nugget</div>
          <div style={{ fontSize: 30, fontWeight: 800, color: covColor, lineHeight: 1, marginBottom: 8 }}>{pct(nugget_coverage)}</div>
          <GaugePill value={nugget_coverage} color={covColor} />
          <div style={{ marginTop: 8, fontSize: 11, color: 'var(--text-3)', lineHeight: 1.4 }}>{n_covered} coperti su {n_nuggets} nugget</div>
          <div style={{ marginTop: 6, fontSize: 11, color: 'var(--text-3)', fontStyle: 'italic' }}>Percentuale di nugget toccati da almeno un claim.</div>
        </div>
      </div>

      {/* Dettaglio per nugget */}
      {per_nugget.length > 0 && (
        <div>
          <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.7px', marginBottom: 10 }}>
            Dettaglio per nugget — {per_nugget.length} totali
          </div>

          {per_nugget.map((nug, i) => {
            const excluded = nug.excluded_no_golden
            const score = nug.nugget_precision_score
            const isCited = nug.cited
            const statusLabel = excluded ? 'Escluso (no golden evidence)' : isCited ? 'Citato ✓' : nug.covered ? 'Coperto, non citato' : 'Non coperto'
            const statusBg = excluded ? '#F3F4F6' : isCited ? '#DCFCE7' : nug.covered ? '#FEF9C3' : '#FEE2E2'
            const statusFg = excluded ? '#6B7280' : isCited ? '#166534' : nug.covered ? '#713F12' : '#991B1B'

            return (
              <div key={i} style={{ marginBottom: 8, border: `1px solid ${excluded ? '#E5E7EB' : isCited ? '#A7F3D0' : nug.covered ? '#FDE68A' : '#FECACA'}`, borderRadius: 8, overflow: 'hidden', opacity: excluded ? 0.6 : 1 }}>
                <div onClick={() => setExpanded(e => ({ ...e, [i]: !e[i] }))}
                  style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '10px 14px', background: excluded ? '#F9FAFB' : isCited ? '#F0FDF4' : nug.covered ? '#FFFBEB' : '#FFF1F2', cursor: 'pointer' }}>
                  <span style={{ fontSize: 10, fontWeight: 700, background: statusBg, color: statusFg, padding: '2px 8px', borderRadius: 10, whiteSpace: 'nowrap', flexShrink: 0 }}>{statusLabel}</span>
                  <span style={{ fontFamily: 'var(--mono)', fontSize: 10, fontWeight: 600, color: 'var(--text-3)', flexShrink: 0 }}>{nug.nugget_id}</span>
                  <span style={{ fontSize: 12, color: 'var(--text)', flex: 1, lineHeight: 1.4 }}>{nug.nugget_text}</span>
                  {nug.required && (
                    <span style={{ fontSize: 9, fontWeight: 700, background: '#EDE9FE', color: '#5B21B6', padding: '2px 6px', borderRadius: 4, flexShrink: 0 }}>REQUIRED</span>
                  )}
                  {score != null && (
                    <span style={{ fontSize: 10, fontWeight: 700, fontFamily: 'var(--mono)', background: score >= 0.45 ? '#ECFDF5' : '#FEF2F2', color: score >= 0.45 ? '#166534' : '#991B1B', padding: '2px 6px', borderRadius: 6, flexShrink: 0 }}>{score.toFixed(2)}</span>
                  )}
                  <span style={{ color: 'var(--text-3)', fontSize: 12 }}>{expanded[i] ? '▲' : '▼'}</span>
                </div>

                {expanded[i] && (
                  <div style={{ padding: '12px 16px', background: 'white' }}>
                    {excluded && (
                      <div style={{ marginBottom: 10, padding: '8px 12px', background: '#F3F4F6', border: '1px solid #E5E7EB', borderRadius: 6, fontSize: 12, color: '#6B7280' }}>
                        ⚠️ Nugget escluso dalle metriche continue: manca `golden_evidence`.
                      </div>
                    )}
                    {nug.keywords?.length > 0 && (
                      <div style={{ marginBottom: 10, display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
                        <span style={{ fontSize: 11, color: 'var(--text-3)', fontWeight: 600 }}>Keywords:</span>
                        {nug.keywords.map((k, ki) => (
                          <span key={ki} style={{ fontSize: 11, background: '#EDE9FE', color: '#5B21B6', padding: '1px 7px', borderRadius: 10, fontFamily: 'var(--mono)' }}>{k}</span>
                        ))}
                      </div>
                    )}
                    {nug.golden_evidence && (
                      <div style={{ marginBottom: 10, padding: '8px 12px', background: '#F0F9FF', border: '1px solid #BAE6FD', borderRadius: 6, fontSize: 12, color: '#0C4A6E' }}>
                        <strong>Golden evidence:</strong> {nug.golden_evidence}
                        {nug.golden_passage_title && (
                          <span style={{ marginLeft: 6, fontFamily: 'var(--mono)', fontSize: 10, color: '#075985' }}>[{nug.golden_passage_title}]</span>
                        )}
                      </div>
                    )}
                    {nug.best_covering_claim && (
                      <div style={{ marginBottom: 10, padding: '8px 12px', background: '#F0FDF4', border: '1px solid #86EFAC', borderRadius: 6, fontSize: 12 }}>
                        <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--text-3)', textTransform: 'uppercase', marginBottom: 4 }}>Miglior claim che copre il nugget</div>
                        <span style={{ color: '#166534' }}>{nug.best_covering_claim}</span>
                        {score != null && (
                          <span style={{ marginLeft: 8, fontSize: 10, color: 'var(--text-3)', fontFamily: 'var(--mono)' }}>(score: {score.toFixed(2)})</span>
                        )}
                      </div>
                    )}
                    {nug.all_evidence && nug.all_evidence.length > 0 && (
                      <div style={{ marginTop: 8 }}>
                        <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.6px', marginBottom: 8 }}>
                          Tutte le evidenze ({nug.all_evidence.length}) — ordinate per similarità decrescente
                        </div>
                        <div style={{ maxHeight: 400, overflowY: 'auto', border: '1px solid var(--border)', borderRadius: 6, padding: '0 8px' }}>
                          {nug.all_evidence.map((ev, j) => {
                            const evScore = ev.evidence_score
                            const scoreColor = evScore >= 0.6 ? '#166534' : evScore >= 0.3 ? '#92400E' : '#991B1B'
                            const scoreBg = evScore >= 0.6 ? '#DCFCE7' : evScore >= 0.3 ? '#FEF9C3' : '#FEE2E2'
                            return (
                              <div key={j} style={{ padding: '8px 0', borderBottom: '1px solid var(--border)', display: 'flex', flexDirection: 'column', gap: 4 }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                                  <span style={{ fontSize: 10, fontWeight: 700, background: scoreBg, color: scoreColor, padding: '1px 6px', borderRadius: 10, fontFamily: 'var(--mono)' }}>{evScore.toFixed(2)}</span>
                                  {ev.entailment_score != null && (
                                    <span style={{ fontSize: 10, color: 'var(--text-3)' }}>ent: {ev.entailment_score.toFixed(2)}</span>
                                  )}
                                  {ev.is_noise && (
                                    <span style={{ fontSize: 9, color: 'var(--amber)', fontWeight: 600 }}>⚠️ rumore</span>
                                  )}
                                  <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-2)' }}>{ev.passage_title || 'Senza titolo'}</span>
                                </div>
                                {ev.span && (
                                  <div style={{ marginLeft: 12, padding: '4px 8px', background: '#F0F9FF', borderLeft: '3px solid #38BDF8', borderRadius: 4, fontSize: 11, color: '#0C4A6E', fontStyle: 'italic' }}>«{ev.span}»</div>
                                )}
                                <div style={{ marginLeft: 12, fontSize: 10, color: 'var(--text-3)' }}>Claim: {ev.claim}{ev.claim.length >= 200 ? '…' : ''}</div>
                              </div>
                            )
                          })}
                        </div>
                      </div>
                    )}
                    {(!nug.all_evidence || nug.all_evidence.length === 0) && nug.covered && !excluded && (
                      <div style={{ fontSize: 12, color: 'var(--text-3)', fontStyle: 'italic', marginTop: 8 }}>Nessun passaggio con evidenza disponibile.</div>
                    )}
                    {!nug.covered && !excluded && (
                      <div style={{ fontSize: 12, color: 'var(--text-3)', fontStyle: 'italic' }}>Nessun claim generato copre questo nugget.</div>
                    )}
                    {nug.covered && !nug.cited && !excluded && (
                      <div style={{ fontSize: 12, color: '#92400E', fontStyle: 'italic' }}>Il nugget è coperto da {nug.n_covering_claims} claim, ma il punteggio di evidenza rimane sotto la soglia ({score?.toFixed(2)} &lt; 0.45).</div>
                    )}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}

      {(onSave || onDownload) && (
        <div style={{ marginTop: 20, display: 'flex', gap: 12 }}>
          {onSave && (
            <button className="btn btn-primary" onClick={onSave}>
              <Icon name="download" size={13} color="white" strokeWidth={2} /> Salva in Esplora
            </button>
          )}
          {onDownload && (
            <button className="btn btn-secondary" onClick={onDownload}>
              <Icon name="download" size={13} strokeWidth={1.75} /> Scarica dati
            </button>
          )}
        </div>
      )}
    </div>
  )
}

// ── DeepSeekMetricsView ─────────────────────────────────────────────────────────

export function DeepSeekMetricsView({ metrics, onSave, onDownload }) {
  const [expanded, setExpanded] = useState({})
  const {
    citation_precision, citation_recall,
    n_claims, n_pairs, n_pairs_supported, per_claim = [],
  } = metrics

  const nFull    = metrics.n_full    ?? 0
  const nPartial = metrics.n_partial ?? 0
  const nNone    = metrics.n_none    ?? Math.max(0, (n_pairs ?? 0) - nFull - nPartial)
  const pctFull    = metrics.pct_full    ?? (n_pairs > 0 ? nFull    / n_pairs : 0)
  const pctPartial = metrics.pct_partial ?? (n_pairs > 0 ? nPartial / n_pairs : 0)
  const pctNone    = metrics.pct_none    ?? (n_pairs > 0 ? nNone    / n_pairs : 0)

  const pct = v => `${Math.round(v * 100)}%`

  const STYLE_BY_VERDICT = {
    supported:     { bg: '#F0FDF4', bd: '#86EFAC',       pillBg: '#DCFCE7', pillFg: '#166534', label: 'SUPPORTED' },
    partial:       { bg: '#FFFBEB', bd: '#FDE68A',       pillBg: '#FEF3C7', pillFg: '#92400E', label: 'PARTIAL'   },
    not_supported: { bg: '#FAFAF9', bd: 'var(--border)', pillBg: '#FEE2E2', pillFg: '#991B1B', label: 'NOT SUPPORTED' },
  }
  const verdictOf = (j) => j.verdict || (j.supported ? 'supported' : 'not_supported')

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16, flexWrap: 'wrap', padding: '8px 14px', background: '#F0F9FF', border: '1px solid #BAE6FD', borderRadius: 8, fontSize: 12, color: '#0C4A6E' }}>
        <Icon name="search" size={14} color="#0284C7" strokeWidth={2} />
        <span>Giudizio LLM-as-judge via DeepSeek · {n_pairs} coppie valutate</span>
        {n_pairs > 0 && (
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}>
            ·
            <span style={{ color: '#166534', fontWeight: 600 }}>{nFull} full</span>
            <span style={{ color: 'var(--text-3)' }}>·</span>
            <span style={{ color: '#92400E', fontWeight: 600 }}>{nPartial} partial</span>
            <span style={{ color: 'var(--text-3)' }}>·</span>
            <span style={{ color: '#991B1B', fontWeight: 600 }}>{nNone} none</span>
          </span>
        )}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12, marginBottom: 12 }}>
        {[
          { key: 'citation_precision', value: citation_precision, sub: `(${nFull} full + 0.5·${nPartial} partial) / ${n_pairs} coppie` },
          { key: 'citation_recall',    value: citation_recall,    sub: `claim con ≥1 evidenza full o partial su ${n_claims} totali` },
        ].map(({ key, value, sub }) => {
          const color = metricColor(key, value)
          const bd = color === 'var(--green)' ? '#A7F3D0' : color === 'var(--amber)' ? '#FDE68A' : '#FECACA'
          return (
            <div key={key} style={{ padding: '16px 18px', background: 'white', border: `1px solid ${bd}`, borderTop: `3px solid ${color}`, borderRadius: 10 }}>
              <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.6px', marginBottom: 8 }}>{METRIC_INFO_DEEPSEEK[key].label}</div>
              <div style={{ fontSize: 30, fontWeight: 800, color, lineHeight: 1, marginBottom: 8 }}>{pct(value)}</div>
              <div style={{ fontSize: 11, color: 'var(--text-3)', lineHeight: 1.4 }}>{sub}</div>
              <div style={{ marginTop: 6, fontSize: 11, color: 'var(--text-3)', fontStyle: 'italic' }}>{METRIC_INFO_DEEPSEEK[key].desc}</div>
            </div>
          )
        })}
      </div>

      {n_pairs > 0 && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, marginBottom: 20 }}>
          {[
            { label: 'Full support',    value: pctFull,    count: nFull,    color: '#166534', bg: '#ECFDF5', bd: '#A7F3D0' },
            { label: 'Partial support', value: pctPartial, count: nPartial, color: '#92400E', bg: '#FFFBEB', bd: '#FDE68A' },
            { label: 'Not supported',   value: pctNone,    count: nNone,    color: '#991B1B', bg: '#FEF2F2', bd: '#FECACA' },
          ].map(t => (
            <div key={t.label} style={{ padding: '12px 14px', background: t.bg, border: `1px solid ${t.bd}`, borderRadius: 10 }}>
              <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.6px', marginBottom: 6 }}>{t.label}</div>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
                <span style={{ fontSize: 22, fontWeight: 800, color: t.color }}>{pct(t.value)}</span>
                <span style={{ fontSize: 11, fontFamily: 'var(--mono)', color: 'var(--text-3)' }}>({t.count})</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {per_claim.length > 0 && (
        <div>
          <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.7px', marginBottom: 10 }}>
            Dettaglio per claim — {per_claim.length} totali
          </div>
          {per_claim.map((c, i) => {
            const ok = c.any_supported
            const nFullC    = c.n_full    ?? c.n_supported ?? 0
            const nPartialC = c.n_partial ?? 0
            const pillLabel = ok ? `${nFullC} full${nPartialC > 0 ? ` + ${nPartialC} partial` : ''} / ${c.n_passages}` : 'nessun supporto'

            return (
              <div key={i} style={{ marginBottom: 8, border: `1px solid ${ok ? '#A7F3D0' : '#FECACA'}`, borderRadius: 8, overflow: 'hidden' }}>
                <div onClick={() => setExpanded(e => ({ ...e, [i]: !e[i] }))}
                  style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '10px 14px', background: ok ? '#F0FDF4' : '#FFF1F2', cursor: 'pointer' }}>
                  <span style={{ fontSize: 10, fontWeight: 700, padding: '2px 8px', borderRadius: 10, whiteSpace: 'nowrap', flexShrink: 0, background: ok ? '#DCFCE7' : '#FEE2E2', color: ok ? '#166534' : '#991B1B' }}>{pillLabel}</span>
                  <span style={{ fontSize: 12, color: 'var(--text)', flex: 1, lineHeight: 1.4 }}>{c.claim}</span>
                  <span style={{ color: 'var(--text-3)', fontSize: 12 }}>{expanded[i] ? '▲' : '▼'}</span>
                </div>
                {expanded[i] && (
                  <div style={{ padding: '12px 16px', background: 'white' }}>
                    {c.judgments.length === 0 ? (
                      <div style={{ fontSize: 12, color: 'var(--text-3)', fontStyle: 'italic' }}>Nessun passaggio citato per questo claim.</div>
                    ) : c.judgments.map((j, ji) => {
                      const v = verdictOf(j)
                      const st = STYLE_BY_VERDICT[v]
                      return (
                        <div key={ji} style={{ marginBottom: 8, padding: '8px 12px', borderRadius: 6, background: st.bg, border: `1px solid ${st.bd}` }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                            <span style={{ fontSize: 10, fontWeight: 700, padding: '1px 7px', borderRadius: 8, background: st.pillBg, color: st.pillFg }}>{st.label}</span>
                            <span style={{ fontSize: 12, fontWeight: 600 }}>{j.passage_title || '—'}</span>
                          </div>
                          <div style={{ fontSize: 11, color: st.pillFg, lineHeight: 1.5, marginBottom: 6, padding: '6px 10px', background: 'white', borderRadius: 6, borderLeft: `3px solid ${st.bd}` }}>
                            <strong style={{ color: 'var(--text-3)', fontWeight: 700 }}>Evidenza: </strong>{j.evidence || '(nessuno span estratto)'}
                          </div>
                          {j.reason && (
                            <div style={{ fontSize: 11, color: '#0C4A6E', fontStyle: 'italic', padding: '6px 10px', background: '#F0F9FF', borderRadius: 6, borderLeft: '3px solid #BAE6FD' }}>
                              <strong>DeepSeek:</strong> {j.reason}
                            </div>
                          )}
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}

      {(onSave || onDownload) && (
        <div style={{ marginTop: 20, display: 'flex', gap: 12 }}>
          {onSave && (
            <button className="btn btn-primary" onClick={onSave}>
              <Icon name="download" size={13} color="white" strokeWidth={2} /> Salva in Esplora
            </button>
          )}
          {onDownload && (
            <button className="btn btn-secondary" onClick={onDownload}>
              <Icon name="download" size={13} strokeWidth={1.75} /> Scarica dati
            </button>
          )}
        </div>
      )}
    </div>
  )
}