/**
 * MetricsViews.jsx — Viste di dettaglio delle metriche, condivise fra
 * Pipeline (step 6) ed Explore (tab Metriche).
 *
 * Esporta NuggetMetricsView e DeepSeekMetricsView. I bottoni di azione
 * (Salva in Esplora / Scarica dati) sono OPZIONALI.
 *
 * Entrambe le lenti: precision sulle coppie (claim, evidenza), recall sui
 * CLAIM. La lente NUGGET espone in piu' la COVERAGE dei nugget (totale +
 * required/optional) come diagnostica. La lente DEEPSEEK e' a verdetto
 * BINARIO (supported / not_supported).
 */

import { useState } from 'react'
import Icon from './Icon'

export const METRIC_INFO_DEEPSEEK = {
  citation_precision: {
    label: 'Citation Precision',
    desc: 'Delle coppie (claim, evidenza), quante il giudice ritiene supportate.',
  },
  citation_recall: {
    label: 'Citation Recall',
    desc: 'Dei claim, quanti hanno almeno un\'evidenza giudicata supportata.',
  },
}

export function metricColor(key, v) {
  if (key === 'unsupported_ratio') {
    return v <= 0.2 ? 'var(--green)' : v <= 0.5 ? 'var(--amber)' : 'var(--red)'
  }
  return v >= 0.7 ? 'var(--green)' : v >= 0.4 ? 'var(--amber)' : 'var(--red)'
}

const cardBorder = (c) => c === 'var(--green)' ? '#A7F3D0' : c === 'var(--amber)' ? '#FDE68A' : '#FECACA'

// ── NuggetMetricsView ─────────────────────────────────────────────────────────

export function NuggetMetricsView({ metrics, onSave, onDownload }) {
  const [expanded, setExpanded] = useState({})

  const {
    nugget_precision, nugget_recall, nugget_coverage,
    n_claims, n_claims_covered, n_pairs, n_pairs_correct,
    n_nuggets, n_covered,
    n_required, n_required_covered, required_coverage,
    n_optional, n_optional_covered, optional_coverage,
    n_pairs_from_noise = 0, n_pairs_correct_from_noise = 0,
    per_nugget = [],
  } = metrics

  const pct = v => `${Math.round((v ?? 0) * 100)}%`
  const mc = (val) => val >= 0.6 ? 'var(--green)' : val >= 0.3 ? 'var(--amber)' : 'var(--red)'

  function GaugePill({ value, color }) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <div style={{ width: 80, height: 6, background: 'var(--border)', borderRadius: 3, overflow: 'hidden' }}>
          <div style={{ height: '100%', borderRadius: 3, width: `${Math.min(100, (value ?? 0) * 100)}%`, background: color, transition: 'width 0.4s ease' }} />
        </div>
        <span style={{ fontFamily: 'var(--mono)', fontSize: 12, fontWeight: 700, color }}>{pct(value)}</span>
      </div>
    )
  }

  function BigCard({ title, value, color, lines }) {
    return (
      <div style={{ padding: '16px 18px', background: 'white', border: `1px solid ${cardBorder(color)}`, borderTop: `3px solid ${color}`, borderRadius: 10 }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.6px', marginBottom: 8 }}>{title}</div>
        <div style={{ fontSize: 30, fontWeight: 800, color, lineHeight: 1, marginBottom: 8 }}>{pct(value)}</div>
        <GaugePill value={value} color={color} />
        {lines?.map((l, i) => (
          <div key={i} style={{ marginTop: i === 0 ? 8 : 4, fontSize: 11, color: 'var(--text-3)', lineHeight: 1.4, fontStyle: i === 0 ? 'normal' : 'italic' }}>{l}</div>
        ))}
      </div>
    )
  }

  return (
    <div>
      {/* Precision (coppie) + Recall (claim) */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12, marginBottom: 12 }}>
        <BigCard title="Citation Precision" value={nugget_precision} color={mc(nugget_precision)}
          lines={[`${n_pairs_correct ?? 0} coppie corrette su ${n_pairs ?? 0} prodotte`,
                  'Delle citazioni prodotte, quante hanno evidenza che matcha la golden.']} />
        <BigCard title="Citation Recall" value={nugget_recall} color={mc(nugget_recall)}
          lines={[`${n_claims_covered ?? 0} claim fondati su ${n_claims ?? 0}`,
                  'Dei claim prodotti, quanti hanno almeno un\'evidenza valida.']} />
      </div>

      {/* Coverage dei nugget (diagnostica), totale + required/optional */}
      <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.7px', margin: '4px 0 8px' }}>
        Nugget coverage — quali fatti del gold sono coperti
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, marginBottom: 20 }}>
        <BigCard title="Coverage (tutti)" value={nugget_coverage} color={mc(nugget_coverage)}
          lines={[`${n_covered ?? 0} coperti su ${n_nuggets ?? 0} nugget`]} />
        <BigCard title="Coverage required" value={required_coverage} color={mc(required_coverage)}
          lines={[`${n_required_covered ?? 0} su ${n_required ?? 0} required`]} />
        <BigCard title="Coverage optional" value={optional_coverage} color={mc(optional_coverage)}
          lines={[`${n_optional_covered ?? 0} su ${n_optional ?? 0} optional`]} />
      </div>

      {/* Evidenze provenienti da noise (solo se c'e' noise) */}
      {n_pairs_from_noise > 0 && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 14, flexWrap: 'wrap', marginBottom: 20, padding: '8px 14px', background: '#FFFBEB', border: '1px solid #FDE68A', borderRadius: 8, fontSize: 12, color: '#92400E' }}>
          <span style={{ fontWeight: 700 }}>⚠ Noise</span>
          <span><strong>{n_pairs_from_noise}</strong> coppie con evidenza da un passaggio di noise</span>
          <span style={{ color: 'var(--text-3)' }}>·</span>
          <span>di cui <strong>{n_pairs_correct_from_noise}</strong> matchano comunque una golden</span>
        </div>
      )}

      {/* Dettaglio per nugget */}
      {per_nugget.length > 0 && (
        <div>
          <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.7px', marginBottom: 10 }}>
            Dettaglio per nugget — {per_nugget.length} totali
          </div>

          {per_nugget.map((nug, i) => {
            const covered = nug.covered
            const score = nug.nugget_precision_score
            const statusLabel = covered ? 'Coperto ✓' : 'Non coperto'
            const statusBg = covered ? '#DCFCE7' : '#FEE2E2'
            const statusFg = covered ? '#166534' : '#991B1B'

            return (
              <div key={i} style={{ marginBottom: 8, border: `1px solid ${covered ? '#A7F3D0' : '#FECACA'}`, borderRadius: 8, overflow: 'hidden' }}>
                <div onClick={() => setExpanded(e => ({ ...e, [i]: !e[i] }))}
                  style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '10px 14px', background: covered ? '#F0FDF4' : '#FFF1F2', cursor: 'pointer' }}>
                  <span style={{ fontSize: 10, fontWeight: 700, background: statusBg, color: statusFg, padding: '2px 8px', borderRadius: 10, whiteSpace: 'nowrap', flexShrink: 0 }}>{statusLabel}</span>
                  <span style={{ fontFamily: 'var(--mono)', fontSize: 10, fontWeight: 600, color: 'var(--text-3)', flexShrink: 0 }}>{nug.nugget_id}</span>
                  <span style={{ fontSize: 12, color: 'var(--text)', flex: 1, lineHeight: 1.4 }}>{nug.nugget_text}</span>
                  {nug.required
                    ? <span style={{ fontSize: 9, fontWeight: 700, background: '#EDE9FE', color: '#5B21B6', padding: '2px 6px', borderRadius: 4, flexShrink: 0 }}>REQUIRED</span>
                    : <span style={{ fontSize: 9, fontWeight: 700, background: '#F3F4F6', color: '#6B7280', padding: '2px 6px', borderRadius: 4, flexShrink: 0 }}>OPTIONAL</span>}
                  {score != null && (
                    <span title="Miglior match evidenza ↔ golden evidence" style={{ fontSize: 10, fontWeight: 700, fontFamily: 'var(--mono)', background: score >= 0.5 ? '#ECFDF5' : '#FEF2F2', color: score >= 0.5 ? '#166534' : '#991B1B', padding: '2px 6px', borderRadius: 6, flexShrink: 0 }}>{score.toFixed(2)}</span>
                  )}
                  <span style={{ color: 'var(--text-3)', fontSize: 12 }}>{expanded[i] ? '▲' : '▼'}</span>
                </div>

                {expanded[i] && (
                  <div style={{ padding: '12px 16px', background: 'white' }}>
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
                        <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--text-3)', textTransform: 'uppercase', marginBottom: 4 }}>Claim con l'evidenza piu' vicina alla golden</div>
                        <span style={{ color: '#166534' }}>{nug.best_covering_claim}</span>
                        {score != null && (
                          <span style={{ marginLeft: 8, fontSize: 10, color: 'var(--text-3)', fontFamily: 'var(--mono)' }}>(match evidenza: {score.toFixed(2)})</span>
                        )}
                      </div>
                    )}
                    {nug.all_evidence && nug.all_evidence.length > 0 && (
                      <div style={{ marginTop: 8 }}>
                        <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.6px', marginBottom: 8 }}>
                          Evidenze che matchano la golden ({nug.all_evidence.length}) — ordinate per similarità decrescente
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
                    {!covered && (
                      <div style={{ fontSize: 12, color: 'var(--text-3)', fontStyle: 'italic' }}>Nessuna citazione prodotta matcha la golden evidence di questo nugget.</div>
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

// ── DeepSeekMetricsView (verdetto binario) ──────────────────────────────────────

export function DeepSeekMetricsView({ metrics, onSave, onDownload }) {
  const [expanded, setExpanded] = useState({})
  const {
    citation_precision, citation_recall,
    n_claims, n_pairs, n_supported, n_not_supported, per_claim = [],
  } = metrics

  const nSup = n_supported ?? metrics.n_pairs_supported ?? 0
  const nNot = n_not_supported ?? Math.max(0, (n_pairs ?? 0) - nSup)

  const pct = v => `${Math.round((v ?? 0) * 100)}%`

  const STYLE_BY_VERDICT = {
    supported:     { bg: '#F0FDF4', bd: '#86EFAC',       pillBg: '#DCFCE7', pillFg: '#166534', label: 'SUPPORTED' },
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
            <span style={{ color: '#166534', fontWeight: 600 }}>{nSup} supported</span>
            <span style={{ color: 'var(--text-3)' }}>·</span>
            <span style={{ color: '#991B1B', fontWeight: 600 }}>{nNot} not supported</span>
          </span>
        )}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12, marginBottom: 20 }}>
        {[
          { key: 'citation_precision', value: citation_precision, sub: `${nSup} coppie supported / ${n_pairs} coppie` },
          { key: 'citation_recall',    value: citation_recall,    sub: `claim con ≥1 evidenza supported su ${n_claims} totali` },
        ].map(({ key, value, sub }) => {
          const color = metricColor(key, value)
          return (
            <div key={key} style={{ padding: '16px 18px', background: 'white', border: `1px solid ${cardBorder(color)}`, borderTop: `3px solid ${color}`, borderRadius: 10 }}>
              <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.6px', marginBottom: 8 }}>{METRIC_INFO_DEEPSEEK[key].label}</div>
              <div style={{ fontSize: 30, fontWeight: 800, color, lineHeight: 1, marginBottom: 8 }}>{pct(value)}</div>
              <div style={{ fontSize: 11, color: 'var(--text-3)', lineHeight: 1.4 }}>{sub}</div>
              <div style={{ marginTop: 6, fontSize: 11, color: 'var(--text-3)', fontStyle: 'italic' }}>{METRIC_INFO_DEEPSEEK[key].desc}</div>
            </div>
          )
        })}
      </div>

      {per_claim.length > 0 && (
        <div>
          <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.7px', marginBottom: 10 }}>
            Dettaglio per claim — {per_claim.length} totali
          </div>
          {per_claim.map((c, i) => {
            const ok = c.any_supported
            const nSupC = c.n_supported ?? 0
            const pillLabel = ok ? `${nSupC} supported / ${c.n_passages}` : 'nessun supporto'

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
                      const st = STYLE_BY_VERDICT[v] || STYLE_BY_VERDICT.not_supported
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