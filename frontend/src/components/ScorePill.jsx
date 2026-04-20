export default function ScorePill({ score }) {
  const v = typeof score === 'number' ? score : 0
  const cls = v >= 0.8 ? 'score-high' : v >= 0.5 ? 'score-mid' : 'score-low'
  return <span className={`score-pill ${cls}`}>{v.toFixed(3)}</span>
}