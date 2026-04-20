export default function LayerChart({ layers }) {
  if (!layers || !layers.length) {
    return <div style={{ color: 'var(--text-3)', fontSize: 12 }}>Nessun dato layer.</div>
  }

  const W = 460, H = 160, padL = 36, padR = 12, padT = 12, padB = 24
  const w = W - padL - padR, h = H - padT - padB
  const maxY = 1.0, minY = 0.3

  const xOf = i => padL + (i / Math.max(layers.length - 1, 1)) * w
  const yOf = v => padT + h - ((v - minY) / (maxY - minY)) * h

  const pts = layers.map((l, i) =>
    `${xOf(i).toFixed(1)},${yOf(l.mean_hyp_dominance).toFixed(1)}`
  ).join(' ')

  const thresholdY = yOf(0.65)

  return (
    <svg width={W} height={H} style={{ overflow: 'visible' }}>
      {[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0].map(v => (
        <g key={v}>
          <line x1={padL} x2={padL + w} y1={yOf(v)} y2={yOf(v)} stroke="#E2E8F0" strokeWidth={0.5} />
          <text x={padL - 4} y={yOf(v) + 4} fill="#94A3B8" fontSize={9} textAnchor="end">
            {v.toFixed(1)}
          </text>
        </g>
      ))}
      <line
        x1={padL} x2={padL + w} y1={thresholdY} y2={thresholdY}
        stroke="#F59E0B" strokeWidth={1} strokeDasharray="4,3" opacity={0.6}
      />
      <polyline points={pts} fill="none" stroke="#EF4444" strokeWidth={2} strokeLinejoin="round" />
      {layers.map((l, i) => (
        <circle key={i} cx={xOf(i)} cy={yOf(l.mean_hyp_dominance)} r={3} fill="#EF4444" />
      ))}
      {layers.map((l, i) => (
        <text key={`lbl-${i}`} x={xOf(i)} y={padT + h + 14}
          fill="#94A3B8" fontSize={8} textAnchor="middle">L{l.layer}</text>
      ))}
      <line x1={padL} x2={padL} y1={padT} y2={padT + h} stroke="#E2E8F0" strokeWidth={1} />
      <line x1={padL} x2={padL + w} y1={padT + h} y2={padT + h} stroke="#E2E8F0" strokeWidth={1} />
    </svg>
  )
}