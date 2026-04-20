export default function LayerChart({ layers }) {
  if (!layers || !layers.length) {
    return <div style={{ color: 'var(--text-3)', fontSize: 12 }}>Nessun dato layer.</div>
  }

  const n = layers.length
  // Width scala col numero di layer (min 28px per layer, min 500px totale)
  const perLayer = 28
  const W = Math.max(500, n * perLayer + 48)
  const H = 180
  const padL = 36, padR = 12, padT = 12, padB = 32
  const w = W - padL - padR, h = H - padT - padB
  const maxY = 1.0, minY = 0.3

  const xOf = i => padL + (n === 1 ? w / 2 : (i / (n - 1)) * w)
  const yOf = v => padT + h - ((v - minY) / (maxY - minY)) * h

  const pts = layers.map((l, i) =>
    `${xOf(i).toFixed(1)},${yOf(l.mean_hyp_dominance).toFixed(1)}`
  ).join(' ')

  const thresholdY = yOf(0.65)

  // Con molti layer, mostra label solo ogni N (es. ogni 2 o 3)
  const labelEvery = n > 16 ? 2 : 1

  return (
    <div style={{ overflowX: 'auto' }}>
      <svg width={W} height={H} style={{ display: 'block' }}>
        {/* Gridlines orizzontali */}
        {[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0].map(v => (
          <g key={v}>
            <line x1={padL} x2={padL + w} y1={yOf(v)} y2={yOf(v)} stroke="#E2E8F0" strokeWidth={0.5} />
            <text x={padL - 4} y={yOf(v) + 4} fill="#94A3B8" fontSize={9} textAnchor="end">
              {v.toFixed(1)}
            </text>
          </g>
        ))}

        {/* Threshold 0.65 */}
        <line
          x1={padL} x2={padL + w} y1={thresholdY} y2={thresholdY}
          stroke="#F59E0B" strokeWidth={1} strokeDasharray="4,3" opacity={0.6}
        />

        {/* Linea */}
        <polyline points={pts} fill="none" stroke="#EF4444" strokeWidth={2} strokeLinejoin="round" />

        {/* Punti */}
        {layers.map((l, i) => (
          <circle key={i} cx={xOf(i)} cy={yOf(l.mean_hyp_dominance)} r={3} fill="#EF4444" />
        ))}

        {/* Label layer sotto l'asse X */}
        {layers.map((l, i) => {
          if (i % labelEvery !== 0 && i !== n - 1) return null
          return (
            <text
              key={`lbl-${i}`}
              x={xOf(i)}
              y={padT + h + 16}
              fill="#94A3B8"
              fontSize={9}
              textAnchor="middle"
            >
              L{l.layer}
            </text>
          )
        })}

        {/* Assi */}
        <line x1={padL} x2={padL} y1={padT} y2={padT + h} stroke="#E2E8F0" strokeWidth={1} />
        <line x1={padL} x2={padL + w} y1={padT + h} y2={padT + h} stroke="#E2E8F0" strokeWidth={1} />
      </svg>
    </div>
  )
}