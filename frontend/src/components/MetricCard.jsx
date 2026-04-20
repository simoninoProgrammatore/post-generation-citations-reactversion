export default function MetricCard({ label, value, color, desc, isUnsupported }) {
  const v = typeof value === 'number' ? value : 0
  const pct = isUnsupported ? (1 - v) * 100 : v * 100
  return (
    <div className="metric-card">
      <div className="metric-value" style={{ color }}>{v.toFixed(3)}</div>
      <div className="metric-label">{label}</div>
      <div className="metric-bar-track">
        <div className="metric-bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      {desc && (
        <div style={{ marginTop: 8, fontSize: 11, color: 'var(--text-3)', lineHeight: 1.4 }}>
          {desc}
        </div>
      )}
    </div>
  )
}