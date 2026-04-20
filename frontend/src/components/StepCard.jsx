import { useState, useEffect } from 'react'
import Icon from './Icon'

export default function StepCard({
  num, title, status, children, onRun, running, runLabel = 'Run', disabled = false,
}) {
  const [open, setOpen] = useState(status !== 'done')
  useEffect(() => { setOpen(status !== 'done') }, [status])

  const borderColor =
    status === 'active' || status === 'running' ? 'var(--accent)' :
    status === 'done' ? '#A7F3D0' :
    'var(--border)'

  const headerBg = status === 'active' || status === 'running'
    ? 'linear-gradient(90deg,#F5F3FF,#F8FAFC)'
    : 'var(--card)'

  const numBg =
    status === 'done' ? '#DCFCE7' :
    (status === 'active' || status === 'running') ? 'var(--accent)' :
    'var(--border-2)'

  const numColor =
    status === 'done' ? '#166534' :
    (status === 'active' || status === 'running') ? 'white' :
    'var(--text-3)'

  return (
    <div style={{
      marginBottom: 12, borderRadius: 12, overflow: 'hidden',
      border: `1px solid ${borderColor}`,
      background: 'var(--card)',
      boxShadow: 'var(--shadow)',
      opacity: status === 'locked' ? 0.5 : 1,
      transition: 'opacity 0.3s, border-color 0.3s',
    }}>
      {/* Header */}
      <div
        style={{
          display: 'flex', alignItems: 'center', gap: 12,
          padding: '14px 20px',
          cursor: status === 'done' ? 'pointer' : 'default',
          borderBottom: open ? '1px solid var(--border-2)' : 'none',
          background: headerBg,
        }}
        onClick={() => status === 'done' && setOpen(o => !o)}
      >
        <div style={{
          width: 28, height: 28, borderRadius: 8, flexShrink: 0,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 12, fontWeight: 700,
          background: numBg, color: numColor,
          border: status === 'done'
            ? '1px solid #A7F3D0'
            : (status === 'active' || status === 'running')
              ? 'none'
              : '1px solid var(--border)',
        }}>
          {status === 'done'
            ? <Icon name="check" size={14} strokeWidth={2.5} />
            : status === 'running'
              ? <span className="spinner" style={{ width: 14, height: 14 }} />
              : num}
        </div>

        <span style={{ fontSize: 15, fontWeight: 700, flex: 1, color: 'var(--text)' }}>
          {title}
        </span>

        {status === 'done' && (
          <span className="badge badge-green">
            <Icon name="check" size={11} strokeWidth={2.5} /> Completato
          </span>
        )}
        {status === 'locked' && (
          <span style={{ fontSize: 11, color: 'var(--text-3)' }}>Bloccato</span>
        )}
        {status === 'done' && (
          <Icon name={open ? 'chevronUp' : 'chevronDown'} size={13} strokeWidth={2} color="var(--text-3)" />
        )}
      </div>

      {/* Body */}
      {open && status !== 'locked' && (
        <div style={{ padding: 20 }}>
          {children}
          {onRun && status !== 'done' && (
            <button
              className="btn btn-primary"
              style={{ marginTop: 16 }}
              onClick={onRun}
              disabled={running || disabled}
            >
              {running
                ? <>
                    <span className="spinner" style={{
                      width: 14, height: 14,
                      borderColor: 'rgba(255,255,255,0.3)',
                      borderTopColor: 'white',
                    }} /> Calcolo...
                  </>
                : <>
                    <Icon name="play" size={13} color="white" strokeWidth={2} /> {runLabel}
                  </>}
            </button>
          )}
        </div>
      )}
    </div>
  )
}