/**
 * Sidebar — basata sul mockup.
 * Usa NavLink di react-router per gestire lo stato attivo automaticamente.
 */

import { NavLink } from 'react-router-dom'
import Icon from './Icon'

const NAV = [
  { path: '/pipeline',  icon: 'pipeline',  label: 'Pipeline interattivo' },
  { path: '/explore',   icon: 'explore',   label: 'Esplora risultati' },
  { path: '/metrics',   icon: 'metrics',   label: 'Metriche' },
  { path: '/dataset',   icon: 'database',  label: 'Valutazione Dataset'},
  { path: '/attention', icon: 'attention', label: 'Attention Analysis' },
  { path: '/interpret', icon: 'interpret', label: 'Interpretability' },
]

export default function Sidebar() {
  return (
    <div className="sidebar">
      {/* Logo */}
      <div className="sb-logo">
        <div className="sb-logo-mark">
          <div className="sb-logo-icon">
            <Icon name="book" size={18} color="white" strokeWidth={1.75} />
          </div>
          <div>
            <div className="sb-logo-text">Citation Pipeline</div>
            <div className="sb-logo-sub">Post-Generation System</div>
          </div>
        </div>
      </div>

      {/* Nav */}
      <div className="sb-nav">
        <div className="sb-section-label">Navigazione</div>
        {NAV.map(item => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) => `sb-item${isActive ? ' active' : ''}`}
            style={item.indent ? { paddingLeft: 28 } : undefined}
          >
            <span className="sb-item-icon">
              <Icon name={item.icon} size={item.indent ? 13 : 15} strokeWidth={1.6} />
            </span>
            <span style={{ flex: 1, fontSize: item.indent ? 12 : undefined }}>
              {item.label}
            </span>
          </NavLink>
        ))}
      </div>

      {/* Footer */}
      <div className="sb-footer">
        <div style={{ padding: '6px 10px 4px' }}>
          <div style={{ fontSize: 11, color: 'rgba(241,245,249,0.55)', lineHeight: 1.7 }}>
            Post-Generation Citation System
            <br />
            Tesi triennale · 2026
          </div>
          <div style={{
            display: 'flex', alignItems: 'center', gap: 6,
            marginTop: 10, padding: '6px 0',
            borderTop: '1px solid rgba(255,255,255,0.06)'
          }}>
            <div style={{
              width: 6, height: 6, borderRadius: '50%',
              background: '#10B981', flexShrink: 0
            }} />
            <span style={{ fontSize: 11, color: 'rgba(241,245,249,0.5)' }}>
              Backend connesso
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}