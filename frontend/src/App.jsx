/**
 * App.jsx — Shell dell'applicazione.
 * Layout fisso: sidebar a sinistra + main content (topbar + page content).
 * Le pagine vengono montate dinamicamente via <Outlet/> di React Router.
 */

import { Outlet, useLocation } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import Icon from './components/Icon'

const PAGE_TITLES = {
  '/pipeline':  'Pipeline interattivo',
  '/explore':   'Esplora risultati',
  '/metrics':   'Metriche di valutazione',
  '/attention': 'Attention Analysis',
  '/interpret': 'Interpretability',
}

export default function App() {
  const location = useLocation()
  const title = PAGE_TITLES[location.pathname] || 'Citation Pipeline'

  return (
    <>
      <Sidebar />
      <div className="content">
        {/* Topbar */}
        <div className="topbar">
          <span className="topbar-breadcrumb">Post-Gen Citations</span>
          <span style={{ color: 'var(--text-3)', fontSize: 12 }}>›</span>
          <span className="topbar-title">{title}</span>
          <div className="topbar-spacer" />
          <div className="topbar-chip">
            <div className="status-dot" />
            claude-haiku-4-5
          </div>
        </div>

        {/* Page content */}
        <div className="page-content">
          <Outlet />
        </div>
      </div>
    </>
  )
}