export default function Attention() {
  return (
    <div>
      <div className="page-header">
        <div className="page-header-title">Attention Analysis</div>
        <div className="page-header-sub">
          Visualizza gli attention weights di DeBERTa per rilevare parametric knowledge leakage.
        </div>
      </div>

      <div className="empty-state">
        <div className="empty-state-title">Nessun record di attention disponibile</div>
        <div className="empty-state-hint">
          Carica un file JSON di attention records oppure calcola live un esempio.
        </div>
      </div>
    </div>
  )
}