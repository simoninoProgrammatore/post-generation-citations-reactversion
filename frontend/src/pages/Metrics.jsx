export default function Metrics() {
  return (
    <div>
      <div className="page-header">
        <div className="page-header-title">Metriche di valutazione</div>
        <div className="page-header-sub">
          Riepilogo aggregato e distribuzione per esempio.
        </div>
      </div>

      <div className="empty-state">
        <div className="empty-state-title">Nessuna metrica disponibile</div>
        <div className="empty-state-hint">
          Esegui il pipeline oppure carica un file JSON di risultati.
        </div>
      </div>
    </div>
  )
}