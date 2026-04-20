export default function Explore() {
  return (
    <div>
      <div className="page-header">
        <div className="page-header-title">Esplora risultati</div>
        <div className="page-header-sub">
          Naviga i risultati prodotti dal pipeline step-by-step.
        </div>
      </div>

      <div className="empty-state">
        <div className="empty-state-title">Nessun risultato disponibile</div>
        <div className="empty-state-hint">
          Esegui prima il pipeline oppure carica un file JSON di risultati.
        </div>
      </div>
    </div>
  )
}