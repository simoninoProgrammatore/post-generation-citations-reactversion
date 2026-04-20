export default function Interpretability() {
  return (
    <div>
      <div className="page-header">
        <div className="page-header-title">Interpretability</div>
        <div className="page-header-sub">
          Integrated Gradients e Activation Patching per analizzare dove DeBERTa prende decisioni biased.
        </div>
      </div>

      <div className="empty-state">
        <div className="empty-state-title">Nessuna analisi eseguita</div>
        <div className="empty-state-hint">
          Lancia un'analisi Integrated Gradients o Activation Patching per vedere i risultati.
        </div>
      </div>
    </div>
  )
}