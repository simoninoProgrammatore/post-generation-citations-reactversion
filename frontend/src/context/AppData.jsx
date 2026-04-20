/**
 * AppDataContext — state globale condiviso tra le pagine.
 *
 * Contiene:
 *  - pipelineResults: storico delle esecuzioni pipeline complete
 *  - attentionRecords: risultati calcolati nella pagina Attention
 *  - igResult, patchResult: ultimi risultati interpretability
 *
 * Persistenza: salva su localStorage a ogni update.
 */

import { createContext, useContext, useState, useEffect } from 'react'

const KEY = 'citation_pipeline_data_v1'

// Default state — tutto vuoto all'avvio
const DEFAULT = {
  pipelineResults: [],   // [{ question, raw_response, claims, matched, cited_response, references, metrics, created_at }, ...]
  attentionRecords: [],
  igResult: null,
  patchResult: null,
}

const AppDataContext = createContext(null)

export function AppDataProvider({ children }) {
  const [data, setData] = useState(() => {
    try {
      const raw = localStorage.getItem(KEY)
      return raw ? { ...DEFAULT, ...JSON.parse(raw) } : DEFAULT
    } catch {
      return DEFAULT
    }
  })

  useEffect(() => {
    try {
      localStorage.setItem(KEY, JSON.stringify(data))
    } catch {
      // localStorage può fallire se il JSON è troppo grande (>5MB) — ignoriamo
    }
  }, [data])

  // ── Updaters ─────────────────────────────────────────────────────────────
  function addPipelineResult(result) {
    const withMeta = { ...result, created_at: new Date().toISOString() }
    setData(d => ({ ...d, pipelineResults: [...d.pipelineResults, withMeta] }))
  }

  function addAttentionRecord(record) {
    setData(d => ({ ...d, attentionRecords: [record, ...d.attentionRecords] }))
  }

  function setIgResult(result) {
    setData(d => ({ ...d, igResult: result }))
  }

  function setPatchResult(result) {
    setData(d => ({ ...d, patchResult: result }))
  }

  function clearAll() {
    setData(DEFAULT)
  }

  return (
    <AppDataContext.Provider value={{
      ...data,
      addPipelineResult,
      addAttentionRecord,
      setIgResult,
      setPatchResult,
      clearAll,
    }}>
      {children}
    </AppDataContext.Provider>
  )
}

export function useAppData() {
  const ctx = useContext(AppDataContext)
  if (!ctx) throw new Error('useAppData must be used inside <AppDataProvider>')
  return ctx
}