/**
 * api.js — Wrapper attorno a fetch() per chiamare il backend FastAPI.
 *
 * Tutte le chiamate passano per il proxy Vite (/api → localhost:8000).
 * Ogni funzione restituisce una Promise che si risolve con il JSON parsato.
 */

const BASE = '/api'

/**
 * Helper generico per richieste POST con body JSON.
 * In caso di errore HTTP, prova a estrarre il messaggio dal backend e lo lancia.
 */
async function postJson(path, body) {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`
    try {
      const err = await res.json()
      if (err.detail) msg = typeof err.detail === 'string' ? err.detail : JSON.stringify(err.detail)
    } catch {}
    throw new Error(msg)
  }
  return res.json()
}

async function getJson(path) {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`
    try {
      const err = await res.json()
      if (err.detail) msg = typeof err.detail === 'string' ? err.detail : JSON.stringify(err.detail)
    } catch {}
    throw new Error(msg)
  }
  return res.json()
}

// ── Pipeline ────────────────────────────────────────────────────────────
export const pipeline = {
  generate:        (body) => postJson('/pipeline/generate',          body),
  decompose:       (body) => postJson('/pipeline/decompose',         body),
  retrieve:        (body) => postJson('/pipeline/retrieve',          body),
  retrieveSingle:  (body) => postJson('/pipeline/retrieve-single',   body),
  retrieveDebug:   (body) => postJson('/pipeline/retrieve/debug',    body),
  cite:            (body) => postJson('/pipeline/cite',              body),
  evaluate:        (body) => postJson('/pipeline/evaluate',          body),
  evaluateNuggets: (body) => postJson('/pipeline/evaluate-nuggets',  body),
  evaluateDataset: (body) => postJson('/pipeline/evaluate-dataset',  body),
  evaluateExample: (body) => postJson('/pipeline/evaluate-example',  body),
}

// ── NLI ─────────────────────────────────────────────────────────────────
export const nli = {
  predict: (body) => postJson('/nli/predict', body),
}

// ── Interpretability ────────────────────────────────────────────────────
export const interpret = {
  ig:        (body) => postJson('/interpret/ig',        body),
  patching:  (body) => postJson('/interpret/patching',  body),
  attention: (body) => postJson('/interpret/attention', body),
}

// ── Dataset ─────────────────────────────────────────────────────────────
export const dataset = {
  list: ()         => getJson('/dataset/list'),
  load: (filename) => getJson(`/dataset/load/${encodeURIComponent(filename)}`),
}

// ── Default export ──────────────────────────────────────────────────────
export default { pipeline, nli, interpret, dataset }