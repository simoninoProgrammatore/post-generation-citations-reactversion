/**
 * download.js — Helper per scaricare dati come file JSON.
 */

export function downloadJSON(data, filename) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename.endsWith('.json') ? filename : `${filename}.json`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

export function timestampedFilename(prefix) {
  const now = new Date()
  const ts = now.toISOString().slice(0, 19).replace(/[:T]/g, '-')
  return `${prefix}_${ts}.json`
}