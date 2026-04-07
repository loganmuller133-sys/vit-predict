const API_BASE_URL = import.meta.env.VITE_API_URL || ""
const API_KEY = import.meta.env.VITE_API_KEY || "dev_api_key_12345"

function defaultHeaders() {
  const headers = {
    "Content-Type": "application/json"
  }

  if (API_KEY) {
    headers["x-api-key"] = API_KEY
  }

  return headers
}

async function apiFetch(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: defaultHeaders(),
    ...options
  })

  if (!response.ok) {
    const text = await response.text()
    const message = text || response.statusText || "Request failed"
    throw new Error(message)
  }

  return response.json()
}

export async function fetchHealth() {
  return apiFetch("/health")
}

export async function fetchHistory(limit = 10, offset = 0) {
  return apiFetch(`/history?limit=${limit}&offset=${offset}`)
}

export async function predictMatch(matchData) {
  return apiFetch("/predict", {
    method: "POST",
    body: JSON.stringify(matchData)
  })
}
