import { useEffect, useMemo, useState } from 'react'
import { fetchHealth, fetchHistory, predictMatch } from './api'
import './App.css'

const DEFAULT_FORM = {
  home_team: '',
  away_team: '',
  league: 'premier_league',
  kickoff_time: new Date().toISOString().slice(0, 16),
  home: 2.0,
  draw: 3.2,
  away: 3.8
}

const TEAMS = [
  { name: 'Arsenal', league: 'premier_league' },
  { name: 'Chelsea', league: 'premier_league' },
  { name: 'Manchester United', league: 'premier_league' },
  { name: 'Manchester City', league: 'premier_league' },
  { name: 'Liverpool', league: 'premier_league' },
  { name: 'Real Madrid', league: 'la_liga' },
  { name: 'Barcelona', league: 'la_liga' },
  { name: 'Bayern Munich', league: 'bundesliga' },
  { name: 'Borussia Dortmund', league: 'bundesliga' }
]

function App() {
  const [health, setHealth] = useState(null)
  const [history, setHistory] = useState([])
  const [form, setForm] = useState(DEFAULT_FORM)
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [page, setPage] = useState(0)
  const itemsPerPage = 5

  const marketOdds = useMemo(
    () => ({ home: parseFloat(form.home), draw: parseFloat(form.draw), away: parseFloat(form.away) }),
    [form.home, form.draw, form.away]
  )

  useEffect(() => {
    fetchHealthStatus()
    loadHistory()
    const healthInterval = setInterval(fetchHealthStatus, 15000)
    return () => clearInterval(healthInterval)
  }, [])

  async function fetchHealthStatus() {
    try {
      const result = await fetchHealth()
      setHealth(result)
    } catch (err) {
      setError(err.message)
    }
  }

  async function loadHistory() {
    try {
      const result = await fetchHistory(100, 0)
      setHistory(result.predictions || [])
      setPage(0)
    } catch (err) {
      setError(err.message)
    }
  }

  async function submitPrediction(event) {
    event.preventDefault()
    
    if (!form.home_team.trim() || !form.away_team.trim()) {
      setError('Please enter both team names')
      return
    }

    if (form.home_team === form.away_team) {
      setError('Home and away teams must be different')
      return
    }

    setLoading(true)
    setError('')
    setPrediction(null)

    try {
      const payload = {
        home_team: form.home_team.trim(),
        away_team: form.away_team.trim(),
        league: form.league,
        kickoff_time: new Date(form.kickoff_time).toISOString(),
        market_odds: marketOdds
      }

      const result = await predictMatch(payload)
      setPrediction(result)
      loadHistory()
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  function updateField(key, value) {
    setForm((current) => ({ ...current, [key]: value }))
  }

  const paginatedHistory = history.slice(page * itemsPerPage, (page + 1) * itemsPerPage)
  const maxPages = Math.ceil(history.length / itemsPerPage)

  return (
    <div className="app-shell">
      <header className="hero-panel">
        <div>
          <h1>⚽ VIT Predict</h1>
          <p>12-Model Ensemble for Football Match Predictions</p>
        </div>
        <div className="status-card">
          <h2>System Status</h2>
          {health ? (
            <ul>
              <li>
                Status: <strong>{health.status === 'ok' ? '✓ Online' : '✗ Offline'}</strong>
              </li>
              <li>
                Models: <strong>{health.models_loaded || 0}/12</strong>
              </li>
              <li>
                Database: <strong>{health.db_connected ? '✓ Connected' : '✗ Disconnected'}</strong>
              </li>
              <li>
                CLV Tracking: <strong>{health.clv_tracking_enabled ? '✓ Enabled' : '✗ Disabled'}</strong>
              </li>
            </ul>
          ) : (
            <p style={{ color: '#94a3b8', margin: '0' }}>Connecting...</p>
          )}
        </div>
      </header>

      <main>
        <section className="panel">
          <h2>🎯 Make a Prediction</h2>
          <form className="prediction-form" onSubmit={submitPrediction}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '20px' }}>
              <div className="field-group">
                <label htmlFor="home_team">Home Team</label>
                <input
                  id="home_team"
                  type="text"
                  placeholder="e.g., Arsenal"
                  value={form.home_team}
                  onChange={(e) => updateField('home_team', e.target.value)}
                  required
                />
              </div>
              <div className="field-group">
                <label htmlFor="away_team">Away Team</label>
                <input
                  id="away_team"
                  type="text"
                  placeholder="e.g., Chelsea"
                  value={form.away_team}
                  onChange={(e) => updateField('away_team', e.target.value)}
                  required
                />
              </div>
              <div className="field-group">
                <label htmlFor="league">League</label>
                <select
                  id="league"
                  value={form.league}
                  onChange={(e) => updateField('league', e.target.value)}
                >
                  <option value="premier_league">Premier League</option>
                  <option value="la_liga">La Liga</option>
                  <option value="bundesliga">Bundesliga</option>
                  <option value="serie_a">Serie A</option>
                  <option value="ligue_1">Ligue 1</option>
                </select>
              </div>
            </div>

            <div style={{ marginTop: '20px' }}>
              <div className="field-group">
                <label htmlFor="kickoff_time">Kickoff Time</label>
                <input
                  id="kickoff_time"
                  type="datetime-local"
                  value={form.kickoff_time}
                  onChange={(e) => updateField('kickoff_time', e.target.value)}
                  required
                />
              </div>
            </div>

            <div style={{ marginTop: '20px' }}>
              <label style={{ fontWeight: 600, color: '#334155', marginBottom: '12px', display: 'block' }}>
                Market Odds
              </label>
              <div className="market-grid">
                <div className="field-group">
                  <label htmlFor="home">Home Win</label>
                  <input
                    id="home"
                    type="number"
                    min="1"
                    step="0.01"
                    placeholder="2.00"
                    value={form.home}
                    onChange={(e) => updateField('home', e.target.value)}
                    required
                  />
                </div>
                <div className="field-group">
                  <label htmlFor="draw">Draw</label>
                  <input
                    id="draw"
                    type="number"
                    min="1"
                    step="0.01"
                    placeholder="3.50"
                    value={form.draw}
                    onChange={(e) => updateField('draw', e.target.value)}
                    required
                  />
                </div>
                <div className="field-group">
                  <label htmlFor="away">Away Win</label>
                  <input
                    id="away"
                    type="number"
                    min="1"
                    step="0.01"
                    placeholder="3.00"
                    value={form.away}
                    onChange={(e) => updateField('away', e.target.value)}
                    required
                  />
                </div>
              </div>
            </div>

            <button type="submit" className="primary-button" disabled={loading}>
              {loading ? 'Generating Prediction...' : 'Get Prediction'}
            </button>
          </form>

          {error && <div className="alert error">{error}</div>}

          {prediction && (
            <div className="result-card">
              <h3>📊 Prediction Results</h3>
              <dl>
                <div>
                  <dt>Match ID</dt>
                  <dd style={{ fontSize: '0.9rem', color: '#64748b' }}>#{prediction.match_id}</dd>
                </div>
                <div>
                  <dt>Home Win</dt>
                  <dd>{(prediction.home_prob * 100).toFixed(1)}%</dd>
                </div>
                <div>
                  <dt>Draw</dt>
                  <dd>{(prediction.draw_prob * 100).toFixed(1)}%</dd>
                </div>
                <div>
                  <dt>Away Win</dt>
                  <dd>{(prediction.away_prob * 100).toFixed(1)}%</dd>
                </div>
                {prediction.over_25_prob !== undefined && (
                  <>
                    <div>
                      <dt>Over 2.5 Goals</dt>
                      <dd>{(prediction.over_25_prob * 100).toFixed(1)}%</dd>
                    </div>
                    <div>
                      <dt>BTTS</dt>
                      <dd>{(prediction.btts_prob * 100).toFixed(1)}%</dd>
                    </div>
                  </>
                )}
                {prediction.consensus_prob !== undefined && (
                  <div>
                    <dt>Consensus</dt>
                    <dd>{(prediction.consensus_prob * 100).toFixed(1)}%</dd>
                  </div>
                )}
                <div>
                  <dt>Expected Value</dt>
                  <dd style={{ color: prediction.final_ev > 0 ? '#10b981' : '#ef4444', fontWeight: 700 }}>
                    {(prediction.final_ev * 100).toFixed(2)}%
                  </dd>
                </div>
                <div>
                  <dt>Recommended Stake</dt>
                  <dd style={{ color: '#0ea5e9', fontWeight: 700 }}>
                    {(prediction.recommended_stake * 100).toFixed(2)}%
                  </dd>
                </div>
                {prediction.confidence !== undefined && (
                  <div>
                    <dt>Confidence</dt>
                    <dd style={{ color: '#f97316' }}>
                      {(prediction.confidence * 100).toFixed(0)}%
                    </dd>
                  </div>
                )}
              </dl>
            </div>
          )}
        </section>

        {history.length > 0 && (
          <section className="panel history-panel">
            <div className="panel-header">
              <h2>📈 Prediction History</h2>
              <button type="button" onClick={loadHistory} className="secondary-button">
                Refresh
              </button>
            </div>

            {history.length === 0 ? (
              <p style={{ color: '#64748b' }}>No history yet. Generate a prediction to begin.</p>
            ) : (
              <>
                <div className="history-table-wrapper">
                  <table className="history-table">
                    <thead>
                      <tr>
                        <th>Match</th>
                        <th>Home %</th>
                        <th>Draw %</th>
                        <th>Away %</th>
                        <th>Edge</th>
                        <th>Stake</th>
                        <th>Time</th>
                      </tr>
                    </thead>
                    <tbody>
                      {paginatedHistory.map((item) => (
                        <tr key={`${item.match_id}-${item.timestamp}`}>
                          <td style={{ fontWeight: 500 }}>
                            <span style={{ color: '#64748b', fontSize: '0.85rem' }}>#{item.match_id}</span> {item.home_team.split(' ').slice(-1)[0]} v {item.away_team.split(' ').slice(-1)[0]}
                          </td>
                          <td>{(item.home_prob * 100).toFixed(1)}%</td>
                          <td>{(item.draw_prob * 100).toFixed(1)}%</td>
                          <td>{(item.away_prob * 100).toFixed(1)}%</td>
                          <td style={{ color: (item.final_ev || item.edge) > 0 ? '#10b981' : '#ef4444', fontWeight: 600 }}>
                            {((item.final_ev || item.edge) * 100).toFixed(2)}%
                          </td>
                          <td>{(item.recommended_stake * 100).toFixed(2)}%</td>
                          <td style={{ color: '#94a3b8', fontSize: '0.9rem' }}>
                            {item.timestamp ? new Date(item.timestamp).toLocaleString('en-US', { 
                              month: 'short', 
                              day: '2-digit', 
                              hour: '2-digit', 
                              minute: '2-digit'
                            }) : '-'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {maxPages > 1 && (
                  <div style={{ marginTop: '20px', display: 'flex', gap: '10px', justifyContent: 'center', alignItems: 'center' }}>
                    <button
                      className="secondary-button"
                      onClick={() => setPage(Math.max(0, page - 1))}
                      disabled={page === 0}
                    >
                      ← Previous
                    </button>
                    <span style={{ color: '#64748b', fontWeight: 500 }}>
                      Page {page + 1} of {maxPages}
                    </span>
                    <button
                      className="secondary-button"
                      onClick={() => setPage(Math.min(maxPages - 1, page + 1))}
                      disabled={page >= maxPages - 1}
                    >
                      Next →
                    </button>
                  </div>
                )}
              </>
            )}
          </section>
        )}
      </main>
    </div>
  )
}

export default App
