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

function App() {
  const [health, setHealth] = useState(null)
  const [history, setHistory] = useState([])
  const [form, setForm] = useState(DEFAULT_FORM)
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const marketOdds = useMemo(
    () => ({ home: parseFloat(form.home), draw: parseFloat(form.draw), away: parseFloat(form.away) }),
    [form.home, form.draw, form.away]
  )

  useEffect(() => {
    fetchHealthStatus()
    loadHistory()
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
      const result = await fetchHistory(8, 0)
      setHistory(result.predictions || [])
    } catch (err) {
      setError(err.message)
    }
  }

  async function submitPrediction(event) {
    event.preventDefault()
    setLoading(true)
    setError('')
    setPrediction(null)

    try {
      const payload = {
        home_team: form.home_team,
        away_team: form.away_team,
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

  return (
    <div className="app-shell">
      <header className="hero-panel">
        <div>
          <h1>VIT Network Dashboard</h1>
          <p>Generate football predictions and review recent model history.</p>
        </div>
        <div className="status-card">
          <h2>Health</h2>
          {health ? (
            <ul>
              <li>Status: <strong>{health.status}</strong></li>
              <li>Models loaded: <strong>{health.models_loaded}</strong></li>
              <li>DB connected: <strong>{health.db_connected ? 'yes' : 'no'}</strong></li>
            </ul>
          ) : (
            <p>Loading health...</p>
          )}
        </div>
      </header>

      <main>
        <section className="panel">
          <h2>Predict a Match</h2>
          <form className="prediction-form" onSubmit={submitPrediction}>
            <div className="field-group">
              <label htmlFor="home_team">Home team</label>
              <input
                id="home_team"
                value={form.home_team}
                onChange={(e) => updateField('home_team', e.target.value)}
                required
              />
            </div>
            <div className="field-group">
              <label htmlFor="away_team">Away team</label>
              <input
                id="away_team"
                value={form.away_team}
                onChange={(e) => updateField('away_team', e.target.value)}
                required
              />
            </div>
            <div className="field-group">
              <label htmlFor="league">League</label>
              <input
                id="league"
                value={form.league}
                onChange={(e) => updateField('league', e.target.value)}
                required
              />
            </div>
            <div className="field-group">
              <label htmlFor="kickoff_time">Kickoff time</label>
              <input
                id="kickoff_time"
                type="datetime-local"
                value={form.kickoff_time}
                onChange={(e) => updateField('kickoff_time', e.target.value)}
                required
              />
            </div>
            <div className="market-grid">
              <div className="field-group">
                <label htmlFor="home">Home odds</label>
                <input
                  id="home"
                  type="number"
                  min="1"
                  step="0.01"
                  value={form.home}
                  onChange={(e) => updateField('home', e.target.value)}
                  required
                />
              </div>
              <div className="field-group">
                <label htmlFor="draw">Draw odds</label>
                <input
                  id="draw"
                  type="number"
                  min="1"
                  step="0.01"
                  value={form.draw}
                  onChange={(e) => updateField('draw', e.target.value)}
                  required
                />
              </div>
              <div className="field-group">
                <label htmlFor="away">Away odds</label>
                <input
                  id="away"
                  type="number"
                  min="1"
                  step="0.01"
                  value={form.away}
                  onChange={(e) => updateField('away', e.target.value)}
                  required
                />
              </div>
            </div>
            <button type="submit" className="primary-button" disabled={loading}>
              {loading ? 'Predicting…' : 'Generate Prediction'}
            </button>
          </form>

          {error && <div className="alert error">{error}</div>}

          {prediction && (
            <div className="result-card">
              <h3>Prediction Result</h3>
              <dl>
                <div>
                  <dt>Home</dt>
                  <dd>{(prediction.home_prob * 100).toFixed(1)}%</dd>
                </div>
                <div>
                  <dt>Draw</dt>
                  <dd>{(prediction.draw_prob * 100).toFixed(1)}%</dd>
                </div>
                <div>
                  <dt>Away</dt>
                  <dd>{(prediction.away_prob * 100).toFixed(1)}%</dd>
                </div>
                <div>
                  <dt>Consensus</dt>
                  <dd>{(prediction.consensus_prob * 100).toFixed(1)}%</dd>
                </div>
                <div>
                  <dt>Expected value</dt>
                  <dd>{prediction.final_ev.toFixed(3)}</dd>
                </div>
                <div>
                  <dt>Recommended stake</dt>
                  <dd>{(prediction.recommended_stake * 100).toFixed(1)}%</dd>
                </div>
              </dl>
            </div>
          )}
        </section>

        <section className="panel history-panel">
          <div className="panel-header">
            <h2>Prediction History</h2>
            <button type="button" onClick={loadHistory} className="secondary-button">
              Refresh
            </button>
          </div>

          {history.length === 0 ? (
            <p>No history yet. Generate a prediction to begin.</p>
          ) : (
            <div className="history-table-wrapper">
              <table className="history-table">
                <thead>
                  <tr>
                    <th>Match</th>
                    <th>League</th>
                    <th>Edge</th>
                    <th>Stake</th>
                    <th>Outcome</th>
                    <th>Profit</th>
                  </tr>
                </thead>
                <tbody>
                  {history.map((item) => (
                    <tr key={`${item.match_id}-${item.timestamp}`}>
                      <td>{item.home_team} vs {item.away_team}</td>
                      <td>{item.league}</td>
                      <td>{item.edge?.toFixed(3)}</td>
                      <td>{(item.recommended_stake * 100).toFixed(1)}%</td>
                      <td>{item.actual_outcome || 'pending'}</td>
                      <td>{item.profit != null ? item.profit.toFixed(2) : '-'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </main>
    </div>
  )
}

export default App
