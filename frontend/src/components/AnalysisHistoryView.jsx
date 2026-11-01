import React, { useEffect, useState } from 'react';
import { Download, FileText, RefreshCw } from 'lucide-react';

const API_BASE = '';

export default function AnalysisHistoryView({ token, refreshKey }) {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const loadHistory = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/analyses`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = await res.json().catch(() => []);
      if (!res.ok) {
        throw new Error(data.detail || 'Could not load history');
      }
      setItems(data);
    } catch (err) {
      setError(err.message || 'Could not load history');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadHistory();
  }, [token, refreshKey]);

  const download = async (url, filename) => {
    const res = await fetch(`${API_BASE}${url}`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (!res.ok) {
      setError('Download failed');
      return;
    }
    const blob = await res.blob();
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    link.click();
    URL.revokeObjectURL(link.href);
  };

  return (
    <div className="view-container">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '12px' }}>
        <div>
          <h2 style={{ fontSize: '1.14rem', fontWeight: 700 }}>History</h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.86rem', marginTop: '4px' }}>
            Saved reviews for this account.
          </p>
        </div>
        <button className="nav-tab-btn" type="button" onClick={loadHistory}>
          <RefreshCw size={15} /> Refresh
        </button>
      </div>

      {error && <div className="glass-panel" style={{ padding: '14px', color: 'var(--accent-rose)' }}>{error}</div>}
      {loading && <div className="glass-panel" style={{ padding: '18px', color: 'var(--text-muted)' }}>Loading...</div>}
      {!loading && items.length === 0 && (
        <div className="glass-panel" style={{ padding: '18px', color: 'var(--text-muted)', lineHeight: 1.5 }}>
          No saved reviews yet. Upload a video and run a review to see it here.
        </div>
      )}

      <div className="history-list">
        {items.map((item) => {
          const created = new Date(item.created_at * 1000).toLocaleString();
          return (
            <article className="glass-panel history-item" key={item.id}>
              <div>
                <h3>{item.filename}</h3>
                <p>{created}</p>
                <div className="history-meta">
                  <span>{item.summary.primary_strike_type.replace('_', ' ')}</span>
                  <span>{item.feedback.risk_level} risk</span>
                </div>
              </div>
              <div className="history-actions">
                <button type="button" onClick={() => download(item.annotated_video_url, `runform-${item.id}.mp4`)}>
                  <Download size={15} /> Video
                </button>
                <button type="button" onClick={() => download(item.report_download_url, `runform-${item.id}.json`)}>
                  <FileText size={15} /> Report
                </button>
              </div>
            </article>
          );
        })}
      </div>
    </div>
  );
}
