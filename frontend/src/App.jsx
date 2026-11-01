import React, { useEffect, useState } from 'react';
import './App.css';
import { UploadCloud, Camera, BookOpen, Footprints, History, LogOut } from 'lucide-react';
import VideoUploadView from './components/VideoUploadView';
import LiveCameraView from './components/LiveCameraView';
import BiomechanicsGuideView from './components/BiomechanicsGuideView';
import AuthView from './components/AuthView';
import AnalysisHistoryView from './components/AnalysisHistoryView';

const SESSION_KEY = 'runform_session';

export default function App() {
  const [activeTab, setActiveTab] = useState('upload');
  const [session, setSession] = useState(null);
  const [historyRefreshKey, setHistoryRefreshKey] = useState(0);

  useEffect(() => {
    const saved = localStorage.getItem(SESSION_KEY);
    if (saved) {
      setSession(JSON.parse(saved));
    }
  }, []);

  const handleAuth = (authSession) => {
    localStorage.setItem(SESSION_KEY, JSON.stringify(authSession));
    setSession(authSession);
  };

  const logout = () => {
    localStorage.removeItem(SESSION_KEY);
    setSession(null);
    setActiveTab('upload');
  };

  if (!session) {
    return <AuthView onAuth={handleAuth} />;
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="brand">
          <div className="brand-icon" aria-hidden="true">
            <Footprints size={19} />
          </div>
          <div className="brand-text">
            <h1>RunForm</h1>
            <p>{session.user.name}</p>
          </div>
        </div>

        <div className="top-actions">
          <nav className="nav-tabs" aria-label="Primary navigation">
            <button
              className={`nav-tab-btn ${activeTab === 'upload' ? 'active' : ''}`}
              onClick={() => setActiveTab('upload')}
            >
              <UploadCloud size={16} />
              Upload
            </button>
            <button
              className={`nav-tab-btn ${activeTab === 'camera' ? 'active' : ''}`}
              onClick={() => setActiveTab('camera')}
            >
              <Camera size={16} />
              Camera
            </button>
            <button
              className={`nav-tab-btn ${activeTab === 'history' ? 'active' : ''}`}
              onClick={() => setActiveTab('history')}
            >
              <History size={16} />
              History
            </button>
            <button
              className={`nav-tab-btn ${activeTab === 'guide' ? 'active' : ''}`}
              onClick={() => setActiveTab('guide')}
            >
              <BookOpen size={16} />
              Guide
            </button>
          </nav>
          <button className="icon-btn" type="button" onClick={logout} aria-label="Log out" title="Log out">
            <LogOut size={17} />
          </button>
        </div>
      </header>

      <main>
        {activeTab === 'upload' && (
          <VideoUploadView
            token={session.token}
            onSaved={() => setHistoryRefreshKey((key) => key + 1)}
          />
        )}
        {activeTab === 'camera' && <LiveCameraView />}
        {activeTab === 'history' && <AnalysisHistoryView token={session.token} refreshKey={historyRefreshKey} />}
        {activeTab === 'guide' && <BiomechanicsGuideView />}
      </main>
    </div>
  );
}
