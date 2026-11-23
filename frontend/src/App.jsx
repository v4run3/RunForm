import React, { useState } from 'react';
import './App.css';
import { UploadCloud, Camera, BookOpen, Footprints, History, LogOut, Home } from 'lucide-react';
import VideoUploadView from './components/VideoUploadView';
import LiveCameraView from './components/LiveCameraView';
import BiomechanicsGuideView from './components/BiomechanicsGuideView';
import AnalysisHistoryView from './components/AnalysisHistoryView';
import LandingPage from './components/LandingPage';
import { getStoredSession, saveSession, clearSession } from './utils/authStorage';

const workspaceTabs = [
  { id: 'upload', label: 'Upload', icon: UploadCloud },
  { id: 'camera', label: 'Camera', icon: Camera },
  { id: 'history', label: 'History', icon: History },
  { id: 'guide', label: 'Guide', icon: BookOpen },
];

export default function App() {
  const [activeTab, setActiveTab] = useState('upload');
  const [session, setSession] = useState(() => getStoredSession());
  const [historyRefreshKey, setHistoryRefreshKey] = useState(0);
  const [viewingLanding, setViewingLanding] = useState(false);

  const handleAuth = (authSession, rememberOnLaptop = false) => {
    saveSession(authSession, rememberOnLaptop);
    setSession(authSession);
    setViewingLanding(false);
  };

  const logout = () => {
    clearSession();
    setSession(null);
    setViewingLanding(false);
    setActiveTab('upload');
  };

  // If user is not authenticated or explicitly chose to view the landing page
  if (!session || viewingLanding) {
    return (
      <LandingPage
        onAuth={handleAuth}
        currentSession={session}
        onBackToApp={session ? () => setViewingLanding(false) : null}
      />
    );
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="brand">
          <div className="brand-icon" aria-hidden="true">
            <Footprints size={19} />
          </div>
          <div className="brand-text">
            <span className="brand-title">RunForm</span>
            <p>Running analysis</p>
          </div>
        </div>

        <nav className="nav-tabs" aria-label="Primary navigation">
          {workspaceTabs.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              type="button"
              className={`nav-tab-btn ${activeTab === id ? 'active' : ''}`}
              aria-current={activeTab === id ? 'page' : undefined}
              onClick={() => setActiveTab(id)}
            >
              <Icon size={19} aria-hidden="true" />
              <span>{label}</span>
            </button>
          ))}
        </nav>

        <div className="workspace-account">
          <p className="workspace-user" title={session.user.name}>{session.user.name}</p>
          <div className="user-nav-actions">
            <button
              className="icon-btn"
              type="button"
              onClick={() => setViewingLanding(true)}
              aria-label="View Landing Page"
              title="Home / Landing Page"
            >
              <Home size={17} />
            </button>
            <button
              className="icon-btn"
              type="button"
              onClick={logout}
              aria-label="Log out"
              title="Log out"
            >
              <LogOut size={17} />
            </button>
          </div>
        </div>
      </header>

      <main className="workspace-main">
        {activeTab === 'upload' && (
          <VideoUploadView
            token={session.token}
            onSaved={() => setHistoryRefreshKey((key) => key + 1)}
          />
        )}
        {activeTab === 'camera' && <LiveCameraView />}
        {activeTab === 'history' && (
          <AnalysisHistoryView token={session.token} refreshKey={historyRefreshKey} />
        )}
        {activeTab === 'guide' && <BiomechanicsGuideView />}
      </main>
    </div>
  );
}
