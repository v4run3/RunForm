import React, { useState } from 'react';
import {
  Footprints,
  LogIn,
  UserPlus,
  Video,
  Camera,
  Activity,
  BookOpen,
  ShieldCheck,
  CheckCircle,
  Eye,
  EyeOff,
  Laptop,
  Smartphone,
} from 'lucide-react';
import { isMobileDevice } from '../utils/authStorage';

const API_BASE = '';

export default function LandingPage({ onAuth, currentSession = null, onBackToApp = null }) {
  const [authMode, setAuthMode] = useState('login'); // 'login' | 'register'
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [rememberOnLaptop, setRememberOnLaptop] = useState(false);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const isMobile = isMobileDevice();

  const handleAuthSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    const endpoint = authMode === 'login' ? '/api/auth/login' : '/api/auth/register';
    const payload = authMode === 'login' ? { email, password } : { name, email, password };

    try {
      const res = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const data = await res.json().catch(() => ({}));

      if (!res.ok) {
        throw new Error(data.detail || 'Authentication failed. Please check your credentials.');
      }

      onAuth(data, rememberOnLaptop);
    } catch (err) {
      setError(err.message || 'Something went wrong. Please ensure the backend server is running.');
    } finally {
      setLoading(false);
    }
  };

  const scrollToAuth = (mode) => {
    setAuthMode(mode);
    const authCard = document.getElementById('auth-section');
    if (authCard) {
      authCard.scrollIntoView({ behavior: 'smooth' });
      const emailInput = document.getElementById('auth-email-input');
      if (emailInput) emailInput.focus();
    }
  };

  return (
    <div className="landing-container">
      {/* Top Navigation */}
      <header className="landing-navbar">
        <div className="landing-brand">
          <div className="landing-brand-icon">
            <Footprints size={22} />
          </div>
          <div className="landing-brand-text">
            <span className="brand-title">RunForm</span>
            <span className="brand-tag">AI Gait & Biomechanics</span>
          </div>
        </div>

        <nav className="landing-nav-links" aria-label="About RunForm">
          <a href="#features">Features</a>
          <a href="#how-it-works">How It Works</a>
          <a href="#biomechanics">Biomechanics</a>
        </nav>

        <div className="landing-nav-actions">
          {currentSession && onBackToApp ? (
            <button
              type="button"
              className="btn-nav-register"
              onClick={onBackToApp}
            >
              <Activity size={15} />
              <span>Open Dashboard</span>
            </button>
          ) : (
            <>
              <button
                type="button"
                className="btn-nav-login"
                onClick={() => scrollToAuth('login')}
              >
                <LogIn size={15} />
                <span>Sign In</span>
              </button>
              <button
                type="button"
                className="btn-nav-register"
                onClick={() => scrollToAuth('register')}
              >
                <UserPlus size={15} />
                <span>Get Started</span>
              </button>
            </>
          )}
        </div>
      </header>

      {/* Hero Section */}
      <section className="landing-hero-section">
        <div className="hero-content">
          <div className="hero-pill"><Activity size={16} /> Running gait analysis</div>

          <h1 className="hero-headline">
            See how you run.
          </h1>

          <p className="hero-subhead">
            Upload a clip or record with your camera to measure knee motion, trunk lean,
            and left–right symmetry across your stride.
          </p>

          <div className="hero-actions">
            <button
              type="button"
              className="btn-hero-primary"
              onClick={() => scrollToAuth('register')}
            >
              <span>Get Started Free</span>
            </button>
            <button
              type="button"
              className="btn-hero-secondary"
              onClick={() => scrollToAuth('login')}
            >
              <LogIn size={16} />
              <span>Sign In to Account</span>
            </button>
          </div>

        </div>

        {/* Embedded Hero Auth Card */}
        <div className="hero-auth-card-wrapper" id="auth-section">
          <div className="glass-panel auth-card">
            <div className="auth-card-header">
              <h2>{authMode === 'login' ? 'Welcome Back' : 'Create Your Account'}</h2>
              <p>
                {authMode === 'login'
                  ? 'Sign in to access your running analyses and live camera tracking.'
                  : 'Get started with AI-powered running form assessments in seconds.'}
              </p>
            </div>

            {/* Auth Mode Toggle */}
            <div className="auth-tab-switch" role="group" aria-label="Account access">
              <button
                type="button"
                className={`auth-tab-btn ${authMode === 'login' ? 'active' : ''}`}
                aria-pressed={authMode === 'login'}
                onClick={() => {
                  setAuthMode('login');
                  setError(null);
                }}
              >
                <LogIn size={15} />
                <span>Sign In</span>
              </button>
              <button
                type="button"
                className={`auth-tab-btn ${authMode === 'register' ? 'active' : ''}`}
                aria-pressed={authMode === 'register'}
                onClick={() => {
                  setAuthMode('register');
                  setError(null);
                }}
              >
                <UserPlus size={15} />
                <span>Sign Up</span>
              </button>
            </div>

            <form onSubmit={handleAuthSubmit} className="landing-auth-form">
              {authMode === 'register' && (
                <div className="form-group">
                  <label htmlFor="auth-name-input">Full Name</label>
                  <input
                    id="auth-name-input"
                    type="text"
                    autoComplete="name"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder="e.g. Alex Miller"
                    required
                  />
                </div>
              )}

              <div className="form-group">
                <label htmlFor="auth-email-input">Email Address</label>
                <input
                  id="auth-email-input"
                  type="email"
                  autoComplete="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="runner@example.com"
                  required
                />
              </div>

              <div className="form-group">
                <label htmlFor="auth-password-input">Password</label>
                <div className="password-input-wrapper">
                  <input
                    id="auth-password-input"
                    type={showPassword ? 'text' : 'password'}
                    autoComplete={authMode === 'login' ? 'current-password' : 'new-password'}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="At least 6 characters"
                    required
                    minLength={6}
                  />
                  <button
                    type="button"
                    className="password-toggle-btn"
                    onClick={() => setShowPassword(!showPassword)}
                    aria-label={showPassword ? 'Hide password' : 'Show password'}
                  >
                    {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                  </button>
                </div>
              </div>

              {/* Laptop remember me checkbox */}
              {!isMobile && (
                <div className="remember-me-checkbox">
                  <label className="checkbox-label">
                    <input
                      type="checkbox"
                      checked={rememberOnLaptop}
                      onChange={(e) => setRememberOnLaptop(e.target.checked)}
                    />
                    <span>Remember me on this laptop</span>
                  </label>
                </div>
              )}

              {error && <div className="auth-error-banner" role="alert">{error}</div>}

              <button type="submit" className="btn-submit-auth" disabled={loading}>
                {loading ? (
                  <span>Authenticating...</span>
                ) : authMode === 'login' ? (
                  <>
                    <span>Sign In to RunForm</span>
                  </>
                ) : (
                  <>
                    <span>Create Free Account</span>
                  </>
                )}
              </button>
            </form>

            <div className="auth-card-footer">
              {authMode === 'login' ? (
                <p>
                  Don't have an account yet?{' '}
                  <button
                    type="button"
                    className="link-switch"
                    onClick={() => {
                      setAuthMode('register');
                      setError(null);
                    }}
                  >
                    Sign up now
                  </button>
                </p>
              ) : (
                <p>
                  Already have an account?{' '}
                  <button
                    type="button"
                    className="link-switch"
                    onClick={() => {
                      setAuthMode('login');
                      setError(null);
                    }}
                  >
                    Log in here
                  </button>
                </p>
              )}
            </div>
            <div className="device-badge">
              {isMobile ? <Smartphone size={15} /> : <Laptop size={15} />}
              <span>{isMobile ? 'Stay signed in on this device until you log out.' : 'By default, sign-in lasts for this browser session.'}</span>
            </div>
          </div>
        </div>
      </section>

      {/* Core Capabilities Section */}
      <section className="landing-section" id="features">
        <div className="section-header">
          <div className="section-badge">
            <Activity size={14} />
            <span>Core Capabilities</span>
          </div>
          <h2>Engineered for Accurate Biomechanical Analysis</h2>
          <p>
            From casual 5K runners to competitive athletes, RunForm delivers quantitative
            biomechanical analysis directly in your web browser.
          </p>
        </div>

        <div className="features-grid">
          <div className="feature-card glass-panel">
            <div className="feature-icon-box">
              <Video size={22} />
            </div>
            <h3>Video Upload Analysis</h3>
            <p>
              Upload pre-recorded running clips from your phone or camera. Choose between
              MediaPipe Pose for high-speed joint tracking or YOLO for multi-runner detection.
            </p>
            <ul className="feature-perks">
              <li>
                <CheckCircle size={14} /> Frame-by-frame joint angles
              </li>
              <li>
                <CheckCircle size={14} /> Side, front, & rear view support
              </li>
              <li>
                <CheckCircle size={14} /> Annotated video download
              </li>
            </ul>
          </div>

          <div className="feature-card glass-panel">
            <div className="feature-icon-box">
              <Camera size={22} />
            </div>
            <h3>Live Webcam Form Tracking</h3>
            <p>
              Position your laptop next to a treadmill and analyze your running posture
              live. Watch real-time joint skeleton overlays and immediate posture cues.
            </p>
            <ul className="feature-perks">
              <li>
                <CheckCircle size={14} /> Live 30+ FPS skeleton overlay
              </li>
              <li>
                <CheckCircle size={14} /> Immediate trunk lean feedback
              </li>
              <li>
                <CheckCircle size={14} /> Browser-based recording & export
              </li>
            </ul>
          </div>

          <div className="feature-card glass-panel">
            <div className="feature-icon-box">
              <Activity size={22} />
            </div>
            <h3>Temporal Movement Kinematics</h3>
            <p>
              Window-based statistical analysis across 1s, 2s, and 3s intervals. Evaluates
              knee flexion range of motion, trunk pitch, and bilateral left/right symmetry.
            </p>
            <ul className="feature-perks">
              <li>
                <CheckCircle size={14} /> Plausibility quality validation
              </li>
              <li>
                <CheckCircle size={14} /> Angle-wrapping discontinuity fix
              </li>
              <li>
                <CheckCircle size={14} /> ML-ready CSV & JSON exports
              </li>
            </ul>
          </div>

          <div className="feature-card glass-panel">
            <div className="feature-icon-box">
              <BookOpen size={22} />
            </div>
            <h3>Biomechanics Knowledge Base</h3>
            <p>
              Learn the science behind running form. Built-in reference guides for optimal
              knee flexion at contact, hip extension at toe-off, and footstrike mechanics.
            </p>
            <ul className="feature-perks">
              <li>
                <CheckCircle size={14} /> Joint angle reference ranges
              </li>
              <li>
                <CheckCircle size={14} /> Footstrike classification guide
              </li>
              <li>
                <CheckCircle size={14} /> Overstride & cadence insights
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="landing-section" id="how-it-works">
        <div className="section-header">
          <div className="section-badge">
            <ShieldCheck size={14} />
            <span>Workflow</span>
          </div>
          <h2>How RunForm Works</h2>
          <p>Three streamlined steps to comprehensive running gait feedback.</p>
        </div>

        <div className="workflow-steps">
          <div className="step-card glass-panel">
            <div className="step-number">1</div>
            <h4>Record or Upload</h4>
            <p>
              Record a 5–15 second clip from the side (sagittal view) on a treadmill or
              running track, or turn on your laptop camera.
            </p>
          </div>

          <div className="step-card glass-panel">
            <div className="step-number">2</div>
            <h4>Automated Pose Extraction</h4>
            <p>
              MediaPipe Pose tracks 33 normalized body landmarks. Algorithms calculate
              knee, hip, elbow, and trunk angles per frame.
            </p>
          </div>

          <div className="step-card glass-panel">
            <div className="step-number">3</div>
            <h4>Review & Export Metrics</h4>
            <p>
              Inspect knee range of motion, symmetry differences, and trunk lean. Download
              annotated videos and ML-ready temporal CSVs.
            </p>
          </div>
        </div>
      </section>

      {/* Biomechanics Preview Section */}
      <section className="landing-section biomechanics-preview" id="biomechanics">
        <div className="glass-panel preview-card">
          <div className="preview-content">
            <span className="section-badge">
              <Footprints size={14} />
              <span>Gait Metrics</span>
            </span>
            <h2>Clinically Informed, Objectively Measured</h2>
            <p>
              RunForm avoids arbitrary health labels and rule-based injury claims. Instead,
              we focus on validated geometric kinematics:
            </p>
            <div className="metric-highlights">
              <div className="metric-highlight-item">
                <span className="metric-name">Knee Range of Motion</span>
                <span className="metric-desc">Target 115°–130° for efficient swing phase flexion and absorption.</span>
              </div>
              <div className="metric-highlight-item">
                <span className="metric-name">Trunk Forward Pitch</span>
                <span className="metric-desc">Optimal 5°–12° forward lean from the vertical axis for optimal ground force.</span>
              </div>
              <div className="metric-highlight-item">
                <span className="metric-name">Bilateral Knee Symmetry</span>
                <span className="metric-desc">Detect left vs right discrepancy across continuous running strides.</span>
              </div>
            </div>
            <button
              type="button"
              className="btn-primary start-now-btn"
              onClick={() => scrollToAuth('register')}
            >
              <span>Analyze Your Form Now</span>
            </button>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="landing-footer">
        <div className="footer-top">
          <div className="footer-brand">
            <div className="landing-brand-icon">
              <Footprints size={18} />
            </div>
            <span className="brand-title">RunForm</span>
          </div>
          <p className="footer-disclaimer">
            RunForm is an AI-powered running form analysis system. Measurements represent
            software-level geometric kinematics and computer vision tracking for athletic feedback and
            research preparation.
          </p>
        </div>
        <div className="footer-bottom">
          <span>&copy; {new Date().getFullYear()} RunForm. All rights reserved.</span>
          <div className="footer-links">
            <button type="button" onClick={() => scrollToAuth('login')}>
              Sign In
            </button>
            <button type="button" onClick={() => scrollToAuth('register')}>
              Create Account
            </button>
            <a href="#features">Features</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
