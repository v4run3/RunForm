import React, { useState, useRef } from 'react';
import {
  Upload,
  Play,
  RefreshCw,
  AlertTriangle,
  CheckCircle2,
  ShieldAlert,
  Activity,
  ChevronRight,
  Info,
  Sparkles,
  Layers,
} from 'lucide-react';

const API_BASE = '';

export default function VideoUploadView({ token, onSaved }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [analysisMode, setAnalysisMode] = useState('mediapipe'); // 'mediapipe' | 'yolo'
  const [analyzing, setAnalyzing] = useState(false);
  const [poseResult, setPoseResult] = useState(null);
  const [yoloReport, setYoloReport] = useState(null);
  const [error, setError] = useState(null);
  const [videoError, setVideoError] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setPoseResult(null);
      setYoloReport(null);
      setError(null);
      setVideoError(null);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('video/')) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setPoseResult(null);
      setYoloReport(null);
      setError(null);
      setVideoError(null);
    }
  };

  const clearVideo = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setPoseResult(null);
    setYoloReport(null);
    setError(null);
    setVideoError(null);
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setAnalyzing(true);
    setError(null);
    setVideoError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      if (analysisMode === 'mediapipe') {
        // MediaPipe Pose Running Form Analysis
        const res = await fetch(`${API_BASE}/api/pose-analysis`, {
          method: 'POST',
          body: formData,
        });

        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.detail || `Server error: ${res.statusText}`);
        }

        const data = await res.json();
        setPoseResult(data);
      } else {
        // Full YOLO Pipeline Analysis
        const res = await fetch(`${API_BASE}/api/analyze-video`, {
          method: 'POST',
          headers: { Authorization: `Bearer ${token}` },
          body: formData,
        });

        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.detail || `Server error: ${res.statusText}`);
        }

        const data = await res.json();
        setYoloReport(data);
        onSaved?.();
      }
    } catch (err) {
      setError(err.message || 'Could not analyze this video. Make sure the backend is running.');
    } finally {
      setAnalyzing(false);
    }
  };

  const poseSummary = poseResult?.summary || poseResult?.metrics;
  const detectionPct =
    poseSummary?.detection_rate != null ? (poseSummary.detection_rate * 100).toFixed(0) : null;
  const poseVideoUrl = poseResult?.video_url || poseResult?.annotated_video_url;

  return (
    <div className="view-container">
      {/* Educational & Disclaimer Banner */}
      <div
        className="glass-panel"
        style={{
          padding: '12px 16px',
          display: 'flex',
          gap: '10px',
          alignItems: 'flex-start',
          borderLeft: '3px solid var(--accent-amber)',
          marginBottom: '16px',
        }}
      >
        <Info size={16} color="var(--accent-amber)" style={{ flexShrink: 0, marginTop: '2px' }} />
        <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>
          RunForm provides 2D movement/form observations and is not a medical diagnostic or injury-prediction system.
        </p>
      </div>

      {/* Analysis Mode Selector */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '12px',
          marginBottom: '16px',
        }}
      >
        <div>
          <h2 style={{ fontSize: '1.2rem', fontWeight: 700 }}>Running Form Analysis</h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.86rem', marginTop: '2px' }}>
            Upload a running video to evaluate joint angles, symmetry, and running mechanics.
          </p>
        </div>

        <div style={{ display: 'flex', gap: '8px', background: 'var(--bg-subtle)', padding: '4px', borderRadius: '8px' }}>
          <button
            type="button"
            className={`nav-tab-btn ${analysisMode === 'mediapipe' ? 'active' : ''}`}
            onClick={() => setAnalysisMode('mediapipe')}
            style={{ padding: '6px 12px', fontSize: '0.82rem' }}
          >
            <Sparkles size={14} /> Running Form (MediaPipe)
          </button>
          <button
            type="button"
            className={`nav-tab-btn ${analysisMode === 'yolo' ? 'active' : ''}`}
            onClick={() => setAnalysisMode('yolo')}
            style={{ padding: '6px 12px', fontSize: '0.82rem' }}
          >
            <Layers size={14} /> Full Gait (YOLO)
          </button>
        </div>
      </div>

      {!selectedFile && (
        <div
          className="upload-dropzone"
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept="video/*,video/mp4,video/quicktime,video/mov,video/webm,video/avi"
            style={{ display: 'none' }}
          />
          <div className="upload-icon">
            <Upload size={28} />
          </div>
          <h3 style={{ fontSize: '1.2rem', fontWeight: 700, marginBottom: '8px' }}>
            Upload a running video
          </h3>
          <p
            style={{
              color: 'var(--text-muted)',
              fontSize: '0.9rem',
              lineHeight: 1.5,
              maxWidth: '420px',
              margin: '0 auto 16px',
            }}
          >
            Choose a side-view running clip. Keep the runner fully visible in the frame for optimal landmark tracking.
          </p>
          <span style={{ fontSize: '0.78rem', color: 'var(--text-soft)', fontWeight: 600 }}>
            MP4, MOV or WEBM up to 150MB
          </span>
        </div>
      )}

      {selectedFile && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              flexWrap: 'wrap',
              gap: '12px',
            }}
          >
            <div style={{ minWidth: 0 }}>
              <h3 style={{ fontSize: '1rem', fontWeight: 700, overflowWrap: 'anywhere' }}>
                {selectedFile.name}
              </h3>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.82rem', marginTop: '3px' }}>
                {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB &bull; Mode:{' '}
                {analysisMode === 'mediapipe' ? 'MediaPipe Pose' : 'YOLOv8 Gait'}
              </p>
            </div>
            <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
              <button className="nav-tab-btn" onClick={clearVideo} disabled={analyzing}>
                Change
              </button>
              <button className="btn-primary" onClick={handleAnalyze} disabled={analyzing}>
                {analyzing ? (
                  <>
                    <RefreshCw size={18} style={{ animation: 'spin 1s linear infinite' }} />
                    Analyzing form...
                  </>
                ) : (
                  <>
                    <Play size={18} />
                    {analysisMode === 'mediapipe' ? 'Analyze Form' : 'Review Form'}
                  </>
                )}
              </button>
            </div>
          </div>

          {error && (
            <div
              className="glass-panel"
              style={{
                padding: '14px',
                borderLeft: '3px solid var(--accent-rose)',
                display: 'flex',
                gap: '10px',
                alignItems: 'flex-start',
              }}
            >
              <AlertTriangle color="var(--accent-rose)" size={20} />
              <p style={{ fontSize: '0.86rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>
                {error}
              </p>
            </div>
          )}

          {videoError && (
            <div
              className="glass-panel"
              style={{
                padding: '14px',
                borderLeft: '3px solid var(--accent-rose)',
                display: 'flex',
                gap: '10px',
                alignItems: 'flex-start',
              }}
            >
              <AlertTriangle color="var(--accent-rose)" size={20} />
              <p style={{ fontSize: '0.86rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>
                {videoError}
              </p>
            </div>
          )}

          {/* Videos Grid */}
          <div className="grid-2col">
            <div className="glass-panel" style={{ padding: '12px' }}>
              <h4 style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '10px' }}>
                Original Video
              </h4>
              <div className="video-wrapper">
                <video src={previewUrl} controls playsInline />
              </div>
            </div>

            <div className="glass-panel" style={{ padding: '12px' }}>
              <h4 style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '10px' }}>
                Skeleton Overlay
              </h4>
              <div className="video-wrapper">
                {analysisMode === 'mediapipe' && poseResult && poseVideoUrl ? (
                  <video
                    key={poseVideoUrl}
                    controls
                    autoPlay
                    loop
                    playsInline
                    onError={() =>
                      setVideoError('Annotated video could not be played. Check the generated video codec/format.')
                    }
                  >
                    <source src={`${API_BASE}${poseVideoUrl}`} type="video/mp4" />
                    Annotated video could not be played. Check the generated video codec/format.
                  </video>
                ) : analysisMode === 'yolo' && yoloReport ? (
                  <video
                    key={yoloReport.annotated_video_url}
                    controls
                    autoPlay
                    loop
                    playsInline
                    onError={() =>
                      setVideoError('Annotated video could not be played. Check the generated video codec/format.')
                    }
                  >
                    <source src={`${API_BASE}${yoloReport.annotated_video_url}`} type="video/mp4" />
                    Annotated video could not be played. Check the generated video codec/format.
                  </video>
                ) : (
                  <div
                    style={{
                      minHeight: '220px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: 'var(--text-muted)',
                      fontSize: '0.88rem',
                      padding: '18px',
                      textAlign: 'center',
                    }}
                  >
                    {analyzing
                      ? 'Processing video & generating skeleton overlay...'
                      : 'Click "Analyze Form" to generate the skeleton overlay.'}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* MediaPipe Running Form Metrics & Observations */}
          {analysisMode === 'mediapipe' && poseSummary && (
            <div>
              <h3
                style={{
                  fontSize: '1.05rem',
                  fontWeight: 700,
                  marginBottom: '12px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                }}
              >
                <Activity color="var(--accent-cyan)" size={19} /> Running Form Metrics
              </h3>

              <div className="metrics-grid">
                <div className="glass-panel metric-card">
                  <span className="metric-label">Pose detection</span>
                  <div
                    className="metric-value"
                    style={{
                      color:
                        detectionPct >= 60 ? 'var(--accent-emerald)' : 'var(--accent-amber)',
                    }}
                  >
                    {detectionPct != null ? `${detectionPct}%` : 'N/A'}
                  </div>
                  <span className="metric-sub">
                    {poseSummary.pose_detected_frames} / {poseSummary.frames_processed} frames
                  </span>
                </div>

                <div className="glass-panel metric-card">
                  <span className="metric-label">Left knee</span>
                  <div className="metric-value">
                    {poseSummary.average_left_knee_angle != null
                      ? `${poseSummary.average_left_knee_angle}°`
                      : 'N/A'}
                  </div>
                  <span className="metric-sub">Target: 155-165°</span>
                </div>

                <div className="glass-panel metric-card">
                  <span className="metric-label">Right knee</span>
                  <div className="metric-value">
                    {poseSummary.average_right_knee_angle != null
                      ? `${poseSummary.average_right_knee_angle}°`
                      : 'N/A'}
                  </div>
                  <span className="metric-sub">Target: 155-165°</span>
                </div>

                <div className="glass-panel metric-card">
                  <span className="metric-label">Knee difference</span>
                  <div
                    className="metric-value"
                    style={{
                      color:
                        poseSummary.average_knee_asymmetry != null &&
                        poseSummary.average_knee_asymmetry < 10
                          ? 'var(--accent-emerald)'
                          : 'var(--accent-amber)',
                    }}
                  >
                    {poseSummary.average_knee_asymmetry != null
                      ? `${poseSummary.average_knee_asymmetry}°`
                      : 'N/A'}
                  </div>
                  <span className="metric-sub">Left/right asymmetry</span>
                </div>

                <div className="glass-panel metric-card">
                  <span className="metric-label">Left hip</span>
                  <div className="metric-value">
                    {poseSummary.average_left_hip_angle != null
                      ? `${poseSummary.average_left_hip_angle}°`
                      : 'N/A'}
                  </div>
                  <span className="metric-sub">Average angle</span>
                </div>

                <div className="glass-panel metric-card">
                  <span className="metric-label">Right hip</span>
                  <div className="metric-value">
                    {poseSummary.average_right_hip_angle != null
                      ? `${poseSummary.average_right_hip_angle}°`
                      : 'N/A'}
                  </div>
                  <span className="metric-sub">Average angle</span>
                </div>

                <div className="glass-panel metric-card">
                  <span className="metric-label">Trunk lean</span>
                  <div className="metric-value">
                    {poseSummary.average_trunk_lean != null
                      ? `${poseSummary.average_trunk_lean}°`
                      : 'N/A'}
                  </div>
                  <span className="metric-sub">Target: 5-12°</span>
                </div>
              </div>

              {/* Form Observations */}
              {poseSummary.observations && poseSummary.observations.length > 0 && (
                <div className="glass-panel" style={{ padding: '16px', marginTop: '12px' }}>
                  <h4 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '14px' }}>
                    Observations
                  </h4>
                  <ul
                    style={{
                      listStyle: 'none',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '10px',
                    }}
                  >
                    {poseSummary.observations.map((obs, idx) => (
                      <li
                        key={idx}
                        style={{
                          display: 'flex',
                          alignItems: 'flex-start',
                          gap: '10px',
                          fontSize: '0.88rem',
                          lineHeight: 1.45,
                        }}
                      >
                        <CheckCircle2
                          size={16}
                          color="var(--accent-emerald)"
                          style={{ flexShrink: 0, marginTop: '2px' }}
                        />
                        {obs}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          {/* YOLO Gait Summary */}
          {analysisMode === 'yolo' && yoloReport && (
            <div>
              <h3
                style={{
                  fontSize: '1.05rem',
                  fontWeight: 700,
                  marginBottom: '12px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                }}
              >
                <Activity color="var(--accent-cyan)" size={19} /> YOLO Gait Summary
              </h3>

              <div className="metrics-grid">
                <div className="glass-panel metric-card">
                  <span className="metric-label">Foot strike</span>
                  <div
                    className="metric-value"
                    style={{
                      color: 'var(--accent-cyan)',
                      fontSize: '1.18rem',
                      textTransform: 'capitalize',
                    }}
                  >
                    {yoloReport.summary.primary_strike_type.replace('_', ' ')}
                  </div>
                  <span className="metric-sub">At first contact</span>
                </div>

                <div className="glass-panel metric-card">
                  <span className="metric-label">Left knee</span>
                  <div className="metric-value">
                    {yoloReport.summary.avg_knee_flexion_left
                      ? `${yoloReport.summary.avg_knee_flexion_left}°`
                      : 'N/A'}
                  </div>
                  <span className="metric-sub">Target: 155-165°</span>
                </div>

                <div className="glass-panel metric-card">
                  <span className="metric-label">Right knee</span>
                  <div className="metric-value">
                    {yoloReport.summary.avg_knee_flexion_right
                      ? `${yoloReport.summary.avg_knee_flexion_right}°`
                      : 'N/A'}
                  </div>
                  <span className="metric-sub">Target: 155-165°</span>
                </div>

                <div className="glass-panel metric-card">
                  <span className="metric-label">Trunk lean</span>
                  <div className="metric-value">
                    {yoloReport.summary.avg_trunk_lean
                      ? `${yoloReport.summary.avg_trunk_lean}°`
                      : 'N/A'}
                  </div>
                  <span className="metric-sub">Target: 5-12°</span>
                </div>
              </div>

              <div className="glass-panel" style={{ padding: '16px', marginTop: '12px' }}>
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    marginBottom: '14px',
                    flexWrap: 'wrap',
                    gap: '8px',
                  }}
                >
                  <h4 style={{ fontSize: '1rem', fontWeight: 700 }}>Risk Assessment</h4>
                  <span className={`risk-badge ${yoloReport.feedback.risk_level}`}>
                    {yoloReport.feedback.risk_level === 'low' && <CheckCircle2 size={14} />}
                    {yoloReport.feedback.risk_level === 'moderate' && <AlertTriangle size={14} />}
                    {yoloReport.feedback.risk_level === 'high' && <ShieldAlert size={14} />}
                    {yoloReport.feedback.risk_level}
                  </span>
                </div>

                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(230px, 1fr))',
                    gap: '16px',
                  }}
                >
                  <div>
                    <h5
                      style={{
                        fontSize: '0.76rem',
                        color: 'var(--text-soft)',
                        textTransform: 'uppercase',
                        marginBottom: '10px',
                      }}
                    >
                      Observations
                    </h5>
                    <ul
                      style={{
                        listStyle: 'none',
                        display: 'flex',
                        flexDirection: 'column',
                        gap: '8px',
                      }}
                    >
                      {yoloReport.feedback.flags.map((flag, idx) => (
                        <li
                          key={idx}
                          style={{
                            display: 'flex',
                            alignItems: 'flex-start',
                            gap: '8px',
                            fontSize: '0.88rem',
                            lineHeight: 1.45,
                          }}
                        >
                          <ChevronRight
                            size={15}
                            color="var(--text-soft)"
                            style={{ flexShrink: 0, marginTop: '2px' }}
                          />
                          {flag}
                        </li>
                      ))}
                    </ul>
                  </div>

                  <div>
                    <h5
                      style={{
                        fontSize: '0.76rem',
                        color: 'var(--text-soft)',
                        textTransform: 'uppercase',
                        marginBottom: '10px',
                      }}
                    >
                      Coaching Tips
                    </h5>
                    <ul
                      style={{
                        listStyle: 'none',
                        display: 'flex',
                        flexDirection: 'column',
                        gap: '8px',
                      }}
                    >
                      {yoloReport.feedback.recommendations.map((rec, idx) => (
                        <li
                          key={idx}
                          style={{
                            display: 'flex',
                            alignItems: 'flex-start',
                            gap: '8px',
                            fontSize: '0.88rem',
                            lineHeight: 1.45,
                          }}
                        >
                          <ChevronRight
                            size={15}
                            color="var(--text-soft)"
                            style={{ flexShrink: 0, marginTop: '2px' }}
                          />
                          {rec}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
