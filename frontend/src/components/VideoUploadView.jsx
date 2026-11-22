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
  Download,
  Clock,
  TrendingUp,
  Wand2,
  Sliders,
  Eye,
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
  const [cameraView, setCameraView] = useState('side');
  const [enhancementFilter, setEnhancementFilter] = useState('none');
  const [enhancementStrength, setEnhancementStrength] = useState(1.0);
  const [previewingFilter, setPreviewingFilter] = useState(false);
  const [filterPreviewData, setFilterPreviewData] = useState(null);
  const [showPreviewModal, setShowPreviewModal] = useState(false);
  const [originalFrameB64, setOriginalFrameB64] = useState(null);
  const fileInputRef = useRef(null);
  const videoPlayerRef = useRef(null);

  const handlePreviewEnhancement = async () => {
    if (!videoPlayerRef.current) return;
    setPreviewingFilter(true);
    setFilterPreviewData(null);
    try {
      const video = videoPlayerRef.current;
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 360;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const frameDataUrl = canvas.toDataURL('image/jpeg', 0.85);
      setOriginalFrameB64(frameDataUrl);

      const targetFilter = enhancementFilter === 'none' ? 'unblur' : enhancementFilter;
      const res = await fetch(`${API_BASE}/api/pose-enhance-preview`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_base64: frameDataUrl,
          filter_type: targetFilter,
          strength: parseFloat(enhancementStrength) || 1.0,
        }),
      });

      if (!res.ok) {
        throw new Error('Failed to generate preview from server');
      }

      const data = await res.json();
      setFilterPreviewData(data);
      setShowPreviewModal(true);
    } catch (err) {
      setError(`Enhancement preview failed: ${err.message}`);
    } finally {
      setPreviewingFilter(false);
    }
  };

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
        formData.append('camera_view', cameraView);
        formData.append('enhancement_filter', enhancementFilter);
        formData.append('enhancement_strength', String(enhancementStrength));

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

          {/* Video Enhancement & Viewpoint Controls Panel */}
          {analysisMode === 'mediapipe' && (
            <div className="glass-panel enhancement-panel" style={{ padding: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '10px', marginBottom: '14px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Wand2 size={18} color="var(--accent-cyan)" />
                  <span style={{ fontSize: '0.92rem', fontWeight: 700, color: 'var(--text-main)' }}>
                    Video Processing & Quality Enhancements
                  </span>
                  <span className="badge-pill" style={{ background: 'rgba(56, 189, 248, 0.15)', color: 'var(--accent-cyan)', fontSize: '0.7rem' }}>
                    New
                  </span>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <button
                    type="button"
                    className="btn-preview-filter"
                    onClick={handlePreviewEnhancement}
                    disabled={previewingFilter || analyzing}
                    title="Test selected enhancement filter on the current video frame"
                  >
                    {previewingFilter ? (
                      <>
                        <RefreshCw size={14} style={{ animation: 'spin 1s linear infinite' }} />
                        Testing...
                      </>
                    ) : (
                      <>
                        <Eye size={14} />
                        Preview Filter on Frame
                      </>
                    )}
                  </button>
                </div>
              </div>

              {/* Viewpoint Selector */}
              <div style={{ marginBottom: '14px' }}>
                <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '6px', fontWeight: 600 }}>
                  Camera Viewpoint:
                </div>
                <div className="filter-chips-grid">
                  <button
                    type="button"
                    className={`enhancement-chip ${cameraView === 'side' ? 'active' : ''}`}
                    onClick={() => setCameraView('side')}
                  >
                    <span className="chip-title">Side View</span>
                    <span className="chip-desc">Sagittal plane (Recommended)</span>
                  </button>
                  <button
                    type="button"
                    className={`enhancement-chip ${cameraView === 'front' ? 'active' : ''}`}
                    onClick={() => setCameraView('front')}
                  >
                    <span className="chip-title">Front View</span>
                    <span className="chip-desc">Lateral tilt analysis</span>
                  </button>
                  <button
                    type="button"
                    className={`enhancement-chip ${cameraView === 'rear' ? 'active' : ''}`}
                    onClick={() => setCameraView('rear')}
                  >
                    <span className="chip-title">Rear View</span>
                    <span className="chip-desc">Pronation & pelvic drop</span>
                  </button>
                </div>
              </div>

              {/* Filter Type Selector */}
              <div>
                <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '6px', fontWeight: 600 }}>
                  Enhancement Filter:
                </div>
                <div className="filter-chips-grid">
                  <button
                    type="button"
                    className={`enhancement-chip ${enhancementFilter === 'none' ? 'active' : ''}`}
                    onClick={() => setEnhancementFilter('none')}
                  >
                    <span className="chip-icon">🚫</span>
                    <span className="chip-title">None</span>
                    <span className="chip-desc">Original untouched frames</span>
                  </button>
                  <button
                    type="button"
                    className={`enhancement-chip ${enhancementFilter === 'auto' ? 'active' : ''}`}
                    onClick={() => setEnhancementFilter('auto')}
                  >
                    <span className="chip-icon">⚡</span>
                    <span className="chip-title">Auto Enhance</span>
                    <span className="chip-desc">Adaptive unblur + lighting</span>
                  </button>
                  <button
                    type="button"
                    className={`enhancement-chip ${enhancementFilter === 'unblur' ? 'active' : ''}`}
                    onClick={() => setEnhancementFilter('unblur')}
                  >
                    <span className="chip-icon">🔍</span>
                    <span className="chip-title">Unblur / Sharpen</span>
                    <span className="chip-desc">Fixes motion blur & soft focus</span>
                  </button>
                  <button
                    type="button"
                    className={`enhancement-chip ${enhancementFilter === 'contrast' ? 'active' : ''}`}
                    onClick={() => setEnhancementFilter('contrast')}
                  >
                    <span className="chip-icon">☀️</span>
                    <span className="chip-title">Contrast (CLAHE)</span>
                    <span className="chip-desc">Adaptive shadow/highlight balance</span>
                  </button>
                  <button
                    type="button"
                    className={`enhancement-chip ${enhancementFilter === 'low_light' ? 'active' : ''}`}
                    onClick={() => setEnhancementFilter('low_light')}
                  >
                    <span className="chip-icon">🌙</span>
                    <span className="chip-title">Low-Light Boost</span>
                    <span className="chip-desc">Brightens underexposed runners</span>
                  </button>
                  <button
                    type="button"
                    className={`enhancement-chip ${enhancementFilter === 'denoise' ? 'active' : ''}`}
                    onClick={() => setEnhancementFilter('denoise')}
                  >
                    <span className="chip-icon">🧼</span>
                    <span className="chip-title">Denoise</span>
                    <span className="chip-desc">Cleans camera grain & sensor noise</span>
                  </button>
                </div>
              </div>

              {/* Intensity Slider (only when a filter is chosen) */}
              {enhancementFilter !== 'none' && (
                <div style={{ marginTop: '14px', paddingTop: '12px', borderTop: '1px solid var(--border-card)', display: 'flex', alignItems: 'center', gap: '16px', flexWrap: 'wrap' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Sliders size={15} color="var(--accent-cyan)" />
                    <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-main)' }}>
                      Filter Intensity:
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0.5"
                    max="2.0"
                    step="0.1"
                    value={enhancementStrength}
                    onChange={(e) => setEnhancementStrength(parseFloat(e.target.value))}
                    style={{ flex: '1', minWidth: '160px', accentColor: 'var(--accent-cyan)', cursor: 'pointer' }}
                  />
                  <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--accent-cyan)', minWidth: '40px' }}>
                    {enhancementStrength.toFixed(1)}x
                  </span>
                </div>
              )}
            </div>
          )}

          {/* Interactive Before/After Preview Modal */}
          {showPreviewModal && filterPreviewData && (
            <div className="preview-modal-overlay" onClick={() => setShowPreviewModal(false)}>
              <div className="preview-modal-card" onClick={(e) => e.stopPropagation()}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
                  <div>
                    <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-main)' }}>
                      Enhancement Preview: {filterPreviewData.filter_type?.toUpperCase()}
                    </h3>
                    <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                      Intensity: {filterPreviewData.strength}x &bull; Sharpness gain: +{filterPreviewData.sharpness_improvement_pct}%
                    </p>
                  </div>
                  <button
                    type="button"
                    className="nav-tab-btn"
                    style={{ padding: '4px 10px', fontSize: '0.8rem' }}
                    onClick={() => setShowPreviewModal(false)}
                  >
                    Close
                  </button>
                </div>

                <div className="grid-2col" style={{ gap: '14px' }}>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                    <span style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-muted)' }}>
                      Original Frame (Sharpness: {filterPreviewData.initial_quality?.blur_score})
                    </span>
                    <div className="preview-frame-box">
                      <img src={originalFrameB64} alt="Original frame" style={{ width: '100%', height: 'auto', borderRadius: '6px' }} />
                    </div>
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                    <span style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--accent-cyan)' }}>
                      Enhanced Frame (Sharpness: {filterPreviewData.enhanced_quality?.blur_score})
                    </span>
                    <div className="preview-frame-box">
                      <img src={filterPreviewData.enhanced_image} alt="Enhanced frame" style={{ width: '100%', height: 'auto', borderRadius: '6px' }} />
                    </div>
                  </div>
                </div>

                <div style={{ marginTop: '14px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '10px' }}>
                  <div style={{ display: 'flex', gap: '12px', fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                    <span>Initial brightness: {filterPreviewData.initial_quality?.brightness}</span>
                    <span>Enhanced brightness: {filterPreviewData.enhanced_quality?.brightness}</span>
                  </div>
                  <button
                    type="button"
                    className="btn-primary"
                    style={{ padding: '6px 14px', fontSize: '0.82rem' }}
                    onClick={() => setShowPreviewModal(false)}
                  >
                    Use This Filter
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Videos Grid */}
          <div className="grid-2col">
            <div className="glass-panel" style={{ padding: '12px' }}>
              <h4 style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '10px' }}>
                Original Video
              </h4>
              <div className="video-wrapper">
                <video ref={videoPlayerRef} src={previewUrl} controls playsInline crossOrigin="anonymous" />
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

              {/* Quality Enhancement Summary Badge */}
              {poseResult?.quality_info?.applied && (
                <div
                  className="glass-panel"
                  style={{
                    padding: '12px 16px',
                    marginBottom: '16px',
                    borderLeft: '3px solid var(--accent-cyan)',
                    background: 'rgba(56, 189, 248, 0.06)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    flexWrap: 'wrap',
                    gap: '12px',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <Wand2 size={18} color="var(--accent-cyan)" />
                    <div>
                      <div style={{ fontSize: '0.86rem', fontWeight: 700, color: 'var(--text-main)' }}>
                        Quality Filter Active: {String(poseResult.quality_info.enhancement_filter).toUpperCase()} ({poseResult.quality_info.enhancement_strength}x)
                      </div>
                      <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                        Raw blur score: {poseResult.quality_info.initial_blur_score} &bull; Enhanced sharpness score: {poseResult.quality_info.enhanced_blur_score}
                      </div>
                    </div>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span
                      style={{
                        padding: '4px 10px',
                        borderRadius: '999px',
                        background: 'rgba(52, 211, 153, 0.15)',
                        color: 'var(--accent-emerald)',
                        fontSize: '0.78rem',
                        fontWeight: 700,
                      }}
                    >
                      +{poseResult.quality_info.sharpness_improvement_pct}% Sharpness
                    </span>
                  </div>
                </div>
              )}

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

              {/* Milestone 1: Temporal Movement Analysis */}
              {poseResult?.temporal_analysis && (
                <div className="glass-panel" style={{ padding: '16px', marginTop: '14px' }}>
                  <div
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      flexWrap: 'wrap',
                      gap: '10px',
                      marginBottom: '14px',
                    }}
                  >
                    <div>
                      <h4
                        style={{
                          fontSize: '1rem',
                          fontWeight: 700,
                          display: 'flex',
                          alignItems: 'center',
                          gap: '8px',
                        }}
                      >
                        <TrendingUp size={18} color="var(--accent-cyan)" />
                        Temporal Movement Analysis (Milestone 1)
                      </h4>
                      <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginTop: '3px' }}>
                        Multi-frame movement kinematics & window-based feature extraction for ML training
                      </p>
                    </div>
                    {poseResult.temporal_csv_url && (
                      <a
                        href={`${API_BASE}${poseResult.temporal_csv_url}`}
                        download={`runform_temporal_${poseResult.analysis_id}.csv`}
                        className="nav-tab-btn"
                        style={{
                          textDecoration: 'none',
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: '6px',
                          fontSize: '0.8rem',
                          padding: '6px 12px',
                        }}
                      >
                        <Download size={14} />
                        Export Windows CSV
                      </a>
                    )}
                  </div>

                  {/* Temporal Highlights Cards */}
                  {poseResult.temporal_analysis.video_features && (
                    <div
                      style={{
                        display: 'grid',
                        gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))',
                        gap: '10px',
                        marginBottom: '16px',
                      }}
                    >
                      <div className="glass-panel metric-card">
                        <span className="metric-label">Left Knee ROM</span>
                        <div className="metric-value">
                          {poseResult.temporal_analysis.video_features.knee_left_rom != null
                            ? `${poseResult.temporal_analysis.video_features.knee_left_rom.toFixed(1)}°`
                            : 'N/A'}
                        </div>
                        <span className="metric-sub">
                          SD: ±{poseResult.temporal_analysis.video_features.knee_left_std != null
                            ? `${poseResult.temporal_analysis.video_features.knee_left_std.toFixed(1)}°`
                            : 'N/A'}
                        </span>
                      </div>

                      <div className="glass-panel metric-card">
                        <span className="metric-label">Right Knee ROM</span>
                        <div className="metric-value">
                          {poseResult.temporal_analysis.video_features.knee_right_rom != null
                            ? `${poseResult.temporal_analysis.video_features.knee_right_rom.toFixed(1)}°`
                            : 'N/A'}
                        </div>
                        <span className="metric-sub">
                          SD: ±{poseResult.temporal_analysis.video_features.knee_right_std != null
                            ? `${poseResult.temporal_analysis.video_features.knee_right_std.toFixed(1)}°`
                            : 'N/A'}
                        </span>
                      </div>

                      <div className="glass-panel metric-card">
                        <span className="metric-label">Left Hip ROM</span>
                        <div className="metric-value">
                          {poseResult.temporal_analysis.video_features.hip_left_rom != null
                            ? `${poseResult.temporal_analysis.video_features.hip_left_rom.toFixed(1)}°`
                            : 'N/A'}
                        </div>
                        <span className="metric-sub">
                          SD: ±{poseResult.temporal_analysis.video_features.hip_left_std != null
                            ? `${poseResult.temporal_analysis.video_features.hip_left_std.toFixed(1)}°`
                            : 'N/A'}
                        </span>
                      </div>

                      <div className="glass-panel metric-card">
                        <span className="metric-label">Right Hip ROM</span>
                        <div className="metric-value">
                          {poseResult.temporal_analysis.video_features.hip_right_rom != null
                            ? `${poseResult.temporal_analysis.video_features.hip_right_rom.toFixed(1)}°`
                            : 'N/A'}
                        </div>
                        <span className="metric-sub">
                          SD: ±{poseResult.temporal_analysis.video_features.hip_right_std != null
                            ? `${poseResult.temporal_analysis.video_features.hip_right_std.toFixed(1)}°`
                            : 'N/A'}
                        </span>
                      </div>

                      <div className="glass-panel metric-card">
                        <span className="metric-label">Trunk Lean ROM</span>
                        <div className="metric-value">
                          {poseResult.temporal_analysis.video_features.trunk_lean_rom != null
                            ? `${poseResult.temporal_analysis.video_features.trunk_lean_rom.toFixed(1)}°`
                            : 'N/A'}
                        </div>
                        <span className="metric-sub">
                          SD: ±{poseResult.temporal_analysis.video_features.trunk_lean_std != null
                            ? `${poseResult.temporal_analysis.video_features.trunk_lean_std.toFixed(1)}°`
                            : 'N/A'}
                        </span>
                      </div>

                      <div className="glass-panel metric-card">
                        <span className="metric-label">Knee Diff Var</span>
                        <div className="metric-value">
                          {poseResult.temporal_analysis.video_features.knee_asymmetry_std != null
                            ? `±${poseResult.temporal_analysis.video_features.knee_asymmetry_std.toFixed(1)}°`
                            : 'N/A'}
                        </div>
                        <span className="metric-sub">Diff std dev</span>
                      </div>
                    </div>
                  )}

                  {/* Window Breakdown Table */}
                  {poseResult.temporal_analysis.windows && poseResult.temporal_analysis.windows.length > 0 && (
                    <div style={{ marginTop: '12px' }}>
                      <div
                        style={{
                          fontSize: '0.85rem',
                          fontWeight: 600,
                          color: 'var(--text-soft)',
                          marginBottom: '8px',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '6px',
                        }}
                      >
                        <Clock size={14} /> Time Window Breakdown ({poseResult.temporal_analysis.window_duration_sec}s windows)
                      </div>
                      <div style={{ overflowX: 'auto' }}>
                        <table
                          style={{
                            width: '100%',
                            borderCollapse: 'collapse',
                            fontSize: '0.8rem',
                            textAlign: 'left',
                          }}
                        >
                          <thead>
                            <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.08)', color: 'var(--text-muted)' }}>
                              <th style={{ padding: '8px 10px' }}>Window</th>
                              <th style={{ padding: '8px 10px' }}>Time Span</th>
                              <th style={{ padding: '8px 10px' }}>Valid Frames</th>
                              <th style={{ padding: '8px 10px' }}>L Knee ROM</th>
                              <th style={{ padding: '8px 10px' }}>R Knee ROM</th>
                              <th style={{ padding: '8px 10px' }}>Trunk ROM</th>
                              <th style={{ padding: '8px 10px' }}>Quality</th>
                            </tr>
                          </thead>
                          <tbody>
                            {poseResult.temporal_analysis.windows.map((w) => (
                              <tr key={w.window_index} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                                <td style={{ padding: '8px 10px', fontWeight: 600 }}>#{w.window_index + 1}</td>
                                <td style={{ padding: '8px 10px', color: 'var(--text-soft)' }}>
                                  {w.start_time.toFixed(1)}s – {w.end_time.toFixed(1)}s
                                </td>
                                <td style={{ padding: '8px 10px' }}>
                                  {w.valid_frames} / {w.total_frames}
                                </td>
                                <td style={{ padding: '8px 10px' }}>
                                  {w.features.knee_left_rom != null ? `${w.features.knee_left_rom.toFixed(1)}°` : '—'}
                                </td>
                                <td style={{ padding: '8px 10px' }}>
                                  {w.features.knee_right_rom != null ? `${w.features.knee_right_rom.toFixed(1)}°` : '—'}
                                </td>
                                <td style={{ padding: '8px 10px' }}>
                                  {w.features.trunk_lean_rom != null ? `${w.features.trunk_lean_rom.toFixed(1)}°` : '—'}
                                </td>
                                <td style={{ padding: '8px 10px' }}>
                                  <span
                                    style={{
                                      padding: '2px 8px',
                                      borderRadius: '4px',
                                      fontSize: '0.72rem',
                                      fontWeight: 600,
                                      background: w.is_reliable ? 'rgba(16,185,129,0.15)' : 'rgba(239,68,68,0.15)',
                                      color: w.is_reliable ? 'var(--accent-emerald)' : 'var(--accent-rose)',
                                    }}
                                  >
                                    {w.is_reliable ? 'Reliable' : 'Low frames'}
                                  </span>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}
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
