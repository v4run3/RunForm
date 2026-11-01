import React, { useState, useRef, useEffect } from 'react';
import {
  Camera,
  Square,
  Circle,
  RotateCcw,
  Download,
  Sparkles,
  Activity,
  CheckCircle2,
  AlertTriangle,
  Info,
  SwitchCamera,
  RefreshCw,
} from 'lucide-react';

const API_BASE = '';

/**
 * Determine the best supported recording MIME type in the current browser.
 */
export const getSupportedRecordingMimeType = () => {
  if (typeof window === 'undefined' || typeof MediaRecorder === 'undefined' || !MediaRecorder.isTypeSupported) {
    return '';
  }
  const candidateTypes = [
    'video/webm;codecs=vp9,opus',
    'video/webm;codecs=vp8,opus',
    'video/webm;codecs=vp9',
    'video/webm;codecs=vp8',
    'video/webm',
    'video/mp4;codecs=avc1,mp4a.40.2',
    'video/mp4',
  ];
  for (const type of candidateTypes) {
    try {
      if (MediaRecorder.isTypeSupported(type)) {
        return type;
      }
    } catch {
      // Ignore syntax check failures on specific browsers
    }
  }
  return '';
};

/**
 * Extract a suitable file extension (.webm or .mp4) based on the recorded MIME type.
 */
export const getExtensionForMimeType = (mimeType) => {
  if (!mimeType) return 'webm';
  const lower = mimeType.toLowerCase();
  if (lower.includes('mp4')) return 'mp4';
  if (lower.includes('webm')) return 'webm';
  if (lower.includes('quicktime') || lower.includes('mov')) return 'mov';
  return 'webm';
};

/**
 * Generate a timestamped download filename for the annotated video.
 */
export const getAnnotatedDownloadFilename = (date = new Date()) => {
  const pad = (n) => String(n).padStart(2, '0');
  const yyyy = date.getFullYear();
  const MM = pad(date.getMonth() + 1);
  const dd = pad(date.getDate());
  const hh = pad(date.getHours());
  const mm = pad(date.getMinutes());
  const ss = pad(date.getSeconds());
  return `RunForm_Annotated_${yyyy}${MM}${dd}_${hh}${mm}${ss}.mp4`;
};

export default function LiveCameraView() {
  // State machine: 'IDLE' | 'RECORDING' | 'RECORDED' | 'ANALYZING' | 'RESULTS'
  const [status, setStatus] = useState('IDLE');
  const [facingMode, setFacingMode] = useState('environment'); // 'environment' (rear) | 'user' (front)
  const [cameraActive, setCameraActive] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [recordedBlob, setRecordedBlob] = useState(null);
  const [recordedVideoUrl, setRecordedVideoUrl] = useState(null);
  const [recordedMimeType, setRecordedMimeType] = useState('');
  const [analysisResult, setAnalysisResult] = useState(null);
  const [error, setError] = useState(null);
  const [videoError, setVideoError] = useState(null);
  const [downloading, setDownloading] = useState(false);

  const videoPreviewRef = useRef(null);
  const streamRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const timerIntervalRef = useRef(null);
  const recordedUrlRef = useRef(null);

  // Stop active camera stream tracks
  const stopCameraStream = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => {
        try {
          track.stop();
        } catch {
          // ignore track stop errors
        }
      });
      streamRef.current = null;
    }
    if (videoPreviewRef.current) {
      videoPreviewRef.current.srcObject = null;
    }
    setCameraActive(false);
  };

  // Sync ref to manage clean URL revoking
  useEffect(() => {
    recordedUrlRef.current = recordedVideoUrl;
  }, [recordedVideoUrl]);

  // Clean up stream tracks and object URLs on unmount
  useEffect(() => {
    return () => {
      stopCameraStream();
      if (timerIntervalRef.current) {
        clearInterval(timerIntervalRef.current);
      }
      if (recordedUrlRef.current) {
        URL.revokeObjectURL(recordedUrlRef.current);
      }
    };
  }, []);

  /**
   * Start camera preview stream
   */
  const startCameraStream = async (targetFacingMode = facingMode) => {
    setError(null);

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      if (typeof window !== 'undefined' && !window.isSecureContext) {
        setError(
          'Mobile browsers strictly require an HTTPS connection to access the camera. Please open RunForm using an HTTPS URL.'
        );
      } else {
        setError('Camera access is not supported on this browser. Please try Chrome, Safari, or Edge.');
      }
      return null;
    }

    stopCameraStream();

    try {
      let stream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1280, max: 1920 },
            height: { ideal: 720, max: 1080 },
            facingMode: { ideal: targetFacingMode },
          },
          audio: true, // capture audio if available
        });
      } catch {
        // Fall back to video only if audio permission fails or mic is missing
        try {
          stream = await navigator.mediaDevices.getUserMedia({
            video: {
              width: { ideal: 1280 },
              height: { ideal: 720 },
              facingMode: { ideal: targetFacingMode },
            },
            audio: false,
          });
        } catch (constraintErr) {
          if (
            constraintErr.name === 'OverconstrainedError' ||
            constraintErr.name === 'ConstraintNotSatisfiedError'
          ) {
            stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
          } else {
            throw constraintErr;
          }
        }
      }

      streamRef.current = stream;
      if (videoPreviewRef.current) {
        videoPreviewRef.current.srcObject = stream;
        try {
          await videoPreviewRef.current.play();
        } catch {
          // User interaction might be required to play
        }
      }
      setCameraActive(true);
      return stream;
    } catch (err) {
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        setError('Camera permission was denied. Please allow camera permissions in your browser settings.');
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        setError('No camera device was detected on this device.');
      } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
        setError('Camera is currently in use by another app or tab. Please close other camera apps and retry.');
      } else if (typeof window !== 'undefined' && !window.isSecureContext) {
        setError('Mobile browsers block camera access on insecure HTTP. Please access this page over HTTPS.');
      } else {
        setError(`Could not access camera: ${err.message || err.name || 'Unknown error'}`);
      }
      return null;
    }
  };

  /**
   * Switch between front and rear cameras
   */
  const handleToggleFacingMode = async () => {
    const nextMode = facingMode === 'environment' ? 'user' : 'environment';
    setFacingMode(nextMode);
    if (cameraActive && status === 'IDLE') {
      await startCameraStream(nextMode);
    }
  };

  /**
   * Start recording running movement
   */
  const handleStartRecording = async () => {
    setError(null);
    setVideoError(null);

    let stream = streamRef.current;
    if (!stream || !stream.active) {
      stream = await startCameraStream(facingMode);
      if (!stream) return;
    }

    if (typeof MediaRecorder === 'undefined') {
      setError('MediaRecorder API is not supported on this browser.');
      return;
    }

    const mimeType = getSupportedRecordingMimeType();
    setRecordedMimeType(mimeType);

    const recorderOptions = mimeType ? { mimeType } : {};
    let mediaRecorder;
    try {
      mediaRecorder = new MediaRecorder(stream, recorderOptions);
    } catch (recorderErr) {
      try {
        mediaRecorder = new MediaRecorder(stream);
      } catch {
        setError(`Could not initialize video recording: ${recorderErr.message}`);
        return;
      }
    }

    chunksRef.current = [];

    mediaRecorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        chunksRef.current.push(event.data);
      }
    };

    mediaRecorder.onstop = () => {
      const actualType = mediaRecorder.mimeType || mimeType || 'video/webm';
      const blob = new Blob(chunksRef.current, { type: actualType });
      setRecordedBlob(blob);

      if (recordedUrlRef.current) {
        URL.revokeObjectURL(recordedUrlRef.current);
      }
      const newUrl = URL.createObjectURL(blob);
      setRecordedVideoUrl(newUrl);

      // Cleanly release active camera hardware once recording is complete
      stopCameraStream();
      setStatus('RECORDED');
    };

    mediaRecorderRef.current = mediaRecorder;

    try {
      mediaRecorder.start(1000); // chunk every second
      setStatus('RECORDING');
      setElapsedSeconds(0);

      if (timerIntervalRef.current) clearInterval(timerIntervalRef.current);
      const startTime = Date.now();
      timerIntervalRef.current = setInterval(() => {
        setElapsedSeconds(Math.floor((Date.now() - startTime) / 1000));
      }, 500);
    } catch (startErr) {
      setError(`Failed to start recording: ${startErr.message}`);
      setStatus('IDLE');
    }
  };

  /**
   * Stop recording
   */
  const handleStopRecording = () => {
    if (timerIntervalRef.current) {
      clearInterval(timerIntervalRef.current);
      timerIntervalRef.current = null;
    }

    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      try {
        mediaRecorderRef.current.stop();
      } catch (stopErr) {
        console.error('Error stopping MediaRecorder:', stopErr);
      }
    }
  };

  /**
   * Reset back to IDLE to record another video
   */
  const handleRecordAgain = () => {
    if (timerIntervalRef.current) {
      clearInterval(timerIntervalRef.current);
      timerIntervalRef.current = null;
    }
    stopCameraStream();

    if (recordedUrlRef.current) {
      URL.revokeObjectURL(recordedUrlRef.current);
      recordedUrlRef.current = null;
    }

    setRecordedBlob(null);
    setRecordedVideoUrl(null);
    setAnalysisResult(null);
    setError(null);
    setVideoError(null);
    setElapsedSeconds(0);
    setStatus('IDLE');
  };

  /**
   * Send the recorded video Blob to FastAPI MediaPipe pose-analysis endpoint
   */
  const handleAnalyzeRecording = async () => {
    if (!recordedBlob) {
      setError('No recorded video found to analyze.');
      return;
    }

    setStatus('ANALYZING');
    setError(null);
    setVideoError(null);

    const ext = getExtensionForMimeType(recordedBlob.type || recordedMimeType);
    const filename = `camera_recording_${Date.now()}.${ext}`;

    const formData = new FormData();
    formData.append('file', recordedBlob, filename);

    try {
      const res = await fetch(`${API_BASE}/api/pose-analysis`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error during analysis: ${res.statusText}`);
      }

      const data = await res.json();
      setAnalysisResult(data);
      setStatus('RESULTS');
    } catch (err) {
      setError(err.message || 'Could not analyze recording. Make sure the backend server is running.');
      setStatus('RECORDED'); // Remain in recorded state so user can retry or re-record
    }
  };

  /**
   * Download the annotated skeleton video from backend
   */
  const handleDownloadAnnotatedVideo = async () => {
    const videoUrl = analysisResult?.video_url || analysisResult?.annotated_video_url;
    if (!videoUrl) {
      setError('Annotated video URL is not available.');
      return;
    }

    setDownloading(true);
    setError(null);

    try {
      const res = await fetch(`${API_BASE}${videoUrl}`);
      if (!res.ok) {
        throw new Error(`Failed to fetch video file (${res.status} ${res.statusText})`);
      }
      const blob = await res.blob();
      const downloadBlobUrl = URL.createObjectURL(blob);

      const filename = getAnnotatedDownloadFilename();

      const a = document.createElement('a');
      a.href = downloadBlobUrl;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);

      setTimeout(() => {
        URL.revokeObjectURL(downloadBlobUrl);
      }, 2000);
    } catch (err) {
      setError(`Download failed: ${err.message}`);
    } finally {
      setDownloading(false);
    }
  };

  const formatTimer = (totalSeconds) => {
    const mins = Math.floor(totalSeconds / 60);
    const secs = totalSeconds % 60;
    return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
  };

  // Metrics extraction for RESULTS view
  const summary = analysisResult?.summary || analysisResult?.metrics;
  const detectionPct =
    summary?.detection_rate != null ? (summary.detection_rate * 100).toFixed(0) : null;
  const annotatedVideoUrl = analysisResult?.video_url || analysisResult?.annotated_video_url;

  return (
    <div className="view-container">
      {/* Disclaimer Banner */}
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

      {/* Header and Stage Navigation */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '12px',
          marginBottom: '16px',
        }}
      >
        <div>
          <h2 style={{ fontSize: '1.2rem', fontWeight: 700 }}>
            {status === 'RESULTS' ? 'Running Form Analysis Results' : 'Camera Recording & Form Analysis'}
          </h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.86rem', marginTop: '2px' }}>
            {status === 'IDLE' && 'Position your camera to capture full-body side view during running.'}
            {status === 'RECORDING' && 'Recording running movement in progress. Keep runner fully in frame.'}
            {status === 'RECORDED' && 'Recording complete. Review your clip and proceed with MediaPipe analysis.'}
            {status === 'ANALYZING' && 'MediaPipe Pose is evaluating joint kinematics and body angles...'}
            {status === 'RESULTS' && 'MediaPipe pose landmarks and running-form observations.'}
          </p>
        </div>

        {/* Top Controls depending on status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
          {status === 'IDLE' && (
            <button
              type="button"
              className="nav-tab-btn"
              onClick={handleToggleFacingMode}
              title={facingMode === 'environment' ? 'Switch to front camera' : 'Switch to rear camera'}
              style={{ padding: '8px 12px', display: 'flex', alignItems: 'center', gap: '6px' }}
            >
              <SwitchCamera size={16} />
              <span style={{ fontSize: '0.82rem' }}>
                {facingMode === 'environment' ? 'Rear Cam' : 'Front Cam'}
              </span>
            </button>
          )}

          {(status === 'RECORDED' || status === 'RESULTS') && (
            <button
              type="button"
              className="nav-tab-btn"
              onClick={handleRecordAgain}
              style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
            >
              <RotateCcw size={15} />
              <span>Record Again</span>
            </button>
          )}

          {status === 'RESULTS' && annotatedVideoUrl && (
            <button
              type="button"
              className="btn-primary"
              onClick={handleDownloadAnnotatedVideo}
              disabled={downloading}
              style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
            >
              {downloading ? (
                <>
                  <RefreshCw size={16} style={{ animation: 'spin 1s linear infinite' }} />
                  Downloading...
                </>
              ) : (
                <>
                  <Download size={16} />
                  Download Annotated Video
                </>
              )}
            </button>
          )}
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
            marginBottom: '16px',
          }}
        >
          <AlertTriangle color="var(--accent-rose)" size={20} style={{ flexShrink: 0, marginTop: '2px' }} />
          <p style={{ fontSize: '0.86rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>{error}</p>
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
            marginBottom: '16px',
          }}
        >
          <AlertTriangle color="var(--accent-rose)" size={20} style={{ flexShrink: 0, marginTop: '2px' }} />
          <p style={{ fontSize: '0.86rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>{videoError}</p>
        </div>
      )}

      {/* =========================================================================
          STAGE 1: IDLE / RECORDING / RECORDED / ANALYZING View (Single Main Camera)
          ========================================================================= */}
      {status !== 'RESULTS' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {/* Main Single Camera / Video Area */}
          <div
            className="glass-panel"
            style={{
              padding: '16px',
              position: 'relative',
              borderRadius: '16px',
              border: status === 'RECORDING' ? '2px solid var(--accent-rose)' : '1px solid var(--border-glass)',
              transition: 'border-color 0.3s ease',
            }}
          >
            {/* Live Camera Viewfinder (Visible in IDLE when camera active, or while RECORDING) */}
            <div
              className="video-wrapper"
              style={{
                minHeight: '360px',
                display: status === 'RECORDED' || status === 'ANALYZING' ? 'none' : 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                background: '#090d14',
                borderRadius: '12px',
                overflow: 'hidden',
                position: 'relative',
              }}
            >
              <video
                ref={videoPreviewRef}
                playsInline
                autoPlay
                muted
                style={{
                  width: '100%',
                  height: 'auto',
                  maxHeight: '70vh',
                  display: cameraActive ? 'block' : 'none',
                  objectFit: 'contain',
                }}
              />

              {/* Inactive Camera Placeholder in IDLE */}
              {!cameraActive && status === 'IDLE' && (
                <div style={{ textAlign: 'center', padding: '32px 18px', maxWidth: '440px' }}>
                  <div
                    style={{
                      width: '64px',
                      height: '64px',
                      borderRadius: '50%',
                      background: 'var(--bg-subtle)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      margin: '0 auto 16px',
                    }}
                  >
                    <Camera size={32} color="var(--accent-cyan)" />
                  </div>
                  <h3 style={{ fontSize: '1.15rem', fontWeight: 700, marginBottom: '8px' }}>
                    Camera Ready
                  </h3>
                  <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', lineHeight: 1.5, marginBottom: '16px' }}>
                    Position yourself so your full body is visible, then start recording.
                  </p>
                  <p style={{ fontSize: '0.78rem', color: 'var(--text-soft)' }}>
                    Captures high-resolution video for subsequent 33-point MediaPipe Pose biomechanics evaluation.
                  </p>
                </div>
              )}

              {/* Active Recording Badge Overlay */}
              {status === 'RECORDING' && (
                <div
                  style={{
                    position: 'absolute',
                    top: '16px',
                    left: '16px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    background: 'rgba(18, 7, 10, 0.85)',
                    border: '1px solid var(--accent-rose)',
                    padding: '6px 14px',
                    borderRadius: '999px',
                    backdropFilter: 'blur(8px)',
                    zIndex: 10,
                  }}
                >
                  <span
                    style={{
                      width: '10px',
                      height: '10px',
                      borderRadius: '50%',
                      background: 'var(--accent-rose)',
                      boxShadow: '0 0 8px var(--accent-rose)',
                      display: 'inline-block',
                      animation: 'pulse 1s infinite',
                    }}
                  />
                  <span style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--accent-rose)' }}>REC</span>
                  <span style={{ fontSize: '0.82rem', fontWeight: 600, color: '#fff', marginLeft: '4px' }}>
                    {formatTimer(elapsedSeconds)}
                  </span>
                </div>
              )}
            </div>

            {/* Recorded Video Playback (Visible in RECORDED or ANALYZING) */}
            {(status === 'RECORDED' || status === 'ANALYZING') && recordedVideoUrl && (
              <div
                className="video-wrapper"
                style={{
                  minHeight: '360px',
                  background: '#090d14',
                  borderRadius: '12px',
                  overflow: 'hidden',
                  position: 'relative',
                }}
              >
                <video
                  src={recordedVideoUrl}
                  controls={status !== 'ANALYZING'}
                  playsInline
                  style={{ width: '100%', height: 'auto', maxHeight: '70vh', objectFit: 'contain' }}
                />

                {status === 'ANALYZING' && (
                  <div
                    style={{
                      position: 'absolute',
                      inset: 0,
                      background: 'rgba(10, 14, 23, 0.85)',
                      backdropFilter: 'blur(6px)',
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: '14px',
                      padding: '24px',
                      textAlign: 'center',
                      zIndex: 10,
                    }}
                  >
                    <RefreshCw size={36} color="var(--accent-cyan)" style={{ animation: 'spin 1s linear infinite' }} />
                    <h3 style={{ fontSize: '1.1rem', fontWeight: 700 }}>Analyzing Running Form...</h3>
                    <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem', maxWidth: '380px', lineHeight: 1.5 }}>
                      MediaPipe Pose is evaluating joint angles, knee asymmetry, and trunk kinematics.
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Prominent Action Controls Bar */}
          <div
            className="glass-panel"
            style={{
              padding: '16px 20px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              flexWrap: 'wrap',
              gap: '14px',
            }}
          >
            {/* Status Information */}
            <div>
              {status === 'IDLE' && (
                <div>
                  <h4 style={{ fontSize: '0.96rem', fontWeight: 600 }}>Ready to Record</h4>
                  <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                    Click "Start Recording" to capture your running gait.
                  </p>
                </div>
              )}
              {status === 'RECORDING' && (
                <div>
                  <h4 style={{ fontSize: '0.96rem', fontWeight: 600, color: 'var(--accent-rose)' }}>
                    Recording Active ({formatTimer(elapsedSeconds)})
                  </h4>
                  <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                    Capture 5–15 seconds of steady running form, then click "Stop Recording".
                  </p>
                </div>
              )}
              {status === 'RECORDED' && (
                <div>
                  <h4 style={{ fontSize: '0.96rem', fontWeight: 600 }}>Recorded Video Ready</h4>
                  <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                    Duration: {formatTimer(elapsedSeconds)} &bull; Ready to run MediaPipe pose estimation.
                  </p>
                </div>
              )}
              {status === 'ANALYZING' && (
                <div>
                  <h4 style={{ fontSize: '0.96rem', fontWeight: 600, color: 'var(--accent-cyan)' }}>
                    Analyzing Gait Biomechanics
                  </h4>
                  <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                    Please wait while MediaPipe Pose processes all frames.
                  </p>
                </div>
              )}
            </div>

            {/* Buttons */}
            <div style={{ display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap' }}>
              {status === 'IDLE' && (
                <button
                  type="button"
                  className="btn-primary"
                  onClick={handleStartRecording}
                  style={{
                    padding: '12px 24px',
                    fontSize: '0.96rem',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                  }}
                >
                  <Circle size={18} fill="currentColor" color="currentColor" />
                  Start Recording
                </button>
              )}

              {status === 'RECORDING' && (
                <button
                  type="button"
                  className="btn-primary"
                  onClick={handleStopRecording}
                  style={{
                    background: 'var(--accent-rose)',
                    color: '#12070a',
                    padding: '12px 24px',
                    fontSize: '0.96rem',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                  }}
                >
                  <Square size={18} fill="currentColor" />
                  Stop Recording
                </button>
              )}

              {status === 'RECORDED' && (
                <>
                  <button
                    type="button"
                    className="nav-tab-btn"
                    onClick={handleRecordAgain}
                    style={{ padding: '10px 18px', display: 'flex', alignItems: 'center', gap: '6px' }}
                  >
                    <RotateCcw size={16} />
                    Discard &amp; Record Again
                  </button>
                  <button
                    type="button"
                    className="btn-primary"
                    onClick={handleAnalyzeRecording}
                    style={{
                      padding: '10px 22px',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px',
                    }}
                  >
                    <Sparkles size={18} />
                    Analyze Recording
                  </button>
                </>
              )}

              {status === 'ANALYZING' && (
                <button
                  type="button"
                  className="btn-primary"
                  disabled
                  style={{
                    opacity: 0.7,
                    padding: '10px 22px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                  }}
                >
                  <RefreshCw size={18} style={{ animation: 'spin 1s linear infinite' }} />
                  Processing...
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* =========================================================================
          STAGE 2: RESULTS View (Annotated Video, Metrics, Observations, Download)
          ========================================================================= */}
      {status === 'RESULTS' && analysisResult && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
          {/* Action Row */}
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              flexWrap: 'wrap',
              gap: '12px',
            }}
          >
            <div>
              <h3 style={{ fontSize: '1.05rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Activity color="var(--accent-cyan)" size={19} /> Form Analysis Summary
              </h3>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.82rem', marginTop: '3px' }}>
                Annotated overlay generated by MediaPipe Pose with 2D running kinematics.
              </p>
            </div>
            <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
              <button
                type="button"
                className="nav-tab-btn"
                onClick={handleRecordAgain}
                style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
              >
                <RotateCcw size={15} />
                Record Another Clip
              </button>
              <button
                type="button"
                className="btn-primary"
                onClick={handleDownloadAnnotatedVideo}
                disabled={downloading}
                style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
              >
                {downloading ? (
                  <>
                    <RefreshCw size={16} style={{ animation: 'spin 1s linear infinite' }} />
                    Downloading...
                  </>
                ) : (
                  <>
                    <Download size={16} />
                    Download Annotated Video
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Main Video View: Annotated Skeleton Video */}
          <div className="glass-panel" style={{ padding: '16px', borderRadius: '16px' }}>
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '10px',
              }}
            >
              <h4 style={{ fontSize: '0.86rem', color: 'var(--text-muted)' }}>Annotated Skeleton Video</h4>
              <span
                style={{
                  fontSize: '0.74rem',
                  background: 'var(--bg-subtle)',
                  color: 'var(--text-muted)',
                  padding: '3px 8px',
                  borderRadius: '999px',
                }}
              >
                MediaPipe H.264
              </span>
            </div>

            <div
              className="video-wrapper"
              style={{
                minHeight: '360px',
                background: '#090d14',
                borderRadius: '12px',
                overflow: 'hidden',
              }}
            >
              <video
                key={annotatedVideoUrl}
                controls
                autoPlay
                loop
                playsInline
                onError={() =>
                  setVideoError('Annotated video could not be played. Check the generated video codec/format.')
                }
                style={{ width: '100%', height: 'auto', maxHeight: '70vh', objectFit: 'contain' }}
              >
                <source src={`${API_BASE}${annotatedVideoUrl}`} type="video/mp4" />
                Annotated video could not be played.
              </video>
            </div>
          </div>

          {/* Running Form Metrics Cards */}
          {summary && (
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
                        detectionPct != null && Number(detectionPct) >= 60
                          ? 'var(--accent-emerald)'
                          : 'var(--accent-amber)',
                    }}
                  >
                    {detectionPct != null ? `${detectionPct}%` : 'N/A'}
                  </div>
                  <span className="metric-sub">
                    {summary.pose_detected_frames} / {summary.frames_processed} frames
                  </span>
                </div>

                <div className="glass-panel metric-card">
                  <span className="metric-label">Left knee</span>
                  <div className="metric-value">
                    {summary.average_left_knee_angle != null ? `${summary.average_left_knee_angle}°` : 'N/A'}
                  </div>
                  <span className="metric-sub">Target: 155-165°</span>
                </div>

                <div className="glass-panel metric-card">
                  <span className="metric-label">Right knee</span>
                  <div className="metric-value">
                    {summary.average_right_knee_angle != null ? `${summary.average_right_knee_angle}°` : 'N/A'}
                  </div>
                  <span className="metric-sub">Target: 155-165°</span>
                </div>

                <div className="glass-panel metric-card">
                  <span className="metric-label">Knee difference</span>
                  <div
                    className="metric-value"
                    style={{
                      color:
                        summary.average_knee_asymmetry != null && summary.average_knee_asymmetry < 10
                          ? 'var(--accent-emerald)'
                          : 'var(--accent-amber)',
                    }}
                  >
                    {summary.average_knee_asymmetry != null ? `${summary.average_knee_asymmetry}°` : 'N/A'}
                  </div>
                  <span className="metric-sub">Left/right asymmetry</span>
                </div>

                <div className="glass-panel metric-card">
                  <span className="metric-label">Left hip</span>
                  <div className="metric-value">
                    {summary.average_left_hip_angle != null ? `${summary.average_left_hip_angle}°` : 'N/A'}
                  </div>
                  <span className="metric-sub">Average angle</span>
                </div>

                <div className="glass-panel metric-card">
                  <span className="metric-label">Right hip</span>
                  <div className="metric-value">
                    {summary.average_right_hip_angle != null ? `${summary.average_right_hip_angle}°` : 'N/A'}
                  </div>
                  <span className="metric-sub">Average angle</span>
                </div>

                <div className="glass-panel metric-card">
                  <span className="metric-label">Trunk lean</span>
                  <div className="metric-value">
                    {summary.average_trunk_lean != null ? `${summary.average_trunk_lean}°` : 'N/A'}
                  </div>
                  <span className="metric-sub">Target: 5-12°</span>
                </div>
              </div>

              {/* Form Observations */}
              {summary.observations && summary.observations.length > 0 && (
                <div className="glass-panel" style={{ padding: '16px', marginTop: '14px' }}>
                  <h4 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '14px' }}>Observations</h4>
                  <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '10px' }}>
                    {summary.observations.map((obs, idx) => (
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
        </div>
      )}
    </div>
  );
}
