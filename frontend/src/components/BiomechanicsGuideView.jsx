import React from 'react';
import { BookOpen, ShieldCheck, Layers, Activity } from 'lucide-react';

export default function BiomechanicsGuideView() {
  return (
    <div className="view-container">
      <div className="glass-panel" style={{ padding: '18px' }}>
        <h2 style={{ fontSize: '1.12rem', fontWeight: 700, marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <BookOpen color="var(--accent-cyan)" size={19} /> How the review works
        </h2>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', lineHeight: 1.6 }}>
          RunForm uses video pose points to estimate a few practical signals: foot strike, knee angle, trunk lean and risk flags.
        </p>
      </div>

      <div className="grid-2col">
        <div className="glass-panel" style={{ padding: '18px' }}>
          <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Activity size={18} color="var(--accent-cyan)" /> Foot strike
          </h3>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.86rem', lineHeight: 1.6, marginBottom: '12px' }}>
            The landing pattern is estimated at touchdown from the lower-leg angle.
          </p>
          <ul style={{ display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '0.86rem', lineHeight: 1.5, paddingLeft: '18px' }}>
            <li><strong>Heel:</strong> can increase knee loading.</li>
            <li><strong>Midfoot:</strong> often spreads load more evenly.</li>
            <li><strong>Forefoot:</strong> places more demand on calves and Achilles.</li>
          </ul>
        </div>

        <div className="glass-panel" style={{ padding: '18px' }}>
          <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <ShieldCheck size={18} color="var(--accent-emerald)" /> Knee angle
          </h3>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.86rem', lineHeight: 1.6, marginBottom: '12px' }}>
            Knee angle gives a quick signal for stiffness or overstriding at contact.
          </p>
          <ul style={{ display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '0.86rem', lineHeight: 1.5, paddingLeft: '18px' }}>
            <li><strong>155-165 deg:</strong> usually shows better shock absorption.</li>
            <li><strong>Above 165 deg:</strong> can create extra braking force and impact.</li>
          </ul>
        </div>
      </div>

      <div className="glass-panel" style={{ padding: '18px' }}>
        <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Layers size={18} color="var(--accent-cyan)" /> References
        </h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '10px' }}>
          <div style={{ padding: '12px', background: 'var(--bg-subtle)', borderRadius: '8px' }}>
            <strong style={{ fontSize: '0.84rem' }}>VanKeersbilck (2025)</strong>
            <p style={{ fontSize: '0.76rem', color: 'var(--text-muted)', marginTop: '5px', lineHeight: 1.45 }}>
              Smartphone video and monocular pose estimation for remote gait review.
            </p>
          </div>
          <div style={{ padding: '12px', background: 'var(--bg-subtle)', borderRadius: '8px' }}>
            <strong style={{ fontSize: '0.84rem' }}>Stenum et al. (2024)</strong>
            <p style={{ fontSize: '0.76rem', color: 'var(--text-muted)', marginTop: '5px', lineHeight: 1.45 }}>
              Video-based pose estimation across camera perspectives.
            </p>
          </div>
          <div style={{ padding: '12px', background: 'var(--bg-subtle)', borderRadius: '8px' }}>
            <strong style={{ fontSize: '0.84rem' }}>Ceyssens et al. (2026)</strong>
            <p style={{ fontSize: '0.76rem', color: 'var(--text-muted)', marginTop: '5px', lineHeight: 1.45 }}>
              Running injury prediction using biomechanics and machine learning.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
