import React, { useState, useRef, useEffect, useCallback } from 'react';
import ThreeViewer from './ThreeViewer';
import './index.css';

const API_BASE = 'http://localhost:8000/api';

const PIPELINE_STAGES = [
  'Loading',
  'Preprocessing',
  'Depth inference',
  'Structure analysis',
  'Candidate generation',
  'Counterfactual testing',
  'Verification',
  'Self-disproof',
  'Uncertainty',
  'DSM',
  'Mesh',
  'Complete',
];

// ─── Inline SVG Height Fingerprint Chart ───────────────────────
function HeightFingerprintChart({ fingerprint, survivors }) {
  if (!fingerprint || Object.keys(fingerprint).length === 0) return null;

  const firstObjId = Object.keys(fingerprint)[0];
  const fp = fingerprint[firstObjId];
  if (!fp) return null;

  const heights = fp.heights || [];
  const scores  = fp.scores  || [];
  if (heights.length === 0) return null;

  const W = 280, H = 120;
  const PAD = { top: 10, right: 12, bottom: 28, left: 36 };
  const iW = W - PAD.left - PAD.right;
  const iH = H - PAD.top  - PAD.bottom;

  const minH = Math.min(...heights);
  const maxH = Math.max(...heights);
  const hRange = Math.max(maxH - minH, 1e-6);

  const toX = h => PAD.left + ((h - minH) / hRange) * iW;
  const toY = s => PAD.top  + (1 - s) * iH;

  const survivorEntry = survivors ? Object.values(survivors)[0] : null;
  const survivorH = survivorEntry?.height_value;
  const THRESH = 0.35;

  const points = heights.map((h, i) => `${toX(h)},${toY(scores[i])}`).join(' ');

  return (
    <div className="chart-container" style={{ height: H }}>
      <svg width="100%" height={H} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
        {/* Rejection threshold line */}
        <line
          x1={PAD.left} y1={toY(THRESH)}
          x2={W - PAD.right} y2={toY(THRESH)}
          stroke="#ffc107" strokeWidth="1" strokeDasharray="4,3" opacity="0.6"
        />
        <text x={W - PAD.right - 2} y={toY(THRESH) - 3}
          fill="#ffc107" fontSize="8" textAnchor="end" opacity="0.7">
          reject
        </text>

        {/* Score curve */}
        <polyline
          points={points}
          fill="none"
          stroke="rgba(78,205,196,0.4)"
          strokeWidth="1.5"
        />

        {/* Points */}
        {heights.map((h, i) => {
          const isSurvivor = survivorH !== undefined && Math.abs(h - survivorH) < 0.5;
          return (
            <circle key={i}
              cx={toX(h)} cy={toY(scores[i])}
              r={isSurvivor ? 5 : 3}
              fill={isSurvivor ? '#00e676' : scores[i] < THRESH ? '#ff5252' : '#4ecdc4'}
              stroke={isSurvivor ? '#fff' : 'none'}
              strokeWidth="1"
            />
          );
        })}

        {/* Survivor annotation */}
        {survivorH !== undefined && survivorEntry && (
          <>
            <line
              x1={toX(survivorH)} y1={toY(survivorEntry.overall_score)}
              x2={toX(survivorH)} y2={toY(survivorEntry.overall_score) - 16}
              stroke="#00e676" strokeWidth="1"
            />
            <text
              x={toX(survivorH)} y={toY(survivorEntry.overall_score) - 18}
              fill="#00e676" fontSize="9" textAnchor="middle" fontWeight="600"
            >
              SURVIVED
            </text>
          </>
        )}

        {/* X-axis label */}
        <text x={PAD.left + iW / 2} y={H - 4}
          fill="#5c6480" fontSize="9" textAnchor="middle">
          Height ({fp.height_unit || 'rel.'})
        </text>
        {/* Y-axis label (rotated) */}
        <text
          x={10} y={PAD.top + iH / 2}
          fill="#5c6480" fontSize="8" textAnchor="middle"
          transform={`rotate(-90 10 ${PAD.top + iH / 2})`}
        >
          Score
        </text>

        {/* X ticks */}
        {[minH, (minH + maxH) / 2, maxH].map((h, i) => (
          <text key={i}
            x={toX(h)} y={H - PAD.bottom + 12}
            fill="#5c6480" fontSize="8" textAnchor="middle"
          >
            {h.toFixed(1)}
          </text>
        ))}
      </svg>
    </div>
  );
}

// ─── AERIS Verification Table ───────────────────────────────────
function VerificationTable({ candidatesTable, survivors }) {
  if (!candidatesTable || candidatesTable.length === 0) return null;

  const survivorEntries = Object.entries(survivors || {});
  const survivorIds = new Set(
    Object.values(survivors || {}).map(s => s.candidate_id)
  );

  // Use the first survivor's object_id to drive the table (matches hero card)
  const heroObjId = survivorEntries.length > 0 ? survivorEntries[0][0] : null;
  const firstObjId = heroObjId || candidatesTable[0]?.object_id;

  // Sort: survivors first, then by score descending; show max 10 rows
  const rows = candidatesTable
    .filter(c => c.object_id === firstObjId)
    .sort((a, b) => {
      const aS = survivorIds.has(a.candidate_id) ? 1 : 0;
      const bS = survivorIds.has(b.candidate_id) ? 1 : 0;
      return bS - aS || b.overall_score - a.overall_score;
    })
    .slice(0, 10);

  return (
    <table className="verif-table">
      <thead>
        <tr>
          <th>Candidate</th>
          <th>Height</th>
          <th>Score</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(c => {
          const isSurvivor = survivorIds.has(c.candidate_id);
          return (
            <tr key={c.candidate_id} className={isSurvivor ? 'row-survived' : 'row-rejected'}>
              <td className="mono" style={{ fontSize: 10 }}>{c.candidate_id}</td>
              <td className="mono">{c.height_value.toFixed(1)}</td>
              <td className="mono">{c.overall_score.toFixed(4)}</td>
              <td>
                <span className={`badge ${isSurvivor ? 'badge-survived' : 'badge-rejected'}`}>
                  {isSurvivor ? '✓ Survived' : '✗ Rejected'}
                </span>
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

// ─── Evidence Component Bars ────────────────────────────────────
function EvidenceBars({ componentScores }) {
  if (!componentScores) return null;
  const entries = Object.entries(componentScores);
  if (entries.length === 0) return null;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      {entries.map(([key, val]) => (
        <div key={key} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ width: 120, fontSize: 10, color: 'var(--text-muted)', flexShrink: 0, fontFamily: 'var(--text-mono)' }}>
            {key.replace(/_/g, ' ')}
          </span>
          <div className="evidence-bar-track">
            <div className="evidence-bar-fill" style={{ width: `${Math.min(val * 100, 100)}%` }} />
          </div>
          <span style={{ width: 36, fontSize: 10, color: 'var(--text-dim)', fontFamily: 'var(--text-mono)', textAlign: 'right' }}>
            {val.toFixed(3)}
          </span>
        </div>
      ))}
    </div>
  );
}

// ─── Section header ─────────────────────────────────────────────
function SectionHeader({ title }) {
  return (
    <div className="section-header">
      <span className="section-title">{title}</span>
      <div className="section-divider" />
    </div>
  );
}

// ─── Stat card ──────────────────────────────────────────────────
function StatCard({ label, value, color }) {
  return (
    <div className="stat-card">
      <div className="stat-label">{label}</div>
      <div className="stat-value" style={{ color: color || 'var(--text-main)' }}>
        {value}
      </div>
    </div>
  );
}

// ─── Slope Assessment ───────────────────────────────────────────
function SlopeAssessment({ results }) {
  if (!results) return null;
  const { terrain, structure } = results;
  if (!terrain) return null;

  const roughness = terrain.terrain_roughness ?? 0;
  const roughnessPct = Math.round(Math.min(roughness * 200, 100));
  const slopeLabel = roughness < 0.05 ? 'Flat / Low Relief'
    : roughness < 0.15 ? 'Moderate Slope'
    : 'High Relief / Steep';

  return (
    <div style={{ fontSize: 12 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
        <span style={{ color: 'var(--text-dim)' }}>Terrain type</span>
        <span style={{ color: 'var(--accent)', fontWeight: 600, fontFamily: 'var(--text-mono)' }}>
          {slopeLabel}
        </span>
      </div>
      <div style={{ marginBottom: 8 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, marginBottom: 4 }}>
          <span style={{ color: 'var(--text-muted)' }}>Terrain roughness index</span>
          <span style={{ color: 'var(--text-dim)', fontFamily: 'var(--text-mono)' }}>
            {roughness.toFixed(4)}
          </span>
        </div>
        <div className="progress-track">
          <div className="progress-fill" style={{ width: `${roughnessPct}%`, background: 'linear-gradient(90deg, #4ecdc4, #f39c12)' }} />
        </div>
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
        <span style={{ color: 'var(--text-dim)' }}>Scale mode</span>
        <span className={`badge ${terrain.scale_mode === 'RELATIVE' ? 'badge-relative' : 'badge-metric'}`}>
          {terrain.scale_mode}
        </span>
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
        <span style={{ color: 'var(--text-dim)' }}>Regions / Buildings</span>
        <span style={{ color: 'var(--text-main)', fontFamily: 'var(--text-mono)' }}>
          {structure?.num_regions ?? 0} / {structure?.num_buildings ?? 0}
        </span>
      </div>
      <p style={{ color: 'var(--text-muted)', fontSize: 10, marginTop: 8, lineHeight: 1.5 }}>
        ⚠ Slope assessment is derived from the monocular depth model and is
        approximate. Aerial/nadir imagery may show domain-shift artifacts.
        Use as structural guidance only, not as survey-grade measurement.
      </p>
    </div>
  );
}

// ─── Idle Center State ──────────────────────────────────────────
function IdleCenterState({ isProcessing }) {
  return (
    <div style={{
      width: '100%', height: '100%',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      flexDirection: 'column', gap: 0,
      position: 'relative',
    }}>
      {/* Animated topographic globe */}
      <svg
        width="340" height="340" viewBox="0 0 340 340"
        style={{ opacity: 0.55, position: 'absolute' }}
      >
        <defs>
          <radialGradient id="glow-center" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#4ecdc4" stopOpacity="0.15" />
            <stop offset="100%" stopColor="#4ecdc4" stopOpacity="0" />
          </radialGradient>
          <filter id="blur-glow">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>

        {/* Background glow */}
        <circle cx="170" cy="170" r="140" fill="url(#glow-center)" />

        {/* Concentric topographic rings */}
        {[140, 120, 100, 82, 66, 52, 40, 30, 20, 12].map((r, i) => (
          <circle
            key={r}
            cx="170" cy="170" r={r}
            fill="none"
            stroke="#4ecdc4"
            strokeWidth={i === 0 ? 0.6 : 0.4}
            opacity={0.12 + i * 0.04}
          />
        ))}

        {/* Lat/lon grid lines */}
        {[-60, -30, 0, 30, 60].map(deg => {
          const y = 170 + (deg / 90) * 140;
          return (
            <line key={deg}
              x1="30" y1={y} x2="310" y2={y}
              stroke="#4ecdc4" strokeWidth="0.3" opacity="0.12"
            />
          );
        })}
        {[0, 45, 90, 135].map(angle => {
          const rad = angle * Math.PI / 180;
          return (
            <line key={angle}
              x1={170 + Math.cos(rad) * 140} y1={170 + Math.sin(rad) * 140}
              x2={170 - Math.cos(rad) * 140} y2={170 - Math.sin(rad) * 140}
              stroke="#4ecdc4" strokeWidth="0.3" opacity="0.12"
            />
          );
        })}

        {/* Bright outer ring */}
        <circle cx="170" cy="170" r="140"
          fill="none" stroke="#4ecdc4" strokeWidth="1"
          opacity="0.3"
          strokeDasharray="6 6"
          style={{ animation: 'spin 30s linear infinite' }}
          filter="url(#blur-glow)"
        />

        {/* Scanning beam */}
        <line
          x1="170" y1="170" x2="310" y2="170"
          stroke="url(#scan-grad)" strokeWidth="1.5"
          opacity="0.5"
          style={{ transformOrigin: '170px 170px', animation: 'spin 4s linear infinite' }}
        />
        <defs>
          <linearGradient id="scan-grad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#4ecdc4" stopOpacity="0" />
            <stop offset="100%" stopColor="#4ecdc4" stopOpacity="0.9" />
          </linearGradient>
        </defs>

        {/* Crosshair */}
        <circle cx="170" cy="170" r="4" fill="#4ecdc4" opacity="0.7" />
        <circle cx="170" cy="170" r="10" fill="none" stroke="#4ecdc4" strokeWidth="0.8" opacity="0.4" />

        {/* Corner brackets */}
        {[[30,30], [310,30], [30,310], [310,310]].map(([x, y], i) => {
          const dx = x < 170 ? 1 : -1;
          const dy = y < 170 ? 1 : -1;
          return (
            <g key={i} opacity="0.4">
              <line x1={x} y1={y} x2={x + dx*18} y2={y} stroke="#4ecdc4" strokeWidth="1.5" />
              <line x1={x} y1={y} x2={x} y2={y + dy*18} stroke="#4ecdc4" strokeWidth="1.5" />
            </g>
          );
        })}

        {/* Data points */}
        {[[220, 120], [130, 200], [260, 220], [100, 140], [190, 250]].map(([x, y], i) => (
          <circle key={i} cx={x} cy={y} r="2.5"
            fill="#4ecdc4" opacity={0.4 + i * 0.06}
          />
        ))}
      </svg>

      {/* Text overlay */}
      <div style={{
        position: 'relative', textAlign: 'center', zIndex: 1,
        marginTop: 200,
      }}>
        <div style={{
          fontSize: 10, letterSpacing: '0.3em', textTransform: 'uppercase',
          color: 'rgba(78,205,196,0.4)', fontFamily: 'var(--font-mono)',
          marginBottom: 8,
        }}>
          {isProcessing ? 'Pipeline running…' : 'Awaiting input'}
        </div>
        <div style={{
          fontSize: 11, letterSpacing: '0.18em', textTransform: 'uppercase',
          color: 'rgba(78,205,196,0.2)', fontFamily: 'var(--font-mono)',
        }}>
          AERIS-3D · SIH26175 · DepthWizard
        </div>
      </div>

      {/* Bottom coordinate HUD */}
      <div style={{
        position: 'absolute', bottom: 24, left: '50%', transform: 'translateX(-50%)',
        display: 'flex', gap: 20,
      }}>
        {['LAT 28.6139° N', 'LON 77.2090° E', 'ALT --- m', 'MODE RELATIVE'].map(label => (
          <span key={label} style={{
            fontSize: 9, color: 'rgba(78,205,196,0.2)', fontFamily: 'var(--font-mono)',
            letterSpacing: '0.12em',
          }}>{label}</span>
        ))}
      </div>
    </div>
  );
}

// ─── Main App ───────────────────────────────────────────────────
export default function App() {
  const [file, setFile]         = useState(null);
  const [jobId, setJobId]       = useState(null);
  const [status, setStatus]     = useState('');
  const [progress, setProgress] = useState(0);
  const [stage, setStage]       = useState('');
  const [results, setResults]   = useState(null);
  const [error, setError]       = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [activeTab, setActiveTab] = useState('verify');
  const [viewerKey, setViewerKey] = useState(0);

  const fileInputRef = useRef(null);
  const rightPanelRef = useRef(null);

  // Determine current stage index
  const stageIdx = PIPELINE_STAGES.findIndex(
    s => s.toLowerCase() === stage.toLowerCase()
  );

  // Poll job status
  useEffect(() => {
    if (!jobId || (status !== 'queued' && status !== 'running')) return;
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/job/${jobId}`);
        if (!res.ok) return;
        const data = await res.json();
        setStatus(data.status);
        setProgress(data.progress ?? 0);
        setStage(data.stage ?? '');
        if (data.status === 'done') {
          setResults(data.result);
          setViewerKey(k => k + 1);
        } else if (data.status === 'error') {
          setError(data.error ?? 'Unknown pipeline error');
        }
      } catch (e) {
        console.error('Poll error:', e);
      }
    }, 800);
    return () => clearInterval(interval);
  }, [jobId, status]);

  const handleFile = useCallback(f => {
    if (!f) return;
    setFile(f);
    setJobId(null);
    setStatus('');
    setResults(null);
    setError(null);
    setProgress(0);
    setStage('');
  }, []);

  const handleFileChange = e => {
    if (e.target.files?.[0]) handleFile(e.target.files[0]);
  };

  const handleDrop = e => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files?.[0];
    if (f) handleFile(f);
  };

  const handleUpload = async () => {
    if (!file) return;
    setStatus('uploading');
    setError(null);
    const formData = new FormData();
    formData.append('file', file);
    try {
      const res = await fetch(`${API_BASE}/process`, { method: 'POST', body: formData });
      if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
      const data = await res.json();
      setJobId(data.job_id);
      setStatus(data.status);
    } catch (err) {
      setError(err.message);
      setStatus('error');
    }
  };

  const isProcessing = status === 'queued' || status === 'running' || status === 'uploading';
  const isDone = status === 'done';

  // Best survivor (first object)
  const firstSurvivorObjId = results?.survivors
    ? Object.keys(results.survivors)[0]
    : null;
  const firstSurvivor = firstSurvivorObjId
    ? results.survivors[firstSurvivorObjId]
    : null;

  // Find uncertainty for best survivor
  const bestUnc = firstSurvivor && results?.uncertainty
    ? results.uncertainty.find(u => u.candidate_id === firstSurvivor.candidate_id)
    : null;

  const glbUrl = isDone && jobId ? `${API_BASE}/file/${jobId}/mesh.glb` : null;

  const tabs = [
    { id: 'verify',   label: 'AERIS Verify' },
    { id: 'evidence', label: 'Evidence' },
    { id: 'slope',    label: 'Slope' },
    { id: 'export',   label: 'Export' },
  ];

  return (
    <div style={{ width: '100vw', height: '100vh', position: 'relative', background: 'var(--bg-void)' }}>

      {/* ── Background layers ── */}
      <div className="blob-br" />
      <div className="grid-overlay" />

      {/* ── 3D Viewer: fullscreen background ── */}
      <div style={{ position: 'absolute', inset: 0, zIndex: 1 }}>
        {glbUrl ? (
          <ThreeViewer key={viewerKey} glbUrl={glbUrl} />
        ) : (
          <IdleCenterState isProcessing={isProcessing} />
        )}
      </div>

      {/* ── HUD overlay ── */}
      <div style={{
        position: 'absolute', inset: 0, zIndex: 2,
        display: 'flex', padding: 16, gap: 14,
        pointerEvents: 'none',
      }}>

        {/* Top-right status watermark */}
        <div style={{
          position: 'absolute', top: 16, right: 16, zIndex: 10,
          display: 'flex', alignItems: 'center', gap: 8,
          pointerEvents: 'none',
          fontFamily: 'var(--font-mono)', fontSize: 10,
          color: 'rgba(78,205,196,0.6)',
          background: 'rgba(10,25,30,0.5)',
          backdropFilter: 'blur(8px)',
          border: '1px solid rgba(78,205,196,0.15)',
          borderRadius: 4, padding: '4px 10px',
        }}>
          <div style={{
            width: 6, height: 6, borderRadius: '50%',
            background: isProcessing ? '#f39c12' : isDone ? '#00e676' : '#4ecdc4',
            boxShadow: '0 0 6px currentColor',
          }} />
          <span style={{ letterSpacing: '0.1em' }}>
            {isProcessing ? `STATUS: ${status.toUpperCase()}` : isDone ? 'STATUS: RECONSTRUCTED' : 'STATUS: STANDBY'}
          </span>
        </div>

        {/* ══ LEFT PANEL ══════════════════════════════════════ */}
        <div className="glass-panel animate-fade-in" style={{
          width: 280, display: 'flex', flexDirection: 'column',
          pointerEvents: 'auto', overflow: 'hidden',
        }}>
          {/* Header */}
          <div style={{
            padding: '18px 20px 16px',
            borderBottom: '1px solid rgba(78,205,196,0.1)',
            background: 'linear-gradient(160deg, rgba(78,205,196,0.08) 0%, rgba(124,111,247,0.04) 100%)',
          }}>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
              <h1 style={{ fontSize: 20, fontWeight: 800, color: 'var(--accent)', letterSpacing: '-0.02em' }}>
                AERIS-3D
              </h1>
              <span style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--text-mono)' }}>v0.2</span>
            </div>
            <p style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 3, lineHeight: 1.4 }}>
              Hypothesis-Testing 3D Reconstruction · SIH26175
            </p>
          </div>

          <div style={{ padding: '16px 20px', display: 'flex', flexDirection: 'column', gap: 14, flex: 1, overflowY: 'auto' }}>

            {/* Upload zone */}
            <div>
              <SectionHeader title="Input" />
              <input
                type="file"
                accept=".jpg,.jpeg,.png,.tif,.tiff"
                style={{ display: 'none' }}
                ref={fileInputRef}
                onChange={handleFileChange}
              />
              <div
                className={`drop-zone ${dragOver ? 'drag-over' : ''}`}
                onClick={() => !isProcessing && fileInputRef.current?.click()}
                onDrop={handleDrop}
                onDragOver={e => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                style={{ padding: 16, cursor: isProcessing ? 'default' : 'pointer' }}
              >
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="1.5" style={{ marginBottom: 8, opacity: 0.7 }}>
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                  <polyline points="17 8 12 3 7 8"/>
                  <line x1="12" y1="3" x2="12" y2="15"/>
                </svg>
                <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                  Drop JPG / PNG / GeoTIFF<br />
                  <span style={{ color: 'var(--accent)', fontSize: 10 }}>or click to browse</span>
                </div>
              </div>
              {file && (
                <div className="filename-chip" style={{ marginTop: 8 }}>
                  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="2">
                    <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/>
                  </svg>
                  {file.name}
                </div>
              )}
              <button
                className={`btn ${file && !isProcessing ? 'btn-primary' : 'btn'}`}
                style={{ width: '100%', marginTop: 10 }}
                onClick={handleUpload}
                disabled={!file || isProcessing}
                id="btn-process"
              >
                {isProcessing ? (
                  <>
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ animation: 'spin 1s linear infinite' }}>
                      <path d="M21 12a9 9 0 1 1-6.219-8.56"/>
                    </svg>
                    Processing…
                  </>
                ) : '▶ Run AERIS Pipeline'}
              </button>
              <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
            </div>

            {/* Pipeline stages */}
            {(isProcessing || isDone) && (
              <div>
                <SectionHeader title="Pipeline" />
                <div>
                  {PIPELINE_STAGES.map((s, i) => {
                    const isActive = i === stageIdx;
                    const isDoneStage = isDone || i < stageIdx;
                    return (
                      <div key={s} className={`stage-item ${isActive ? 'active' : isDoneStage ? 'done' : ''}`}>
                        <div className={`stage-dot ${isActive ? 'active' : isDoneStage ? 'done' : ''}`} />
                        {s}
                        {isDoneStage && (
                          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="var(--success)" strokeWidth="3" style={{ marginLeft: 'auto' }}>
                            <polyline points="20 6 9 17 4 12" />
                          </svg>
                        )}
                      </div>
                    );
                  })}
                </div>
                <div style={{ marginTop: 12 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, marginBottom: 6 }}>
                    <span style={{ color: 'var(--text-muted)' }}>{stage || 'Initializing…'}</span>
                    <span style={{ color: 'var(--accent)', fontFamily: 'var(--text-mono)' }}>{progress}%</span>
                  </div>
                  <div className="progress-track">
                    <div className="progress-fill" style={{ width: `${progress}%` }} />
                  </div>
                </div>
              </div>
            )}

            {/* Error */}
            {error && (
              <div className="alert alert-warning" style={{ background: 'var(--danger-dim)', borderColor: 'var(--danger-border)', color: 'var(--danger)' }}>
                <strong>Error:</strong> {error}
              </div>
            )}

            {/* Input image preview */}
            {isDone && jobId && (
              <div>
                <SectionHeader title="Input Preview" />
                <div className="thumb" style={{ aspectRatio: 'auto', height: 100, borderRadius: 8 }}>
                  <img
                    src={`${API_BASE}/file/${jobId}/input_preview.png`}
                    alt="Input"
                    style={{ width: '100%', height: '100%', objectFit: 'cover', borderRadius: 8 }}
                  />
                </div>
              </div>
            )}

            {/* Quick stats when done */}
            {results && (
              <div>
                <SectionHeader title="Pipeline Summary" />
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                  <StatCard label="Buildings" value={results.structure?.num_buildings ?? 0} />
                  <StatCard label="Candidates" value={results.candidates?.total ?? 0} />
                  <StatCard label="Survived" value={results.candidates?.survived ?? 0} color="var(--success)" />
                  <StatCard label="Rejected" value={results.candidates?.rejected ?? 0} color="var(--danger)" />
                </div>
                {results.total_runtime_s && (
                  <div style={{ marginTop: 8, fontSize: 11, color: 'var(--text-muted)', textAlign: 'right', fontFamily: 'var(--text-mono)' }}>
                    Total: {results.total_runtime_s}s
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* ══ CENTER: spacer (3D viewer shows through) ══════════ */}
        <div style={{ flex: 1 }} />

        {/* ══ RIGHT PANEL ═════════════════════════════════════ */}
        {results && (
          <div
            ref={rightPanelRef}
            className="glass-panel animate-fade-in"
            style={{
              width: 380, display: 'flex', flexDirection: 'column',
              pointerEvents: 'auto', overflow: 'hidden',
            }}
          >
            {/* Scale mode banner */}
            {results.terrain?.scale_mode === 'RELATIVE' && (
              <div className="alert alert-warning" style={{ borderRadius: 0, marginBottom: 0, borderLeft: 'none', borderRight: 'none', borderTop: 'none' }}>
                ⚠ Heights are in RELATIVE scale — no metric anchor available for this image.
              </div>
            )}

            {/* Best survivor hero card */}
            {firstSurvivor && (
              <div style={{
                padding: '16px 20px',
                background: 'linear-gradient(135deg, rgba(0,230,118,0.06), transparent)',
                borderBottom: '1px solid var(--border)',
              }}>
                <div style={{ fontSize: 10, color: 'var(--success)', fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 4 }}>
                  ✓ Best-Supported Height <span style={{ opacity: 0.6 }}>[{firstSurvivorObjId}]</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
                  <span style={{ fontSize: 28, fontWeight: 800, color: 'var(--success)', fontFamily: 'var(--text-mono)', letterSpacing: '-0.02em' }}>
                    {firstSurvivor.height_value.toFixed(1)}
                  </span>
                  <span style={{ fontSize: 13, color: 'var(--text-dim)', fontFamily: 'var(--text-mono)' }}>
                    {(firstSurvivor.height_unit || 'REL').substring(0, 8)}
                  </span>
                  {bestUnc && (
                    <span style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--text-mono)', marginLeft: 4 }}>
                      ±{((bestUnc.p95 - bestUnc.p05) / 2).toFixed(3)}
                    </span>
                  )}
                </div>
                <div style={{ display: 'flex', gap: 8, marginTop: 6, flexWrap: 'wrap' }}>
                  <span style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--text-mono)' }}>
                    EGSS {firstSurvivor.overall_score.toFixed(4)}
                  </span>
                  {bestUnc && (
                    <span className={`badge ${
                      bestUnc.confidence_label === 'HIGH' ? 'badge-survived'
                      : bestUnc.confidence_label === 'MEDIUM' ? 'badge-relative'
                      : 'badge-rejected'}`}
                    >
                      {bestUnc.confidence_label} confidence
                    </span>
                  )}
                </div>
                <div style={{ fontSize: 9, color: 'var(--text-muted)', marginTop: 6, fontStyle: 'italic' }}>
                  ⚠ Relative estimate — not ground truth unless metric-anchored
                </div>
              </div>
            )}

            {/* Tabs */}
            <div style={{
              display: 'flex',
              borderBottom: '1px solid var(--border)',
              background: 'rgba(0,0,0,0.2)',
            }}>
              {tabs.map(t => (
                <button
                  key={t.id}
                  onClick={() => setActiveTab(t.id)}
                  style={{
                    flex: 1, padding: '9px 0', border: 'none',
                    background: activeTab === t.id ? 'var(--accent-dim)' : 'transparent',
                    color: activeTab === t.id ? 'var(--accent)' : 'var(--text-muted)',
                    fontSize: 10, fontWeight: 700, letterSpacing: '0.06em',
                    textTransform: 'uppercase', cursor: 'pointer', fontFamily: 'inherit',
                    borderBottom: activeTab === t.id ? '2px solid var(--accent)' : '2px solid transparent',
                    transition: 'all 0.15s',
                  }}
                >
                  {t.label}
                </button>
              ))}
            </div>

            {/* Tab body */}
            <div style={{ flex: 1, overflowY: 'auto', padding: '16px 20px' }}>

              {/* ── AERIS VERIFY tab ─────────────────────────── */}
              {activeTab === 'verify' && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                  <div>
                    <SectionHeader title="AERIS Verification" />
                    <VerificationTable
                      candidatesTable={results.candidates_table}
                      survivors={results.survivors}
                    />
                  </div>

                  <div>
                    <SectionHeader title="Height Fingerprint" />
                    <HeightFingerprintChart
                      fingerprint={results.height_fingerprint}
                      survivors={results.survivors}
                    />
                    <p style={{ fontSize: 9, color: 'var(--text-muted)', marginTop: 4, lineHeight: 1.4 }}>
                      Height vs. EGSS consistency score. Green = survived, red = rejected. Dashed line = rejection threshold (0.35).
                    </p>
                  </div>

                  {/* SDRL log preview */}
                  {results.sdrl_log && results.sdrl_log.length > 0 && (
                    <div>
                      <SectionHeader title="SDRL Iterations" />
                      <div style={{ maxHeight: 100, overflowY: 'auto', background: 'var(--bg-card)', borderRadius: 'var(--radius-sm)', padding: '8px 10px' }}>
                        {results.sdrl_log.slice(0, 6).map((row, i) => (
                          <div key={i} style={{ fontSize: 10, fontFamily: 'var(--text-mono)', color: 'var(--text-dim)', padding: '2px 0', borderBottom: i < 5 ? '1px solid rgba(255,255,255,0.04)' : 'none' }}>
                            iter {row.iteration} · {row.candidate_id} · score={typeof row.score === 'number' ? row.score.toFixed(4) : row.score} ·{' '}
                            <span style={{ color: row.status === 'SURVIVED' ? 'var(--success)' : 'var(--danger)' }}>
                              {row.status}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Depth thumbnails */}
                  <div>
                    <SectionHeader title="Depth Views" />
                    <div className="thumb-grid">
                      {[
                        { key: 'depth.png',           label: 'Initial Depth' },
                        { key: 'depth_corrected.png', label: 'HCDC Corrected' },
                        { key: 'structures.png',       label: 'Structures' },
                        { key: 'dsm.png',              label: 'DSM/rDSM' },
                      ].map(({ key, label }) => (
                        <div key={key}>
                          <a href={`${API_BASE}/file/${jobId}/${key}`} target="_blank" rel="noreferrer">
                            <div className="thumb">
                              <img src={`${API_BASE}/file/${jobId}/${key}`} alt={label} />
                            </div>
                          </a>
                          <div className="thumb-label">{label}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* ── EVIDENCE tab ─────────────────────────────── */}
              {activeTab === 'evidence' && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                  {Object.entries(results.survivors || {}).slice(0, 3).map(([objId, surv]) => {
                    const unc = results.uncertainty?.find(u => u.candidate_id === surv.candidate_id);
                    return (
                      <div key={objId}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                          <div style={{ fontFamily: 'var(--text-mono)', fontSize: 12, color: 'var(--text-main)', fontWeight: 600 }}>
                            {objId}
                          </div>
                          <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                            <span style={{ fontFamily: 'var(--text-mono)', fontSize: 11, color: 'var(--success)' }}>
                              {surv.height_value.toFixed(1)} {(surv.height_unit || '').substring(0, 4)}
                            </span>
                            {unc && (
                              <span className={`badge ${
                                unc.confidence_label === 'HIGH' ? 'badge-survived'
                                : unc.confidence_label === 'MEDIUM' ? 'badge-relative'
                                : 'badge-rejected'
                              }`}>{unc.confidence_label}</span>
                            )}
                          </div>
                        </div>

                        <EvidenceBars componentScores={surv.component_scores} />

                        {unc && (
                          <div style={{ marginTop: 10, padding: '8px 10px', background: 'var(--bg-card)', borderRadius: 'var(--radius-sm)', fontSize: 10, fontFamily: 'var(--text-mono)' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-dim)', marginBottom: 3 }}>
                              <span>CI [p05–p95]</span>
                              <span>{unc.p05.toFixed(4)} – {unc.p95.toFixed(4)}</span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-dim)' }}>
                              <span>Stability std</span>
                              <span>{unc.std_score.toFixed(4)}</span>
                            </div>
                          </div>
                        )}

                        {surv.active_evidence && (
                          <div style={{ marginTop: 8, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                            {surv.active_evidence.map(e => (
                              <span key={e} style={{
                                fontSize: 9, padding: '2px 6px', background: 'var(--accent-dim)',
                                border: '1px solid var(--accent-border)', borderRadius: 10,
                                color: 'var(--accent)', fontFamily: 'var(--text-mono)',
                              }}>{e}</span>
                            ))}
                          </div>
                        )}

                        <div style={{ height: 1, background: 'var(--border)', margin: '12px 0' }} />
                      </div>
                    );
                  })}

                  {/* Shadow / occlusion info */}
                  <div>
                    <SectionHeader title="Evidence Quality" />
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 6, fontSize: 11 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ color: 'var(--text-dim)' }}>Shadow available</span>
                        <span style={{ color: results.shadow?.available ? 'var(--success)' : 'var(--danger)', fontFamily: 'var(--text-mono)' }}>
                          {String(results.shadow?.available)}
                        </span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ color: 'var(--text-dim)' }}>Shadow reliable</span>
                        <span style={{ color: results.shadow?.reliable ? 'var(--success)' : 'var(--warning)', fontFamily: 'var(--text-mono)' }}>
                          {String(results.shadow?.reliable)}
                        </span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ color: 'var(--text-dim)' }}>Shadow weight</span>
                        <span style={{ color: 'var(--text-main)', fontFamily: 'var(--text-mono)' }}>
                          {results.shadow?.effective_weight?.toFixed(3)}
                        </span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ color: 'var(--text-dim)' }}>Depth model</span>
                        <span style={{ color: 'var(--text-main)', fontFamily: 'var(--text-mono)', fontSize: 10 }}>
                          {results.depth?.model}
                        </span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ color: 'var(--text-dim)' }}>Inference device</span>
                        <span style={{ color: 'var(--accent)', fontFamily: 'var(--text-mono)' }}>
                          {results.depth?.device}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* ── SLOPE tab ─────────────────────────────────── */}
              {activeTab === 'slope' && (
                <div>
                  <SectionHeader title="Slope &amp; Terrain Assessment" />
                  <SlopeAssessment results={results} />

                  {/* Phase runtimes */}
                  {results.phase_runtimes_s && (
                    <div style={{ marginTop: 16 }}>
                      <SectionHeader title="Phase Runtimes" />
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                        {Object.entries(results.phase_runtimes_s).map(([phase, t]) => (
                          <div key={phase} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10 }}>
                            <span style={{ color: 'var(--text-muted)', fontFamily: 'var(--text-mono)' }}>
                              {phase.replace('phase', 'Ph').replace('_', ' ')}
                            </span>
                            <span style={{ color: 'var(--text-dim)', fontFamily: 'var(--text-mono)' }}>
                              {t.toFixed(3)}s
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* ── EXPORT tab ───────────────────────────────── */}
              {activeTab === 'export' && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                  <div>
                    <SectionHeader title="Downloadable Assets" />
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                      {(results.output_files || []).map(fname => {
                        const icon = fname.endsWith('.glb') ? '🧊'
                          : fname.endsWith('.npz') ? '📦'
                          : fname.endsWith('.json') ? '📋'
                          : fname.endsWith('.tif') || fname.endsWith('.tiff') ? '🗺️'
                          : '🖼️';
                        return (
                          <a
                            key={fname}
                            href={`${API_BASE}/file/${jobId}/${fname}`}
                            target="_blank"
                            rel="noreferrer"
                            style={{
                              display: 'flex', alignItems: 'center', gap: 10,
                              padding: '8px 12px',
                              background: 'var(--bg-card)', borderRadius: 'var(--radius-sm)',
                              border: '1px solid var(--border)',
                              color: 'var(--text-dim)', textDecoration: 'none',
                              fontSize: 11, fontFamily: 'var(--text-mono)',
                              transition: 'all 0.15s',
                            }}
                            onMouseOver={e => e.currentTarget.style.borderColor = 'var(--accent-border)'}
                            onMouseOut={e => e.currentTarget.style.borderColor = 'var(--border)'}
                          >
                            <span>{icon}</span>
                            <span style={{ flex: 1 }}>{fname}</span>
                            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" opacity="0.5">
                              <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>
                              <polyline points="15 3 21 3 21 9"/>
                              <line x1="10" y1="14" x2="21" y2="3"/>
                            </svg>
                          </a>
                        );
                      })}
                    </div>
                  </div>

                  <div className="alert alert-info" style={{ fontSize: 10 }}>
                    <strong>DSM scale:</strong> {results.dsm?.scale_mode || 'RELATIVE'}<br />
                    {results.dsm?.scale_mode === 'RELATIVE'
                      ? 'No metric anchor available. Values are in relative depth units.'
                      : 'GeoTIFF is georeferenced. Units are metric elevation.'}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
