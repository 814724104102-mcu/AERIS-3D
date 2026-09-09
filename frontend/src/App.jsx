import React, { useState, useRef, useEffect } from 'react';
import ThreeViewer from './ThreeViewer';
import './index.css';

const API_BASE = 'http://localhost:8000/api';

function App() {
  const [file, setFile] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState('');
  const [progress, setProgress] = useState(0);
  const [stage, setStage] = useState('');
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  
  const fileInputRef = useRef(null);

  // Poll job status
  useEffect(() => {
    let interval;
    if (jobId && (status === 'queued' || status === 'running')) {
      interval = setInterval(async () => {
        try {
          const res = await fetch(`${API_BASE}/job/${jobId}`);
          if (res.ok) {
            const data = await res.json();
            setStatus(data.status);
            setProgress(data.progress);
            setStage(data.stage);
            if (data.status === 'done') {
              setResults(data.result);
            } else if (data.status === 'error') {
              setError(data.error);
            }
          }
        } catch (err) {
          console.error("Polling error:", err);
        }
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [jobId, status]);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      // Reset state for new upload
      setJobId(null);
      setStatus('');
      setResults(null);
      setError(null);
      setProgress(0);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    
    setStatus('uploading');
    setError(null);
    
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_BASE}/process`, {
        method: 'POST',
        body: formData,
      });
      
      if (!res.ok) throw new Error('Upload failed');
      
      const data = await res.json();
      setJobId(data.job_id);
      setStatus(data.status);
    } catch (err) {
      setError(err.message);
      setStatus('error');
    }
  };

  return (
    <div style={{ width: '100vw', height: '100vh', display: 'flex', flexDirection: 'column' }}>
      
      {/* Background 3D Viewer */}
      {results && results.mesh && (
        <ThreeViewer glbUrl={`${API_BASE}/file/${jobId}/mesh.glb`} />
      )}

      {/* Main HUD overlay */}
      <div style={{ 
        position: 'absolute', 
        top: 0, left: 0, right: 0, bottom: 0, 
        pointerEvents: 'none', 
        display: 'flex', 
        padding: '24px',
        gap: '24px'
      }}>
        
        {/* Left Panel: Upload & Status */}
        <div className="glass-panel animate-fade-in" style={{ 
          width: '350px', 
          display: 'flex', 
          flexDirection: 'column',
          pointerEvents: 'auto',
          padding: '24px'
        }}>
          <h1 style={{ fontSize: '24px', fontWeight: 800, marginBottom: '8px', color: 'var(--accent)' }}>
            AERIS-3D
          </h1>
          <p style={{ fontSize: '13px', color: 'var(--text-muted)', marginBottom: '24px' }}>
            Hypothesis-Testing 3D Reconstruction from Monocular Imagery
          </p>

          <div style={{ marginBottom: '24px' }}>
            <input 
              type="file" 
              accept=".jpg,.jpeg,.png,.tif,.tiff" 
              style={{ display: 'none' }}
              ref={fileInputRef}
              onChange={handleFileChange}
            />
            <button 
              className="btn" 
              style={{ width: '100%', marginBottom: '12px' }}
              onClick={() => fileInputRef.current.click()}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="17 8 12 3 7 8"></polyline>
                <line x1="12" y1="3" x2="12" y2="15"></line>
              </svg>
              Select Imagery
            </button>
            {file && (
              <div style={{ fontSize: '12px', textAlign: 'center', marginBottom: '12px', wordBreak: 'break-all' }}>
                {file.name}
              </div>
            )}
            
            <button 
              className="btn" 
              style={{ 
                width: '100%', 
                background: file && !jobId ? 'var(--accent)' : 'transparent',
                color: file && !jobId ? 'var(--bg-dark)' : 'var(--accent)',
              }}
              onClick={handleUpload}
              disabled={!file || (status !== '' && status !== 'error')}
            >
              Process Pipeline
            </button>
          </div>

          {/* Progress / Status display */}
          {(status === 'queued' || status === 'running' || status === 'uploading') && (
            <div style={{ marginTop: 'auto' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '8px' }}>
                <span style={{ color: 'var(--accent)' }}>{stage || 'Initializing...'}</span>
                <span>{progress}%</span>
              </div>
              <div style={{ height: '4px', background: 'rgba(255,255,255,0.1)', borderRadius: '2px', overflow: 'hidden' }}>
                <div style={{ 
                  height: '100%', 
                  width: `${progress}%`, 
                  background: 'var(--accent)',
                  transition: 'width 0.3s ease'
                }} />
              </div>
            </div>
          )}

          {error && (
            <div style={{ marginTop: '24px', padding: '12px', background: 'rgba(231, 76, 60, 0.1)', border: '1px solid var(--danger)', borderRadius: '6px', color: 'var(--danger)', fontSize: '13px' }}>
              <strong>Error:</strong> {error}
            </div>
          )}
        </div>

        {/* Right Panel: Results Data */}
        {results && (
          <div className="glass-panel animate-fade-in" style={{ 
            width: '400px', 
            marginLeft: 'auto',
            pointerEvents: 'auto',
            display: 'flex',
            flexDirection: 'column',
            maxHeight: '100%',
          }}>
            <div style={{ padding: '20px', borderBottom: '1px solid var(--border)' }}>
              <h2 style={{ fontSize: '16px', fontWeight: 600, display: 'flex', justifyContent: 'space-between' }}>
                Pipeline Results
                <span style={{ fontSize: '12px', color: 'var(--text-muted)', fontWeight: 400 }}>
                  {results.total_runtime_s}s
                </span>
              </h2>
            </div>
            
            <div style={{ padding: '20px', overflowY: 'auto', flex: 1, fontSize: '13px' }}>
              {/* Alert: Relative mode */}
              {results.terrain?.scale_mode === 'RELATIVE' && (
                <div style={{ padding: '12px', background: 'rgba(243, 156, 18, 0.1)', border: '1px solid #f39c12', borderRadius: '6px', color: '#f39c12', marginBottom: '16px' }}>
                  ⚠ Heights and mesh are in RELATIVE scale. No metric anchor available.
                </div>
              )}

              {/* Stats Grid */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '24px' }}>
                <div style={{ background: 'rgba(0,0,0,0.2)', padding: '12px', borderRadius: '6px' }}>
                  <div style={{ color: 'var(--text-muted)', fontSize: '11px', marginBottom: '4px' }}>Buildings Found</div>
                  <div style={{ fontSize: '18px', fontWeight: 600 }}>{results.structure?.num_buildings || 0}</div>
                </div>
                <div style={{ background: 'rgba(0,0,0,0.2)', padding: '12px', borderRadius: '6px' }}>
                  <div style={{ color: 'var(--text-muted)', fontSize: '11px', marginBottom: '4px' }}>Candidates Swept</div>
                  <div style={{ fontSize: '18px', fontWeight: 600 }}>{results.candidates?.total || 0}</div>
                </div>
                <div style={{ background: 'rgba(0,0,0,0.2)', padding: '12px', borderRadius: '6px' }}>
                  <div style={{ color: 'var(--success)', fontSize: '11px', marginBottom: '4px' }}>Candidates Survived</div>
                  <div style={{ fontSize: '18px', fontWeight: 600, color: 'var(--success)' }}>{results.candidates?.survived || 0}</div>
                </div>
                <div style={{ background: 'rgba(0,0,0,0.2)', padding: '12px', borderRadius: '6px' }}>
                  <div style={{ color: 'var(--danger)', fontSize: '11px', marginBottom: '4px' }}>SDRL Rejections</div>
                  <div style={{ fontSize: '18px', fontWeight: 600, color: 'var(--danger)' }}>{results.candidates?.rejected || 0}</div>
                </div>
              </div>

              {/* Survivors List */}
              <h3 style={{ fontSize: '14px', marginBottom: '12px', color: 'var(--accent)' }}>Verified Heights (SDRL Survivors)</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '24px' }}>
                {Object.entries(results.survivors || {}).map(([objId, survivor]) => {
                  // Find uncertainty info for this candidate
                  const unc = (results.uncertainty || []).find(u => u.candidate_id === survivor.candidate_id);
                  return (
                    <div key={objId} style={{ background: 'rgba(0,0,0,0.2)', padding: '12px', borderRadius: '6px', borderLeft: '3px solid var(--success)' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                        <strong style={{ fontFamily: 'monospace' }}>{objId}</strong>
                        <span style={{ color: 'var(--success)', fontWeight: 'bold' }}>
                          {survivor.height_value} {survivor.height_unit.substring(0,3)}
                        </span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: 'var(--text-muted)' }}>
                        <span>EGSS Score: {survivor.overall_score}</span>
                        {unc && (
                          <span style={{ 
                            color: unc.confidence_label === 'HIGH' ? 'var(--success)' : (unc.confidence_label === 'MEDIUM' ? '#f39c12' : 'var(--danger)')
                          }}>
                            {unc.confidence_label} CONFIDENCE
                          </span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Downloads */}
              <h3 style={{ fontSize: '14px', marginBottom: '12px', color: 'var(--accent)' }}>Export Assets</h3>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                {results.output_files?.map(fname => (
                  <a 
                    key={fname}
                    href={`${API_BASE}/file/${jobId}/${fname}`} 
                    target="_blank" 
                    rel="noreferrer"
                    style={{
                      display: 'block', padding: '8px', background: 'rgba(255,255,255,0.05)', 
                      borderRadius: '4px', color: 'var(--text-main)', textDecoration: 'none',
                      fontSize: '11px', textAlign: 'center', border: '1px solid var(--border)',
                      transition: 'background 0.2s'
                    }}
                    onMouseOver={(e) => e.target.style.background = 'rgba(255,255,255,0.1)'}
                    onMouseOut={(e) => e.target.style.background = 'rgba(255,255,255,0.05)'}
                  >
                    {fname}
                  </a>
                ))}
              </div>

            </div>
          </div>
        )}

      </div>
    </div>
  );
}

export default App;
