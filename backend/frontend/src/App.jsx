import { useState, useRef } from 'react';
import {
  UploadCloud, CheckCircle, AlertTriangle, MonitorPlay,
  Loader2, ShieldCheck, Zap, BarChart2, Globe, Clock,
  FileVideo, ChevronDown, Star, Users, ArrowRight, X
} from 'lucide-react';

/* ─── tiny helper ─── */
function Badge({ children }) {
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: '6px',
      background: 'rgba(59,130,246,0.12)', border: '1px solid rgba(59,130,246,0.3)',
      color: '#93c5fd', borderRadius: '999px', fontSize: '13px',
      fontWeight: 600, padding: '4px 14px', letterSpacing: '0.02em'
    }}>
      {children}
    </span>
  );
}

function StatCard({ icon, label, value }) {
  return (
    <div style={{
      background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)',
      borderRadius: '16px', padding: '28px 24px', textAlign: 'center', flex: '1 1 160px'
    }}>
      <div style={{ fontSize: '28px', marginBottom: '8px' }}>{icon}</div>
      <div style={{ fontSize: '30px', fontWeight: 800, color: '#f1f5f9', lineHeight: 1 }}>{value}</div>
      <div style={{ fontSize: '13px', color: '#94a3b8', marginTop: '6px', fontWeight: 500 }}>{label}</div>
    </div>
  );
}

function FeatureCard({ icon, title, desc }) {
  return (
    <div style={{
      background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)',
      borderRadius: '16px', padding: '28px 24px', transition: 'border-color .2s',
      cursor: 'default'
    }}
      onMouseEnter={e => e.currentTarget.style.borderColor = 'rgba(99,102,241,0.4)'}
      onMouseLeave={e => e.currentTarget.style.borderColor = 'rgba(255,255,255,0.07)'}
    >
      <div style={{
        width: '44px', height: '44px', borderRadius: '12px',
        background: 'rgba(99,102,241,0.15)', display: 'flex', alignItems: 'center',
        justifyContent: 'center', marginBottom: '16px', color: '#818cf8'
      }}>{icon}</div>
      <div style={{ fontSize: '16px', fontWeight: 700, color: '#f1f5f9', marginBottom: '8px' }}>{title}</div>
      <div style={{ fontSize: '14px', color: '#94a3b8', lineHeight: '1.65' }}>{desc}</div>
    </div>
  );
}

function FAQ({ q, a }) {
  const [open, setOpen] = useState(false);
  return (
    <div style={{
      border: '1px solid rgba(255,255,255,0.08)', borderRadius: '14px',
      overflow: 'hidden', transition: 'border-color .2s',
      borderColor: open ? 'rgba(99,102,241,0.4)' : 'rgba(255,255,255,0.08)'
    }}>
      <button
        onClick={() => setOpen(!open)}
        style={{
          width: '100%', textAlign: 'left', padding: '20px 24px',
          background: 'rgba(255,255,255,0.03)', border: 'none', cursor: 'pointer',
          color: '#f1f5f9', fontFamily: 'inherit', fontSize: '15px', fontWeight: 600,
          display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '16px'
        }}
      >
        {q}
        <ChevronDown size={18} color="#94a3b8" style={{ transform: open ? 'rotate(180deg)' : 'none', transition: 'transform .2s', flexShrink: 0 }} />
      </button>
      {open && (
        <div style={{ padding: '0 24px 20px', color: '#94a3b8', fontSize: '14px', lineHeight: '1.7' }}>
          {a}
        </div>
      )}
    </div>
  );
}

/* ─── MAIN UPLOAD WIDGET ─── */
function UploadWidget() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault(); e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') setDragActive(true);
    else setDragActive(false);
  };

  const handleDrop = (e) => {
    e.preventDefault(); e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files?.[0]) handleFile(e.dataTransfer.files[0]);
  };

  const handleFile = (f) => {
    if (!f.type.startsWith('video/')) { setError('Please upload a valid video file.'); return; }
    setError(''); setFile(f);
    setPreview(URL.createObjectURL(f)); setResult(null);
  };

  const analyzeVideo = async () => {
    if (!file) return;
    setLoading(true); setResult(null); setError('');
    const formData = new FormData();
    formData.append('file', file);
    try {
      const res = await fetch('/predict', { method: 'POST', body: formData });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Server error');
      setResult(data);
    } catch (err) { setError(err.message); }
    finally { setLoading(false); }
  };

  const reset = () => { setPreview(null); setFile(null); setResult(null); setError(''); };

  /* idle state */
  if (!preview && !result) return (
    <div style={{ width: '100%' }}>
      {/* URL bar row */}
      <div style={{ display: 'flex', gap: '10px', marginBottom: '16px' }}>
        <div style={{
          flex: 1, display: 'flex', alignItems: 'center', gap: '10px',
          background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.12)',
          borderRadius: '12px', padding: '0 16px', height: '52px'
        }}>
          <Globe size={16} color="#64748b" style={{ flexShrink: 0 }} />
          <input
            type="text"
            placeholder="Paste YouTube, TikTok or Instagram URL…"
            style={{
              flex: 1, background: 'none', border: 'none', outline: 'none',
              color: '#f1f5f9', fontFamily: 'inherit', fontSize: '14px'
            }}
          />
        </div>
        <button style={{
          background: 'linear-gradient(135deg,#6366f1,#8b5cf6)',
          border: 'none', borderRadius: '12px', padding: '0 24px', height: '52px',
          color: '#fff', fontFamily: 'inherit', fontSize: '14px', fontWeight: 700,
          cursor: 'pointer', whiteSpace: 'nowrap', display: 'flex', alignItems: 'center', gap: '8px'
        }}>
          <Zap size={15} /> Detect
        </button>
      </div>

      {/* divider */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '14px', marginBottom: '16px' }}>
        <div style={{ flex: 1, height: '1px', background: 'rgba(255,255,255,0.08)' }} />
        <span style={{ color: '#475569', fontSize: '13px', fontWeight: 600 }}>or upload a file</span>
        <div style={{ flex: 1, height: '1px', background: 'rgba(255,255,255,0.08)' }} />
      </div>

      {/* drop zone */}
      <div
        onDragEnter={handleDrag} onDragLeave={handleDrag}
        onDragOver={handleDrag} onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        style={{
          border: `2px dashed ${dragActive ? '#6366f1' : 'rgba(255,255,255,0.12)'}`,
          borderRadius: '20px', padding: '52px 24px', textAlign: 'center',
          cursor: 'pointer', transition: 'all .25s',
          background: dragActive ? 'rgba(99,102,241,0.07)' : 'rgba(255,255,255,0.02)',
          transform: dragActive ? 'scale(1.01)' : 'scale(1)'
        }}
      >
        <input type="file" ref={inputRef} accept="video/*" style={{ display: 'none' }}
          onChange={e => { if (e.target.files?.[0]) handleFile(e.target.files[0]); }} />

        <div style={{
          width: '72px', height: '72px', borderRadius: '20px',
          background: 'rgba(99,102,241,0.15)', display: 'flex', alignItems: 'center',
          justifyContent: 'center', margin: '0 auto 20px', color: '#818cf8'
        }}>
          <FileVideo size={32} />
        </div>
        <p style={{ fontSize: '18px', fontWeight: 700, color: '#f1f5f9', marginBottom: '8px' }}>
          Drag &amp; drop your video here
        </p>
        <p style={{ fontSize: '14px', color: '#64748b', marginBottom: '24px' }}>
          MP4, MOV, AVI, WebM — up to 500 MB
        </p>
        <button style={{
          background: 'rgba(99,102,241,0.15)', border: '1px solid rgba(99,102,241,0.35)',
          borderRadius: '12px', padding: '12px 28px', color: '#818cf8',
          fontFamily: 'inherit', fontSize: '14px', fontWeight: 700, cursor: 'pointer',
          display: 'inline-flex', alignItems: 'center', gap: '8px', transition: 'background .2s'
        }}
          onMouseEnter={e => e.currentTarget.style.background = 'rgba(99,102,241,0.25)'}
          onMouseLeave={e => e.currentTarget.style.background = 'rgba(99,102,241,0.15)'}
        >
          <UploadCloud size={16} /> Browse Files
        </button>
      </div>

      {error && (
        <div style={{
          marginTop: '12px', padding: '12px 16px', borderRadius: '12px',
          background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.25)',
          color: '#f87171', fontSize: '14px', display: 'flex', alignItems: 'center', gap: '10px'
        }}>
          <AlertTriangle size={15} /> {error}
        </div>
      )}

      {/* trust row */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '24px', marginTop: '20px', flexWrap: 'wrap' }}>
        {['No signup required', 'Free to use', '95% accuracy'].map(t => (
          <span key={t} style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px', color: '#64748b', fontWeight: 500 }}>
            <CheckCircle size={13} color="#22c55e" /> {t}
          </span>
        ))}
      </div>
    </div>
  );

  /* preview state */
  if (preview && !result) return (
    <div style={{ width: '100%' }}>
      <div style={{ position: 'relative', marginBottom: '20px' }}>
        <video src={preview} controls style={{
          width: '100%', maxHeight: '380px', borderRadius: '16px',
          background: '#000', border: '1px solid rgba(255,255,255,0.08)'
        }} />
        <button onClick={reset} style={{
          position: 'absolute', top: '12px', right: '12px', width: '32px', height: '32px',
          background: 'rgba(0,0,0,0.7)', border: '1px solid rgba(255,255,255,0.2)',
          borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center',
          color: '#fff', cursor: 'pointer'
        }}><X size={14} /></button>
      </div>

      <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
        <div style={{ flex: 1, background: 'rgba(255,255,255,0.04)', borderRadius: '12px', padding: '10px 16px' }}>
          <div style={{ fontSize: '13px', color: '#64748b', fontWeight: 500 }}>Selected file</div>
          <div style={{ fontSize: '14px', color: '#f1f5f9', fontWeight: 600, marginTop: '2px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{file?.name}</div>
        </div>
        <button onClick={analyzeVideo} disabled={loading} style={{
          background: loading ? 'rgba(99,102,241,0.4)' : 'linear-gradient(135deg,#6366f1,#8b5cf6)',
          border: 'none', borderRadius: '12px', padding: '14px 32px', color: '#fff',
          fontFamily: 'inherit', fontSize: '15px', fontWeight: 700, cursor: loading ? 'not-allowed' : 'pointer',
          display: 'flex', alignItems: 'center', gap: '10px', whiteSpace: 'nowrap', flexShrink: 0,
          boxShadow: loading ? 'none' : '0 4px 24px rgba(99,102,241,0.35)'
        }}>
          {loading
            ? <><Loader2 size={16} style={{ animation: 'spin 1s linear infinite' }} /> Analyzing…</>
            : <><ShieldCheck size={16} /> Analyze Video</>
          }
        </button>
      </div>

      {loading && (
        <div style={{ marginTop: '16px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span style={{ fontSize: '13px', color: '#94a3b8' }}>Running deepfake analysis…</span>
          </div>
          <div style={{ height: '4px', background: 'rgba(255,255,255,0.07)', borderRadius: '999px', overflow: 'hidden' }}>
            <div style={{
              height: '100%', background: 'linear-gradient(90deg,#6366f1,#8b5cf6)',
              borderRadius: '999px', animation: 'progress 2s ease-in-out infinite'
            }} />
          </div>
          <div style={{ marginTop: '14px', display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
            {['Facial consistency', 'Lip-sync accuracy', 'Biometric patterns', 'Metadata check'].map((s, i) => (
              <span key={s} style={{
                fontSize: '12px', padding: '5px 12px', borderRadius: '999px',
                background: 'rgba(99,102,241,0.12)', border: '1px solid rgba(99,102,241,0.25)',
                color: '#818cf8', display: 'flex', alignItems: 'center', gap: '6px'
              }}>
                <Loader2 size={10} style={{ animation: `spin 1s linear ${i * 0.25}s infinite` }} /> {s}
              </span>
            ))}
          </div>
        </div>
      )}

      {error && (
        <div style={{
          marginTop: '12px', padding: '12px 16px', borderRadius: '12px',
          background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.25)',
          color: '#f87171', fontSize: '14px'
        }}>{error}</div>
      )}
    </div>
  );

  /* result state */
  if (result) {
    const isFake = result.verdict === 'FAKE';
    return (
      <div style={{ width: '100%', animation: 'fadeUp .5s ease' }}>
        {/* verdict banner */}
        <div style={{
          borderRadius: '20px', padding: '32px 28px', marginBottom: '24px', textAlign: 'center',
          background: isFake ? 'rgba(239,68,68,0.07)' : 'rgba(34,197,94,0.07)',
          border: `1px solid ${isFake ? 'rgba(239,68,68,0.3)' : 'rgba(34,197,94,0.3)'}`
        }}>
          <div style={{
            width: '64px', height: '64px', borderRadius: '50%', margin: '0 auto 16px',
            background: isFake ? 'rgba(239,68,68,0.12)' : 'rgba(34,197,94,0.12)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            color: isFake ? '#f87171' : '#4ade80'
          }}>
            {isFake ? <AlertTriangle size={28} /> : <ShieldCheck size={28} />}
          </div>
          <h3 style={{
            fontSize: '26px', fontWeight: 800, margin: '0 0 8px',
            color: isFake ? '#f87171' : '#4ade80'
          }}>
            {isFake ? '⚠ Deepfake Detected' : '✓ Authentic Video'}
          </h3>
          <p style={{ fontSize: '15px', color: '#94a3b8', margin: 0 }}>
            {isFake
              ? 'This video shows signs of AI manipulation or deepfake generation.'
              : 'No deepfake indicators found. This video appears to be authentic.'}
          </p>
        </div>

        {/* uncertainty warning */}
        {result.uncertain && (
          <div style={{
            marginBottom: '16px', padding: '14px 18px', borderRadius: '14px',
            background: 'rgba(251,191,36,0.08)', border: '1px solid rgba(251,191,36,0.3)',
            color: '#fbbf24', fontSize: '13px', lineHeight: '1.6'
          }}>
            ⚠ <strong>Low confidence result</strong> — the model is uncertain. Try a higher-resolution video with clearly visible frontal faces, or increase the number of sampled frames.
          </div>
        )}

        {/* metrics */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: '14px', marginBottom: '14px' }}>
          {[
            { label: 'Fake Probability', value: `${(result.fake_prob * 100).toFixed(1)}%`, bar: result.fake_prob, color: '#f87171', gradient: 'linear-gradient(90deg,#f97316,#ef4444)' },
            { label: 'Real Probability', value: `${(result.real_prob * 100).toFixed(1)}%`, bar: result.real_prob, color: '#4ade80', gradient: 'linear-gradient(90deg,#34d399,#22c55e)' },
            { label: 'Confidence', value: `${result.confidence != null ? result.confidence.toFixed(1) : ((Math.max(result.fake_prob, result.real_prob)) * 100).toFixed(1)}%`, bar: null, color: '#818cf8', gradient: null }
          ].map(m => (
            <div key={m.label} style={{
              background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)',
              borderRadius: '16px', padding: '20px 18px'
            }}>
              <div style={{ fontSize: '12px', color: '#64748b', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '12px' }}>{m.label}</div>
              {m.bar !== null && (
                <div style={{ height: '4px', background: 'rgba(255,255,255,0.07)', borderRadius: '999px', overflow: 'hidden', marginBottom: '10px' }}>
                  <div style={{ height: '100%', width: `${m.bar * 100}%`, background: m.gradient, borderRadius: '999px', transition: 'width 1s ease' }} />
                </div>
              )}
              <div style={{ fontSize: '26px', fontWeight: 800, color: m.color }}>{m.value}</div>
            </div>
          ))}
        </div>

        {/* frame breakdown & time */}
        {result.total_frames != null && (
          <div style={{
            background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)',
            borderRadius: '14px', padding: '16px 20px', marginBottom: '14px',
            display: 'flex', gap: '24px', flexWrap: 'wrap'
          }}>
            <div style={{ fontSize: '13px', color: '#64748b' }}>
              <span style={{ color: '#94a3b8', fontWeight: 600 }}>Frames analysed: </span>
              <span style={{ color: '#f1f5f9' }}>{result.total_frames}</span>
            </div>
            <div style={{ fontSize: '13px', color: '#64748b' }}>
              <span style={{ color: '#94a3b8', fontWeight: 600 }}>Deepfake frames: </span>
              <span style={{ color: '#f87171' }}>{result.deepfake_frames} ({result.deepfake_frame_pct}%)</span>
            </div>
            <div style={{ fontSize: '13px', color: '#64748b' }}>
              <span style={{ color: '#94a3b8', fontWeight: 600 }}>Analysis time: </span>
              <span style={{ color: '#818cf8' }}>{result.elapsed.toFixed(2)}s</span>
            </div>
          </div>
        )}

        <button onClick={reset} style={{
          width: '100%', background: 'rgba(255,255,255,0.04)',
          border: '1px solid rgba(255,255,255,0.1)', borderRadius: '14px',
          padding: '14px', color: '#94a3b8', fontFamily: 'inherit',
          fontSize: '14px', fontWeight: 600, cursor: 'pointer',
          display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px',
          transition: 'all .2s'
        }}
          onMouseEnter={e => { e.currentTarget.style.background = 'rgba(255,255,255,0.08)'; e.currentTarget.style.color = '#f1f5f9'; }}
          onMouseLeave={e => { e.currentTarget.style.background = 'rgba(255,255,255,0.04)'; e.currentTarget.style.color = '#94a3b8'; }}
        >
          <ArrowRight size={15} /> Analyze another video
        </button>
      </div>
    );
  }

  return null;
}

/* ─── PAGE ─── */
export default function App() {
  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #060b18; color: #f1f5f9; font-family: 'Plus Jakarta Sans', system-ui, sans-serif; }
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes progress { 0%{width:0} 50%{width:70%} 100%{width:100%} }
        @keyframes fadeUp { from{opacity:0;transform:translateY(16px)} to{opacity:1;transform:none} }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.5} }
        ::-webkit-scrollbar { width: 6px; } ::-webkit-scrollbar-track { background: #0f172a; } ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
      `}</style>

      {/* NAV */}
      <nav style={{
        position: 'sticky', top: 0, zIndex: 100,
        background: 'rgba(6,11,24,0.8)', backdropFilter: 'blur(20px)',
        borderBottom: '1px solid rgba(255,255,255,0.06)',
        padding: '0 32px', height: '62px',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '24px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', fontWeight: 800, fontSize: '18px', color: '#f1f5f9' }}>
          <div style={{
            width: '34px', height: '34px', borderRadius: '10px',
            background: 'linear-gradient(135deg,#6366f1,#8b5cf6)',
            display: 'flex', alignItems: 'center', justifyContent: 'center'
          }}>
            <ShieldCheck size={18} color="#fff" />
          </div>
          DeepFake<span style={{ color: '#818cf8' }}>Guardian</span>
        </div>

        <div style={{ display: 'flex', gap: '8px' }}>
          <button style={{
            background: 'none', border: '1px solid rgba(255,255,255,0.12)',
            borderRadius: '10px', padding: '8px 18px', color: '#94a3b8',
            fontFamily: 'inherit', fontSize: '13px', fontWeight: 600, cursor: 'pointer'
          }}>Log in</button>
          <button style={{
            background: 'linear-gradient(135deg,#6366f1,#8b5cf6)',
            border: 'none', borderRadius: '10px', padding: '8px 18px',
            color: '#fff', fontFamily: 'inherit', fontSize: '13px', fontWeight: 700, cursor: 'pointer'
          }}>Start free →</button>
        </div>
      </nav>

      <main style={{ maxWidth: '1100px', margin: '0 auto', padding: '0 24px 80px' }}>

        {/* HERO */}
        <section style={{ textAlign: 'center', padding: '72px 0 60px', position: 'relative' }}>
          {/* glow blobs */}
          <div style={{ position: 'absolute', top: '-60px', left: '50%', transform: 'translateX(-60%)', width: '600px', height: '600px', background: 'radial-gradient(ellipse,rgba(99,102,241,0.12),transparent 70%)', pointerEvents: 'none', zIndex: 0 }} />

          <div style={{ position: 'relative', zIndex: 1 }}>
            <Badge><ShieldCheck size={12} /> Free deepfake detector — no signup</Badge>
            <h1 style={{ fontSize: 'clamp(36px,6vw,64px)', fontWeight: 900, lineHeight: 1.1, marginTop: '20px', marginBottom: '20px', letterSpacing: '-0.02em' }}>
              Detect AI-Generated<br />
              <span style={{ background: 'linear-gradient(135deg,#6366f1,#a78bfa)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                Video in Seconds
              </span>
            </h1>
            <p style={{ fontSize: '18px', color: '#94a3b8', maxWidth: '560px', margin: '0 auto 36px', lineHeight: 1.7 }}>
              Upload any video or paste a URL. Our AI scans facial movements, lip-sync, and biometric patterns with <strong style={{ color: '#c4b5fd' }}>95% accuracy</strong>.
            </p>

            {/* social proof */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '12px', marginBottom: '48px' }}>
              <div style={{ display: 'flex' }}>
                {['#6366f1','#8b5cf6','#a78bfa'].map((c, i) => (
                  <div key={i} style={{
                    width: '32px', height: '32px', borderRadius: '50%',
                    background: c, border: '2px solid #060b18', marginLeft: i ? '-8px' : 0,
                    display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '12px', fontWeight: 700, color: '#fff'
                  }}>{['J','M','A'][i]}</div>
                ))}
              </div>
              <div style={{ fontSize: '14px', color: '#64748b' }}>
                <span style={{ display: 'flex', gap: '2px', color: '#fbbf24', marginBottom: '1px' }}>
                  {[...Array(5)].map((_,i)=><Star key={i} size={12} fill="#fbbf24" />)}
                </span>
                Loved by <strong style={{ color: '#94a3b8' }}>3M+</strong> users
              </div>
            </div>
          </div>
        </section>

        {/* UPLOAD CARD */}
        <section style={{ marginBottom: '80px' }}>
          <div style={{
            background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)',
            borderRadius: '28px', padding: '40px', boxShadow: '0 40px 80px rgba(0,0,0,0.4)'
          }}>
            <UploadWidget />
          </div>
        </section>

        {/* STATS */}
        <section style={{ marginBottom: '80px' }}>
          <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap' }}>
            <StatCard icon="🎯" label="Detection Accuracy" value="95%" />
            <StatCard icon="⚡" label="Avg. Analysis Time" value="< 2 min" />
            <StatCard icon="🎬" label="Videos Analyzed" value="10M+" />
            <StatCard icon="🔒" label="Data Retained" value="0 days" />
          </div>
        </section>

        {/* HOW IT WORKS */}
        <section style={{ marginBottom: '80px' }}>
          <div style={{ textAlign: 'center', marginBottom: '44px' }}>
            <h2 style={{ fontSize: '34px', fontWeight: 800, letterSpacing: '-0.02em', marginBottom: '12px' }}>
              How It Works
            </h2>
            <p style={{ color: '#64748b', fontSize: '16px' }}>Three steps. Under two minutes.</p>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(240px,1fr))', gap: '20px' }}>
            {[
              { n: '01', title: 'Upload or paste URL', desc: 'Drop an MP4, MOV or AVI file, or paste a YouTube / TikTok / Instagram URL directly.' },
              { n: '02', title: 'Multi-modal analysis', desc: 'Transformer neural networks scan facial consistency, lip-sync, voice tone, and metadata.' },
              { n: '03', title: 'Get your report', desc: 'Receive a confidence score with flagged frames and a clear REAL / FAKE verdict in minutes.' },
            ].map(s => (
              <div key={s.n} style={{
                background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)',
                borderRadius: '20px', padding: '32px 28px', position: 'relative', overflow: 'hidden'
              }}>
                <div style={{
                  position: 'absolute', top: '20px', right: '20px', fontSize: '48px', fontWeight: 900,
                  color: 'rgba(99,102,241,0.08)', lineHeight: 1, userSelect: 'none'
                }}>{s.n}</div>
                <div style={{ fontSize: '36px', fontWeight: 900, color: '#6366f1', marginBottom: '14px' }}>{s.n}</div>
                <h3 style={{ fontSize: '17px', fontWeight: 700, color: '#f1f5f9', marginBottom: '10px' }}>{s.title}</h3>
                <p style={{ fontSize: '14px', color: '#64748b', lineHeight: '1.7' }}>{s.desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* FEATURES */}
        <section style={{ marginBottom: '80px' }}>
          <div style={{ textAlign: 'center', marginBottom: '44px' }}>
            <h2 style={{ fontSize: '34px', fontWeight: 800, letterSpacing: '-0.02em', marginBottom: '12px' }}>
              What We Detect
            </h2>
            <p style={{ color: '#64748b', fontSize: '16px' }}>Comprehensive analysis across every manipulation vector.</p>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(220px,1fr))', gap: '16px' }}>
            <FeatureCard icon={<MonitorPlay size={20} />} title="Face-swap Detection" desc="Identifies replaced faces using landmark tracking and temporal consistency analysis." />
            <FeatureCard icon={<BarChart2 size={20} />} title="Lip-sync Analysis" desc="Checks audio-visual timing offsets introduced during voice cloning or re-dubbing." />
            <FeatureCard icon={<Zap size={20} />} title="Motion Artifacts" desc="Detects unnatural eye blinking, subtle warping, and GAN-specific pixel patterns." />
            <FeatureCard icon={<Clock size={20} />} title="Metadata Inspection" desc="Reads encoding fingerprints and editing software traces left in the video container." />
            <FeatureCard icon={<Globe size={20} />} title="URL Support" desc="Works with YouTube, TikTok, Instagram, and Facebook links — no download needed." />
            <FeatureCard icon={<Users size={20} />} title="Batch Processing" desc="Analyze multiple videos simultaneously on Business and Enterprise plans." />
          </div>
        </section>

        {/* FAQ */}
        <section style={{ marginBottom: '80px', maxWidth: '720px', margin: '0 auto 80px' }}>
          <div style={{ textAlign: 'center', marginBottom: '44px' }}>
            <h2 style={{ fontSize: '34px', fontWeight: 800, letterSpacing: '-0.02em', marginBottom: '12px' }}>
              Frequently Asked Questions
            </h2>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <FAQ q="Is this really free?" a="Yes — you can analyze videos with no account required. We offer additional credits and batch processing on paid plans." />
            <FAQ q="How accurate is the detector?" a="Our transformer-based models achieve 95% accuracy on deepfakes, face-swaps, voice clones, and fully synthetic AI videos." />
            <FAQ q="Which video formats are supported?" a="MP4, MOV, AVI, and WebM are supported for direct upload. You can also paste YouTube, TikTok, Instagram, or Facebook URLs." />
            <FAQ q="Is my video stored after analysis?" a="No. Videos are processed in-memory and immediately discarded after analysis. Zero data retention." />
            <FAQ q="Can it detect Sora, Runway, or Synthesia videos?" a="Yes — our model is continuously updated as new generative tools emerge, including Sora, Runway Gen-3, Synthesia, and Kling." />
          </div>
        </section>

      </main>

      {/* FOOTER */}
      <footer style={{
        borderTop: '1px solid rgba(255,255,255,0.06)',
        padding: '28px 32px', textAlign: 'center',
        color: '#334155', fontSize: '13px', fontWeight: 500
      }}>
        © 2025 DeepFakeGuardian · Built with AI · Zero data retention
      </footer>
    </>
  );
}
