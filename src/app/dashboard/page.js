'use client';

import { useState, useRef, useCallback } from 'react';
import Link from 'next/link';
import { motion, AnimatePresence } from 'framer-motion';
import { Camera, Image as ImageIcon, Video, Scan, X, RefreshCw, AlertCircle, Leaf, CheckCircle2, AlertTriangle, ArrowLeft, Zap, Download, ChevronRight } from 'lucide-react';
import { analyzeImage, MODEL_READY } from '@/lib/model';

const playTone = (type) => {
  if (typeof window === 'undefined') return;
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);
    
    if (type === 'green') {
      osc.type = 'sine';
      osc.frequency.setValueAtTime(523.25, ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(1046.50, ctx.currentTime + 0.2);
      gain.gain.setValueAtTime(0.1, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.5);
    } else if (type === 'amber') {
      osc.type = 'triangle';
      osc.frequency.setValueAtTime(392.00, ctx.currentTime);
      gain.gain.setValueAtTime(0.1, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.5);
    } else if (type === 'red') {
      osc.type = 'square';
      osc.frequency.setValueAtTime(300, ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(200, ctx.currentTime + 0.3);
      gain.gain.setValueAtTime(0.1, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.5);
    }
    osc.start();
    osc.stop(ctx.currentTime + 0.5);
  } catch (e) { console.error('Audio blocked', e); }
};

export default function DashboardPage() {
  const [imagePreview, setImagePreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [inputMode, setInputMode] = useState(null);
  const [cameraStream, setCameraStream] = useState(null);

  const fileInputRef = useRef(null);
  const videoInputRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const imgRef = useRef(null);

  const startCamera = useCallback(async () => {
    setResult(null);
    setImagePreview(null);
    setInputMode('camera');

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      setCameraStream(stream);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
    } catch (err) {
      alert('Could not access camera. Please check your permissions.');
      setInputMode(null);
    }
  }, []);

  const capturePhoto = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
    setImagePreview(dataUrl);
    if (cameraStream) {
      cameraStream.getTracks().forEach(t => t.stop());
      setCameraStream(null);
    }
    setInputMode('captured');
  }, [cameraStream]);

  const handleFileSelect = useCallback((e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setResult(null);
    setInputMode('upload');
    const reader = new FileReader();
    reader.onload = (ev) => setImagePreview(ev.target.result);
    reader.readAsDataURL(file);
  }, []);

  const handleVideoSelect = useCallback((e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setResult(null);
    setInputMode('video');
    setImagePreview(null);
    const url = URL.createObjectURL(file);
    if (videoRef.current) {
      videoRef.current.srcObject = null;
      videoRef.current.src = url;
      videoRef.current.play();
    }
  }, []);

  const captureVideoFrame = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    setImagePreview(canvas.toDataURL('image/jpeg', 0.9));
    video.pause();
    setInputMode('captured');
  }, []);

  const handleAnalyze = useCallback(async () => {
    if (!imagePreview) return;
    setLoading(true);
    setResult(null);
    try {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = reject;
        img.src = imagePreview;
      });
      const analysisResult = await analyzeImage(img);
      setResult(analysisResult);
      playTone(analysisResult.light);
    } catch (err) {
      console.error('Analysis error:', err);
      setResult({ error: 'Analysis failed. Please try again.' });
    } finally {
      setLoading(false);
    }
  }, [imagePreview]);

  const handleDownloadReport = useCallback(() => {
    if (!result) return;
    const reportText = `CropGuard AI - Field Analysis Report\n======================================\nDate: ${new Date().toLocaleString()}\nStatus: ${result.isHealthy ? 'Healthy' : 'Disease Detected'}\nConfidence: ${(result.confidence * 100).toFixed(1)}%\nRecommendation: ${result.shouldSpray ? 'Immediate Spray Recommended' : (result.light === 'amber' ? 'Close Monitoring Required' : 'No Action Needed')}\n\nPrimary Detection: ${result.topPrediction.label}\n${result.topPrediction.diseaseInfo ? `\nDetails:\n- Crop: ${result.topPrediction.diseaseInfo.crop}\n- Disease: ${result.topPrediction.diseaseInfo.disease}\n- Severity: ${result.topPrediction.diseaseInfo.severity} Risk\n- Action Plan: ${result.topPrediction.diseaseInfo.advice}\n` : ''}`.trim();
    const blob = new Blob([reportText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `CropGuard_Report_${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  }, [result]);

  const handleReset = useCallback(() => {
    setImagePreview(null);
    setResult(null);
    setInputMode(null);
    if (cameraStream) {
      cameraStream.getTracks().forEach(t => t.stop());
      setCameraStream(null);
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
      videoRef.current.src = '';
    }
  }, [cameraStream]);

  return (
    <div className="min-h-screen relative flex flex-col selection:bg-[#21A049] selection:text-white overflow-x-hidden">
      {/* Dynamic Background */}
      <div className="fixed inset-0 z-0 pointer-events-none opacity-40">
        <div className="absolute top-[-10%] right-[-10%] w-[500px] h-[500px] bg-[#21A049] blur-[150px] rounded-full opacity-10" />
        <div className="absolute bottom-[-10%] left-[-10%] w-[500px] h-[500px] bg-[#10B981] blur-[150px] rounded-full opacity-10" />
      </div>

      {/* ─── Premium Header ───────────────────────── */}
      <header className="sticky top-0 z-50 glass-header">
        <div className="max-w-5xl mx-auto px-6 h-16 flex items-center justify-between">
          <Link href="/" className="flex items-center gap-2 text-[var(--text-secondary)] hover:text-[var(--text)] transition-colors group">
            <ArrowLeft className="w-5 h-5 group-hover:-translate-x-1 transition-transform" />
            <span className="font-bold text-xs uppercase tracking-widest">Back to Hub</span>
          </Link>
          <div className="flex items-center gap-4">
            <div className="hidden sm:flex items-center gap-2 px-3 py-1 rounded-full bg-[#E8F5E9] dark:bg-[#121F16] border border-[#A5D6A7] dark:border-[#2E7D32]">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#4CAF50] opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-[#4CAF50]"></span>
              </span>
              <span className="text-[10px] font-black uppercase tracking-tighter text-[#2E7D32] dark:text-[#4CAF50]">Edge AI Active</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-[#21A049] to-[#124022] flex items-center justify-center shadow-md">
                <Leaf className="w-4 h-4 text-white" />
              </div>
              <span className="font-black tracking-tighter text-lg text-[var(--green-dark)]">CropGuard</span>
            </div>
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-2xl mx-auto w-full px-6 py-12 relative z-10 flex flex-col">
        {!MODEL_READY && (
          <motion.div 
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="w-full p-4 mb-8 rounded-2xl flex items-center gap-3 text-sm font-bold bg-[#FFF8EB] dark:bg-[#201A0A] border border-[#FDE0A6] dark:border-[#B37400] text-[#B37400]"
          >
            <AlertCircle className="w-5 h-5 flex-shrink-0" />
            <span className="uppercase tracking-widest text-[10px]">Simulation Mode — YOLOv8 Integration Pending</span>
          </motion.div>
        )}

        <AnimatePresence mode="wait">
          {/* ─── Mode Selection ────────────────────── */}
          {!imagePreview && !inputMode && (
            <motion.div 
              key="selection"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="w-full space-y-8"
            >
              <div className="text-center space-y-2">
                <h1 className="text-4xl font-black tracking-tighter text-[var(--text)]">Capture Engine</h1>
                <p className="text-[var(--text-secondary)] font-medium">Deploy neural network to scan crop health.</p>
              </div>

              <div className="grid gap-4">
                <button onClick={startCamera} className="upload-zone !p-12 group card flex flex-col items-center gap-6">
                  <div className="w-20 h-20 rounded-[28px] bg-[#F2F7F0] dark:bg-[#1A261F] flex items-center justify-center group-hover:scale-110 group-hover:rotate-6 transition-all duration-500 shadow-inner">
                    <Camera className="w-10 h-10 text-[#21A049]" />
                  </div>
                  <div className="text-center">
                    <h3 className="text-2xl font-black text-[var(--text)] tracking-tight">Live Field Scan</h3>
                    <p className="text-[var(--text-secondary)] text-sm font-medium">Real-time inference on site</p>
                  </div>
                </button>

                <div className="grid grid-cols-2 gap-4">
                  <button onClick={() => fileInputRef.current?.click()} className="card !p-8 group flex flex-col items-center gap-4">
                    <div className="w-14 h-14 rounded-2xl bg-[#F2F7F0] dark:bg-[#1A261F] flex items-center justify-center group-hover:scale-110 transition-transform">
                      <ImageIcon className="w-6 h-6 text-[#21A049]" />
                    </div>
                    <span className="font-bold text-sm text-[var(--text)] uppercase tracking-widest">Photo File</span>
                  </button>
                  <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={handleFileSelect} />

                  <button onClick={() => videoInputRef.current?.click()} className="card !p-8 group flex flex-col items-center gap-4">
                    <div className="w-14 h-14 rounded-2xl bg-[#F2F7F0] dark:bg-[#1A261F] flex items-center justify-center group-hover:scale-110 transition-transform">
                      <Video className="w-6 h-6 text-[#21A049]" />
                    </div>
                    <span className="font-bold text-sm text-[var(--text)] uppercase tracking-widest">Video Frame</span>
                  </button>
                  <input ref={videoInputRef} type="file" accept="video/*" className="hidden" onChange={handleVideoSelect} />
                </div>
              </div>
            </motion.div>
          )}

          {/* ─── Viewfinders ───────────────────────── */}
          {(inputMode === 'camera' || inputMode === 'video') && !imagePreview && (
            <motion.div key="finder" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="w-full space-y-6">
              <div className="relative aspect-[4/3] rounded-[32px] overflow-hidden bg-black shadow-2xl border-4 border-[var(--border)]">
                <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover" />
                <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                  <div className="w-64 h-64 border-2 border-white/20 rounded-[40px] relative">
                    <div className="absolute top-0 left-0 w-8 h-8 border-t-4 border-l-4 border-[#21A049] rounded-tl-2xl -translate-x-2 -translate-y-2" />
                    <div className="absolute top-0 right-0 w-8 h-8 border-t-4 border-r-4 border-[#21A049] rounded-tr-2xl translate-x-2 -translate-y-2" />
                    <div className="absolute bottom-0 left-0 w-8 h-8 border-b-4 border-l-4 border-[#21A049] rounded-bl-2xl -translate-x-2 translate-y-2" />
                    <div className="absolute bottom-0 right-0 w-8 h-8 border-b-4 border-r-4 border-[#21A049] rounded-br-2xl translate-x-2 translate-y-2" />
                    <div className="absolute inset-0 bg-[#21A049]/10 animate-pulse" />
                  </div>
                </div>
              </div>
              <div className="flex gap-4">
                <button onClick={handleReset} className="btn btn-secondary flex-1">Abort</button>
                <button onClick={inputMode === 'camera' ? capturePhoto : captureVideoFrame} className="btn btn-primary flex-[2] shadow-2xl">
                  {inputMode === 'camera' ? <Camera className="w-6 h-6"/> : <Scan className="w-6 h-6"/>}
                  <span>{inputMode === 'camera' ? 'Capture Frame' : 'Extract Frame'}</span>
                </button>
              </div>
            </motion.div>
          )}

          {/* ─── Analysis Stage ────────────────────── */}
          {imagePreview && !result && (
            <motion.div key="analysis" initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="w-full space-y-8">
              <div className="relative rounded-[40px] overflow-hidden shadow-[0_40px_80px_-20px_rgba(0,0,0,0.3)] border-4 border-[var(--border)]">
                <img ref={imgRef} src={imagePreview} alt="Target" className="w-full h-full object-cover aspect-square" />
                <AnimatePresence>
                  {loading && (
                    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="absolute inset-0 bg-black/40 backdrop-blur-md flex flex-col items-center justify-center">
                      <div className="w-24 h-24 relative mb-6">
                        <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 2, ease: "linear" }} className="absolute inset-0 border-4 border-white/20 rounded-full" />
                        <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 1.5, ease: "linear" }} className="absolute inset-0 border-t-4 border-[#21A049] rounded-full" />
                        <Scan className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 text-[#21A049] animate-pulse" />
                      </div>
                      <span className="font-black text-xs uppercase tracking-[0.3em] text-white">Neural Processing...</span>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
              <div className="flex flex-col sm:flex-row gap-4">
                <button onClick={handleReset} disabled={loading} className="btn btn-secondary flex-1">Reset</button>
                <button onClick={handleAnalyze} disabled={loading} className="btn btn-primary flex-[2] text-xl !py-6 shadow-2xl">
                  {loading ? <RefreshCw className="w-6 h-6 animate-spin" /> : <Zap className="w-6 h-6" />}
                  <span>{loading ? 'Processing...' : 'Run Diagnostics'}</span>
                </button>
              </div>
            </motion.div>
          )}

          {/* ─── Result Stage ──────────────────────── */}
          {result && !result.error && (
            <motion.div key="results" initial={{ opacity: 0, y: 40 }} animate={{ opacity: 1, y: 0 }} className="w-full space-y-8">
              <div className="card !p-10 text-center relative overflow-hidden bg-gradient-to-b from-[var(--surface)] to-[var(--surface-hover)]">
                <div className={`w-32 h-32 rounded-full mx-auto mb-8 light-${result.light}`} />
                <h2 className="text-4xl font-black tracking-tighter mb-4 text-[var(--text)]">
                  {result.isHealthy ? 'Healthy Growth' : result.topPrediction.label}
                </h2>
                
                <div className="inline-flex items-center gap-2 px-6 py-2 rounded-full bg-[var(--surface)] border border-[var(--border)] shadow-sm mb-8">
                  <span className="w-2 h-2 rounded-full" style={{ background: `var(--status-${result.light})` }} />
                  <span className="text-xs font-black uppercase tracking-widest" style={{ color: `var(--status-${result.light})` }}>
                    {result.shouldSpray ? 'Immediate Action Required' : result.light === 'amber' ? 'Increased Monitoring' : 'Stable Condition'}
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-4 pt-8 border-t border-[var(--border)]">
                  <div className="text-left">
                    <p className="text-[10px] font-black uppercase tracking-[0.2em] text-[var(--text-secondary)] mb-1">AI Confidence</p>
                    <p className="text-2xl font-black text-[var(--text)]">{(result.confidence * 100).toFixed(1)}%</p>
                  </div>
                  <div className="text-right">
                    <p className="text-[10px] font-black uppercase tracking-[0.2em] text-[var(--text-secondary)] mb-1">Protocol</p>
                    <p className="text-sm font-bold text-[var(--green-main)]">{result.isHealthy ? 'Standard Care' : 'Treatment Active'}</p>
                  </div>
                </div>
              </div>

              {!result.isHealthy && result.topPrediction.diseaseInfo && (
                <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.3 }} className="card !p-8 space-y-8 border-l-4 border-l-[#21A049]">
                  <div className="flex justify-between items-start">
                    <div>
                      <p className="text-[10px] font-black uppercase tracking-widest text-[var(--text-secondary)] mb-2">Detected Pathogen</p>
                      <h3 className="text-3xl font-black tracking-tight">{result.topPrediction.diseaseInfo.disease}</h3>
                    </div>
                    <span className="px-4 py-2 rounded-xl text-[10px] font-black uppercase tracking-widest border border-[var(--border)] bg-[var(--surface-hover)]">
                      {result.topPrediction.diseaseInfo.severity} Risk
                    </span>
                  </div>

                  <div className="grid grid-cols-2 gap-8">
                    <div>
                      <p className="text-[10px] font-black uppercase tracking-widest text-[var(--text-secondary)] mb-1">Host Crop</p>
                      <p className="font-bold">{result.topPrediction.diseaseInfo.crop}</p>
                    </div>
                    <div>
                      <p className="text-[10px] font-black uppercase tracking-widest text-[var(--text-secondary)] mb-1">Status</p>
                      <p className="font-bold text-[#EF4444]">Action Needed</p>
                    </div>
                  </div>

                  <div className="p-6 rounded-[24px] bg-[#21A049]/5 border border-[#21A049]/20 relative overflow-hidden group hover:bg-[#21A049]/10 transition-colors">
                    <p className="text-[10px] font-black uppercase tracking-widest text-[#21A049] mb-3 flex items-center gap-2">
                      <Zap className="w-3 h-3"/> Recommended Treatment Protocol
                    </p>
                    <p className="text-lg font-medium leading-relaxed text-[var(--text)] italic">
                      "{result.topPrediction.diseaseInfo.advice}"
                    </p>
                  </div>
                </motion.div>
              )}

              <div className="flex flex-col sm:flex-row gap-4">
                <button onClick={handleReset} className="btn btn-secondary flex-1 !py-5">
                  <RefreshCw className="w-5 h-5"/> New Analysis
                </button>
                <button onClick={handleDownloadReport} className="btn btn-primary flex-1 !py-5 shadow-2xl group">
                  <Download className="w-5 h-5 group-hover:translate-y-1 transition-transform"/> Save Report
                </button>
              </div>
            </motion.div>
          )}

          {/* Error State */}
          {result?.error && (
            <motion.div key="error" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="card !p-12 text-center bg-red-500/10 border-red-500/30">
              <AlertCircle className="w-16 h-16 text-red-500 mx-auto mb-6" />
              <h2 className="text-2xl font-black mb-2">Engine Fault</h2>
              <p className="text-red-500/80 font-bold mb-8 tracking-tight">{result.error}</p>
              <button onClick={handleReset} className="btn btn-primary w-full bg-red-500 !shadow-red-500/20">Reinitialize System</button>
            </motion.div>
          )}
        </AnimatePresence>

        <canvas ref={canvasRef} className="hidden" />
      </main>

      <footer className="py-12 text-center">
        <p className="text-[10px] font-black uppercase tracking-[0.4em] text-[var(--text-secondary)] opacity-40">CropGuard Intelligence Protocol v2.4.0</p>
      </footer>
    </div>
  );
}
