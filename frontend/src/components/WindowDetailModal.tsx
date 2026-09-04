import { useEffect, useState } from 'react'
import { getWindow, getWindowAnalysis, getWindowDetections } from '../api'
import type { AnomalyDetection, DetectionWindow, RiskDossier } from '../types'

interface WindowDetailModalProps {
  windowId: number | null
  isOpen: boolean
  onClose: () => void
}

export function WindowDetailModal({ windowId, isOpen, onClose }: WindowDetailModalProps) {
  const [windowData, setWindowData] = useState<DetectionWindow | null>(null)
  const [detections, setDetections] = useState<AnomalyDetection[]>([])
  const [baselineDossier, setBaselineDossier] = useState<RiskDossier | null>(null)
  const [mlDossier, setMlDossier] = useState<RiskDossier | null>(null)

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [noStoredResult, setNoStoredResult] = useState(false)

  // Listen for Escape key press
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isOpen, onClose])

  // Fetch window details and read-only analysis dossiers when modal opens
  useEffect(() => {
    if (!isOpen || windowId === null) {
      return
    }

    let isMounted = true
    const fetchAll = async () => {
      try {
        // 1. Fetch Window metadata
        const win = await getWindow(windowId)
        if (!isMounted) return
        setWindowData(win)

        // 2. Fetch stored detections array
        try {
          const detList = await getWindowDetections(windowId)
          if (isMounted) setDetections(detList)
        } catch {
          if (isMounted) setDetections([])
        }

        // 3. Fetch read-only analysis dossiers (Baseline & ML)
        let baselineRes: RiskDossier | null = null
        let mlRes: RiskDossier | null = null
        let foundStored = false

        try {
          baselineRes = await getWindowAnalysis(windowId, 'baseline')
          foundStored = true
        } catch {
          // No stored baseline analysis
        }

        try {
          mlRes = await getWindowAnalysis(windowId, 'ml')
          foundStored = true
        } catch {
          // No stored ML analysis
        }

        if (isMounted) {
          setBaselineDossier(baselineRes)
          setMlDossier(mlRes)
          if (!foundStored) {
            setNoStoredResult(true)
          }
        }
      } catch (err: unknown) {
        if (isMounted) {
          const msg = err instanceof Error ? err.message : 'Unable to load window investigation'
          setError(msg)
        }
      } finally {
        if (isMounted) {
          setLoading(false)
        }
      }
    }

    fetchAll()

    return () => {
      isMounted = false
    }
  }, [isOpen, windowId])

  if (!isOpen || windowId === null) return null

  const formatCurrency = (val: number) =>
    new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      maximumFractionDigits: 2,
    }).format(val)

  const formatDate = (isoString: string) =>
    new Date(isoString).toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false,
    })

  // Find stored detection rows if dossiers absent
  const storedBaselineDet = detections.find((d) => d.detector_type === 'baseline')
  const storedMlDet = detections.find((d) => d.detector_type === 'ml')

  const baselineScore = baselineDossier?.risk_result.risk_score ?? storedBaselineDet?.risk_score
  const baselineFlagged = baselineDossier?.is_flagged ?? storedBaselineDet?.is_flagged
  const baselineExplanation = baselineDossier?.explanation.summary ?? storedBaselineDet?.explanation

  const mlScore = mlDossier?.risk_result.risk_score ?? storedMlDet?.risk_score
  const mlFlagged = mlDossier?.is_flagged ?? storedMlDet?.is_flagged
  const mlExplanation = mlDossier?.explanation.summary ?? storedMlDet?.explanation

  // Determine agreement if both evaluated
  const bothEvaluated = baselineFlagged !== undefined && mlFlagged !== undefined
  const detectorsAgree = bothEvaluated && baselineFlagged === mlFlagged

  const getRiskScoreColor = (score?: number) => {
    if (score === undefined) return 'bg-text-muted text-text-muted'
    if (score >= 80) return 'bg-status-critical text-status-critical'
    if (score >= 60) return 'bg-status-high text-status-high'
    if (score >= 30) return 'bg-status-medium text-status-medium'
    return 'bg-status-low text-status-low'
  }

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-black/60 backdrop-blur-xs transition-opacity animate-in fade-in duration-200">
      {/* Backdrop click area */}
      <div className="flex-1 cursor-pointer" onClick={onClose} aria-hidden="true" />

      {/* Slide-over Drawer Panel */}
      <div className="w-full max-w-2xl bg-bg-secondary border-l border-border h-full overflow-y-auto flex flex-col shadow-2xl z-50">
        {/* Drawer Header */}
        <div className="p-6 border-b border-border bg-bg-primary/50 sticky top-0 z-10 flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xl">🔍</span>
              <h2 className="text-lg font-bold text-text-primary">
                Window Investigation — #{windowId}
              </h2>
            </div>
            {windowData && (
              <div className="flex items-center gap-2 mt-1">
                <span className="text-xs text-text-secondary font-medium">Merchant:</span>
                <span className="text-xs font-mono font-semibold text-text-primary">
                  {windowData.merchant_id}
                </span>
                <span className="text-text-muted">•</span>
                <span className="px-2 py-0.5 bg-accent/15 text-accent border border-accent/30 text-[10px] font-semibold rounded uppercase">
                  {windowData.split}
                </span>
                {windowData.is_synthetic_fraud_spike !== undefined && (
                  <span
                    className={`px-2 py-0.5 text-[10px] font-semibold rounded uppercase ${
                      windowData.is_synthetic_fraud_spike
                        ? 'bg-status-medium/15 text-status-medium border border-status-medium/30'
                        : 'bg-status-low/15 text-status-low border border-status-low/30'
                    }`}
                  >
                    {windowData.is_synthetic_fraud_spike ? '⚡ Synthetic Spike' : 'Normal'}
                  </span>
                )}
              </div>
            )}
          </div>

          <button
            onClick={onClose}
            className="text-text-muted hover:text-text-primary px-3 py-1.5 rounded-lg border border-border bg-bg-primary hover:border-text-muted text-sm font-semibold transition-colors"
          >
            ✕ Close (Esc)
          </button>
        </div>

        {/* Drawer Body Content */}
        <div className="p-6 space-y-6 flex-1">
          {loading ? (
            <div className="py-20 flex flex-col items-center gap-3 text-text-secondary text-sm">
              <div className="w-8 h-8 border-2 border-accent border-t-transparent rounded-full animate-spin" />
              <span>Loading stored window analysis...</span>
            </div>
          ) : error ? (
            <div className="bg-status-critical/10 border border-status-critical/30 rounded-xl p-5 text-center">
              <p className="text-status-critical font-bold text-sm">❌ Unable to load window analysis</p>
              <p className="text-xs text-text-secondary mt-1">{error}</p>
            </div>
          ) : windowData ? (
            <>
              {/* Section 1: Window Metadata & Feature Breakdown */}
              <div className="bg-bg-card border border-border rounded-xl p-5 space-y-4">
                <h3 className="text-sm font-bold text-text-primary uppercase tracking-wider border-b border-border pb-2 flex items-center gap-2">
                  <span>📊 Transaction Features</span>
                </h3>

                <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 text-xs">
                  <div>
                    <span className="text-text-muted block">Window Start</span>
                    <span className="font-mono text-text-primary font-medium">
                      {formatDate(windowData.window_start)}
                    </span>
                  </div>
                  <div>
                    <span className="text-text-muted block">Window End</span>
                    <span className="font-mono text-text-primary font-medium">
                      {formatDate(windowData.window_end)}
                    </span>
                  </div>
                  <div>
                    <span className="text-text-muted block">Partition Split</span>
                    <span className="font-mono text-text-primary font-medium uppercase">
                      {windowData.split}
                    </span>
                  </div>

                  <div>
                    <span className="text-text-muted block">Transaction Count</span>
                    <span className="text-base font-mono font-bold text-text-primary">
                      {windowData.transaction_count}
                    </span>
                  </div>
                  <div>
                    <span className="text-text-muted block">Total Volume</span>
                    <span className="text-base font-mono font-bold text-text-primary">
                      {formatCurrency(windowData.total_amount)}
                    </span>
                  </div>
                  <div>
                    <span className="text-text-muted block">Avg Transaction Size</span>
                    <span className="text-base font-mono font-bold text-text-primary">
                      {formatCurrency(windowData.avg_transaction_amount)}
                    </span>
                  </div>
                </div>
              </div>

              {/* Section 2: Detector Comparison (Baseline vs ML) */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-bold text-text-primary uppercase tracking-wider flex items-center gap-2">
                    <span>🔬 Detector Results Comparison</span>
                  </h3>

                  {bothEvaluated && (
                    <span
                      className={`px-2.5 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider ${
                        detectorsAgree
                          ? 'bg-status-low/15 text-status-low border border-status-low/30'
                          : 'bg-status-high/15 text-status-high border border-status-high/30'
                      }`}
                    >
                      {detectorsAgree ? 'Agreement: Yes' : 'Disagreement: Split Prediction'}
                    </span>
                  )}
                </div>

                {noStoredResult && (
                  <div className="bg-status-medium/10 border border-status-medium/30 rounded-xl p-4 text-xs text-status-medium space-y-1">
                    <p className="font-bold flex items-center gap-1.5">
                      <span>ℹ️</span> No Stored Detector Results Available
                    </p>
                    <p className="text-text-secondary">
                      No stored detection predictions exist for window #{windowId}. Please run the Anomaly Pipeline from the dashboard header to score and persist results.
                    </p>
                  </div>
                )}

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* Baseline Detector Card */}
                  <div className="bg-bg-card border border-border rounded-xl p-5 space-y-3">
                    <div className="flex items-center justify-between border-b border-border pb-2">
                      <span className="text-xs font-bold text-text-primary">Baseline Detector</span>
                      {baselineFlagged !== undefined ? (
                        baselineFlagged ? (
                          <span className="px-2 py-0.5 bg-status-critical/20 text-status-critical text-[10px] font-bold rounded uppercase">
                            🚨 Flagged
                          </span>
                        ) : (
                          <span className="px-2 py-0.5 bg-status-low/20 text-status-low text-[10px] font-bold rounded uppercase">
                            ✅ Normal
                          </span>
                        )
                      ) : (
                        <span className="text-[10px] text-text-muted">Unscored</span>
                      )}
                    </div>

                    {/* Risk Score Meter */}
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-text-secondary">Risk Score</span>
                        <span className="font-mono font-bold text-text-primary">
                          {baselineScore !== undefined ? `${baselineScore.toFixed(1)} / 100` : 'N/A'}
                        </span>
                      </div>
                      <div className="w-full bg-bg-primary rounded-full h-2 overflow-hidden border border-border">
                        <div
                          className={`h-full transition-all ${getRiskScoreColor(baselineScore).split(' ')[0]}`}
                          style={{ width: `${Math.min(100, Math.max(0, baselineScore ?? 0))}%` }}
                        />
                      </div>
                    </div>

                    {/* Explanation */}
                    <div className="text-xs space-y-1 pt-1">
                      <span className="text-text-muted font-medium block">Explanation:</span>
                      <p className="text-text-secondary leading-relaxed bg-bg-primary/50 p-2.5 rounded border border-border/60">
                        {baselineExplanation || 'No stored baseline explanation available.'}
                      </p>
                    </div>
                  </div>

                  {/* ML Isolation Forest Card */}
                  <div className="bg-bg-card border border-border rounded-xl p-5 space-y-3">
                    <div className="flex items-center justify-between border-b border-border pb-2">
                      <span className="text-xs font-bold text-text-primary">ML Isolation Forest</span>
                      {mlFlagged !== undefined ? (
                        mlFlagged ? (
                          <span className="px-2 py-0.5 bg-status-critical/20 text-status-critical text-[10px] font-bold rounded uppercase">
                            🚨 Flagged
                          </span>
                        ) : (
                          <span className="px-2 py-0.5 bg-status-low/20 text-status-low text-[10px] font-bold rounded uppercase">
                            ✅ Normal
                          </span>
                        )
                      ) : (
                        <span className="text-[10px] text-text-muted">Unscored</span>
                      )}
                    </div>

                    {/* Risk Score Meter */}
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-text-secondary">Risk Score</span>
                        <span className="font-mono font-bold text-text-primary">
                          {mlScore !== undefined ? `${mlScore.toFixed(1)} / 100` : 'N/A'}
                        </span>
                      </div>
                      <div className="w-full bg-bg-primary rounded-full h-2 overflow-hidden border border-border">
                        <div
                          className={`h-full transition-all ${getRiskScoreColor(mlScore).split(' ')[0]}`}
                          style={{ width: `${Math.min(100, Math.max(0, mlScore ?? 0))}%` }}
                        />
                      </div>
                    </div>

                    {/* Explanation */}
                    <div className="text-xs space-y-1 pt-1">
                      <span className="text-text-muted font-medium block">Explanation:</span>
                      <p className="text-text-secondary leading-relaxed bg-bg-primary/50 p-2.5 rounded border border-border/60">
                        {mlExplanation || 'No stored ML explanation available.'}
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Section 3: Risk Scoring & Operational Policy Dossier */}
              {(baselineDossier || mlDossier) && (
                <div className="bg-bg-card border border-border rounded-xl p-5 space-y-4">
                  <h3 className="text-sm font-bold text-text-primary uppercase tracking-wider border-b border-border pb-2 flex items-center gap-2">
                    <span>🛡️ Defensive Risk & Policy Dossier</span>
                  </h3>

                  {(() => {
                    const dossier = baselineDossier || mlDossier
                    if (!dossier) return null
                    return (
                      <div className="space-y-4 text-xs">
                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                          <div className="bg-bg-primary/50 p-3 rounded-lg border border-border">
                            <span className="text-text-muted block">Risk Band</span>
                            <span className="text-sm font-bold text-text-primary capitalize">
                              {dossier.risk_result.risk_band}
                            </span>
                          </div>
                          <div className="bg-bg-primary/50 p-3 rounded-lg border border-border">
                            <span className="text-text-muted block">Estimated Exposure</span>
                            <span className="text-sm font-bold font-mono text-text-primary">
                              {formatCurrency(dossier.risk_result.estimated_exposure)}
                            </span>
                          </div>
                          <div className="bg-bg-primary/50 p-3 rounded-lg border border-border">
                            <span className="text-text-muted block">Policy ID</span>
                            <span className="text-sm font-bold font-mono text-text-primary">
                              {dossier.policy_decision.policy_id}
                            </span>
                          </div>
                          <div className="bg-bg-primary/50 p-3 rounded-lg border border-border">
                            <span className="text-text-muted block">Review SLA</span>
                            <span className="text-sm font-bold text-text-primary">
                              {dossier.policy_decision.review_sla_hours} hours
                            </span>
                          </div>
                        </div>

                        {/* Recommended Action */}
                        <div className="bg-accent/10 border border-accent/30 rounded-lg p-3.5 space-y-1">
                          <span className="font-bold text-accent block">Recommended Triage Action:</span>
                          <p className="text-text-primary font-semibold">
                            {dossier.risk_result.recommended_action}
                          </p>
                          <p className="text-text-muted text-[11px] mt-1">
                            Recommendation for human analyst review only. No automated account restrictions, blocking, or funds freezing are performed.
                          </p>
                        </div>
                      </div>
                    )
                  })()}
                </div>
              )}

              {/* Section 4: Defense-Only System Disclaimer */}
              <div className="bg-bg-primary/40 border border-border rounded-xl p-4 text-[11px] text-text-muted leading-relaxed flex items-start gap-2.5">
                <span className="text-base">🛡️</span>
                <div>
                  <strong className="text-text-secondary">Defensive Risk Operations Notice:</strong> This investigation view exposes persisted detector outputs and natural-language explanations strictly for human analyst evaluation. The system enforces zero automated enforcement, blocking, or account suspension.
                </div>
              </div>
            </>
          ) : null}
        </div>
      </div>
    </div>
  )
}
