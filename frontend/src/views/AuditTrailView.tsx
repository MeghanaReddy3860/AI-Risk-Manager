import { useEffect, useState } from 'react'
import {
  getWindows,
  getWindowAuditReport,
  recordAnalystAction,
  verifyAuditIntegrity,
} from '../api'
import type {
  AuditIntegrityVerification,
  AuditRecord,
  AuditReport,
  DetectionWindow,
} from '../types'

export function AuditTrailView() {
  // Selected window & search state
  const [selectedWindowId, setSelectedWindowId] = useState<number | null>(null)
  const [inputWindowId, setInputWindowId] = useState<string>('1')
  const [recentWindows, setRecentWindows] = useState<DetectionWindow[]>([])

  // Audit report for selected window state
  const [auditReport, setAuditReport] = useState<AuditReport | null>(null)
  const [reportLoading, setReportLoading] = useState<boolean>(false)
  const [reportError, setReportError] = useState<string | null>(null)

  // Integrity verification state
  const [verificationResult, setVerificationResult] = useState<AuditIntegrityVerification | null>(null)
  const [verifying, setVerifying] = useState<boolean>(false)
  const [verificationError, setVerificationError] = useState<string | null>(null)

  // Selected audit record for detail modal
  const [selectedRecord, setSelectedRecord] = useState<AuditRecord | null>(null)
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false)

  // Analyst Action form state
  const [actionDisposition, setActionDisposition] = useState<string>('escalate')
  const [actionActor, setActionActor] = useState<string>('ANALYST:analyst_01')
  const [actionNotes, setActionNotes] = useState<string>('')
  const [isSubmittingAction, setIsSubmittingAction] = useState<boolean>(false)
  const [actionFeedback, setActionFeedback] = useState<{
    type: 'success' | 'error'
    message: string
  } | null>(null)

  // 1. Fetch recent windows on mount for quick selector
  useEffect(() => {
    let isMounted = true
    getWindows({ limit: 10 })
      .then((windows) => {
        if (isMounted) {
          setRecentWindows(windows)
          if (windows.length > 0) {
            setSelectedWindowId(windows[0].id)
            setInputWindowId(String(windows[0].id))
          } else {
            setSelectedWindowId(1)
          }
        }
      })
      .catch(() => {
        if (isMounted) {
          setSelectedWindowId(1)
        }
      })

    return () => {
      isMounted = false
    }
  }, [])

  // 2. Fetch Audit Report whenever selectedWindowId changes (Read-Only GET request)
  useEffect(() => {
    if (selectedWindowId === null) return
    let isMounted = true

    const fetchReport = async () => {
      try {
        const report = await getWindowAuditReport(selectedWindowId)
        if (isMounted) {
          setAuditReport(report)
          setReportError(null)
          setReportLoading(false)
        }
      } catch (err: unknown) {
        if (isMounted) {
          const msg = err instanceof Error ? err.message : 'Failed to load audit report'
          setReportError(msg)
          setAuditReport(null)
          setReportLoading(false)
        }
      }
    }

    fetchReport()

    return () => {
      isMounted = false
    }
  }, [selectedWindowId])

  // Explicit user-triggered integrity verification
  const handleVerifyIntegrity = async () => {
    if (verifying) return
    setVerifying(true)
    setVerificationError(null)

    try {
      const res = await verifyAuditIntegrity()
      setVerificationResult(res)
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Integrity verification failed'
      setVerificationError(msg)
    } finally {
      setVerifying(false)
    }
  }

  // Handle manual window lookup search submit
  const handleSearchWindow = (e: React.FormEvent) => {
    e.preventDefault()
    const id = parseInt(inputWindowId.trim(), 10)
    if (!isNaN(id) && id > 0) {
      setSelectedWindowId(id)
    }
  }

  // Handle submitting an analyst review action into audit trail
  const handleRecordAction = async (e: React.FormEvent) => {
    e.preventDefault()
    if (selectedWindowId === null || isSubmittingAction) return

    setIsSubmittingAction(true)
    setActionFeedback(null)

    try {
      const res = await recordAnalystAction({
        actor: actionActor.trim() || 'ANALYST:test_user',
        window_id: selectedWindowId,
        disposition: actionDisposition,
        notes: actionNotes.trim(),
      })

      setActionFeedback({
        type: 'success',
        message: res.message || 'Analyst disposition recorded in audit trail successfully.',
      })
      setActionNotes('')

      // Re-fetch window audit report to show newly created ANALYST_ACTION event
      const updatedReport = await getWindowAuditReport(selectedWindowId)
      setAuditReport(updatedReport)
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to record analyst action'
      setActionFeedback({
        type: 'error',
        message: msg,
      })
    } finally {
      setIsSubmittingAction(false)
    }
  }

  const getEventTypeBadge = (eventType: string) => {
    switch (eventType) {
      case 'WINDOW_CREATED':
        return <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-accent/10 border border-accent/30 text-accent">Window Created</span>
      case 'DETECTION_FLAGGED':
        return <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-status-critical/10 border border-status-critical/30 text-status-critical">Detection Flagged</span>
      case 'RISK_EVALUATED':
        return <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-status-medium/10 border border-status-medium/30 text-status-medium">Risk Evaluated</span>
      case 'EXPLANATION_GENERATED':
        return <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-purple-500/10 border border-purple-500/30 text-purple-400">Explanation Generated</span>
      case 'POLICY_DECISION':
        return <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-blue-500/10 border border-blue-500/30 text-blue-400">Policy Decision</span>
      case 'ANALYST_ACTION':
        return <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-status-low/10 border border-status-low/30 text-status-low">Analyst Review</span>
      default:
        return <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-bg-card-hover border border-border text-text-secondary">{eventType}</span>
    }
  }

  const formatDate = (isoString: string) => {
    try {
      return new Date(isoString).toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false,
      })
    } catch {
      return isoString
    }
  }

  return (
    <div className="space-y-6">
      {/* Top Title & Header Card */}
      <div className="bg-bg-card border border-border rounded-xl p-6 shadow-md space-y-4">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-border pb-4">
          <div>
            <h2 className="text-xl font-bold text-text-primary flex items-center gap-2">
              <span>🛡️ Cryptographic Audit Trail</span>
            </h2>
            <p className="text-xs text-text-secondary mt-1">
              Immutable operational record of risk-analysis, policy decisions, and analyst reviews with SHA-256 chain verification
            </p>
          </div>

          {/* Explicit User-Triggered Integrity Verification Button */}
          <button
            onClick={handleVerifyIntegrity}
            disabled={verifying}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold text-white shadow-sm transition-all ${
              verifying
                ? 'bg-bg-card-hover cursor-not-allowed opacity-60 text-text-muted'
                : 'bg-accent hover:bg-accent-hover active:scale-98'
            }`}
          >
            {verifying ? (
              <>
                <span className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                <span>Verifying SHA-256 Chain...</span>
              </>
            ) : (
              <>
                <span>🔒</span>
                <span>Verify Audit Integrity</span>
              </>
            )}
          </button>
        </div>

        {/* Verification Status Output Banner */}
        {verificationResult && (
          <div
            className={`p-4 border rounded-xl flex items-start justify-between text-xs ${
              verificationResult.integrity_valid
                ? 'bg-status-low/10 border-status-low/30 text-status-low'
                : 'bg-status-critical/10 border-status-critical/30 text-status-critical'
            }`}
          >
            <div className="space-y-1">
              <div className="font-bold flex items-center gap-2">
                <span>{verificationResult.integrity_valid ? '✅ SHA-256 Audit Chain Validated' : '❌ Cryptographic Chain Tampering Detected'}</span>
                <span className="font-mono text-[11px] px-2 py-0.5 rounded bg-bg-primary/40 border border-border">
                  Total Records: {verificationResult.total_records}
                </span>
              </div>
              {verificationResult.integrity_errors.length > 0 && (
                <ul className="list-disc list-inside text-[11px] space-y-0.5 text-status-critical mt-1 font-mono">
                  {verificationResult.integrity_errors.map((err, idx) => (
                    <li key={idx}>{err}</li>
                  ))}
                </ul>
              )}
            </div>
            <button
              onClick={() => setVerificationResult(null)}
              className="text-text-muted hover:text-text-primary px-2 py-0.5 rounded"
            >
              ✕
            </button>
          </div>
        )}

        {verificationError && (
          <div className="p-4 border rounded-xl bg-status-critical/10 border-status-critical/30 text-status-critical text-xs flex justify-between">
            <span>❌ Integrity Verification Error: {verificationError}</span>
            <button onClick={() => setVerificationError(null)} className="text-text-muted hover:text-text-primary">✕</button>
          </div>
        )}

        {/* Window Lookup Bar & Quick Selectors */}
        <div className="flex flex-col md:flex-row items-stretch md:items-center justify-between gap-4 bg-bg-primary/50 p-4 rounded-lg border border-border">
          <form onSubmit={handleSearchWindow} className="flex items-center gap-2">
            <label htmlFor="window-search-input" className="text-xs font-semibold text-text-secondary whitespace-nowrap">
              Inspect Window ID:
            </label>
            <input
              id="window-search-input"
              type="number"
              min="1"
              value={inputWindowId}
              onChange={(e) => setInputWindowId(e.target.value)}
              className="w-24 bg-bg-card border border-border rounded-lg px-3 py-1.5 text-xs font-mono text-text-primary focus:outline-none focus:border-accent"
              placeholder="e.g. 1"
            />
            <button
              type="submit"
              className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-bg-card-hover border border-border text-text-primary hover:bg-border transition-all"
            >
              Inspect Audit
            </button>
          </form>

          {/* Quick Select Pills for Recent Detection Windows */}
          {recentWindows.length > 0 && (
            <div className="flex items-center gap-1.5 overflow-x-auto text-[11px] py-1">
              <span className="text-text-muted font-medium mr-1">Quick Select:</span>
              {recentWindows.slice(0, 6).map((w) => (
                <button
                  key={w.id}
                  onClick={() => {
                    setSelectedWindowId(w.id)
                    setInputWindowId(String(w.id))
                  }}
                  className={`px-2.5 py-1 rounded font-mono font-semibold border transition-all ${
                    selectedWindowId === w.id
                      ? 'bg-accent/20 border-accent text-accent'
                      : 'bg-bg-card border-border text-text-secondary hover:text-text-primary hover:bg-bg-card-hover'
                  }`}
                >
                  #{w.id} ({w.merchant_id})
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Main Audit Trail Window Report View */}
      {reportLoading ? (
        <div className="bg-bg-card border border-border rounded-xl p-12 text-center flex flex-col items-center justify-center gap-3">
          <div className="w-8 h-8 border-2 border-accent border-t-transparent rounded-full animate-spin" />
          <p className="text-text-secondary text-sm font-medium">Loading audit events for Detection Window #{selectedWindowId}...</p>
        </div>
      ) : reportError ? (
        <div className="bg-status-critical/10 border border-status-critical/30 rounded-xl p-6 text-center">
          <p className="text-status-critical font-bold text-sm">❌ Unable to load audit trail</p>
          <p className="text-xs text-text-secondary mt-1">{reportError}</p>
        </div>
      ) : auditReport ? (
        <div className="space-y-6">
          {/* Audit Report Header Summary */}
          <div className="bg-bg-card border border-border rounded-xl p-6 shadow-md space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-border pb-3">
              <h3 className="text-sm font-bold text-text-primary uppercase tracking-wider flex items-center gap-2">
                <span>📋 Audit Report — Window #{auditReport.window_id}</span>
              </h3>
              <div className="flex items-center gap-3 text-xs">
                <span className="text-text-muted">Total Events: <strong className="text-text-primary font-mono">{auditReport.event_count}</strong></span>
                <span className={`px-2 py-0.5 rounded font-mono text-[11px] font-semibold border ${
                  auditReport.integrity_valid
                    ? 'bg-status-low/10 border-status-low/30 text-status-low'
                    : 'bg-status-critical/10 border-status-critical/30 text-status-critical'
                }`}>
                  {auditReport.integrity_valid ? 'Integrity: Valid ✅' : 'Integrity: Failed ❌'}
                </span>
              </div>
            </div>

            {/* High-Density Audit Trail Event Table */}
            {auditReport.events.length === 0 ? (
              <div className="py-8 text-center text-xs text-text-muted">
                No audit events logged for Window #{auditReport.window_id}.
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr className="border-b border-border text-text-secondary font-semibold uppercase tracking-wider bg-bg-primary/30">
                      <th className="py-2.5 px-3">Timestamp (UTC)</th>
                      <th className="py-2.5 px-3">Event Type</th>
                      <th className="py-2.5 px-3">Actor / System</th>
                      <th className="py-2.5 px-3">Entry ID / Hash</th>
                      <th className="py-2.5 px-3 text-right">Inspect Detail</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border/60 font-mono text-[11px]">
                    {auditReport.events.map((rec) => (
                      <tr key={rec.entry_id} className="hover:bg-bg-card-hover/40 transition-colors">
                        <td className="py-2.5 px-3 text-text-secondary whitespace-nowrap">
                          {formatDate(rec.timestamp)}
                        </td>
                        <td className="py-2.5 px-3">
                          {getEventTypeBadge(rec.event_type)}
                        </td>
                        <td className="py-2.5 px-3 font-semibold text-text-primary whitespace-nowrap">
                          {rec.actor}
                        </td>
                        <td className="py-2.5 px-3 text-text-muted truncate max-w-[180px]" title={rec.entry_id}>
                          {rec.entry_id.slice(0, 18)}...
                        </td>
                        <td className="py-2.5 px-3 text-right">
                          <button
                            onClick={() => {
                              setSelectedRecord(rec)
                              setIsModalOpen(true)
                            }}
                            className="px-2.5 py-1 rounded bg-bg-card-hover border border-border text-text-primary hover:bg-border transition-all text-[11px]"
                          >
                            Inspect 🔍
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Record Analyst Review Disposition Form (Defense-Only Human Triage) */}
          <div className="bg-bg-card border border-border rounded-xl p-6 shadow-md space-y-4">
            <h3 className="text-sm font-bold text-text-primary uppercase tracking-wider border-b border-border pb-2 flex items-center justify-between">
              <span>✍️ Record Analyst Review Action</span>
              <span className="text-xs text-status-low font-semibold border border-status-low/30 bg-status-low/10 px-2 py-0.5 rounded">
                Human Analyst Triage Only
              </span>
            </h3>

            {actionFeedback && (
              <div
                className={`p-3 border rounded-lg text-xs flex items-center justify-between ${
                  actionFeedback.type === 'success'
                    ? 'bg-status-low/10 border-status-low/30 text-status-low'
                    : 'bg-status-critical/10 border-status-critical/30 text-status-critical'
                }`}
              >
                <span>{actionFeedback.message}</span>
                <button onClick={() => setActionFeedback(null)} className="text-text-muted">✕</button>
              </div>
            )}

            <form onSubmit={handleRecordAction} className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
              <div>
                <label htmlFor="action-actor-input" className="block text-text-secondary font-semibold mb-1">
                  Analyst Identity (Actor):
                </label>
                <input
                  id="action-actor-input"
                  type="text"
                  value={actionActor}
                  onChange={(e) => setActionActor(e.target.value)}
                  className="w-full bg-bg-primary border border-border rounded-lg px-3 py-2 text-text-primary font-mono focus:outline-none focus:border-accent"
                  placeholder="ANALYST:analyst_01"
                  required
                />
              </div>

              <div>
                <label htmlFor="action-disposition-select" className="block text-text-secondary font-semibold mb-1">
                  Permitted Review Disposition:
                </label>
                <select
                  id="action-disposition-select"
                  value={actionDisposition}
                  onChange={(e) => setActionDisposition(e.target.value)}
                  className="w-full bg-bg-primary border border-border rounded-lg px-3 py-2 text-text-primary focus:outline-none focus:border-accent"
                >
                  <option value="escalate">Escalate (Dual Review Required)</option>
                  <option value="resolve">Resolve (False Positive Confirmed)</option>
                  <option value="flag_for_followup">Flag for Followup</option>
                  <option value="monitor">Monitor (Enhanced Velocity Tracking)</option>
                </select>
              </div>

              <div className="md:col-span-3">
                <label htmlFor="action-notes-input" className="block text-text-secondary font-semibold mb-1">
                  Analyst Triage Notes / Rationale:
                </label>
                <textarea
                  id="action-notes-input"
                  rows={2}
                  value={actionNotes}
                  onChange={(e) => setActionNotes(e.target.value)}
                  className="w-full bg-bg-primary border border-border rounded-lg px-3 py-2 text-text-primary focus:outline-none focus:border-accent font-sans"
                  placeholder="Provide investigation details per defense policy guidelines..."
                />
              </div>

              <div className="md:col-span-3 flex justify-end">
                <button
                  type="submit"
                  disabled={isSubmittingAction}
                  className={`px-4 py-2 rounded-lg text-xs font-semibold text-white transition-all ${
                    isSubmittingAction
                      ? 'bg-bg-card-hover cursor-not-allowed opacity-60'
                      : 'bg-accent hover:bg-accent-hover'
                  }`}
                >
                  {isSubmittingAction ? 'Logging to Audit Trail...' : 'Log Analyst Action'}
                </button>
              </div>
            </form>
          </div>
        </div>
      ) : null}

      {/* Read-Only Audit Event Detail Drawer Modal */}
      {isModalOpen && selectedRecord && (
        <div className="fixed inset-0 z-50 flex items-center justify-end bg-black/60 backdrop-blur-sm p-4 sm:p-6 transition-opacity">
          <div className="bg-bg-card border border-border rounded-2xl w-full max-w-2xl max-h-[90vh] flex flex-col shadow-2xl overflow-hidden">
            {/* Modal Header */}
            <div className="p-6 border-b border-border flex items-center justify-between bg-bg-primary/50">
              <div>
                <h3 className="text-base font-bold text-text-primary flex items-center gap-2">
                  <span>🔍 Audit Event Detail</span>
                  {getEventTypeBadge(selectedRecord.event_type)}
                </h3>
                <p className="text-xs text-text-secondary font-mono mt-0.5">
                  Entry ID: {selectedRecord.entry_id}
                </p>
              </div>
              <button
                onClick={() => setIsModalOpen(false)}
                className="w-8 h-8 rounded-lg border border-border flex items-center justify-center text-text-secondary hover:text-text-primary hover:bg-bg-card-hover transition-colors"
              >
                ✕
              </button>
            </div>

            {/* Modal Body */}
            <div className="p-6 overflow-y-auto space-y-4 text-xs font-mono">
              {/* Record Metadata Grid */}
              <div className="grid grid-cols-2 gap-3 bg-bg-primary/50 p-4 rounded-xl border border-border">
                <div>
                  <span className="text-text-muted block text-[11px]">Timestamp (UTC):</span>
                  <span className="text-text-primary font-bold">{formatDate(selectedRecord.timestamp)}</span>
                </div>
                <div>
                  <span className="text-text-muted block text-[11px]">Actor Identity:</span>
                  <span className="text-accent font-bold">{selectedRecord.actor}</span>
                </div>
                <div>
                  <span className="text-text-muted block text-[11px]">Associated Window ID:</span>
                  <span className="text-text-primary font-bold">#{selectedRecord.window_id}</span>
                </div>
                <div>
                  <span className="text-text-muted block text-[11px]">Associated Merchant ID:</span>
                  <span className="text-text-primary font-bold">{selectedRecord.merchant_id}</span>
                </div>
              </div>

              {/* Cryptographic SHA-256 Hashing Details */}
              <div className="bg-bg-primary/50 p-4 rounded-xl border border-border space-y-2">
                <div>
                  <span className="text-text-muted block text-[11px]">Previous Record Hash (Chaining):</span>
                  <span className="text-text-secondary text-[11px] break-all">
                    {selectedRecord.previous_hash ?? 'GENESIS (None)'}
                  </span>
                </div>
                <div>
                  <span className="text-text-muted block text-[11px]">Integrity SHA-256 Hash:</span>
                  <span className="text-status-low font-bold text-[11px] break-all">
                    {selectedRecord.integrity_hash}
                  </span>
                </div>
              </div>

              {/* Formatted JSON Payload */}
              <div className="space-y-1">
                <span className="text-text-primary font-bold block text-xs">Event Payload Data:</span>
                <pre className="bg-bg-primary border border-border rounded-xl p-4 text-[11px] text-text-primary overflow-x-auto leading-relaxed">
                  {JSON.stringify(selectedRecord.payload, null, 2)}
                </pre>
              </div>
            </div>

            {/* Modal Footer (Strict Read-Only) */}
            <div className="p-4 border-t border-border bg-bg-primary/50 flex justify-between items-center text-xs">
              <span className="text-text-muted text-[11px]">
                🛡️ Read-only audit inspection. Zero database mutations.
              </span>
              <button
                onClick={() => setIsModalOpen(false)}
                className="px-4 py-2 rounded-lg bg-bg-card-hover border border-border text-text-primary hover:bg-border transition-colors font-semibold"
              >
                Close Inspection
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
