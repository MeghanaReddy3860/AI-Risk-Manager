import { useEffect, useState } from 'react'
import { getHealth, runDetectors } from './api'
import { Header } from './components/Header'
import { Navigation, type TabType } from './components/Navigation'
import { NotificationBanner } from './components/NotificationBanner'
import type { DetectorRunSummary, HealthStatus } from './types'
import { AuditTrailView } from './views/AuditTrailView'
import { EvaluationView } from './views/EvaluationView'
import { StreamMonitorView } from './views/StreamMonitorView'

function App() {
  const [health, setHealth] = useState<HealthStatus | null>(null)
  const [healthLoading, setHealthLoading] = useState(true)
  const [healthError, setHealthError] = useState<string | null>(null)

  const [activeTab, setActiveTab] = useState<TabType>('stream')
  const [isPipelineRunning, setIsPipelineRunning] = useState(false)
  const [notification, setNotification] = useState<{
    type: 'success' | 'error' | 'info'
    message: string
  } | null>(null)

  // Fetch backend health status for manual refresh button
  const fetchHealth = async () => {
    setHealthLoading(true)
    setHealthError(null)
    try {
      const data = await getHealth()
      setHealth(data)
    } catch (err: unknown) {
      const errorMsg = err instanceof Error ? err.message : 'Backend unreachable'
      setHealthError(errorMsg)
      setHealth(null)
    } finally {
      setHealthLoading(false)
    }
  }

  useEffect(() => {
    let isMounted = true
    getHealth()
      .then((data) => {
        if (isMounted) {
          setHealth(data)
          setHealthLoading(false)
        }
      })
      .catch((err: unknown) => {
        if (isMounted) {
          const errorMsg = err instanceof Error ? err.message : 'Backend unreachable'
          setHealthError(errorMsg)
          setHealth(null)
          setHealthLoading(false)
        }
      })
    return () => {
      isMounted = false
    }
  }, [])

  const [refreshTrigger, setRefreshTrigger] = useState<number>(0)

  // Explicit user-triggered pipeline execution
  const handleRunPipeline = async () => {
    if (isPipelineRunning) return

    setIsPipelineRunning(true)
    setNotification({
      type: 'info',
      message: 'Running detectors over synthetic train/dev_test data... Please wait.',
    })

    try {
      const summary: DetectorRunSummary = await runDetectors()
      const baselineFlagged = summary.results?.baseline?.windows_flagged ?? 0
      const mlFlagged = summary.results?.ml?.windows_flagged ?? 0
      setNotification({
        type: 'success',
        message: `Pipeline run completed! Scored baseline (${baselineFlagged} flagged) and ML Isolation Forest (${mlFlagged} flagged).`,
      })
      setRefreshTrigger((prev) => prev + 1)
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Pipeline execution failed'
      setNotification({
        type: 'error',
        message: `Detector pipeline failed: ${msg}. No punitive actions were taken.`,
      })
    } finally {
      setIsPipelineRunning(false)
    }
  }

  return (
    <div className="min-h-screen bg-bg-primary text-text-primary flex flex-col font-sans">
      {/* Application Header */}
      <Header
        health={health}
        healthLoading={healthLoading}
        healthError={healthError}
        onRefreshHealth={fetchHealth}
        onRunPipeline={handleRunPipeline}
        isPipelineRunning={isPipelineRunning}
      />

      {/* Main Tab Navigation */}
      <Navigation activeTab={activeTab} onSelectTab={setActiveTab} />

      {/* Primary Dashboard Content Area */}
      <main className="flex-1 px-6 py-6 max-w-7xl w-full mx-auto space-y-6">
        {/* User Notification / Status Feedback Banner */}
        {notification && (
          <NotificationBanner
            type={notification.type}
            message={notification.message}
            onClose={() => setNotification(null)}
          />
        )}

        {/* Tab View Containers */}
        {activeTab === 'stream' && <StreamMonitorView refreshTrigger={refreshTrigger} />}
        {activeTab === 'evaluation' && <EvaluationView />}
        {activeTab === 'audit' && <AuditTrailView />}
      </main>

      {/* Dashboard Footer */}
      <footer className="border-t border-border py-4 px-6 text-center text-xs text-text-muted">
        <span>AI Risk Manager — Defensive Fraud-Spike Detector</span>
        <span className="mx-2">•</span>
        <span>Phase 12.2B Stream Monitor Dashboard</span>
      </footer>
    </div>
  )
}

export default App
