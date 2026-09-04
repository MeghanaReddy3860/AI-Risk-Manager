import type { HealthStatus } from '../types'

interface HeaderProps {
  health: HealthStatus | null
  healthLoading: boolean
  healthError: string | null
  onRefreshHealth: () => void
  onRunPipeline: () => void
  isPipelineRunning: boolean
}

export function Header({
  health,
  healthLoading,
  healthError,
  onRefreshHealth,
  onRunPipeline,
  isPipelineRunning,
}: HeaderProps) {
  return (
    <header className="bg-bg-secondary border-b border-border px-6 py-4 flex flex-col md:flex-row md:items-center justify-between gap-4 shadow-md">
      {/* Product Identity & Badges */}
      <div className="flex flex-col sm:flex-row sm:items-center gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-2xl">🛡️</span>
            <h1 className="text-xl font-bold text-text-primary tracking-tight">
              AI Risk Manager
            </h1>
          </div>
          <p className="text-xs text-text-secondary font-medium mt-0.5">
            Fraud-Spike Detection System
          </p>
        </div>

        {/* Persistent Defensive & Synthetic Warnings */}
        <div className="flex flex-wrap items-center gap-2">
          <span className="px-2.5 py-1 bg-status-medium/15 border border-status-medium/30 text-status-medium text-xs font-semibold rounded-full uppercase tracking-wider">
            Synthetic / Test Data Only
          </span>
          <span className="px-2.5 py-1 bg-accent/15 border border-accent/30 text-accent text-xs font-semibold rounded-full uppercase tracking-wider">
            Defense-Only • No Automated Enforcement
          </span>
        </div>
      </div>

      {/* Backend Status & Primary Pipeline Trigger */}
      <div className="flex items-center gap-4">
        {/* Backend Health Badge */}
        <button
          onClick={onRefreshHealth}
          title="Click to re-check backend health"
          className="flex items-center gap-2 px-3 py-1.5 bg-bg-primary/50 border border-border rounded-lg text-xs font-medium hover:border-text-muted transition-colors"
        >
          {healthLoading ? (
            <>
              <span className="w-2 h-2 border border-accent border-t-transparent rounded-full animate-spin" />
              <span className="text-text-secondary">Checking backend...</span>
            </>
          ) : healthError ? (
            <>
              <span className="w-2 h-2 bg-status-critical rounded-full" />
              <span className="text-status-critical font-semibold">Backend Unreachable</span>
            </>
          ) : health ? (
            <>
              <span className="w-2 h-2 bg-status-low rounded-full animate-pulse" />
              <span className="text-status-low font-semibold">Backend Healthy</span>
              <span className="text-text-muted">({health.version})</span>
            </>
          ) : null}
        </button>

        {/* Run Anomaly Pipeline Action Button */}
        <button
          onClick={onRunPipeline}
          disabled={isPipelineRunning || Boolean(healthError)}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold text-white shadow-sm transition-all ${
            isPipelineRunning || healthError
              ? 'bg-bg-card-hover cursor-not-allowed opacity-60 text-text-muted'
              : 'bg-accent hover:bg-accent-hover active:scale-98'
          }`}
        >
          {isPipelineRunning ? (
            <>
              <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
              <span>Running Detectors...</span>
            </>
          ) : (
            <>
              <span>⚡</span>
              <span>Run Anomaly Pipeline</span>
            </>
          )}
        </button>
      </div>
    </header>
  )
}
