import type { MerchantSummary } from '../types'

interface OverviewCardsProps {
  merchants: MerchantSummary[]
  loading: boolean
  error: string | null
}

export function OverviewCards({ merchants, loading, error }: OverviewCardsProps) {
  if (loading) {
    return (
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 animate-pulse">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="bg-bg-card border border-border rounded-xl p-5 h-28 flex flex-col justify-between shadow-sm">
            <div className="h-4 bg-bg-card-hover rounded w-1/2" />
            <div className="h-8 bg-bg-card-hover rounded w-3/4" />
          </div>
        ))}
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-status-critical/10 border border-status-critical/30 rounded-xl p-4 text-status-critical text-sm font-medium">
        ⚠️ Unable to calculate merchant KPI metrics: {error}
      </div>
    )
  }

  const totalMerchants = merchants.length
  const totalWindows = merchants.reduce((sum, m) => sum + m.total_windows, 0)
  const totalFlaggedSpikes = merchants.reduce((sum, m) => sum + m.flagged_anomaly_count, 0)
  const totalVolume = merchants.reduce((sum, m) => sum + m.total_monetary_volume, 0)

  // Format currency nicely in INR
  const formattedVolume = new Intl.NumberFormat('en-IN', {
    style: 'currency',
    currency: 'INR',
    maximumFractionDigits: 0,
  }).format(totalVolume)

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      {/* Card 1: Total Monitored Merchants */}
      <div className="bg-bg-card border border-border rounded-xl p-5 shadow-sm hover:border-text-muted transition-colors flex flex-col justify-between">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold uppercase tracking-wider text-text-secondary">
            Monitored Merchants
          </span>
          <span className="text-lg">🏪</span>
        </div>
        <div className="mt-3">
          <div className="text-3xl font-bold text-text-primary">{totalMerchants}</div>
          <span className="text-xs text-text-muted mt-1 block">Active merchant profiles</span>
        </div>
      </div>

      {/* Card 2: Total Detection Windows */}
      <div className="bg-bg-card border border-border rounded-xl p-5 shadow-sm hover:border-text-muted transition-colors flex flex-col justify-between">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold uppercase tracking-wider text-text-secondary">
            Detection Windows
          </span>
          <span className="text-lg">⏱️</span>
        </div>
        <div className="mt-3">
          <div className="text-3xl font-bold text-text-primary">{totalWindows}</div>
          <span className="text-xs text-text-muted mt-1 block">Hourly window partitions</span>
        </div>
      </div>

      {/* Card 3: Active Flagged Fraud Spikes */}
      <div className="bg-bg-card border border-border rounded-xl p-5 shadow-sm hover:border-text-muted transition-colors flex flex-col justify-between">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold uppercase tracking-wider text-text-secondary">
            Flagged Anomalies
          </span>
          <span className="text-lg">🚨</span>
        </div>
        <div className="mt-3">
          <div className="flex items-baseline gap-2">
            <span className={`text-3xl font-bold ${totalFlaggedSpikes > 0 ? 'text-status-critical' : 'text-status-low'}`}>
              {totalFlaggedSpikes}
            </span>
            {totalFlaggedSpikes > 0 && (
              <span className="px-2 py-0.5 bg-status-critical/15 text-status-critical text-xs font-semibold rounded-full">
                Requires Review
              </span>
            )}
          </div>
          <span className="text-xs text-text-muted mt-1 block">Persisted detector flags</span>
        </div>
      </div>

      {/* Card 4: Total Monitored Volume */}
      <div className="bg-bg-card border border-border rounded-xl p-5 shadow-sm hover:border-text-muted transition-colors flex flex-col justify-between">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold uppercase tracking-wider text-text-secondary">
            Monitored Volume
          </span>
          <span className="text-lg">💰</span>
        </div>
        <div className="mt-3">
          <div className="text-2xl font-bold text-text-primary truncate">{formattedVolume}</div>
          <span className="text-xs text-text-muted mt-1 block">Total synthetic transaction size</span>
        </div>
      </div>
    </div>
  )
}
