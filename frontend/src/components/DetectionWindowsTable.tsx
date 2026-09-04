import type { DetectionWindow, MerchantSummary } from '../types'

interface DetectionWindowsTableProps {
  windows: DetectionWindow[]
  loading: boolean
  error: string | null
  merchants: MerchantSummary[]
  selectedMerchant: string
  selectedSplit: string
  flaggedOnly: boolean
  page: number
  limit: number
  onMerchantChange: (merchantId: string) => void
  onSplitChange: (split: string) => void
  onFlaggedToggle: (flaggedOnly: boolean) => void
  onPageChange: (page: number) => void
  onInspectWindow?: (windowId: number) => void
}

export function DetectionWindowsTable({
  windows,
  loading,
  error,
  merchants,
  selectedMerchant,
  selectedSplit,
  flaggedOnly,
  page,
  limit,
  onMerchantChange,
  onSplitChange,
  onFlaggedToggle,
  onPageChange,
  onInspectWindow,
}: DetectionWindowsTableProps) {
  const formatCurrency = (amount: number) =>
    new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      maximumFractionDigits: 0,
    }).format(amount)

  const formatDate = (isoString: string) =>
    new Date(isoString).toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      hour12: false,
    })

  return (
    <div className="bg-bg-card border border-border rounded-xl p-6 shadow-md space-y-4">
      {/* Table Header & Interactive Filter Bar */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-border pb-4">
        <div>
          <h3 className="text-lg font-bold text-text-primary flex items-center gap-2">
            <span>📋 Detection Windows Stream</span>
            {loading && <span className="w-3 h-3 border-2 border-accent border-t-transparent rounded-full animate-spin" />}
          </h3>
          <p className="text-xs text-text-secondary">
            Filter, inspect, and analyze hourly detection window partitions
          </p>
        </div>

        {/* Filter Controls */}
        <div className="flex flex-wrap items-center gap-3">
          {/* Merchant Filter */}
          <div className="flex items-center gap-1.5">
            <label htmlFor="filter-merchant" className="text-xs text-text-muted font-medium">
              Merchant:
            </label>
            <select
              id="filter-merchant"
              value={selectedMerchant}
              onChange={(e) => onMerchantChange(e.target.value)}
              className="bg-bg-primary border border-border rounded-lg px-2.5 py-1 text-xs text-text-primary focus:outline-none focus:border-accent"
            >
              <option value="">All Merchants</option>
              {merchants.map((m) => (
                <option key={m.merchant_id} value={m.merchant_id}>
                  {m.merchant_id}
                </option>
              ))}
            </select>
          </div>

          {/* Split Filter */}
          <div className="flex items-center gap-1.5">
            <label htmlFor="filter-split" className="text-xs text-text-muted font-medium">
              Split:
            </label>
            <select
              id="filter-split"
              value={selectedSplit}
              onChange={(e) => onSplitChange(e.target.value)}
              className="bg-bg-primary border border-border rounded-lg px-2.5 py-1 text-xs text-text-primary focus:outline-none focus:border-accent"
            >
              <option value="">All Permitted</option>
              <option value="dev_test">dev_test</option>
              <option value="train">train</option>
            </select>
          </div>

          {/* Flagged Only Filter */}
          <label className="flex items-center gap-2 cursor-pointer bg-bg-primary/50 border border-border rounded-lg px-2.5 py-1 text-xs text-text-primary font-medium hover:border-text-muted transition-colors">
            <input
              type="checkbox"
              checked={flaggedOnly}
              onChange={(e) => onFlaggedToggle(e.target.checked)}
              className="rounded bg-bg-secondary border-border text-accent focus:ring-0 accent-accent"
            />
            <span>Flagged Only</span>
          </label>
        </div>
      </div>

      {/* Main Table Container */}
      {error ? (
        <div className="bg-status-critical/10 border border-status-critical/30 rounded-lg p-4 text-center">
          <p className="text-status-critical font-semibold text-sm">❌ Unable to query detection windows</p>
          <p className="text-xs text-text-secondary mt-1">{error}</p>
        </div>
      ) : loading && windows.length === 0 ? (
        <div className="py-12 flex flex-col items-center gap-3 text-text-secondary text-sm">
          <div className="w-6 h-6 border-2 border-accent border-t-transparent rounded-full animate-spin" />
          <span>Fetching detection windows...</span>
        </div>
      ) : windows.length === 0 ? (
        <div className="py-12 text-center text-text-muted text-sm space-y-1">
          <p className="font-semibold text-text-secondary">No detection windows match the selected filters.</p>
          <p className="text-xs">Try clearing merchant or split filters.</p>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-border text-text-secondary font-semibold uppercase tracking-wider bg-bg-primary/30">
                <th className="py-3 px-3">ID</th>
                <th className="py-3 px-3">Merchant</th>
                <th className="py-3 px-3">Window Start / End</th>
                <th className="py-3 px-3 text-right">Transactions</th>
                <th className="py-3 px-3 text-right">Total Volume</th>
                <th className="py-3 px-3 text-right">Avg Size</th>
                <th className="py-3 px-3 text-center">Split</th>
                <th className="py-3 px-3 text-center">Ground Truth</th>
                <th className="py-3 px-3 text-center">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/60">
              {windows.map((w) => (
                <tr key={w.id} className="hover:bg-bg-card-hover/40 transition-colors">
                  <td className="py-2.5 px-3 font-mono text-text-muted font-medium">#{w.id}</td>
                  <td className="py-2.5 px-3 font-medium text-text-primary">{w.merchant_id}</td>
                  <td className="py-2.5 px-3 text-text-secondary">
                    {formatDate(w.window_start)} – {formatDate(w.window_end).split(', ')[1] || ''}
                  </td>
                  <td className="py-2.5 px-3 text-right font-mono font-medium text-text-primary">
                    {w.transaction_count}
                  </td>
                  <td className="py-2.5 px-3 text-right font-mono font-medium text-text-primary">
                    {formatCurrency(w.total_amount)}
                  </td>
                  <td className="py-2.5 px-3 text-right font-mono text-text-secondary">
                    {formatCurrency(w.avg_transaction_amount)}
                  </td>
                  <td className="py-2.5 px-3 text-center">
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] font-semibold uppercase ${
                        w.split === 'dev_test'
                          ? 'bg-accent/15 text-accent border border-accent/30'
                          : 'bg-text-muted/15 text-text-secondary border border-border'
                      }`}
                    >
                      {w.split}
                    </span>
                  </td>
                  <td className="py-2.5 px-3 text-center">
                    {w.is_synthetic_fraud_spike !== undefined ? (
                      w.is_synthetic_fraud_spike ? (
                        <span className="px-2 py-0.5 bg-status-medium/15 text-status-medium border border-status-medium/30 rounded text-[10px] font-semibold uppercase">
                          ⚡ Spike
                        </span>
                      ) : (
                        <span className="px-2 py-0.5 bg-status-low/15 text-status-low border border-status-low/30 rounded text-[10px] font-semibold uppercase">
                          Normal
                        </span>
                      )
                    ) : (
                      <span className="text-text-muted text-[11px]">—</span>
                    )}
                  </td>
                  <td className="py-2.5 px-3 text-center">
                    <button
                      onClick={() => onInspectWindow && onInspectWindow(w.id)}
                      className="px-2.5 py-1 bg-bg-primary hover:bg-accent hover:text-white border border-border rounded text-[11px] font-semibold text-text-primary transition-all shadow-2xs"
                    >
                      Inspect
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Pagination Footer */}
      <div className="flex items-center justify-between pt-2 border-t border-border text-xs text-text-secondary">
        <div>
          Showing page <span className="font-semibold text-text-primary">{page}</span> ({windows.length} windows)
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => onPageChange(Math.max(1, page - 1))}
            disabled={page === 1 || loading}
            className="px-3 py-1 bg-bg-primary border border-border rounded disabled:opacity-40 disabled:cursor-not-allowed hover:border-text-muted font-medium transition-colors"
          >
            ← Previous
          </button>
          <button
            onClick={() => onPageChange(page + 1)}
            disabled={windows.length < limit || loading}
            className="px-3 py-1 bg-bg-primary border border-border rounded disabled:opacity-40 disabled:cursor-not-allowed hover:border-text-muted font-medium transition-colors"
          >
            Next →
          </button>
        </div>
      </div>
    </div>
  )
}
