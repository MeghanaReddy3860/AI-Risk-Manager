import {
  Bar,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { MerchantSummary, TimeseriesPoint } from '../types'

interface MerchantTimeseriesChartProps {
  merchantId: string
  merchants: MerchantSummary[]
  onSelectMerchant: (id: string) => void
  data: TimeseriesPoint[]
  loading: boolean
  error: string | null
}

interface CustomTooltipProps {
  active?: boolean
  payload?: Array<{
    name: string
    value: number
    payload: TimeseriesPoint
  }>
  label?: string
}

function CustomTooltip({ active, payload }: CustomTooltipProps) {
  if (!active || !payload || !payload.length) return null

  const point = payload[0].payload
  const formattedDate = new Date(point.timestamp).toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })

  const formattedAmount = new Intl.NumberFormat('en-IN', {
    style: 'currency',
    currency: 'INR',
  }).format(point.total_amount)

  const formattedAvg = new Intl.NumberFormat('en-IN', {
    style: 'currency',
    currency: 'INR',
  }).format(point.avg_transaction_amount)

  return (
    <div className="bg-bg-secondary border border-border rounded-lg p-3 shadow-xl text-xs space-y-1.5 z-50">
      <div className="font-semibold text-text-primary border-b border-border pb-1">
        {formattedDate}
      </div>

      <div className="flex justify-between gap-4 text-text-secondary">
        <span>Transactions:</span>
        <span className="font-medium text-text-primary">{point.transaction_count}</span>
      </div>

      <div className="flex justify-between gap-4 text-text-secondary">
        <span>Total Amount:</span>
        <span className="font-medium text-text-primary">{formattedAmount}</span>
      </div>

      <div className="flex justify-between gap-4 text-text-secondary">
        <span>Avg Size:</span>
        <span className="font-medium text-text-primary">{formattedAvg}</span>
      </div>

      <div className="pt-1 flex items-center justify-between">
        <span className="text-text-muted">Detector Status:</span>
        {point.is_flagged ? (
          <span className="px-2 py-0.5 bg-status-critical/20 text-status-critical font-bold rounded text-[10px] uppercase">
            🚨 Flagged Anomaly
          </span>
        ) : (
          <span className="px-2 py-0.5 bg-status-low/20 text-status-low font-semibold rounded text-[10px] uppercase">
            ✅ Normal
          </span>
        )}
      </div>
    </div>
  )
}

interface CustomDotProps {
  cx?: number
  cy?: number
  payload?: TimeseriesPoint
}

function CustomFlagDot({ cx, cy, payload }: CustomDotProps) {
  if (!cx || !cy || !payload || !payload.is_flagged) return null

  return (
    <g>
      <circle cx={cx} cy={cy} r={7} fill="#ef4444" opacity={0.4} className="animate-ping" />
      <circle cx={cx} cy={cy} r={5} fill="#ef4444" stroke="#ffffff" strokeWidth={1.5} />
    </g>
  )
}

export function MerchantTimeseriesChart({
  merchantId,
  merchants,
  onSelectMerchant,
  data,
  loading,
  error,
}: MerchantTimeseriesChartProps) {
  const chartData = data.map((pt) => ({
    ...pt,
    timeLabel: new Date(pt.timestamp).toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      hour12: false,
    }),
  }))

  return (
    <div className="bg-bg-card border border-border rounded-xl p-6 shadow-md space-y-4">
      {/* Chart Header & Merchant Selector */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-border pb-4">
        <div>
          <h3 className="text-lg font-bold text-text-primary flex items-center gap-2">
            <span>📈 Merchant Activity Stream</span>
            {loading && <span className="w-3 h-3 border-2 border-accent border-t-transparent rounded-full animate-spin" />}
          </h3>
          <p className="text-xs text-text-secondary">
            Hourly transaction count & monetary volume timeseries stream
          </p>
        </div>

        {/* Merchant Dropdown */}
        <div className="flex items-center gap-2">
          <label htmlFor="merchant-select" className="text-xs font-semibold text-text-secondary whitespace-nowrap">
            Select Merchant:
          </label>
          <select
            id="merchant-select"
            value={merchantId}
            onChange={(e) => onSelectMerchant(e.target.value)}
            disabled={loading || merchants.length === 0}
            className="bg-bg-primary border border-border rounded-lg px-3 py-1.5 text-xs text-text-primary font-medium focus:outline-none focus:border-accent transition-colors"
          >
            {merchants.map((m) => (
              <option key={m.merchant_id} value={m.merchant_id}>
                {m.merchant_id} ({m.total_windows} windows, {m.flagged_anomaly_count} flagged)
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Main Chart Content / Loading / Error / Empty States */}
      <div className="h-72 w-full flex items-center justify-center">
        {loading ? (
          <div className="flex flex-col items-center gap-2 text-text-secondary text-sm">
            <div className="w-6 h-6 border-2 border-accent border-t-transparent rounded-full animate-spin" />
            <span>Loading transaction stream for {merchantId}...</span>
          </div>
        ) : error ? (
          <div className="bg-status-critical/10 border border-status-critical/30 rounded-lg p-4 text-center max-w-md">
            <p className="text-status-critical font-semibold text-sm">❌ Unable to load transaction stream</p>
            <p className="text-xs text-text-secondary mt-1">{error}</p>
          </div>
        ) : chartData.length === 0 ? (
          <div className="text-center text-text-muted text-sm py-8">
            ℹ️ No transaction data available for merchant {merchantId}.
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.5} />
              <XAxis dataKey="timeLabel" stroke="#94a3b8" fontSize={11} tickLine={false} />
              <YAxis
                yAxisId="left"
                stroke="#94a3b8"
                fontSize={11}
                tickLine={false}
                label={{ value: 'Transactions', angle: -90, position: 'insideLeft', style: { fill: '#94a3b8', fontSize: 10 } }}
              />
              <YAxis
                yAxisId="right"
                orientation="right"
                stroke="#94a3b8"
                fontSize={11}
                tickLine={false}
                tickFormatter={(val: number) => `₹${(val / 1000).toFixed(0)}k`}
                label={{ value: 'Volume (₹)', angle: 90, position: 'insideRight', style: { fill: '#94a3b8', fontSize: 10 } }}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
              <Bar yAxisId="right" dataKey="total_amount" name="Total Volume (₹)" fill="#3b82f6" opacity={0.3} radius={[4, 4, 0, 0]} />
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="transaction_count"
                name="Transaction Count"
                stroke="#22c55e"
                strokeWidth={2}
                dot={<CustomFlagDot />}
                activeDot={{ r: 6, fill: '#22c55e' }}
              />
            </ComposedChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  )
}
