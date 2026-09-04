import { useEffect, useState } from 'react'
import {
  Bar,
  CartesianGrid,
  Legend,
  BarChart as RechartsBarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { getLatestEvaluations, triggerEvaluation } from '../api'
import type { EvaluationRun } from '../types'

interface CustomTooltipProps {
  active?: boolean
  payload?: Array<{
    name: string
    value: number
    dataKey: string
  }>
  label?: string
}

function MetricTooltip({ active, payload, label }: CustomTooltipProps) {
  if (!active || !payload || !payload.length) return null

  return (
    <div className="bg-bg-secondary border border-border rounded-lg p-3 shadow-xl text-xs space-y-1 z-50">
      <div className="font-bold text-text-primary border-b border-border pb-1 mb-1">{label}</div>
      {payload.map((entry) => (
        <div key={entry.name} className="flex justify-between gap-4">
          <span className="text-text-secondary">{entry.name}:</span>
          <span className="font-mono font-bold text-text-primary">
            {entry.value.toFixed(2)}%
          </span>
        </div>
      ))}
    </div>
  )
}

export function EvaluationView() {
  const [evalRuns, setEvalRuns] = useState<EvaluationRun[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const [isEvaluating, setIsEvaluating] = useState(false)
  const [actionFeedback, setActionFeedback] = useState<{
    type: 'success' | 'error' | 'info'
    message: string
  } | null>(null)

  // Load latest evaluation runs on mount
  useEffect(() => {
    let isMounted = true
    getLatestEvaluations()
      .then((data) => {
        if (isMounted) {
          setEvalRuns(data)
          setLoading(false)
        }
      })
      .catch((err: unknown) => {
        if (isMounted) {
          const msg = err instanceof Error ? err.message : 'Unable to load evaluation metrics'
          setError(msg)
          setLoading(false)
        }
      })

    return () => {
      isMounted = false
    }
  }, [])

  // Explicit user-triggered evaluation runner
  const handleRunEvaluation = async () => {
    if (isEvaluating) return

    setIsEvaluating(true)
    setActionFeedback({
      type: 'info',
      message: 'Evaluating Baseline and ML detectors against dev_test ground-truth partition... Please wait.',
    })

    try {
      const newRuns = await triggerEvaluation('dev_test')
      setEvalRuns(newRuns)
      setActionFeedback({
        type: 'success',
        message: `Evaluation completed successfully for dev_test partition! Updated ${newRuns.length} detector benchmarks.`,
      })
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Evaluation trigger failed'
      setActionFeedback({
        type: 'error',
        message: `Failed to run evaluation: ${msg}. No model hyperparameters or database schemas were altered.`,
      })
    } finally {
      setIsEvaluating(false)
    }
  }

  const formatCurrency = (val: number) =>
    new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      maximumFractionDigits: 0,
    }).format(val)

  const formatDate = (isoString: string) =>
    new Date(isoString).toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      hour12: false,
    })

  // Extract latest run for baseline and ML
  const baselineRun = evalRuns.find((r) => r.detector_type === 'baseline')
  const mlRun = evalRuns.find((r) => r.detector_type === 'ml')

  // Prepare data for percentages grouped chart
  const chartData = [
    {
      metric: 'Precision',
      Baseline: (baselineRun?.precision ?? 0) * 100,
      'ML Isolation Forest': (mlRun?.precision ?? 0) * 100,
    },
    {
      metric: 'Recall',
      Baseline: (baselineRun?.recall ?? 0) * 100,
      'ML Isolation Forest': (mlRun?.recall ?? 0) * 100,
    },
    {
      metric: 'F1 Score',
      Baseline: (baselineRun?.f1_score ?? 0) * 100,
      'ML Isolation Forest': (mlRun?.f1_score ?? 0) * 100,
    },
    {
      metric: 'False Positive Rate (FPR)',
      Baseline: (baselineRun?.false_positive_rate ?? 0) * 100,
      'ML Isolation Forest': (mlRun?.false_positive_rate ?? 0) * 100,
    },
  ]

  return (
    <div className="space-y-6">
      {/* Action Feedback Banner */}
      {actionFeedback && (
        <div
          className={`p-4 border rounded-xl flex items-center justify-between text-xs font-medium ${
            actionFeedback.type === 'success'
              ? 'bg-status-low/10 border-status-low/30 text-status-low'
              : actionFeedback.type === 'error'
              ? 'bg-status-critical/10 border-status-critical/30 text-status-critical'
              : 'bg-accent/10 border-accent/30 text-accent'
          }`}
        >
          <div className="flex items-center gap-2">
            <span>{actionFeedback.type === 'success' ? '✅' : actionFeedback.type === 'error' ? '❌' : 'ℹ️'}</span>
            <span>{actionFeedback.message}</span>
          </div>
          <button
            onClick={() => setActionFeedback(null)}
            className="text-text-muted hover:text-text-primary px-2 py-0.5 rounded"
          >
            ✕
          </button>
        </div>
      )}

      {/* Header Context Card */}
      <div className="bg-bg-card border border-border rounded-xl p-6 shadow-md space-y-4">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-border pb-4">
          <div>
            <h2 className="text-xl font-bold text-text-primary flex items-center gap-2">
              <span>📈 Offline Evaluation Benchmarks</span>
              {loading && <span className="w-3.5 h-3.5 border-2 border-accent border-t-transparent rounded-full animate-spin" />}
            </h2>
            <p className="text-xs text-text-secondary mt-1">
              Statistical baseline vs ML Isolation Forest classification & financial cost metrics on dev_test partition
            </p>
          </div>

          {/* Trigger Evaluation Action Button */}
          <button
            onClick={handleRunEvaluation}
            disabled={isEvaluating}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold text-white shadow-sm transition-all ${
              isEvaluating
                ? 'bg-bg-card-hover cursor-not-allowed opacity-60 text-text-muted'
                : 'bg-accent hover:bg-accent-hover active:scale-98'
            }`}
          >
            {isEvaluating ? (
              <>
                <span className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                <span>Evaluating dev_test...</span>
              </>
            ) : (
              <>
                <span>⚡</span>
                <span>Run Model Evaluation</span>
              </>
            )}
          </button>
        </div>

        {/* Evaluation Metadata / Partition Context */}
        {evalRuns.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs bg-bg-primary/50 p-4 rounded-lg border border-border">
            <div>
              <span className="text-text-muted block font-medium">Evaluation Dataset Partition:</span>
              <span className="font-mono font-bold text-accent uppercase">
                {evalRuns[0].partition} (Permitted Split)
              </span>
            </div>
            <div>
              <span className="text-text-muted block font-medium">Last Benchmark Execution:</span>
              <span className="font-mono font-semibold text-text-primary">
                {formatDate(evalRuns[0].run_timestamp)}
              </span>
            </div>
            <div>
              <span className="text-text-muted block font-medium">Operational Cost Assumptions:</span>
              <span className="text-text-secondary truncate block" title={evalRuns[0].notes}>
                {evalRuns[0].notes}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Main Benchmarks Content / Loading / Error / Empty States */}
      {loading ? (
        <div className="bg-bg-card border border-border rounded-xl p-12 text-center flex flex-col items-center justify-center gap-3">
          <div className="w-8 h-8 border-2 border-accent border-t-transparent rounded-full animate-spin" />
          <p className="text-text-secondary text-sm font-medium">Loading evaluation benchmarks...</p>
        </div>
      ) : error ? (
        <div className="bg-status-critical/10 border border-status-critical/30 rounded-xl p-6 text-center">
          <p className="text-status-critical font-bold text-sm">❌ Unable to load evaluation benchmarks</p>
          <p className="text-xs text-text-secondary mt-1">{error}</p>
        </div>
      ) : evalRuns.length === 0 ? (
        <div className="bg-bg-card border border-border rounded-xl p-12 text-center space-y-3">
          <div className="w-12 h-12 bg-accent/10 border border-accent/30 text-accent rounded-full flex items-center justify-center mx-auto text-xl">
            📈
          </div>
          <h3 className="text-lg font-bold text-text-primary">No Evaluation Results Available</h3>
          <p className="text-xs text-text-secondary max-w-md mx-auto">
            No persisted benchmark runs exist for the dev_test evaluation partition. Click &quot;Run Model Evaluation&quot; above to compute baseline and ML performance metrics.
          </p>
        </div>
      ) : (
        <>
          {/* Side-by-Side Detailed Benchmark Comparison Table */}
          <div className="bg-bg-card border border-border rounded-xl p-6 shadow-md space-y-4">
            <h3 className="text-sm font-bold text-text-primary uppercase tracking-wider border-b border-border pb-2 flex items-center justify-between">
              <span>📊 Performance Metrics Comparison</span>
              <span className="text-xs font-normal text-text-muted">dev_test Ground-Truth Labeled Data</span>
            </h3>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="border-b border-border text-text-secondary font-semibold uppercase tracking-wider bg-bg-primary/30">
                    <th className="py-3 px-4">Metric</th>
                    <th className="py-3 px-4 text-center">Optimal Direction</th>
                    <th className="py-3 px-4 text-right font-bold text-text-primary">Baseline Detector</th>
                    <th className="py-3 px-4 text-right font-bold text-accent">ML Isolation Forest</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/60">
                  {/* Precision */}
                  <tr className="hover:bg-bg-card-hover/40 transition-colors">
                    <td className="py-3 px-4 font-semibold text-text-primary">
                      Precision
                      <span className="text-text-muted font-normal block text-[11px]">
                        Fraction of flagged anomalies that were true fraud spikes
                      </span>
                    </td>
                    <td className="py-3 px-4 text-center text-status-low font-semibold text-[11px]">
                      Higher is better ⬆️
                    </td>
                    <td className="py-3 px-4 text-right font-mono text-sm font-bold text-text-primary">
                      {baselineRun ? `${(baselineRun.precision * 100).toFixed(2)}%` : 'N/A'}
                    </td>
                    <td className="py-3 px-4 text-right font-mono text-sm font-bold text-accent">
                      {mlRun ? `${(mlRun.precision * 100).toFixed(2)}%` : 'N/A'}
                    </td>
                  </tr>

                  {/* Recall */}
                  <tr className="hover:bg-bg-card-hover/40 transition-colors">
                    <td className="py-3 px-4 font-semibold text-text-primary">
                      Recall (Sensitivity)
                      <span className="text-text-muted font-normal block text-[11px]">
                        Fraction of actual fraud spikes successfully detected
                      </span>
                    </td>
                    <td className="py-3 px-4 text-center text-status-low font-semibold text-[11px]">
                      Higher is better ⬆️
                    </td>
                    <td className="py-3 px-4 text-right font-mono text-sm font-bold text-text-primary">
                      {baselineRun ? `${(baselineRun.recall * 100).toFixed(2)}%` : 'N/A'}
                    </td>
                    <td className="py-3 px-4 text-right font-mono text-sm font-bold text-accent">
                      {mlRun ? `${(mlRun.recall * 100).toFixed(2)}%` : 'N/A'}
                    </td>
                  </tr>

                  {/* F1 Score */}
                  <tr className="hover:bg-bg-card-hover/40 transition-colors">
                    <td className="py-3 px-4 font-semibold text-text-primary">
                      F1 Score
                      <span className="text-text-muted font-normal block text-[11px]">
                        Harmonic mean of Precision and Recall
                      </span>
                    </td>
                    <td className="py-3 px-4 text-center text-status-low font-semibold text-[11px]">
                      Higher is better ⬆️
                    </td>
                    <td className="py-3 px-4 text-right font-mono text-sm font-bold text-text-primary">
                      {baselineRun ? `${(baselineRun.f1_score * 100).toFixed(2)}%` : 'N/A'}
                    </td>
                    <td className="py-3 px-4 text-right font-mono text-sm font-bold text-accent">
                      {mlRun ? `${(mlRun.f1_score * 100).toFixed(2)}%` : 'N/A'}
                    </td>
                  </tr>

                  {/* False Positive Rate */}
                  <tr className="hover:bg-bg-card-hover/40 transition-colors">
                    <td className="py-3 px-4 font-semibold text-text-primary">
                      False Positive Rate (FPR)
                      <span className="text-text-muted font-normal block text-[11px]">
                        Fraction of normal windows incorrectly flagged
                      </span>
                    </td>
                    <td className="py-3 px-4 text-center text-status-medium font-semibold text-[11px]">
                      Lower is better ⬇️
                    </td>
                    <td className="py-3 px-4 text-right font-mono text-sm font-bold text-text-primary">
                      {baselineRun ? `${(baselineRun.false_positive_rate * 100).toFixed(2)}%` : 'N/A'}
                    </td>
                    <td className="py-3 px-4 text-right font-mono text-sm font-bold text-accent">
                      {mlRun ? `${(mlRun.false_positive_rate * 100).toFixed(2)}%` : 'N/A'}
                    </td>
                  </tr>

                  {/* False Positive Cost */}
                  <tr className="hover:bg-bg-card-hover/40 transition-colors">
                    <td className="py-3 px-4 font-semibold text-text-primary">
                      False Positive Cost
                      <span className="text-text-muted font-normal block text-[11px]">
                        Total financial cost of unnecessary analyst triage reviews
                      </span>
                    </td>
                    <td className="py-3 px-4 text-center text-status-medium font-semibold text-[11px]">
                      Lower is better ⬇️
                    </td>
                    <td className="py-3 px-4 text-right font-mono text-sm font-bold text-text-primary">
                      {baselineRun ? formatCurrency(baselineRun.fp_cost) : 'N/A'}
                    </td>
                    <td className="py-3 px-4 text-right font-mono text-sm font-bold text-accent">
                      {mlRun ? formatCurrency(mlRun.fp_cost) : 'N/A'}
                    </td>
                  </tr>

                  {/* Total Operational Cost */}
                  <tr className="hover:bg-bg-card-hover/40 transition-colors bg-bg-primary/20 font-bold">
                    <td className="py-3 px-4 text-text-primary">
                      Total Operational Cost
                      <span className="text-text-muted font-normal block text-[11px]">
                        Combined cost of False Positives + False Negatives
                      </span>
                    </td>
                    <td className="py-3 px-4 text-center text-status-medium text-[11px]">
                      Lower is better ⬇️
                    </td>
                    <td className="py-3 px-4 text-right font-mono text-sm text-text-primary">
                      {baselineRun ? formatCurrency(baselineRun.total_cost) : 'N/A'}
                    </td>
                    <td className="py-3 px-4 text-right font-mono text-sm text-accent">
                      {mlRun ? formatCurrency(mlRun.total_cost) : 'N/A'}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* Grouped Bar Chart Visualizing Classification Metrics */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 bg-bg-card border border-border rounded-xl p-6 shadow-md space-y-4">
              <h3 className="text-sm font-bold text-text-primary uppercase tracking-wider border-b border-border pb-2 flex items-center justify-between">
                <span>📊 Classification Metrics Comparison</span>
                <span className="text-xs text-text-muted font-normal">Percentage Scale (0-100%)</span>
              </h3>

              <div className="h-72 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <RechartsBarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.5} />
                    <XAxis dataKey="metric" stroke="#94a3b8" fontSize={11} tickLine={false} />
                    <YAxis stroke="#94a3b8" fontSize={11} tickLine={false} unit="%" domain={[0, 100]} />
                    <Tooltip content={<MetricTooltip />} />
                    <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
                    <Bar dataKey="Baseline" fill="#94a3b8" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="ML Isolation Forest" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                  </RechartsBarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Confusion Matrix Summary Cards */}
            <div className="bg-bg-card border border-border rounded-xl p-6 shadow-md space-y-4 flex flex-col justify-between">
              <h3 className="text-sm font-bold text-text-primary uppercase tracking-wider border-b border-border pb-2">
                <span>📋 Confusion Matrix Counts</span>
              </h3>

              <div className="space-y-4 text-xs">
                {/* Baseline Matrix */}
                <div className="bg-bg-primary/50 border border-border rounded-lg p-3 space-y-2">
                  <span className="font-bold text-text-primary block border-b border-border/60 pb-1">
                    Baseline Detector
                  </span>
                  <div className="grid grid-cols-2 gap-2 text-[11px]">
                    <div>TP: <span className="font-mono font-bold text-status-low">{baselineRun?.true_positives ?? 0}</span></div>
                    <div>FP: <span className="font-mono font-bold text-status-critical">{baselineRun?.false_positives ?? 0}</span></div>
                    <div>FN: <span className="font-mono font-bold text-status-medium">{baselineRun?.false_negatives ?? 0}</span></div>
                    <div>TN: <span className="font-mono font-bold text-text-secondary">{baselineRun?.true_negatives ?? 0}</span></div>
                  </div>
                </div>

                {/* ML Matrix */}
                <div className="bg-bg-primary/50 border border-border rounded-lg p-3 space-y-2">
                  <span className="font-bold text-accent block border-b border-border/60 pb-1">
                    ML Isolation Forest
                  </span>
                  <div className="grid grid-cols-2 gap-2 text-[11px]">
                    <div>TP: <span className="font-mono font-bold text-status-low">{mlRun?.true_positives ?? 0}</span></div>
                    <div>FP: <span className="font-mono font-bold text-status-critical">{mlRun?.false_positives ?? 0}</span></div>
                    <div>FN: <span className="font-mono font-bold text-status-medium">{mlRun?.false_negatives ?? 0}</span></div>
                    <div>TN: <span className="font-mono font-bold text-text-secondary">{mlRun?.true_negatives ?? 0}</span></div>
                  </div>
                </div>
              </div>

              <div className="text-[11px] text-text-muted border-t border-border pt-3 leading-relaxed">
                <span>TP = True Positives • FP = False Positives • FN = False Negatives • TN = True Negatives</span>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
