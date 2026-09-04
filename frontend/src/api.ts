import type {
  AnalystActionRequest,
  AnalystActionResponse,
  AnomalyDetection,
  AuditIntegrityVerification,
  AuditReport,
  DetectionWindow,
  DetectorRunSummary,
  EvaluationRun,
  HealthStatus,
  MerchantSummary,
  RiskDossier,
  TimeseriesPoint,
} from './types'

const BASE_URL = '/api'

export async function getHealth(): Promise<HealthStatus> {
  const res = await fetch(`${BASE_URL}/health`)
  if (!res.ok) throw new Error(`Health check failed: HTTP ${res.status}`)
  return res.json()
}

export async function getMerchants(): Promise<MerchantSummary[]> {
  const res = await fetch(`${BASE_URL}/merchants`)
  if (!res.ok) throw new Error(`Failed to fetch merchants: HTTP ${res.status}`)
  return res.json()
}

export async function getWindows(params?: {
  merchant_id?: string
  split?: string
  is_flagged?: boolean
  limit?: number
  offset?: number
}): Promise<DetectionWindow[]> {
  const query = new URLSearchParams()
  if (params?.merchant_id) query.append('merchant_id', params.merchant_id)
  if (params?.split) query.append('split', params.split)
  if (params?.is_flagged !== undefined) query.append('is_flagged', String(params.is_flagged))
  if (params?.limit !== undefined) query.append('limit', String(params.limit))
  if (params?.offset !== undefined) query.append('offset', String(params.offset))

  const res = await fetch(`${BASE_URL}/windows?${query.toString()}`)
  if (!res.ok) throw new Error(`Failed to fetch windows: HTTP ${res.status}`)
  return res.json()
}

export async function getWindow(windowId: number): Promise<DetectionWindow> {
  const res = await fetch(`${BASE_URL}/windows/${windowId}`)
  if (!res.ok) throw new Error(`Failed to fetch window ${windowId}: HTTP ${res.status}`)
  return res.json()
}

export async function getWindowDetections(windowId: number): Promise<AnomalyDetection[]> {
  const res = await fetch(`${BASE_URL}/windows/${windowId}/detections`)
  if (!res.ok) throw new Error(`Failed to fetch detections for window ${windowId}: HTTP ${res.status}`)
  return res.json()
}

export async function runDetectors(detectorType?: 'baseline' | 'ml'): Promise<DetectorRunSummary> {
  const query = detectorType ? `?detector_type=${detectorType}` : ''
  const res = await fetch(`${BASE_URL}/pipeline/run-detectors${query}`, {
    method: 'POST',
  })
  if (!res.ok) throw new Error(`Failed to run detectors: HTTP ${res.status}`)
  return res.json()
}

export async function analyzeWindow(
  windowId: number,
  detectorType: 'baseline' | 'ml' = 'baseline',
  useLlm: boolean = false
): Promise<RiskDossier> {
  const res = await fetch(
    `${BASE_URL}/pipeline/analyze-window/${windowId}?detector_type=${detectorType}&use_llm=${useLlm}`,
    { method: 'POST' }
  )
  if (!res.ok) throw new Error(`Failed to analyze window ${windowId}: HTTP ${res.status}`)
  return res.json()
}

export async function getWindowAnalysis(
  windowId: number,
  detectorType: 'baseline' | 'ml' = 'baseline',
  useLlm: boolean = false
): Promise<RiskDossier> {
  const res = await fetch(
    `${BASE_URL}/windows/${windowId}/analysis?detector_type=${detectorType}&use_llm=${useLlm}`
  )
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }))
    throw new Error(errorData.detail || `Failed to fetch analysis for window ${windowId}`)
  }
  return res.json()
}

export async function getMerchantTimeseries(
  merchantId: string,
  limit: number = 100
): Promise<TimeseriesPoint[]> {
  const res = await fetch(`${BASE_URL}/merchants/${merchantId}/timeseries?limit=${limit}`)
  if (!res.ok) throw new Error(`Failed to fetch timeseries for merchant ${merchantId}: HTTP ${res.status}`)
  return res.json()
}

export async function getLatestEvaluations(): Promise<EvaluationRun[]> {
  const res = await fetch(`${BASE_URL}/evaluation/latest`)
  if (!res.ok) throw new Error(`Failed to fetch evaluation runs: HTTP ${res.status}`)
  return res.json()
}

export async function triggerEvaluation(partition: string = 'dev_test'): Promise<EvaluationRun[]> {
  const res = await fetch(`${BASE_URL}/evaluation/run?partition=${partition}`, {
    method: 'POST',
  })
  if (!res.ok) {
    const errData = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }))
    throw new Error(errData.detail || `Failed to run evaluation: HTTP ${res.status}`)
  }
  return res.json()
}

export async function getWindowAuditReport(windowId: number): Promise<AuditReport> {
  const res = await fetch(`${BASE_URL}/audit/window/${windowId}`)
  if (!res.ok) {
    const errData = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }))
    throw new Error(errData.detail || `Failed to fetch audit report for window ${windowId}: HTTP ${res.status}`)
  }
  return res.json()
}

export async function verifyAuditIntegrity(): Promise<AuditIntegrityVerification> {
  const res = await fetch(`${BASE_URL}/audit/verify`)
  if (!res.ok) throw new Error(`Failed to verify audit integrity: HTTP ${res.status}`)
  return res.json()
}

export async function recordAnalystAction(action: AnalystActionRequest): Promise<AnalystActionResponse> {
  const res = await fetch(`${BASE_URL}/analyst/action`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(action),
  })
  if (!res.ok) {
    const errData = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }))
    throw new Error(errData.detail || `Failed to record analyst action: HTTP ${res.status}`)
  }
  return res.json()
}
