export interface HealthStatus {
  status: string
  service: string
  version: string
  timestamp: string
  data_mode: string
}

export interface MerchantSummary {
  merchant_id: string
  total_windows: number
  flagged_anomaly_count: number
  total_monetary_volume: number
  active_risk_band: 'low' | 'medium' | 'high' | 'critical'
}

export interface DetectionWindow {
  id: number
  merchant_id: string
  window_start: string
  window_end: string
  transaction_count: number
  total_amount: number
  avg_transaction_amount: number
  is_synthetic_fraud_spike?: boolean
  split: 'train' | 'dev_test'
  created_at: string
}

export interface TimeseriesPoint {
  timestamp: string
  transaction_count: number
  total_amount: number
  avg_transaction_amount: number
  is_flagged: boolean
}

export interface AnomalyDetection {
  id: number
  window_id: number
  detector_type: 'baseline' | 'ml'
  risk_score: number
  is_flagged: boolean
  explanation?: string
  created_at: string
}

export interface DetectorSummaryDetail {
  windows_scored: number
  windows_flagged: number
}

export interface DetectorRunSummary {
  detectors_run: string[]
  results: Record<string, DetectorSummaryDetail>
  run_timestamp: string
}

export interface RiskScoringResult {
  risk_score: number
  risk_band: string
  risk_multiplier: number
  estimated_exposure: number
  recommended_action: string
}

export interface ExplanationResult {
  summary: string
  key_drivers: string[]
  raw_text: string
  generated_by: string
}

export interface PolicyDecision {
  policy_id: string
  action_type: string
  priority: string
  review_sla_hours: number
  require_dual_review: boolean
  routing_tags: string[]
  triggered_rules: string[]
  audit_metadata: Record<string, unknown>
}

export interface RiskDossier {
  window: DetectionWindow
  detector_type: string
  is_flagged: boolean
  risk_result: RiskScoringResult
  explanation: ExplanationResult
  policy_decision: PolicyDecision
  audit_entry_id: string
}

export interface EvaluationRun {
  id?: number
  detector_type: 'baseline' | 'ml' | string
  partition: string
  run_timestamp: string
  precision: number
  recall: number
  f1_score: number
  false_positive_rate: number
  true_positives: number
  false_positives: number
  false_negatives: number
  true_negatives: number
  fp_cost: number
  fn_cost: number
  total_cost: number
  notes: string
}

export interface AuditRecord {
  entry_id: string
  timestamp: string
  event_type: string
  window_id: string
  merchant_id: string
  actor: string
  payload: Record<string, unknown>
  previous_hash: string | null
  integrity_hash: string
}

export interface AuditReport {
  window_id: string
  event_count: number
  events: AuditRecord[]
  integrity_valid: boolean
  integrity_errors: string[]
}

export interface AuditIntegrityVerification {
  integrity_valid: boolean
  integrity_errors: string[]
  total_records: number
}

export interface AnalystActionRequest {
  actor: string
  window_id: number
  disposition: 'escalate' | 'resolve' | 'flag_for_followup' | 'monitor' | string
  notes?: string
}

export interface AnalystActionResponse {
  status: string
  entry_id: string
  message: string
}
