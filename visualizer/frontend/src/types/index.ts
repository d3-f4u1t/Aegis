export interface NeuronWeights {
  neuron_idx: number;
  bias: number;
  weights: number[];
}

export interface LayerParameters {
  layer_idx: number;
  n_inputs: number;
  n_outputs: number;
  activation: string;
  weights: number[][]; // [n_inputs][n_outputs]
  biases: number[];
  neurons: NeuronWeights[];
}

export interface ModelArchitecture {
  model_id: string;
  name: string;
  engine: string;
  format_version: number;
  sizes: number[];
  hidden_activation?: string;
  output_activation?: string;
  total_parameters: number;
  schema_version: string;
  feature_names: string[];
  metadata: Record<string, any>;
}

export interface ModelSummary {
  model_id: string;
  name: string;
  engine: string;
  architecture_desc: string;
  total_parameters: number;
  schema_version: string;
  feature_count: number;
}

export interface NeuronTrace {
  neuron_idx: number;
  incoming_values: number[];
  incoming_weights: number[];
  weighted_terms: number[];
  bias: number;
  z: number;
  a: number;
  is_active: boolean;
}

export interface LayerTrace {
  layer_idx: number;
  layer_type: string;
  activation: string;
  inputs: number[];
  outputs: number[];
  neurons: NeuronTrace[];
  mathematical_formula: string;
}

export interface ForwardTraceResult {
  model_id: string;
  engine: string;
  raw_features: number[];
  scaled_features: number[];
  feature_names: string[];
  layers: LayerTrace[];
  logit: number;
  probability: number;
  threshold: number;
  prediction: number; // 0 = Normal, 1 = Attack
  confidence: number;
  parity_verified: boolean;
  inference_time_ms: number;
}

export interface ScenarioMetric {
  total_samples: number;
  accuracy: number;
  recall: number;
  false_positive_rate: number;
  tp: number;
  fp: number;
  tn: number;
  fn: number;
}

export interface ExperimentModelResult {
  model_name: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  false_positive_rate: number;
  false_negative_rate: number;
  tp: number;
  fp: number;
  tn: number;
  fn: number;
  pr_auc: number;
  average_precision: number;
  inference_latency_ms: number;
  per_scenario: Record<string, ScenarioMetric>;
}

export interface ExperimentReport {
  experiment_id: string;
  name: string;
  description: string;
  dataset_size: number;
  n_train: number;
  n_val: number;
  n_test: number;
  positives_test: number;
  split_strategy: string;
  models: Record<string, ExperimentModelResult>;
}

export interface TrafficSampleItem {
  sample_id: string;
  scenario: string;
  source: string;
  ground_truth_label: number;
  client_ip_hash: string;
  window_start_utc: string;
  raw_features: number[];
  scaled_features: number[];
  feature_names: string[];
  description: string;
  is_failure_case: boolean;
  failure_model?: string;
  failure_explanation?: string;
}

export interface ModelComparisonItem {
  model_id: string;
  model_name: string;
  engine: string;
  prediction: number;
  probability: number;
  threshold: number;
  is_correct: boolean;
  latency_ms: number;
  agreement: boolean;
}

export interface ComparisonResponse {
  sample_id?: string;
  ground_truth_label?: number;
  results: ModelComparisonItem[];
}
