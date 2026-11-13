"""Pydantic schemas for the Neural Network Visualization Laboratory API."""

from __future__ import annotations

from typing import Any, List, Optional
from pydantic import BaseModel, Field


class NeuronWeights(BaseModel):
    neuron_idx: int
    bias: float
    weights: List[float]


class LayerParameters(BaseModel):
    layer_idx: int
    n_inputs: int
    n_outputs: int
    activation: str
    weights: List[List[float]]  # shape (n_inputs, n_outputs)
    biases: List[float]         # shape (n_outputs,)
    neurons: List[NeuronWeights]


class ModelArchitecture(BaseModel):
    model_id: str
    name: str
    engine: str  # "scalar", "numpy", "logistic_regression", "threshold"
    format_version: int
    sizes: List[int]
    hidden_activation: Optional[str] = None
    output_activation: Optional[str] = None
    total_parameters: int
    schema_version: str = "1"
    feature_names: List[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelSummary(BaseModel):
    model_id: str
    name: str
    engine: str
    architecture_desc: str
    total_parameters: int
    schema_version: str
    feature_count: int


class NeuronTrace(BaseModel):
    neuron_idx: int
    incoming_values: List[float]
    incoming_weights: List[float]
    weighted_terms: List[float]  # x_i * w_ij
    bias: float
    z: float                      # pre-activation sum
    a: float                      # post-activation output
    is_active: bool               # z > 0 for ReLU


class LayerTrace(BaseModel):
    layer_idx: int
    layer_type: str              # "input", "hidden", "output"
    activation: str
    inputs: List[float]
    outputs: List[float]
    neurons: List[NeuronTrace]
    mathematical_formula: str


class ForwardTraceResult(BaseModel):
    model_id: str
    engine: str
    raw_features: List[float]
    scaled_features: List[float]
    feature_names: List[str]
    layers: List[LayerTrace]
    logit: float
    probability: float
    threshold: float
    prediction: int             # 0 = Normal, 1 = Attack
    confidence: float           # distance from threshold or probability
    parity_verified: bool       # matches engine predict_proba within 1e-12
    inference_time_ms: float


class TraceRequest(BaseModel):
    features: List[float]
    is_scaled: bool = False
    threshold: float = 0.5


class ScenarioMetric(BaseModel):
    total_samples: float
    accuracy: float
    recall: float
    false_positive_rate: float
    tp: float
    fp: float
    tn: float
    fn: float


class ExperimentModelResult(BaseModel):
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    false_negative_rate: float
    tp: float
    fp: float
    tn: float
    fn: float
    pr_auc: float
    average_precision: float
    inference_latency_ms: float
    per_scenario: dict[str, ScenarioMetric] = Field(default_factory=dict)


class ExperimentReport(BaseModel):
    experiment_id: str
    name: str
    description: str
    dataset_size: int
    n_train: int
    n_val: int
    n_test: int
    positives_test: int
    split_strategy: str
    models: dict[str, ExperimentModelResult]


class TrafficSampleItem(BaseModel):
    sample_id: str
    scenario: str
    source: str                 # "normal" or "attack"
    ground_truth_label: int     # 0 or 1
    client_ip_hash: str
    window_start_utc: str
    raw_features: List[float]
    scaled_features: List[float]
    feature_names: List[str]
    description: str
    is_failure_case: bool = False
    failure_model: Optional[str] = None
    failure_explanation: Optional[str] = None


class ModelComparisonItem(BaseModel):
    model_id: str
    model_name: str
    engine: str
    prediction: int
    probability: float
    threshold: float
    is_correct: bool
    latency_ms: float
    agreement: bool


class ComparisonResponse(BaseModel):
    sample_id: Optional[str] = None
    ground_truth_label: Optional[int] = None
    results: List[ModelComparisonItem]
