import {
  ComparisonResponse,
  ExperimentReport,
  ForwardTraceResult,
  LayerParameters,
  ModelArchitecture,
  ModelSummary,
  TrafficSampleItem,
} from "../types";

const API_BASE = "/api/v1";

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, options);
  if (!res.ok) {
    let errDetail = `${res.status} ${res.statusText}`;
    try {
      const errJson = await res.json();
      if (errJson.detail) errDetail = errJson.detail;
    } catch {
      // ignore
    }
    throw new Error(errDetail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  async getHealth(): Promise<{ status: string; models_count: number; default_model: string }> {
    return fetchJson(`${API_BASE}/health`);
  },

  async listModels(): Promise<ModelSummary[]> {
    return fetchJson(`${API_BASE}/models`);
  },

  async getModelArchitecture(modelId: string): Promise<ModelArchitecture> {
    return fetchJson(`${API_BASE}/models/${encodeURIComponent(modelId)}`);
  },

  async getLayerParameters(modelId: string, layerIdx: number): Promise<LayerParameters> {
    return fetchJson(`${API_BASE}/models/${encodeURIComponent(modelId)}/layers/${layerIdx}`);
  },

  async traceForward(
    modelId: string,
    features: number[],
    isScaled = false,
    threshold = 0.5
  ): Promise<ForwardTraceResult> {
    return fetchJson(`${API_BASE}/models/${encodeURIComponent(modelId)}/trace`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        features,
        is_scaled: isScaled,
        threshold,
      }),
    });
  },

  async compareModels(
    features: number[],
    isScaled = false,
    sampleId?: string,
    groundTruthLabel?: number,
    threshold = 0.5
  ): Promise<ComparisonResponse> {
    return fetchJson(`${API_BASE}/compare`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        features,
        is_scaled: isScaled,
        sample_id: sampleId,
        ground_truth_label: groundTruthLabel,
        threshold,
      }),
    });
  },

  async listExperiments(): Promise<Array<{ experiment_id: string; name: string; description: string; dataset_size: number }>> {
    return fetchJson(`${API_BASE}/experiments`);
  },

  async getExperimentReport(experimentId: string): Promise<ExperimentReport> {
    return fetchJson(`${API_BASE}/experiments/${encodeURIComponent(experimentId)}`);
  },

  async getTelemetrySamples(scenario?: string): Promise<TrafficSampleItem[]> {
    const url = scenario
      ? `${API_BASE}/telemetry/samples?scenario=${encodeURIComponent(scenario)}`
      : `${API_BASE}/telemetry/samples`;
    return fetchJson(url);
  },

  async getSingleSample(sampleId: string): Promise<TrafficSampleItem> {
    return fetchJson(`${API_BASE}/telemetry/samples/${encodeURIComponent(sampleId)}`);
  },
};
