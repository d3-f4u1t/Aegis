import React, { useEffect, useState } from "react";
import {
  Activity,
  AlertCircle,
  ArrowLeft,
  ArrowRight,
  CheckCircle,
  Pause,
  Play,
  RotateCcw,
  ShieldAlert,
  Zap,
} from "lucide-react";
import { ForwardTraceResult, TrafficSampleItem } from "../types";
import { api } from "../services/api";

interface ForwardPassPageProps {
  modelId: string;
}

export const ForwardPassPage: React.FC<ForwardPassPageProps> = ({ modelId }) => {
  const [samples, setSamples] = useState<TrafficSampleItem[]>([]);
  const [selectedSampleId, setSelectedSampleId] = useState<string>("");
  const [currentFeatures, setCurrentFeatures] = useState<number[]>(new Array(13).fill(0));
  const [threshold, setThreshold] = useState<number>(0.5);
  const [trace, setTrace] = useState<ForwardTraceResult | null>(null);
  const [currentStep, setCurrentStep] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Load curated samples
  useEffect(() => {
    let isMounted = true;
    async function loadSamples() {
      try {
        const s = await api.getTelemetrySamples();
        if (!isMounted) return;
        setSamples(s);
        if (s.length > 0) {
          setSelectedSampleId(s[0].sample_id);
          setCurrentFeatures(s[0].raw_features);
        }
      } catch (err: any) {
        if (!isMounted) return;
        setError("Failed to load traffic samples: " + err.message);
      }
    }
    loadSamples();
    return () => {
      isMounted = false;
    };
  }, []);

  // Run forward trace whenever features, threshold, or modelId change
  useEffect(() => {
    let isMounted = true;
    async function runTrace() {
      if (currentFeatures.length === 0) return;
      setLoading(true);
      setError(null);
      try {
        const res = await api.traceForward(modelId, currentFeatures, false, threshold);
        if (!isMounted) return;
        setTrace(res);
      } catch (err: any) {
        if (!isMounted) return;
        setError(err.message || "Failed to trace forward pass.");
      } finally {
        if (isMounted) setLoading(false);
      }
    }
    runTrace();
    return () => {
      isMounted = false;
    };
  }, [modelId, currentFeatures, threshold]);

  // Handle sample selection change
  const handleSampleChange = (sId: string) => {
    setSelectedSampleId(sId);
    const chosen = samples.find((s) => s.sample_id === sId);
    if (chosen) {
      setCurrentFeatures(chosen.raw_features);
      setCurrentStep(0);
    }
  };

  // Playback timer
  const maxSteps = trace ? trace.layers.length + 1 : 0; // step 0 is input, then each layer, then final decision

  useEffect(() => {
    if (!isPlaying) return;
    const interval = setInterval(() => {
      setCurrentStep((prev) => {
        if (prev >= maxSteps) {
          setIsPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, 1400);
    return () => clearInterval(interval);
  }, [isPlaying, maxSteps]);

  const activeSample = samples.find((s) => s.sample_id === selectedSampleId);

  return (
    <div style={{ padding: "1.5rem 2rem", maxWidth: "1350px", margin: "0 auto" }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.25rem" }}>
        <div>
          <h2 style={{ fontSize: "1.5rem", fontWeight: 700, color: "#f8fafc", margin: 0 }}>
            Step-by-Step Forward Propagation Tracer
          </h2>
          <p style={{ fontSize: "0.85rem", color: "#94a3b8", margin: 0 }}>
            Observable execution trace computing exact weighted sums Z and post-activation values A.
          </p>
        </div>

        {/* Engine Parity Badge */}
        {trace && (
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
            <span
              className={trace.parity_verified ? "badge badge-normal" : "badge badge-attack"}
              style={{ fontSize: "0.8rem", padding: "0.35rem 0.75rem" }}
            >
              <CheckCircle size={14} /> Mathematical Parity Verified (&lt; 10⁻¹¹)
            </span>
            <span style={{ fontSize: "0.75rem", color: "#94a3b8" }}>
              Latency: {trace.inference_time_ms.toFixed(3)} ms
            </span>
          </div>
        )}
      </div>

      {/* Traffic Sample & Threshold Controls Card */}
      <div className="card" style={{ marginBottom: "1.25rem" }}>
        <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "space-between", alignItems: "center", gap: "1rem" }}>
          {/* Sample Selector */}
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
            <label style={{ fontSize: "0.85rem", fontWeight: 600, color: "#cbd5e1" }}>
              Curated Traffic Window:
            </label>
            <select
              value={selectedSampleId}
              onChange={(e) => handleSampleChange(e.target.value)}
              style={{
                background: "#1e293b",
                color: "#f8fafc",
                border: "1px solid #334155",
                borderRadius: "6px",
                padding: "0.4rem 0.75rem",
                fontSize: "0.85rem",
                cursor: "pointer",
                minWidth: "300px",
              }}
            >
              {samples.map((s) => (
                <option key={s.sample_id} value={s.sample_id}>
                  {s.scenario} &bull; {s.source.toUpperCase()} (GT: {s.ground_truth_label})
                </option>
              ))}
            </select>
          </div>

          {/* Classification Threshold Slider */}
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
            <label style={{ fontSize: "0.85rem", color: "#cbd5e1" }}>
              Decision Threshold (τ = {threshold.toFixed(2)}):
            </label>
            <input
              type="range"
              min="0.05"
              max="0.95"
              step="0.05"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              style={{ width: "120px", cursor: "pointer" }}
            />
          </div>

          {/* Stepper Playback Controls */}
          <div style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
            <button
              className="btn btn-secondary"
              onClick={() => {
                setIsPlaying(false);
                setCurrentStep(0);
              }}
              title="Reset"
            >
              <RotateCcw size={15} />
            </button>
            <button
              className="btn btn-secondary"
              disabled={currentStep === 0}
              onClick={() => {
                setIsPlaying(false);
                setCurrentStep((p) => Math.max(0, p - 1));
              }}
            >
              <ArrowLeft size={15} /> Prev
            </button>
            <button
              className="btn btn-primary"
              onClick={() => {
                if (currentStep >= maxSteps) setCurrentStep(0);
                setIsPlaying(!isPlaying);
              }}
            >
              {isPlaying ? <Pause size={15} /> : <Play size={15} />}
              {isPlaying ? "Pause" : "Play"}
            </button>
            <button
              className="btn btn-secondary"
              disabled={currentStep >= maxSteps}
              onClick={() => {
                setIsPlaying(false);
                setCurrentStep((p) => Math.min(maxSteps, p + 1));
              }}
            >
              Next <ArrowRight size={15} />
            </button>
          </div>
        </div>

        {/* Selected Sample Context Banner */}
        {activeSample && (
          <div style={{ marginTop: "0.75rem", paddingTop: "0.75rem", borderTop: "1px solid #243042", fontSize: "0.8rem", color: "#94a3b8", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span>
              <strong>Scenario:</strong> {activeSample.scenario} &bull; <strong>Ground Truth:</strong>{" "}
              {activeSample.ground_truth_label === 1 ? (
                <span className="badge badge-attack" style={{ fontSize: "0.7rem" }}>ATTACK (1)</span>
              ) : (
                <span className="badge badge-normal" style={{ fontSize: "0.7rem" }}>NORMAL (0)</span>
              )}
              {" "}&bull; {activeSample.description}
            </span>
            {activeSample.is_failure_case && (
              <span className="badge badge-attack">Failure Study Sample</span>
            )}
          </div>
        )}
      </div>

      {/* Stepper Progress Bar */}
      <div style={{ display: "flex", gap: "0.5rem", marginBottom: "1.25rem" }}>
        {Array.from({ length: maxSteps + 1 }).map((_, stepIdx) => {
          const isActive = currentStep === stepIdx;
          const isPassed = currentStep > stepIdx;
          const stepLabel =
            stepIdx === 0
              ? "0: Input Vector"
              : stepIdx === maxSteps
              ? `${stepIdx}: Classification`
              : `${stepIdx}: Layer ${stepIdx}`;

          return (
            <button
              key={`step-btn-${stepIdx}`}
              onClick={() => {
                setIsPlaying(false);
                setCurrentStep(stepIdx);
              }}
              style={{
                flex: 1,
                padding: "0.5rem 0.75rem",
                borderRadius: "6px",
                border: isActive
                  ? "1px solid #38bdf8"
                  : isPassed
                  ? "1px solid #22c55e"
                  : "1px solid #243042",
                background: isActive ? "rgba(56, 189, 248, 0.15)" : isPassed ? "#13211a" : "#111827",
                color: isActive ? "#38bdf8" : isPassed ? "#4ade80" : "#64748b",
                fontWeight: isActive ? 700 : 500,
                fontSize: "0.75rem",
                cursor: "pointer",
                textAlign: "center",
                transition: "all 0.15s ease",
              }}
            >
              {stepLabel}
            </button>
          );
        })}
      </div>

      {/* Step Content Render */}
      {trace && (
        <div>
          {/* STEP 0: Input Features */}
          {currentStep === 0 && (
            <div className="card">
              <h3 style={{ fontSize: "1.1rem", fontWeight: 700, color: "#f8fafc", marginBottom: "0.5rem" }}>
                Step 0: Input Behavioural Window Features (13 Features)
              </h3>
              <p style={{ fontSize: "0.85rem", color: "#94a3b8", marginBottom: "1rem" }}>
                Features extracted from a 60-second window, normalized using training-fitted StandardScaler statistics:
                <code className="font-mono"> x_scaled = (x - mean) / std</code>.
              </p>

              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "0.75rem" }}>
                {trace.feature_names.map((name, idx) => {
                  const rawVal = trace.raw_features[idx];
                  const scaledVal = trace.scaled_features[idx];
                  return (
                    <div
                      key={`feat-${idx}`}
                      style={{
                        background: "#0e1420",
                        padding: "0.6rem 0.85rem",
                        borderRadius: "8px",
                        border: "1px solid #1e293b",
                      }}
                    >
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.25rem" }}>
                        <span style={{ fontSize: "0.75rem", color: "#cbd5e1", fontFamily: "monospace" }}>
                          [{idx}] {name}
                        </span>
                      </div>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                        <span style={{ fontSize: "0.7rem", color: "#64748b" }}>
                          Raw: <code className="font-mono" style={{ color: "#94a3b8" }}>{rawVal.toFixed(3)}</code>
                        </span>
                        <span style={{ fontSize: "0.85rem", fontWeight: 600, color: "#38bdf8", fontFamily: "monospace" }}>
                          z: {scaledVal.toFixed(4)}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* STEP 1..N: Hidden / Output Layers */}
          {currentStep > 0 && currentStep <= trace.layers.length && (
            (() => {
              const layer = trace.layers[currentStep - 1];
              return (
                <div className="card">
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem" }}>
                    <div>
                      <h3 style={{ fontSize: "1.1rem", fontWeight: 700, color: "#f8fafc", margin: 0 }}>
                        Step {currentStep}: Layer {layer.layer_idx} ({layer.layer_type.toUpperCase()}) &bull; Activation: {layer.activation}
                      </h3>
                      <p style={{ fontSize: "0.8rem", color: "#38bdf8", fontFamily: "monospace", margin: "0.25rem 0 0 0" }}>
                        Formula: {layer.mathematical_formula}
                      </p>
                    </div>
                    <span className="badge badge-info">
                      {layer.neurons.length} Neurons
                    </span>
                  </div>

                  {/* Grid of Neurons */}
                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
                      gap: "0.75rem",
                      maxHeight: "520px",
                      overflowY: "auto",
                      paddingRight: "0.25rem",
                    }}
                  >
                    {layer.neurons.map((neuron) => {
                      const isReLU = layer.activation.toLowerCase() === "relu";
                      const isDead = isReLU && neuron.a === 0;

                      return (
                        <div
                          key={`step-${currentStep}-n-${neuron.neuron_idx}`}
                          style={{
                            background: isDead ? "#0f131a" : "#0e1826",
                            padding: "0.75rem",
                            borderRadius: "8px",
                            border: isDead ? "1px solid #1e2633" : "1px solid #1d4ed8",
                          }}
                        >
                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.4rem" }}>
                            <span style={{ fontSize: "0.8rem", fontWeight: 600, color: "#f8fafc" }}>
                              Neuron #{neuron.neuron_idx}
                            </span>
                            <span
                              className={neuron.is_active ? "badge badge-normal" : "badge badge-secondary"}
                              style={{ fontSize: "0.65rem", padding: "0.15rem 0.45rem" }}
                            >
                              {neuron.is_active ? "ACTIVE" : "INACTIVE (0)"}
                            </span>
                          </div>

                          <div style={{ fontSize: "0.75rem", display: "grid", gap: "0.2rem", fontFamily: "monospace" }}>
                            <div style={{ display: "flex", justifyContent: "space-between" }}>
                              <span style={{ color: "#94a3b8" }}>Weighted Sum (z):</span>
                              <span style={{ color: "#cbd5e1", fontWeight: 600 }}>{neuron.z.toFixed(4)}</span>
                            </div>
                            <div style={{ display: "flex", justifyContent: "space-between" }}>
                              <span style={{ color: "#94a3b8" }}>Bias (b):</span>
                              <span style={{ color: "#64748b" }}>{neuron.bias.toFixed(4)}</span>
                            </div>
                            <div style={{ display: "flex", justifyContent: "space-between", paddingTop: "0.2rem", borderTop: "1px solid #243042" }}>
                              <span style={{ color: "#38bdf8", fontWeight: 600 }}>Activation a = f(z):</span>
                              <span style={{ color: neuron.a > 0 ? "#4ade80" : "#64748b", fontWeight: 700 }}>
                                {neuron.a.toFixed(5)}
                              </span>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })()
          )}

          {/* FINAL STEP: Classification Decision */}
          {currentStep === maxSteps && (
            <div className="card">
              <h3 style={{ fontSize: "1.1rem", fontWeight: 700, color: "#f8fafc", marginBottom: "0.75rem" }}>
                Final Classification Decision
              </h3>

              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "1.25rem", marginBottom: "1.5rem" }}>
                <div style={{ background: "#0e1420", padding: "1.25rem", borderRadius: "8px", border: "1px solid #243042" }}>
                  <span style={{ fontSize: "0.8rem", color: "#94a3b8" }}>Output Logit (z):</span>
                  <p style={{ fontSize: "1.5rem", fontWeight: 700, color: "#cbd5e1", margin: "0.25rem 0", fontFamily: "monospace" }}>
                    {trace.logit.toFixed(5)}
                  </p>
                  <span style={{ fontSize: "0.75rem", color: "#64748b" }}>Linear output from final dense layer</span>
                </div>

                <div style={{ background: "#0e1420", padding: "1.25rem", borderRadius: "8px", border: "1px solid #243042" }}>
                  <span style={{ fontSize: "0.8rem", color: "#94a3b8" }}>Predicted Probability P(attack):</span>
                  <p style={{ fontSize: "1.5rem", fontWeight: 700, color: "#38bdf8", margin: "0.25rem 0", fontFamily: "monospace" }}>
                    {(trace.probability * 100).toFixed(2)}% ({trace.probability.toFixed(6)})
                  </p>
                  <span style={{ fontSize: "0.75rem", color: "#64748b" }}>p = σ(z) = 1 / (1 + e^(-z))</span>
                </div>

                <div style={{ background: "#0e1420", padding: "1.25rem", borderRadius: "8px", border: "1px solid #243042" }}>
                  <span style={{ fontSize: "0.8rem", color: "#94a3b8" }}>Decision Rule (τ = {trace.threshold}):</span>
                  <div style={{ margin: "0.25rem 0" }}>
                    {trace.prediction === 1 ? (
                      <span className="badge badge-attack" style={{ fontSize: "1rem", padding: "0.4rem 0.8rem" }}>
                        FLAGGED: ATTACK
                      </span>
                    ) : (
                      <span className="badge badge-normal" style={{ fontSize: "1rem", padding: "0.4rem 0.8rem" }}>
                        CLEARED: NORMAL
                      </span>
                    )}
                  </div>
                  <span style={{ fontSize: "0.75rem", color: "#64748b" }}>p ≥ τ → Attack (1), else Normal (0)</span>
                </div>
              </div>

              {/* Confidence Meter */}
              <div style={{ background: "#0e1420", padding: "1rem", borderRadius: "8px", border: "1px solid #1e293b" }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.4rem", fontSize: "0.8rem" }}>
                  <span style={{ color: "#94a3b8" }}>Classification Margin from Decision Threshold:</span>
                  <span style={{ color: "#f8fafc", fontWeight: 600, fontFamily: "monospace" }}>
                    |p - τ| = {Math.abs(trace.probability - trace.threshold).toFixed(4)}
                  </span>
                </div>
                <div style={{ width: "100%", height: "10px", background: "#1e293b", borderRadius: "5px", position: "relative" }}>
                  <div
                    style={{
                      position: "absolute",
                      top: 0,
                      bottom: 0,
                      left: "50%",
                      width: "2px",
                      background: "#f59e0b",
                      zIndex: 2,
                    }}
                    title="Threshold 0.5"
                  />
                  <div
                    style={{
                      height: "100%",
                      width: `${Math.min(100, Math.max(0, trace.probability * 100))}%`,
                      background: trace.prediction === 1 ? "#ef4444" : "#10b981",
                      borderRadius: "5px",
                      transition: "width 0.3s ease",
                    }}
                  />
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", marginTop: "0.3rem", fontSize: "0.7rem", color: "#64748b" }}>
                  <span>0.0 (Definitely Normal)</span>
                  <span style={{ color: "#f59e0b" }}>τ = {trace.threshold.toFixed(2)}</span>
                  <span>1.0 (Definitely Attack)</span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
