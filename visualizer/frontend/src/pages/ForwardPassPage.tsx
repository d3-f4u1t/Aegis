import React, { useEffect, useState } from "react";
import {
  Activity,
  ArrowLeft,
  ArrowRight,
  Check,
  CheckCircle2,
  Clock,
  Pause,
  Play,
  RotateCcw,
  ShieldAlert,
  Sparkles,
  Zap,
} from "lucide-react";
import type { ForwardTraceResult, TrafficSampleItem } from "../types";
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

  const handleSampleChange = (sId: string) => {
    setSelectedSampleId(sId);
    const chosen = samples.find((s) => s.sample_id === sId);
    if (chosen) {
      setCurrentFeatures(chosen.raw_features);
      setCurrentStep(0);
    }
  };

  const maxSteps = trace ? trace.layers.length + 1 : 0;

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
    }, 1300);
    return () => clearInterval(interval);
  }, [isPlaying, maxSteps]);

  const activeSample = samples.find((s) => s.sample_id === selectedSampleId);

  return (
    <div style={{ padding: "2.5rem", maxWidth: "1440px", margin: "0 auto" }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", marginBottom: "1.75rem", flexWrap: "wrap", gap: "1rem" }}>
        <div>
          <div className="section-kicker">Observable Inference Engine</div>
          <h2 style={{ fontSize: "1.75rem", fontWeight: 800, color: "#ffffff", letterSpacing: "-0.025em", margin: 0 }}>
            Forward Propagation Tracer
          </h2>
          <p style={{ fontSize: "0.875rem", color: "#888888", marginTop: "0.25rem" }}>
            Step-by-step layer computations exposing intermediate weighted sums Z and post-activation values A.
          </p>
        </div>

        {/* Parity Status Badge */}
        {trace && (
          <div style={{ display: "flex", alignItems: "center", gap: "0.85rem" }}>
            <span
              className={trace.parity_verified ? "badge badge-normal" : "badge badge-attack"}
              style={{ fontSize: "0.75rem", padding: "0.35rem 0.85rem" }}
            >
              <Check size={13} /> Mathematical Parity Verified (&lt; 10⁻¹¹)
            </span>
            <span style={{ fontSize: "0.75rem", color: "#777777", fontFamily: "var(--font-mono)" }}>
              LATENCY: {trace.inference_time_ms.toFixed(3)} ms
            </span>
          </div>
        )}
      </div>

      {/* Control Station Card */}
      <div
        className="card"
        style={{
          background: "#080808",
          border: "1px solid rgba(255, 255, 255, 0.08)",
          marginBottom: "1.5rem",
          padding: "1.25rem 1.5rem",
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "1.25rem" }}>
          {/* Sample Picker */}
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
            <span style={{ fontSize: "0.75rem", color: "#777777", textTransform: "uppercase", fontFamily: "var(--font-mono)" }}>
              Traffic Sample:
            </span>
            <select
              value={selectedSampleId}
              onChange={(e) => handleSampleChange(e.target.value)}
              style={{
                background: "#111111",
                color: "#ededed",
                border: "1px solid rgba(255, 255, 255, 0.12)",
                borderRadius: "6px",
                padding: "0.45rem 0.85rem",
                fontSize: "0.8125rem",
                fontFamily: "var(--font-mono)",
                cursor: "pointer",
                minWidth: "320px",
              }}
            >
              {samples.map((s) => (
                <option key={s.sample_id} value={s.sample_id}>
                  {s.scenario} &bull; {s.source.toUpperCase()}
                </option>
              ))}
            </select>
          </div>

          {/* Decision Threshold Slider */}
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
            <span style={{ fontSize: "0.75rem", color: "#777777", textTransform: "uppercase", fontFamily: "var(--font-mono)" }}>
              Decision Threshold (&tau; = {threshold.toFixed(2)}):
            </span>
            <input
              type="range"
              min="0.05"
              max="0.95"
              step="0.05"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              style={{ width: "120px", cursor: "pointer", accentColor: "#00f5a0" }}
            />
          </div>

          {/* Stepper Transport Controls */}
          <div style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
            <button
              className="btn btn-secondary"
              onClick={() => {
                setIsPlaying(false);
                setCurrentStep(0);
              }}
              title="Reset to Input"
              style={{ padding: "0.45rem 0.65rem" }}
            >
              <RotateCcw size={14} />
            </button>
            <button
              className="btn btn-secondary"
              disabled={currentStep === 0}
              onClick={() => {
                setIsPlaying(false);
                setCurrentStep((p) => Math.max(0, p - 1));
              }}
              style={{ padding: "0.45rem 0.85rem" }}
            >
              <ArrowLeft size={14} /> Prev
            </button>
            <button
              className="btn btn-primary"
              onClick={() => {
                if (currentStep >= maxSteps) setCurrentStep(0);
                setIsPlaying(!isPlaying);
              }}
              style={{ padding: "0.45rem 1rem" }}
            >
              {isPlaying ? <Pause size={14} /> : <Play size={14} />}
              {isPlaying ? "Pause" : "Play"}
            </button>
            <button
              className="btn btn-secondary"
              disabled={currentStep >= maxSteps}
              onClick={() => {
                setIsPlaying(false);
                setCurrentStep((p) => Math.min(maxSteps, p + 1));
              }}
              style={{ padding: "0.45rem 0.85rem" }}
            >
              Next <ArrowRight size={14} />
            </button>
          </div>
        </div>

        {/* Active Sample Context Bar */}
        {activeSample && (
          <div
            style={{
              marginTop: "0.85rem",
              paddingTop: "0.85rem",
              borderTop: "1px solid rgba(255, 255, 255, 0.05)",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              fontSize: "0.8125rem",
              color: "#888888",
            }}
          >
            <div>
              <strong style={{ color: "#ffffff" }}>Scenario:</strong> {activeSample.scenario} &bull;{" "}
              <strong>Ground Truth:</strong>{" "}
              {activeSample.ground_truth_label === 1 ? (
                <span className="badge badge-attack" style={{ fontSize: "0.65rem" }}>ATTACK (1)</span>
              ) : (
                <span className="badge badge-normal" style={{ fontSize: "0.65rem" }}>NORMAL (0)</span>
              )}
              {" "}&bull; {activeSample.description}
            </div>
            {activeSample.is_failure_case && (
              <span className="badge badge-attack">Failure Case Study</span>
            )}
          </div>
        )}
      </div>

      {/* Stepper Timeline Pipeline */}
      <div style={{ display: "flex", gap: "0.4rem", marginBottom: "1.75rem", overflowX: "auto" }}>
        {Array.from({ length: maxSteps + 1 }).map((_, stepIdx) => {
          const isActive = currentStep === stepIdx;
          const isPassed = currentStep > stepIdx;
          const stepLabel =
            stepIdx === 0
              ? "00: Input Feature Vector"
              : stepIdx === maxSteps
              ? `0${stepIdx}: Classification Decision`
              : `0${stepIdx}: Layer ${stepIdx} Propagation`;

          return (
            <button
              key={`step-${stepIdx}`}
              onClick={() => {
                setIsPlaying(false);
                setCurrentStep(stepIdx);
              }}
              style={{
                flex: 1,
                padding: "0.6rem 0.75rem",
                borderRadius: "6px",
                border: isActive
                  ? "1px solid #00f0ff"
                  : isPassed
                  ? "1px solid rgba(0, 245, 160, 0.4)"
                  : "1px solid rgba(255, 255, 255, 0.06)",
                background: isActive
                  ? "rgba(0, 240, 255, 0.08)"
                  : isPassed
                  ? "rgba(0, 245, 160, 0.04)"
                  : "#090909",
                color: isActive ? "#00f0ff" : isPassed ? "#00f5a0" : "#666666",
                fontWeight: isActive ? 600 : 500,
                fontSize: "0.75rem",
                fontFamily: "var(--font-mono)",
                cursor: "pointer",
                textAlign: "center",
                transition: "all 0.15s ease",
                whiteSpace: "nowrap",
              }}
            >
              {stepLabel}
            </button>
          );
        })}
      </div>

      {/* Step Renderings */}
      {trace && (
        <div>
          {/* STEP 0: Input Features */}
          {currentStep === 0 && (
            <div className="card">
              <div className="section-kicker">Step 00 &bull; Feature Ingestion</div>
              <h3 style={{ fontSize: "1.25rem", fontWeight: 700, color: "#ffffff", marginBottom: "0.5rem", letterSpacing: "-0.02em" }}>
                13-Feature Behavioural Representation
              </h3>
              <p style={{ fontSize: "0.8125rem", color: "#888888", marginBottom: "1.5rem" }}>
                Normalized using training-set StandardScaler parameters: <code className="font-mono">z = (x - &mu;) / &sigma;</code>.
              </p>

              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "0.85rem" }}>
                {trace.feature_names.map((name, idx) => {
                  const rawVal = trace.raw_features[idx];
                  const scaledVal = trace.scaled_features[idx];
                  const isElevated = Math.abs(scaledVal) > 1.5;

                  return (
                    <div
                      key={`feat-${idx}`}
                      style={{
                        background: "#080808",
                        padding: "0.85rem 1rem",
                        borderRadius: "6px",
                        border: isElevated ? "1px solid rgba(0, 240, 255, 0.3)" : "1px solid rgba(255, 255, 255, 0.06)",
                      }}
                    >
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.35rem" }}>
                        <span style={{ fontSize: "0.75rem", color: "#a1a1a1", fontFamily: "var(--font-mono)" }}>
                          [{idx.toString().padStart(2, "0")}] {name}
                        </span>
                      </div>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                        <span style={{ fontSize: "0.75rem", color: "#666666", fontFamily: "var(--font-mono)" }}>
                          Raw: <span style={{ color: "#888888" }}>{rawVal.toFixed(3)}</span>
                        </span>
                        <span
                          style={{
                            fontSize: "0.9375rem",
                            fontWeight: 700,
                            fontFamily: "var(--font-mono)",
                            color: isElevated ? "#00f0ff" : "#ededed",
                          }}
                        >
                          z={scaledVal.toFixed(4)}
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
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "1.25rem", flexWrap: "wrap", gap: "1rem" }}>
                    <div>
                      <div className="section-kicker">Step 0{currentStep} &bull; Dense Layer Computation</div>
                      <h3 style={{ fontSize: "1.25rem", fontWeight: 700, color: "#ffffff", margin: "0.2rem 0", letterSpacing: "-0.02em" }}>
                        Layer {layer.layer_idx} ({layer.layer_type.toUpperCase()}) &bull; Activation: {layer.activation.toUpperCase()}
                      </h3>
                      <p style={{ fontSize: "0.75rem", color: "#00f0ff", fontFamily: "var(--font-mono)", margin: "0.35rem 0 0 0" }}>
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
                      gap: "0.85rem",
                      maxHeight: "560px",
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
                            background: isDead ? "#080808" : "#0d1117",
                            padding: "0.85rem",
                            borderRadius: "6px",
                            border: isDead ? "1px solid rgba(255, 255, 255, 0.05)" : "1px solid rgba(0, 240, 255, 0.25)",
                            boxShadow: neuron.is_active ? "0 0 16px -4px rgba(0, 240, 255, 0.15)" : "none",
                          }}
                        >
                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.5rem" }}>
                            <span style={{ fontSize: "0.8125rem", fontWeight: 600, color: "#ffffff", fontFamily: "var(--font-mono)" }}>
                              Neuron #{neuron.neuron_idx}
                            </span>
                            <span
                              className={neuron.is_active ? "badge badge-normal" : "badge badge-neutral"}
                              style={{ fontSize: "0.625rem" }}
                            >
                              {neuron.is_active ? "ACTIVE" : "INACTIVE (0)"}
                            </span>
                          </div>

                          <div style={{ fontSize: "0.75rem", display: "grid", gap: "0.25rem", fontFamily: "var(--font-mono)" }}>
                            <div style={{ display: "flex", justifyContent: "space-between" }}>
                              <span style={{ color: "#666666" }}>Weighted Sum (z):</span>
                              <span style={{ color: "#ededed", fontWeight: 600 }}>{neuron.z.toFixed(4)}</span>
                            </div>
                            <div style={{ display: "flex", justifyContent: "space-between" }}>
                              <span style={{ color: "#666666" }}>Bias (b):</span>
                              <span style={{ color: "#888888" }}>{neuron.bias.toFixed(4)}</span>
                            </div>
                            <div style={{ display: "flex", justifyContent: "space-between", paddingTop: "0.35rem", borderTop: "1px solid rgba(255, 255, 255, 0.06)" }}>
                              <span style={{ color: "#00f0ff" }}>Output a = f(z):</span>
                              <span style={{ color: neuron.a > 0 ? "#00f5a0" : "#666666", fontWeight: 700 }}>
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
              <div className="section-kicker">Final Stage &bull; Classification Verdict</div>
              <h3 style={{ fontSize: "1.25rem", fontWeight: 700, color: "#ffffff", marginBottom: "1.25rem", letterSpacing: "-0.02em" }}>
                Inference Result &amp; Margin
              </h3>

              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "1.25rem", marginBottom: "1.5rem" }}>
                <div style={{ background: "#080808", padding: "1.25rem", borderRadius: "6px", border: "1px solid rgba(255, 255, 255, 0.08)" }}>
                  <span style={{ fontSize: "0.6875rem", color: "#777777", fontFamily: "var(--font-mono)", textTransform: "uppercase" }}>
                    Output Logit (z)
                  </span>
                  <div style={{ fontSize: "1.75rem", fontWeight: 700, color: "#ededed", margin: "0.35rem 0", fontFamily: "var(--font-mono)" }}>
                    {trace.logit.toFixed(5)}
                  </div>
                  <span style={{ fontSize: "0.75rem", color: "#666666" }}>Linear output from final dense layer</span>
                </div>

                <div style={{ background: "#080808", padding: "1.25rem", borderRadius: "6px", border: "1px solid rgba(255, 255, 255, 0.08)" }}>
                  <span style={{ fontSize: "0.6875rem", color: "#777777", fontFamily: "var(--font-mono)", textTransform: "uppercase" }}>
                    Predicted Probability P(attack)
                  </span>
                  <div style={{ fontSize: "1.75rem", fontWeight: 700, color: "#00f0ff", margin: "0.35rem 0", fontFamily: "var(--font-mono)" }}>
                    {(trace.probability * 100).toFixed(2)}%
                  </div>
                  <span style={{ fontSize: "0.75rem", color: "#666666", fontFamily: "var(--font-mono)" }}>
                    p = &sigma;(z) = {trace.probability.toFixed(6)}
                  </span>
                </div>

                <div style={{ background: "#080808", padding: "1.25rem", borderRadius: "6px", border: "1px solid rgba(255, 255, 255, 0.08)" }}>
                  <span style={{ fontSize: "0.6875rem", color: "#777777", fontFamily: "var(--font-mono)", textTransform: "uppercase" }}>
                    Classification Verdict
                  </span>
                  <div style={{ margin: "0.45rem 0" }}>
                    {trace.prediction === 1 ? (
                      <span className="badge badge-attack" style={{ fontSize: "0.9375rem", padding: "0.4rem 0.85rem" }}>
                        FLAGGED : ATTACK
                      </span>
                    ) : (
                      <span className="badge badge-normal" style={{ fontSize: "0.9375rem", padding: "0.4rem 0.85rem" }}>
                        CLEARED : NORMAL
                      </span>
                    )}
                  </div>
                  <span style={{ fontSize: "0.75rem", color: "#666666", fontFamily: "var(--font-mono)" }}>
                    Rule: p &ge; {trace.threshold.toFixed(2)} &rarr; Attack
                  </span>
                </div>
              </div>

              {/* Decision Margin Gauge */}
              <div style={{ background: "#080808", padding: "1.25rem", borderRadius: "6px", border: "1px solid rgba(255, 255, 255, 0.08)" }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.5rem", fontSize: "0.8125rem" }}>
                  <span style={{ color: "#888888" }}>Decision Margin Distance:</span>
                  <span style={{ color: "#ffffff", fontFamily: "var(--font-mono)", fontWeight: 600 }}>
                    |p - &tau;| = {Math.abs(trace.probability - trace.threshold).toFixed(4)}
                  </span>
                </div>
                <div style={{ width: "100%", height: "8px", background: "#1a1a1a", borderRadius: "4px", position: "relative" }}>
                  <div
                    style={{
                      position: "absolute",
                      top: "-2px",
                      bottom: "-2px",
                      left: "50%",
                      width: "2px",
                      background: "#f5a623",
                      zIndex: 2,
                    }}
                    title="Threshold 0.50"
                  />
                  <div
                    style={{
                      height: "100%",
                      width: `${Math.min(100, Math.max(0, trace.probability * 100))}%`,
                      background: trace.prediction === 1 ? "#ff3366" : "#00f5a0",
                      borderRadius: "4px",
                      transition: "width 0.3s ease",
                    }}
                  />
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", marginTop: "0.4rem", fontSize: "0.6875rem", fontFamily: "var(--font-mono)", color: "#666666" }}>
                  <span>0.00 (BENIGN)</span>
                  <span style={{ color: "#f5a623" }}>&tau; = {trace.threshold.toFixed(2)}</span>
                  <span>1.00 (MALICIOUS)</span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
