import React, { useEffect, useState } from "react";
import {
  Check,
  CheckCircle2,
  Cpu,
  Layers,
  ShieldAlert,
  X,
  XCircle,
  Zap,
} from "lucide-react";
import type { ComparisonResponse, TrafficSampleItem } from "../types";
import { api } from "../services/api";

export const ModelComparisonPage: React.FC = () => {
  const [samples, setSamples] = useState<TrafficSampleItem[]>([]);
  const [selectedSampleId, setSelectedSampleId] = useState<string>("");
  const [comparison, setComparison] = useState<ComparisonResponse | null>(null);

  useEffect(() => {
    let isMounted = true;
    async function loadSamples() {
      try {
        const s = await api.getTelemetrySamples();
        if (!isMounted) return;
        setSamples(s);
        if (s.length > 0) {
          setSelectedSampleId(s[0].sample_id);
        }
      } catch (err: any) {
        // non-blocking
      }
    }
    loadSamples();
    return () => {
      isMounted = false;
    };
  }, []);

  useEffect(() => {
    let isMounted = true;
    async function runComparison() {
      if (!selectedSampleId) return;
      const s = samples.find((item) => item.sample_id === selectedSampleId);
      if (!s) return;

      try {
        const res = await api.compareModels(
          s.raw_features,
          false,
          s.sample_id,
          s.ground_truth_label,
          0.5
        );
        if (!isMounted) return;
        setComparison(res);
      } catch (err: any) {
        // non-blocking
      }
    }
    runComparison();
    return () => {
      isMounted = false;
    };
  }, [selectedSampleId, samples]);

  const activeSample = samples.find((s) => s.sample_id === selectedSampleId);

  return (
    <div style={{ padding: "2.5rem", maxWidth: "1440px", margin: "0 auto" }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", marginBottom: "1.75rem", flexWrap: "wrap", gap: "1rem" }}>
        <div>
          <div className="section-kicker">Multi-Engine Benchmark</div>
          <h2 style={{ fontSize: "1.75rem", fontWeight: 800, color: "#ffffff", letterSpacing: "-0.025em", margin: 0 }}>
            Simultaneous Engine Evaluation
          </h2>
          <p style={{ fontSize: "0.875rem", color: "#888888", marginTop: "0.25rem" }}>
            Parallel inference benchmark comparing baselines, scalar reference MLP, and vectorized NumPy MLP.
          </p>
        </div>

        {/* Sample Picker */}
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <span style={{ fontSize: "0.75rem", color: "#777777", fontFamily: "var(--font-mono)", textTransform: "uppercase" }}>
            Test Sample:
          </span>
          <select
            value={selectedSampleId}
            onChange={(e) => setSelectedSampleId(e.target.value)}
            style={{
              background: "#111111",
              color: "#ededed",
              border: "1px solid rgba(255, 255, 255, 0.12)",
              borderRadius: "6px",
              padding: "0.4rem 0.85rem",
              fontSize: "0.75rem",
              fontFamily: "var(--font-mono)",
              cursor: "pointer",
              minWidth: "280px",
            }}
          >
            {samples.map((s) => (
              <option key={s.sample_id} value={s.sample_id}>
                {s.scenario} ({s.source.toUpperCase()})
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Selected Sample Information */}
      {activeSample && (
        <div className="card" style={{ marginBottom: "1.75rem", display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "1rem" }}>
          <div>
            <span style={{ fontSize: "0.6875rem", color: "#777777", fontFamily: "var(--font-mono)", textTransform: "uppercase" }}>
              Selected Evaluation Window
            </span>
            <h3 style={{ fontSize: "1.125rem", fontWeight: 700, color: "#ffffff", margin: "0.2rem 0", letterSpacing: "-0.01em" }}>
              {activeSample.scenario} &bull; Client: <code className="font-mono">{activeSample.client_ip_hash}</code>
            </h3>
            <p style={{ fontSize: "0.8125rem", color: "#a1a1a1", margin: 0 }}>
              {activeSample.description}
            </p>
          </div>
          <div style={{ textAlign: "right" }}>
            <span style={{ fontSize: "0.6875rem", color: "#777777", fontFamily: "var(--font-mono)", textTransform: "uppercase" }}>
              Ground Truth
            </span>
            <div style={{ marginTop: "0.25rem" }}>
              {activeSample.ground_truth_label === 1 ? (
                <span className="badge badge-attack" style={{ fontSize: "0.8125rem", padding: "0.3rem 0.75rem" }}>
                  ATTACK (1)
                </span>
              ) : (
                <span className="badge badge-normal" style={{ fontSize: "0.8125rem", padding: "0.3rem 0.75rem" }}>
                  NORMAL (0)
                </span>
              )}
            </div>
          </div>
        </div>
      )}

      {/* 4 Models Comparison Grid */}
      {comparison && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
            gap: "1.25rem",
            marginBottom: "2rem",
          }}
        >
          {comparison.results.map((res) => {
            const isAttack = res.prediction === 1;
            const isNumPy = res.engine === "numpy";

            return (
              <div
                key={res.model_id}
                className="card"
                style={{
                  borderTop: isAttack ? "2px solid #ff3366" : "2px solid #00f5a0",
                  display: "flex",
                  flexDirection: "column",
                  justifyContent: "space-between",
                  background: isNumPy ? "#0d1117" : "#080808",
                }}
              >
                <div>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem" }}>
                    <span className="badge badge-neutral" style={{ fontSize: "0.625rem" }}>
                      {res.engine.toUpperCase()}
                    </span>
                    {res.is_correct ? (
                      <span style={{ display: "flex", alignItems: "center", gap: "0.3rem", color: "#00f5a0", fontSize: "0.75rem", fontWeight: 600, fontFamily: "var(--font-mono)" }}>
                        <Check size={12} /> CORRECT
                      </span>
                    ) : (
                      <span style={{ display: "flex", alignItems: "center", gap: "0.3rem", color: "#ff3366", fontSize: "0.75rem", fontWeight: 600, fontFamily: "var(--font-mono)" }}>
                        <X size={12} /> INCORRECT
                      </span>
                    )}
                  </div>

                  <h3 style={{ fontSize: "1rem", fontWeight: 700, color: "#ffffff", marginBottom: "0.85rem", letterSpacing: "-0.01em" }}>
                    {res.model_name}
                  </h3>

                  <div style={{ background: "#050505", padding: "0.85rem 1rem", borderRadius: "6px", border: "1px solid rgba(255, 255, 255, 0.06)", marginBottom: "0.85rem" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.4rem" }}>
                      <span style={{ fontSize: "0.75rem", color: "#777777" }}>Prediction:</span>
                      <span
                        className={isAttack ? "badge badge-attack" : "badge badge-normal"}
                        style={{ fontSize: "0.6875rem" }}
                      >
                        {isAttack ? "ATTACK (1)" : "NORMAL (0)"}
                      </span>
                    </div>

                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.4rem" }}>
                      <span style={{ fontSize: "0.75rem", color: "#777777" }}>Probability:</span>
                      <span style={{ fontSize: "0.875rem", fontWeight: 700, fontFamily: "var(--font-mono)", color: "#00f0ff" }}>
                        {(res.probability * 100).toFixed(2)}%
                      </span>
                    </div>

                    <div style={{ display: "flex", justifyContent: "space-between" }}>
                      <span style={{ fontSize: "0.75rem", color: "#777777" }}>Decision Rule:</span>
                      <span style={{ fontSize: "0.75rem", color: "#666666", fontFamily: "var(--font-mono)" }}>
                        p &ge; {res.threshold.toFixed(2)}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Footer with Latency and Speedup */}
                <div style={{ borderTop: "1px solid rgba(255, 255, 255, 0.06)", paddingTop: "0.75rem", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span style={{ fontSize: "0.75rem", color: "#777777", display: "flex", alignItems: "center", gap: "0.35rem" }}>
                    <Zap size={12} color="#f5a623" /> Latency:
                  </span>
                  <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                    {isNumPy && (
                      <span className="badge badge-normal" style={{ fontSize: "0.6rem" }}>
                        ~170&times; SPEEDUP
                      </span>
                    )}
                    <span style={{ fontSize: "0.8125rem", fontWeight: 600, fontFamily: "var(--font-mono)", color: "#ffffff" }}>
                      {res.latency_ms.toFixed(3)} ms
                    </span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Latency & Mathematical Parity Takeaway Card */}
      <div className="card">
        <div className="section-kicker">Theoretical &amp; Empirical Parity</div>
        <h3 style={{ fontSize: "1.125rem", fontWeight: 700, color: "#ffffff", margin: "0.2rem 0 0.5rem 0", letterSpacing: "-0.01em" }}>
          Numerical Equivalence &amp; Vectorization Gains
        </h3>
        <p style={{ fontSize: "0.8125rem", color: "#b5b5b5", lineHeight: 1.6, marginBottom: "0.65rem" }}>
          The Scalar reference engine (<code className="font-mono">nn/</code>) and NumPy vectorized engine (<code className="font-mono">nn_numpy/</code>) share
          verified mathematical parity (&lt; 10⁻¹¹) under identical parameters. In Phase 5, both independent implementations were trained
          on the 2,695-window dataset under identical configurations, both achieving <strong>99.49% accuracy</strong> and <strong>0.9952 F1</strong>.
        </p>
        <p style={{ fontSize: "0.8125rem", color: "#888888", lineHeight: 1.6 }}>
          However, because the NumPy engine vectorizes layer multiplications using contiguous BLAS matrix operations (<code className="font-mono">Z = X @ W + b</code>),
          it achieved a verified <strong>172.9&times; speedup</strong> in batch inference latency (0.00028 ms vs 0.04826 ms/sample) compared to pure Python object iteration over individual scalar neurons.
        </p>
      </div>
    </div>
  );
};
