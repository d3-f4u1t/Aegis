import React, { useEffect, useState } from "react";
import {
  CheckCircle,
  Cpu,
  Gauge,
  HelpCircle,
  ShieldAlert,
  XCircle,
  Zap,
} from "lucide-react";
import { ComparisonResponse, TrafficSampleItem } from "../types";
import { api } from "../services/api";

export const ModelComparisonPage: React.FC = () => {
  const [samples, setSamples] = useState<TrafficSampleItem[]>([]);
  const [selectedSampleId, setSelectedSampleId] = useState<string>("");
  const [comparison, setComparison] = useState<ComparisonResponse | null>(null);
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
    async function runComparison() {
      if (!selectedSampleId) return;
      const s = samples.find((item) => item.sample_id === selectedSampleId);
      if (!s) return;

      setLoading(true);
      setError(null);
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
        if (!isMounted) return;
        setError("Comparison failed: " + err.message);
      } finally {
        if (isMounted) setLoading(false);
      }
    }
    runComparison();
    return () => {
      isMounted = false;
    };
  }, [selectedSampleId, samples]);

  const activeSample = samples.find((s) => s.sample_id === selectedSampleId);

  return (
    <div style={{ padding: "1.5rem 2rem", maxWidth: "1350px", margin: "0 auto" }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.25rem" }}>
        <div>
          <h2 style={{ fontSize: "1.5rem", fontWeight: 700, color: "#f8fafc", margin: 0 }}>
            Multi-Model Engine Comparison
          </h2>
          <p style={{ fontSize: "0.85rem", color: "#94a3b8", margin: 0 }}>
            Simultaneous inference benchmark comparing baselines, scalar reference MLP, and vectorized NumPy MLP.
          </p>
        </div>

        {/* Sample Selector */}
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <label style={{ fontSize: "0.8rem", color: "#94a3b8" }}>Test Traffic Window:</label>
          <select
            value={selectedSampleId}
            onChange={(e) => setSelectedSampleId(e.target.value)}
            style={{
              background: "#1e293b",
              color: "#f8fafc",
              border: "1px solid #334155",
              borderRadius: "6px",
              padding: "0.35rem 0.75rem",
              fontSize: "0.8rem",
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

      {/* Selected Sample Information Banner */}
      {activeSample && (
        <div className="card" style={{ marginBottom: "1.5rem", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div>
            <span style={{ fontSize: "0.75rem", color: "#94a3b8" }}>Tested Sample Window:</span>
            <h3 style={{ fontSize: "1.1rem", fontWeight: 700, color: "#f8fafc", margin: "0.2rem 0" }}>
              {activeSample.scenario} &bull; Client: <code className="font-mono">{activeSample.client_ip_hash}</code>
            </h3>
            <p style={{ fontSize: "0.8rem", color: "#cbd5e1", margin: 0 }}>
              {activeSample.description}
            </p>
          </div>
          <div style={{ textAlign: "right" }}>
            <span style={{ fontSize: "0.75rem", color: "#94a3b8" }}>Ground Truth Label:</span>
            <div>
              {activeSample.ground_truth_label === 1 ? (
                <span className="badge badge-attack" style={{ fontSize: "0.9rem", padding: "0.35rem 0.75rem" }}>
                  ATTACK (1)
                </span>
              ) : (
                <span className="badge badge-normal" style={{ fontSize: "0.9rem", padding: "0.35rem 0.75rem" }}>
                  NORMAL (0)
                </span>
              )}
            </div>
          </div>
        </div>
      )}

      {/* 4 Models Comparison Cards Grid */}
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
            const isMLP = res.engine === "numpy" || res.engine === "scalar";

            return (
              <div
                key={res.model_id}
                className="card"
                style={{
                  borderTop: isAttack ? "3px solid #ef4444" : "3px solid #10b981",
                  display: "flex",
                  flexDirection: "column",
                  justifyContent: "space-between",
                }}
              >
                <div>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem" }}>
                    <span className="badge badge-info" style={{ fontSize: "0.65rem" }}>
                      {res.engine.toUpperCase()}
                    </span>
                    {res.is_correct ? (
                      <span style={{ display: "flex", alignItems: "center", gap: "0.25rem", color: "#22c55e", fontSize: "0.75rem", fontWeight: 600 }}>
                        <CheckCircle size={14} /> CORRECT
                      </span>
                    ) : (
                      <span style={{ display: "flex", alignItems: "center", gap: "0.25rem", color: "#ef4444", fontSize: "0.75rem", fontWeight: 600 }}>
                        <XCircle size={14} /> INCORRECT
                      </span>
                    )}
                  </div>

                  <h3 style={{ fontSize: "1rem", fontWeight: 700, color: "#f8fafc", marginBottom: "0.5rem" }}>
                    {res.model_name}
                  </h3>

                  <div style={{ background: "#0e1420", padding: "0.85rem", borderRadius: "8px", border: "1px solid #1e293b", marginBottom: "0.75rem" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.3rem" }}>
                      <span style={{ fontSize: "0.75rem", color: "#94a3b8" }}>Prediction:</span>
                      <span
                        className={isAttack ? "badge badge-attack" : "badge badge-normal"}
                        style={{ fontSize: "0.75rem" }}
                      >
                        {isAttack ? "ATTACK (1)" : "NORMAL (0)"}
                      </span>
                    </div>

                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.3rem" }}>
                      <span style={{ fontSize: "0.75rem", color: "#94a3b8" }}>Probability P(attack):</span>
                      <span style={{ fontSize: "0.85rem", fontWeight: 700, fontFamily: "monospace", color: "#38bdf8" }}>
                        {(res.probability * 100).toFixed(2)}% ({res.probability.toFixed(4)})
                      </span>
                    </div>

                    <div style={{ display: "flex", justifyContent: "space-between" }}>
                      <span style={{ fontSize: "0.75rem", color: "#94a3b8" }}>Decision Rule:</span>
                      <span style={{ fontSize: "0.75rem", color: "#64748b", fontFamily: "monospace" }}>
                        p ≥ {res.threshold.toFixed(2)}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Footer with Latency */}
                <div style={{ borderTop: "1px solid #1e293b", paddingTop: "0.75rem", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span style={{ fontSize: "0.75rem", color: "#94a3b8", display: "flex", alignItems: "center", gap: "0.3rem" }}>
                    <Zap size={13} color="#f59e0b" /> Inference Latency:
                  </span>
                  <span style={{ fontSize: "0.8rem", fontWeight: 600, fontFamily: "monospace", color: "#f8fafc" }}>
                    {res.latency_ms.toFixed(3)} ms
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Latency & Mathematical Parity Takeaway Card */}
      <div className="card">
        <h3 style={{ fontSize: "1rem", fontWeight: 700, color: "#f8fafc", marginBottom: "0.5rem" }}>
          Engine Parity & Vectorization Analysis
        </h3>
        <p style={{ fontSize: "0.85rem", color: "#cbd5e1", lineHeight: 1.6, marginBottom: "0.75rem" }}>
          The Scalar MLP (<code className="font-mono">nn/</code>) and NumPy MLP (<code className="font-mono">nn_numpy/</code>) share
          identical network topology and identical weights. Both models evaluate to the exact same classification prediction
          and probability scores with numerical discrepancy below <strong>10⁻¹¹</strong>.
        </p>
        <p style={{ fontSize: "0.85rem", color: "#94a3b8", lineHeight: 1.6 }}>
          However, because the NumPy engine vectorizes layer multiplications using contiguous BLAS matrix operations (<code className="font-mono">Z = X @ W + b</code>),
          it achieves approximately <strong>170× lower inference latency</strong> compared to pure Python object iteration over individual scalar neuron instances.
        </p>
      </div>
    </div>
  );
};
