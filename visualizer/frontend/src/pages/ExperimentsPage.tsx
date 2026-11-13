import React, { useEffect, useState } from "react";
import { BarChart3, CheckCircle, ChevronRight, FileText, Layers, ShieldAlert, Sparkles } from "lucide-react";
import type { ExperimentModelResult, ExperimentReport } from "../types";
import { api } from "../services/api";

export const ExperimentsPage: React.FC = () => {
  const [experiments, setExperiments] = useState<Array<{ experiment_id: string; name: string; description: string }>>([]);
  const [selectedExpId, setSelectedExpId] = useState<string>("exp1_indistribution");
  const [report, setReport] = useState<ExperimentReport | null>(null);
  const [selectedModelKey, setSelectedModelKey] = useState<string>("numpy_mlp");

  useEffect(() => {
    let isMounted = true;
    async function loadExps() {
      try {
        const exps = await api.listExperiments();
        if (!isMounted) return;
        setExperiments(exps);
        if (exps.length > 0) {
          setSelectedExpId(exps[0].experiment_id);
        }
      } catch (err: any) {
        // non-blocking
      }
    }
    loadExps();
    return () => {
      isMounted = false;
    };
  }, []);

  useEffect(() => {
    let isMounted = true;
    async function loadReport() {
      if (!selectedExpId) return;
      try {
        const r = await api.getExperimentReport(selectedExpId);
        if (!isMounted) return;
        setReport(r);
        if (r.models["numpy_mlp"]) {
          setSelectedModelKey("numpy_mlp");
        } else {
          setSelectedModelKey(Object.keys(r.models)[0] || "");
        }
      } catch (err: any) {
        // non-blocking
      }
    }
    loadReport();
    return () => {
      isMounted = false;
    };
  }, [selectedExpId]);

  const activeModel: ExperimentModelResult | null =
    report && selectedModelKey ? report.models[selectedModelKey] : null;

  return (
    <div style={{ padding: "2.5rem", maxWidth: "1440px", margin: "0 auto" }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", marginBottom: "1.75rem", flexWrap: "wrap", gap: "1rem" }}>
        <div>
          <div className="section-kicker">Empirical Benchmarks</div>
          <h2 style={{ fontSize: "1.75rem", fontWeight: 800, color: "#ffffff", letterSpacing: "-0.025em", margin: 0 }}>
            Phase 5 Experimental Logs
          </h2>
          <p style={{ fontSize: "0.875rem", color: "#888888", marginTop: "0.25rem" }}>
            Rigorous evaluation across 2,695 windows, client-isolated partitions, and 13 attack scenarios.
          </p>
        </div>

        {/* Experiment Switcher */}
        <div
          style={{
            display: "flex",
            gap: "0.25rem",
            background: "#080808",
            padding: "0.25rem",
            borderRadius: "6px",
            border: "1px solid rgba(255, 255, 255, 0.08)",
          }}
        >
          {experiments.map((exp) => {
            const isSelected = selectedExpId === exp.experiment_id;
            const label =
              exp.experiment_id === "exp1_indistribution"
                ? "Exp 1: In-Distribution"
                : exp.experiment_id === "exp2_unseen_attacks"
                ? "Exp 2: Unseen Attacks"
                : "Exp 3: Unseen Polling";

            return (
              <button
                key={exp.experiment_id}
                onClick={() => setSelectedExpId(exp.experiment_id)}
                style={{
                  padding: "0.45rem 0.85rem",
                  borderRadius: "4px",
                  border: "none",
                  background: isSelected ? "#ffffff" : "transparent",
                  color: isSelected ? "#000000" : "#888888",
                  fontSize: "0.75rem",
                  fontFamily: "var(--font-mono)",
                  fontWeight: isSelected ? 700 : 500,
                  cursor: "pointer",
                  transition: "all 0.15s ease",
                }}
              >
                {label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Experiment Overview Banner */}
      {report && (
        <div className="card" style={{ marginBottom: "1.75rem" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "0.75rem", flexWrap: "wrap", gap: "1rem" }}>
            <div>
              <h3 style={{ fontSize: "1.125rem", fontWeight: 700, color: "#ffffff", margin: 0, letterSpacing: "-0.01em" }}>
                {report.name}
              </h3>
              <p style={{ fontSize: "0.8125rem", color: "#a1a1a1", marginTop: "0.25rem" }}>
                {report.description}
              </p>
            </div>
            <span className="badge badge-info">{report.split_strategy}</span>
          </div>

          <div style={{ display: "flex", gap: "2.5rem", marginTop: "1rem", paddingTop: "0.85rem", borderTop: "1px solid rgba(255, 255, 255, 0.05)", fontSize: "0.8125rem", fontFamily: "var(--font-mono)", color: "#777777", flexWrap: "wrap" }}>
            <span>DATASET: <strong style={{ color: "#ededed" }}>{report.dataset_size}</strong></span>
            <span>TRAIN: <strong style={{ color: "#ededed" }}>{report.n_train}</strong></span>
            <span>VAL: <strong style={{ color: "#ededed" }}>{report.n_val}</strong></span>
            <span>TEST: <strong style={{ color: "#ededed" }}>{report.n_test}</strong> ({report.positives_test} attacks)</span>
          </div>
        </div>
      )}

      {/* Model Benchmark Comparison Table */}
      {report && (
        <div className="card" style={{ marginBottom: "1.75rem", padding: "1.25rem" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
            <h3 style={{ fontSize: "0.9375rem", fontWeight: 700, color: "#ffffff", margin: 0 }}>
              Model Performance Comparison Matrix
            </h3>
            <span style={{ fontSize: "0.75rem", color: "#666666", fontFamily: "var(--font-mono)" }}>
              CLICK ROW TO INSPECT CONFUSION MATRIX
            </span>
          </div>

          <div style={{ overflowX: "auto" }}>
            <table className="clean-table">
              <thead>
                <tr>
                  <th>Model Architecture</th>
                  <th style={{ textAlign: "right" }}>Accuracy</th>
                  <th style={{ textAlign: "right" }}>Precision</th>
                  <th style={{ textAlign: "right" }}>Recall</th>
                  <th style={{ textAlign: "right" }}>F1 Score</th>
                  <th style={{ textAlign: "right" }}>FPR</th>
                  <th style={{ textAlign: "right" }}>PR-AUC</th>
                  <th style={{ textAlign: "right" }}>Avg Precision</th>
                  <th style={{ textAlign: "right" }}>Latency (ms)</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(report.models).map(([mKey, mRes]) => {
                  const isSelected = selectedModelKey === mKey;
                  const isNumPy = mKey === "numpy_mlp";

                  return (
                    <tr
                      key={mKey}
                      onClick={() => setSelectedModelKey(mKey)}
                      style={{
                        background: isSelected ? "rgba(0, 240, 255, 0.05)" : "transparent",
                        cursor: "pointer",
                      }}
                    >
                      <td style={{ fontWeight: 600, color: isSelected ? "#00f0ff" : "#ededed" }}>
                        {mRes.model_name || mKey}
                        {isNumPy && <span className="badge badge-normal" style={{ marginLeft: "0.6rem", fontSize: "0.6rem" }}>PRODUCTION</span>}
                      </td>
                      <td style={{ textAlign: "right" }}>{(mRes.accuracy * 100).toFixed(2)}%</td>
                      <td style={{ textAlign: "right" }}>{(mRes.precision * 100).toFixed(2)}%</td>
                      <td style={{ textAlign: "right" }}>{(mRes.recall * 100).toFixed(2)}%</td>
                      <td style={{ textAlign: "right", color: "#00f5a0", fontWeight: 700 }}>{mRes.f1.toFixed(4)}</td>
                      <td style={{ textAlign: "right", color: mRes.false_positive_rate > 0.05 ? "#ff3366" : "#888888" }}>
                        {(mRes.false_positive_rate * 100).toFixed(2)}%
                      </td>
                      <td style={{ textAlign: "right" }}>{mRes.pr_auc.toFixed(4)}</td>
                      <td style={{ textAlign: "right" }}>{mRes.average_precision.toFixed(4)}</td>
                      <td style={{ textAlign: "right", color: "#8b5cf6" }}>{mRes.inference_latency_ms.toFixed(4)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Confusion Matrix & Scenario Breakdown */}
      {activeModel && (
        <div style={{ display: "grid", gridTemplateColumns: "360px 1fr", gap: "1.5rem" }}>
          {/* 2x2 Matrix */}
          <div className="card">
            <h3 style={{ fontSize: "0.9375rem", fontWeight: 700, color: "#ffffff", marginBottom: "0.35rem" }}>
              Confusion Matrix: {activeModel.model_name}
            </h3>
            <p style={{ fontSize: "0.75rem", color: "#888888", marginBottom: "1.25rem" }}>
              Evaluated on {report?.n_test} test windows with client-isolated partitioning.
            </p>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.75rem" }}>
              {/* TP */}
              <div style={{ background: "#05140b", border: "1px solid rgba(0, 245, 160, 0.3)", padding: "1.1rem", borderRadius: "6px", textAlign: "center" }}>
                <span style={{ fontSize: "0.6875rem", color: "#00f5a0", fontFamily: "var(--font-mono)", fontWeight: 600 }}>TRUE POSITIVE</span>
                <div style={{ fontSize: "1.875rem", fontWeight: 800, color: "#00f5a0", fontFamily: "var(--font-mono)", margin: "0.2rem 0" }}>
                  {activeModel.tp}
                </div>
                <span style={{ fontSize: "0.6875rem", color: "#666666" }}>Attacks detected</span>
              </div>

              {/* FP */}
              <div style={{ background: "#1a060a", border: "1px solid rgba(255, 51, 102, 0.3)", padding: "1.1rem", borderRadius: "6px", textAlign: "center" }}>
                <span style={{ fontSize: "0.6875rem", color: "#ff3366", fontFamily: "var(--font-mono)", fontWeight: 600 }}>FALSE POSITIVE</span>
                <div style={{ fontSize: "1.875rem", fontWeight: 800, color: "#ff3366", fontFamily: "var(--font-mono)", margin: "0.2rem 0" }}>
                  {activeModel.fp}
                </div>
                <span style={{ fontSize: "0.6875rem", color: "#666666" }}>False alarms</span>
              </div>

              {/* FN */}
              <div style={{ background: "#140e04", border: "1px solid rgba(245, 166, 35, 0.3)", padding: "1.1rem", borderRadius: "6px", textAlign: "center" }}>
                <span style={{ fontSize: "0.6875rem", color: "#f5a623", fontFamily: "var(--font-mono)", fontWeight: 600 }}>FALSE NEGATIVE</span>
                <div style={{ fontSize: "1.875rem", fontWeight: 800, color: "#f5a623", fontFamily: "var(--font-mono)", margin: "0.2rem 0" }}>
                  {activeModel.fn}
                </div>
                <span style={{ fontSize: "0.6875rem", color: "#666666" }}>Missed attacks</span>
              </div>

              {/* TN */}
              <div style={{ background: "#05101a", border: "1px solid rgba(0, 240, 255, 0.3)", padding: "1.1rem", borderRadius: "6px", textAlign: "center" }}>
                <span style={{ fontSize: "0.6875rem", color: "#00f0ff", fontFamily: "var(--font-mono)", fontWeight: 600 }}>TRUE NEGATIVE</span>
                <div style={{ fontSize: "1.875rem", fontWeight: 800, color: "#00f0ff", fontFamily: "var(--font-mono)", margin: "0.2rem 0" }}>
                  {activeModel.tn}
                </div>
                <span style={{ fontSize: "0.6875rem", color: "#666666" }}>Normal cleared</span>
              </div>
            </div>
          </div>

          {/* Scenario Breakdown */}
          <div className="card">
            <h3 style={{ fontSize: "0.9375rem", fontWeight: 700, color: "#ffffff", marginBottom: "0.35rem" }}>
              Per-Scenario Recall Breakdown
            </h3>
            <p style={{ fontSize: "0.75rem", color: "#888888", marginBottom: "1.25rem" }}>
              Detection recall across individual traffic generation profiles.
            </p>

            <div style={{ display: "flex", flexDirection: "column", gap: "0.65rem", maxHeight: "420px", overflowY: "auto" }}>
              {Object.entries(activeModel.per_scenario).map(([scenName, scenData]) => {
                const isAttack = scenName.startsWith("attack");
                const recallPct = scenData.recall * 100;
                const isFailedScraping = scenName === "attack_high_volume_scraping" && recallPct === 0;

                return (
                  <div
                    key={scenName}
                    style={{
                      background: isFailedScraping ? "rgba(255, 51, 102, 0.08)" : "#080808",
                      padding: "0.6rem 0.85rem",
                      borderRadius: "6px",
                      border: isFailedScraping ? "1px solid #ff3366" : "1px solid rgba(255, 255, 255, 0.06)",
                    }}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.35rem" }}>
                      <span style={{ fontSize: "0.75rem", color: "#ededed", fontFamily: "var(--font-mono)" }}>
                        {scenName}
                        {isFailedScraping && (
                          <span className="badge badge-attack" style={{ marginLeft: "0.6rem", fontSize: "0.6rem" }}>
                            CRITICAL FAILURE (0%)
                          </span>
                        )}
                      </span>
                      <span style={{ fontSize: "0.75rem", fontFamily: "var(--font-mono)", color: recallPct > 90 ? "#00f5a0" : recallPct === 0 ? "#ff3366" : "#f5a623" }}>
                        {recallPct.toFixed(1)}% ({scenData.tp}/{scenData.tp + scenData.fn})
                      </span>
                    </div>

                    <div style={{ width: "100%", height: "4px", background: "#1a1a1a", borderRadius: "2px", overflow: "hidden" }}>
                      <div
                        style={{
                          height: "100%",
                          width: `${recallPct}%`,
                          background: isFailedScraping ? "#ff3366" : isAttack ? "#00f0ff" : "#00f5a0",
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
