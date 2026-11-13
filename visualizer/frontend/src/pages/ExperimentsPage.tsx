import React, { useEffect, useState } from "react";
import { AlertCircle, BarChart2, CheckCircle2, ChevronRight, FileText, Info } from "lucide-react";
import { ExperimentModelResult, ExperimentReport } from "../types";
import { api } from "../services/api";

export const ExperimentsPage: React.FC = () => {
  const [experiments, setExperiments] = useState<Array<{ experiment_id: string; name: string; description: string }>>([]);
  const [selectedExpId, setSelectedExpId] = useState<string>("exp1_indistribution");
  const [report, setReport] = useState<ExperimentReport | null>(null);
  const [selectedModelKey, setSelectedModelKey] = useState<string>("numpy_mlp");
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Load experiment list
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
        if (!isMounted) return;
        setError("Failed to load experiments list: " + err.message);
      }
    }
    loadExps();
    return () => {
      isMounted = false;
    };
  }, []);

  // Load report when selectedExpId changes
  useEffect(() => {
    let isMounted = true;
    async function loadReport() {
      if (!selectedExpId) return;
      setLoading(true);
      setError(null);
      try {
        const r = await api.getExperimentReport(selectedExpId);
        if (!isMounted) return;
        setReport(r);
        // Default to numpy_mlp if present
        if (r.models["numpy_mlp"]) {
          setSelectedModelKey("numpy_mlp");
        } else {
          setSelectedModelKey(Object.keys(r.models)[0] || "");
        }
      } catch (err: any) {
        if (!isMounted) return;
        setError("Failed to load experiment report: " + err.message);
      } finally {
        if (isMounted) setLoading(false);
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
    <div style={{ padding: "1.5rem 2rem", maxWidth: "1350px", margin: "0 auto" }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.25rem" }}>
        <div>
          <h2 style={{ fontSize: "1.5rem", fontWeight: 700, color: "#f8fafc", margin: 0 }}>
            Empirical Evaluation & Experiment Logs
          </h2>
          <p style={{ fontSize: "0.85rem", color: "#94a3b8", margin: 0 }}>
            Scientific benchmarks across 2,695 behavioral windows and 13 traffic scenarios.
          </p>
        </div>

        {/* Experiment Selector Tabs */}
        <div style={{ display: "flex", gap: "0.4rem", background: "#0e1420", padding: "0.25rem", borderRadius: "8px", border: "1px solid #1e293b" }}>
          {experiments.map((exp) => {
            const isSelected = selectedExpId === exp.experiment_id;
            return (
              <button
                key={exp.experiment_id}
                onClick={() => setSelectedExpId(exp.experiment_id)}
                style={{
                  padding: "0.45rem 0.85rem",
                  borderRadius: "6px",
                  border: "none",
                  background: isSelected ? "#2563eb" : "transparent",
                  color: isSelected ? "white" : "#94a3b8",
                  fontSize: "0.8rem",
                  fontWeight: isSelected ? 600 : 500,
                  cursor: "pointer",
                  transition: "all 0.15s ease",
                }}
              >
                {exp.experiment_id === "exp1_indistribution"
                  ? "Exp 1: In-Distribution"
                  : exp.experiment_id === "exp2_unseen_attacks"
                  ? "Exp 2: Unseen Attacks"
                  : "Exp 3: Unseen Polling"}
              </button>
            );
          })}
        </div>
      </div>

      {/* Experiment Overview Banner */}
      {report && (
        <div className="card" style={{ marginBottom: "1.5rem" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "0.5rem" }}>
            <div>
              <h3 style={{ fontSize: "1.1rem", fontWeight: 700, color: "#f8fafc", margin: 0 }}>
                {report.name}
              </h3>
              <p style={{ fontSize: "0.85rem", color: "#cbd5e1", marginTop: "0.25rem" }}>
                {report.description}
              </p>
            </div>
            <span className="badge badge-info">{report.split_strategy}</span>
          </div>

          <div style={{ display: "flex", gap: "2rem", marginTop: "0.75rem", fontSize: "0.8rem", color: "#94a3b8" }}>
            <span><strong>Total Dataset:</strong> {report.dataset_size} windows</span>
            <span><strong>Train:</strong> {report.n_train}</span>
            <span><strong>Validation:</strong> {report.n_val}</span>
            <span><strong>Test:</strong> {report.n_test} ({report.positives_test} attacks)</span>
          </div>
        </div>
      )}

      {/* Model Performance Comparison Table */}
      {report && (
        <div className="card" style={{ marginBottom: "1.5rem" }}>
          <h3 style={{ fontSize: "1rem", fontWeight: 700, color: "#f8fafc", marginBottom: "0.75rem" }}>
            Model Benchmark Comparison
          </h3>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.8rem" }}>
              <thead>
                <tr style={{ background: "#0e1420", borderBottom: "1px solid #243042", color: "#94a3b8", textAlign: "left" }}>
                  <th style={{ padding: "0.6rem 0.75rem" }}>Model</th>
                  <th style={{ padding: "0.6rem 0.5rem", textAlign: "right" }}>Accuracy</th>
                  <th style={{ padding: "0.6rem 0.5rem", textAlign: "right" }}>Precision</th>
                  <th style={{ padding: "0.6rem 0.5rem", textAlign: "right" }}>Recall</th>
                  <th style={{ padding: "0.6rem 0.5rem", textAlign: "right" }}>F1 Score</th>
                  <th style={{ padding: "0.6rem 0.5rem", textAlign: "right" }}>FPR</th>
                  <th style={{ padding: "0.6rem 0.5rem", textAlign: "right" }}>PR-AUC</th>
                  <th style={{ padding: "0.6rem 0.5rem", textAlign: "right" }}>Avg Precision</th>
                  <th style={{ padding: "0.6rem 0.75rem", textAlign: "right" }}>Latency (ms/sample)</th>
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
                        borderBottom: "1px solid #1a2332",
                        background: isSelected ? "rgba(56, 189, 248, 0.08)" : "transparent",
                        cursor: "pointer",
                      }}
                    >
                      <td style={{ padding: "0.6rem 0.75rem", fontWeight: 600, color: isSelected ? "#38bdf8" : "#f1f5f9" }}>
                        {mRes.model_name || mKey}
                        {isNumPy && <span className="badge badge-normal" style={{ marginLeft: "0.5rem", fontSize: "0.65rem" }}>PRODUCTION</span>}
                      </td>
                      <td style={{ padding: "0.6rem 0.5rem", textAlign: "right", fontFamily: "monospace" }}>
                        {(mRes.accuracy * 100).toFixed(2)}%
                      </td>
                      <td style={{ padding: "0.6rem 0.5rem", textAlign: "right", fontFamily: "monospace" }}>
                        {(mRes.precision * 100).toFixed(2)}%
                      </td>
                      <td style={{ padding: "0.6rem 0.5rem", textAlign: "right", fontFamily: "monospace" }}>
                        {(mRes.recall * 100).toFixed(2)}%
                      </td>
                      <td style={{ padding: "0.6rem 0.5rem", textAlign: "right", fontFamily: "monospace", fontWeight: 600, color: "#38bdf8" }}>
                        {mRes.f1.toFixed(4)}
                      </td>
                      <td style={{ padding: "0.6rem 0.5rem", textAlign: "right", fontFamily: "monospace", color: mRes.false_positive_rate > 0.05 ? "#ef4444" : "#4ade80" }}>
                        {(mRes.false_positive_rate * 100).toFixed(2)}%
                      </td>
                      <td style={{ padding: "0.6rem 0.5rem", textAlign: "right", fontFamily: "monospace" }}>
                        {mRes.pr_auc.toFixed(4)}
                      </td>
                      <td style={{ padding: "0.6rem 0.5rem", textAlign: "right", fontFamily: "monospace" }}>
                        {mRes.average_precision.toFixed(4)}
                      </td>
                      <td style={{ padding: "0.6rem 0.75rem", textAlign: "right", fontFamily: "monospace", color: "#a855f7" }}>
                        {mRes.inference_latency_ms.toFixed(4)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Model Deep-Dive: Confusion Matrix and Per-Scenario Recall */}
      {activeModel && (
        <div style={{ display: "grid", gridTemplateColumns: "360px 1fr", gap: "1.5rem" }}>
          {/* Confusion Matrix Card */}
          <div className="card">
            <h3 style={{ fontSize: "1rem", fontWeight: 700, color: "#f8fafc", marginBottom: "0.5rem" }}>
              Confusion Matrix: {activeModel.model_name}
            </h3>
            <p style={{ fontSize: "0.75rem", color: "#94a3b8", marginBottom: "1rem" }}>
              Evaluated on {report?.n_test} test samples with client-level isolation.
            </p>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.75rem" }}>
              {/* True Positive */}
              <div style={{ background: "#0e2017", border: "1px solid #15803d", padding: "1rem", borderRadius: "8px", textAlign: "center" }}>
                <span style={{ fontSize: "0.75rem", color: "#86efac", fontWeight: 600 }}>True Positives (TP)</span>
                <p style={{ fontSize: "1.6rem", fontWeight: 800, color: "#22c55e", margin: "0.2rem 0", fontFamily: "monospace" }}>
                  {activeModel.tp}
                </p>
                <span style={{ fontSize: "0.65rem", color: "#86efac" }}>Attacks correctly flagged</span>
              </div>

              {/* False Positive */}
              <div style={{ background: "#210f13", border: "1px solid #b91c1c", padding: "1rem", borderRadius: "8px", textAlign: "center" }}>
                <span style={{ fontSize: "0.75rem", color: "#fca5a5", fontWeight: 600 }}>False Positives (FP)</span>
                <p style={{ fontSize: "1.6rem", fontWeight: 800, color: "#ef4444", margin: "0.2rem 0", fontFamily: "monospace" }}>
                  {activeModel.fp}
                </p>
                <span style={{ fontSize: "0.65rem", color: "#fca5a5" }}>Normal flagged as attack</span>
              </div>

              {/* False Negative */}
              <div style={{ background: "#21160a", border: "1px solid #b45309", padding: "1rem", borderRadius: "8px", textAlign: "center" }}>
                <span style={{ fontSize: "0.75rem", color: "#fde047", fontWeight: 600 }}>False Negatives (FN)</span>
                <p style={{ fontSize: "1.6rem", fontWeight: 800, color: "#f59e0b", margin: "0.2rem 0", fontFamily: "monospace" }}>
                  {activeModel.fn}
                </p>
                <span style={{ fontSize: "0.65rem", color: "#fde047" }}>Attacks missed (blindspot)</span>
              </div>

              {/* True Negative */}
              <div style={{ background: "#0d1b2a", border: "1px solid #1d4ed8", padding: "1rem", borderRadius: "8px", textAlign: "center" }}>
                <span style={{ fontSize: "0.75rem", color: "#93c5fd", fontWeight: 600 }}>True Negatives (TN)</span>
                <p style={{ fontSize: "1.6rem", fontWeight: 800, color: "#38bdf8", margin: "0.2rem 0", fontFamily: "monospace" }}>
                  {activeModel.tn}
                </p>
                <span style={{ fontSize: "0.65rem", color: "#93c5fd" }}>Normal correctly cleared</span>
              </div>
            </div>
          </div>

          {/* Per-Scenario Recall Bars */}
          <div className="card">
            <h3 style={{ fontSize: "1rem", fontWeight: 700, color: "#f8fafc", marginBottom: "0.5rem" }}>
              Per-Scenario Recall & Detection Breakdown
            </h3>
            <p style={{ fontSize: "0.75rem", color: "#94a3b8", marginBottom: "1rem" }}>
              Recall percentage across individual attack and normal behavior patterns.
            </p>

            <div style={{ display: "flex", flexDirection: "column", gap: "0.6rem", maxHeight: "420px", overflowY: "auto" }}>
              {Object.entries(activeModel.per_scenario).map(([scenName, scenData]) => {
                const isAttack = scenName.startsWith("attack");
                const recallPct = scenData.recall * 100;
                const isFailedScraping = scenName === "attack_high_volume_scraping" && recallPct === 0;

                return (
                  <div
                    key={scenName}
                    style={{
                      background: isFailedScraping ? "rgba(239, 68, 68, 0.1)" : "#0e1420",
                      padding: "0.5rem 0.75rem",
                      borderRadius: "6px",
                      border: isFailedScraping ? "1px solid #ef4444" : "1px solid #1e293b",
                    }}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.3rem" }}>
                      <span style={{ fontSize: "0.8rem", color: "#f1f5f9", fontFamily: "monospace" }}>
                        {scenName}
                        {isFailedScraping && (
                          <span className="badge badge-attack" style={{ marginLeft: "0.5rem", fontSize: "0.65rem" }}>
                            CRITICAL FAILURE (0%)
                          </span>
                        )}
                      </span>
                      <span style={{ fontSize: "0.75rem", fontFamily: "monospace", color: recallPct > 90 ? "#4ade80" : recallPct === 0 ? "#ef4444" : "#f59e0b" }}>
                        {recallPct.toFixed(1)}% ({scenData.tp}/{scenData.tp + scenData.fn})
                      </span>
                    </div>

                    <div style={{ width: "100%", height: "6px", background: "#1e293b", borderRadius: "3px", overflow: "hidden" }}>
                      <div
                        style={{
                          height: "100%",
                          width: `${recallPct}%`,
                          background: isFailedScraping ? "#ef4444" : isAttack ? "#38bdf8" : "#10b981",
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
