import React from "react";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BarChart2,
  CheckCircle,
  Cpu,
  Layers,
  Network,
  ShieldAlert,
} from "lucide-react";
import { ModelArchitecture, ModelSummary } from "../types";
import { NavTab } from "../components/Navigation";

interface OverviewPageProps {
  summary: ModelSummary | null;
  architecture: ModelArchitecture | null;
  onNavigate: (tab: NavTab) => void;
}

export const OverviewPage: React.FC<OverviewPageProps> = ({
  summary,
  architecture,
  onNavigate,
}) => {
  return (
    <div style={{ padding: "2rem", maxWidth: "1200px", margin: "0 auto" }}>
      {/* Hero Section */}
      <div style={{ marginBottom: "2rem" }}>
        <div style={{ display: "inline-block", marginBottom: "0.5rem" }}>
          <span className="badge badge-info">Academic Research Visualizer</span>
        </div>
        <h2 style={{ fontSize: "1.875rem", fontWeight: 800, color: "#f8fafc", marginBottom: "0.75rem" }}>
          From-Scratch Neural Network Laboratory
        </h2>
        <p style={{ color: "#94a3b8", fontSize: "1rem", lineHeight: 1.6, maxWidth: "900px" }}>
          This laboratory visualizes the actual internal operations, weights, intermediate forward propagation values,
          and experimental evaluation of our passive website traffic anomaly detection neural network engine.
          Implemented entirely from mathematical foundations without PyTorch, TensorFlow, or autograd libraries.
        </p>
      </div>

      {/* High-Level Spec Grid */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
          gap: "1.25rem",
          marginBottom: "2rem",
        }}
      >
        <div className="card">
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.75rem" }}>
            <Network size={20} color="#38bdf8" />
            <h3 style={{ fontSize: "0.95rem", fontWeight: 600, color: "#f1f5f9" }}>Observation Model</h3>
          </div>
          <p style={{ fontSize: "1.4rem", fontWeight: 700, color: "#38bdf8", marginBottom: "0.25rem" }}>
            13 Features
          </p>
          <p style={{ fontSize: "0.8rem", color: "#94a3b8" }}>
            60-second client-aggregated behavioural windows capturing rate, diversity, error ratios, and temporal intervals.
          </p>
        </div>

        <div className="card">
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.75rem" }}>
            <Cpu size={20} color="#10b981" />
            <h3 style={{ fontSize: "0.95rem", fontWeight: 600, color: "#f1f5f9" }}>Dual Engine Architecture</h3>
          </div>
          <p style={{ fontSize: "1.4rem", fontWeight: 700, color: "#10b981", marginBottom: "0.25rem" }}>
            Scalar &bull; NumPy
          </p>
          <p style={{ fontSize: "0.8rem", color: "#94a3b8" }}>
            Scalar reference engine (<code className="font-mono">nn/</code>) and vectorized NumPy engine (<code className="font-mono">nn_numpy/</code>) with bit-level mathematical parity.
          </p>
        </div>

        <div className="card">
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.75rem" }}>
            <CheckCircle size={20} color="#a855f7" />
            <h3 style={{ fontSize: "0.95rem", fontWeight: 600, color: "#f1f5f9" }}>Current Active Model</h3>
          </div>
          <p style={{ fontSize: "1.2rem", fontWeight: 700, color: "#a855f7", marginBottom: "0.25rem" }}>
            {summary ? summary.architecture_desc : "13 -> 16 -> 8 -> 1"}
          </p>
          <p style={{ fontSize: "0.8rem", color: "#94a3b8" }}>
            {summary ? `${summary.total_parameters} learnable parameters (${summary.engine} engine)` : "Loading..."}
          </p>
        </div>

        <div className="card">
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.75rem" }}>
            <BarChart2 size={20} color="#f59e0b" />
            <h3 style={{ fontSize: "0.95rem", fontWeight: 600, color: "#f1f5f9" }}>In-Distribution Accuracy</h3>
          </div>
          <p style={{ fontSize: "1.4rem", fontWeight: 700, color: "#f59e0b", marginBottom: "0.25rem" }}>
            99.49%
          </p>
          <p style={{ fontSize: "0.8rem", color: "#94a3b8" }}>
            Evaluated on 779 test windows across 13 distinct traffic scenarios (Exp 1). F1 Score: 0.9952.
          </p>
        </div>
      </div>

      {/* Critical Research Finding Callout */}
      <div
        className="card"
        style={{
          borderLeft: "4px solid #ef4444",
          background: "linear-gradient(90deg, rgba(239, 68, 68, 0.08), transparent)",
          marginBottom: "2rem",
          padding: "1.5rem",
        }}
      >
        <div style={{ display: "flex", alignItems: "flex-start", gap: "1rem" }}>
          <AlertTriangle size={28} color="#ef4444" style={{ flexShrink: 0, marginTop: "0.25rem" }} />
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "0.5rem" }}>
              <h3 style={{ fontSize: "1.1rem", fontWeight: 700, color: "#f8fafc" }}>
                Critical Phase 5 Discovery: The Zero-Error Scraping Blindspot
              </h3>
              <span className="badge badge-attack">0% Recall on Novel Scraping</span>
            </div>
            <p style={{ color: "#cbd5e1", fontSize: "0.9rem", lineHeight: 1.6, marginBottom: "0.75rem" }}>
              In Experiment 2 (Unseen Zero-Day Attacks), our supervised MLP achieved 0% detection recall on{" "}
              <code className="font-mono">attack_high_volume_scraping</code>. Because the scraping bot queries valid public URLs,
              all requests return HTTP 200 OK with zero 4xx/5xx errors and zero auth failures.
              Having learned during supervised training that attacks produce elevated error ratios, the network predicted a probability of{" "}
              <strong>p = 0.0001</strong> (normal).
            </p>
            <p style={{ color: "#94a3b8", fontSize: "0.85rem", marginBottom: "1rem" }}>
              <strong>Academic Implication:</strong> This empirical result demonstrates that supervised classification alone is insufficient for website anomaly detection. It establishes the rigorous justification for Phase 7 (Unsupervised Autoencoder Anomaly Detection).
            </p>
            <button
              className="btn btn-secondary"
              onClick={() => onNavigate("traffic")}
              style={{ fontSize: "0.8rem", gap: "0.4rem" }}
            >
              Inspect Scraping Failure in Traffic Lab <ArrowRight size={14} />
            </button>
          </div>
        </div>
      </div>

      {/* Quick Launch Cards */}
      <h3 style={{ fontSize: "1.2rem", fontWeight: 700, color: "#f1f5f9", marginBottom: "1rem" }}>
        Laboratory Modules
      </h3>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
          gap: "1.25rem",
        }}
      >
        <div
          className="card"
          style={{ cursor: "pointer" }}
          onClick={() => onNavigate("architecture")}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.5rem" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <Network size={18} color="#38bdf8" />
              <h4 style={{ fontSize: "1rem", fontWeight: 600, color: "#f8fafc" }}>Network Architecture</h4>
            </div>
            <ArrowRight size={16} color="#64748b" />
          </div>
          <p style={{ fontSize: "0.85rem", color: "#94a3b8" }}>
            Explore the layer graph, neuron connections, weights, biases, and activation functions. Inspect individual neurons interactively.
          </p>
        </div>

        <div
          className="card"
          style={{ cursor: "pointer" }}
          onClick={() => onNavigate("forward")}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.5rem" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <Activity size={18} color="#10b981" />
              <h4 style={{ fontSize: "1rem", fontWeight: 600, color: "#f8fafc" }}>Forward Pass Tracer</h4>
            </div>
            <ArrowRight size={16} color="#64748b" />
          </div>
          <p style={{ fontSize: "0.85rem", color: "#94a3b8" }}>
            Step through inference layer-by-layer. Observe intermediate weighted sums Z = XW + b and post-activation values A = ReLU(Z) in real time.
          </p>
        </div>

        <div
          className="card"
          style={{ cursor: "pointer" }}
          onClick={() => onNavigate("parameters")}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.5rem" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <Layers size={18} color="#a855f7" />
              <h4 style={{ fontSize: "1rem", fontWeight: 600, color: "#f8fafc" }}>Parameters & Heatmaps</h4>
            </div>
            <ArrowRight size={16} color="#64748b" />
          </div>
          <p style={{ fontSize: "0.85rem", color: "#94a3b8" }}>
            Inspect full weight matrices as heatmaps. Examine bias distributions, parameter magnitude histograms, and sparsity metrics.
          </p>
        </div>

        <div
          className="card"
          style={{ cursor: "pointer" }}
          onClick={() => onNavigate("experiments")}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.5rem" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <BarChart2 size={18} color="#f59e0b" />
              <h4 style={{ fontSize: "1rem", fontWeight: 600, color: "#f8fafc" }}>Empirical Experiments</h4>
            </div>
            <ArrowRight size={16} color="#64748b" />
          </div>
          <p style={{ fontSize: "0.85rem", color: "#94a3b8" }}>
            Review Phase 5 results: Exp 1 (In-Distribution), Exp 2 (Unseen Attacks), and Exp 3 (Unseen Polling). Confusion matrices and PR-AUC curves.
          </p>
        </div>

        <div
          className="card"
          style={{ cursor: "pointer" }}
          onClick={() => onNavigate("traffic")}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.5rem" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <ShieldAlert size={18} color="#ef4444" />
              <h4 style={{ fontSize: "1rem", fontWeight: 600, color: "#f8fafc" }}>Traffic Lab & Failures</h4>
            </div>
            <ArrowRight size={16} color="#64748b" />
          </div>
          <p style={{ fontSize: "0.85rem", color: "#94a3b8" }}>
            Analyze 26 curated real behavioural windows across 13 scenarios. Investigate why certain novel attack behaviors evade supervised detection.
          </p>
        </div>

        <div
          className="card"
          style={{ cursor: "pointer" }}
          onClick={() => onNavigate("compare")}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.5rem" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <Cpu size={18} color="#38bdf8" />
              <h4 style={{ fontSize: "1rem", fontWeight: 600, color: "#f8fafc" }}>Model Comparison</h4>
            </div>
            <ArrowRight size={16} color="#64748b" />
          </div>
          <p style={{ fontSize: "0.85rem", color: "#94a3b8" }}>
            Run simultaneous inference across Threshold, Logistic Regression, Scalar MLP, and NumPy MLP. Compare latency and prediction agreement.
          </p>
        </div>
      </div>
    </div>
  );
};
