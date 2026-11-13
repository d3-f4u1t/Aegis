import React from "react";
import {
  Activity,
  AlertOctagon,
  ArrowUpRight,
  BarChart3,
  Binary,
  Cpu,
  Layers,
  Network,
  ShieldAlert,
  Sparkles,
  Zap,
} from "lucide-react";
import type { ModelArchitecture, ModelSummary } from "../types";
import type { NavTab } from "../components/Navigation";

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
    <div style={{ padding: "3rem 2.5rem", maxWidth: "1400px", margin: "0 auto" }}>
      {/* Hero Header */}
      <div style={{ marginBottom: "2.75rem" }}>
        <div className="section-kicker">Academic Deliverable &bull; Passive Telemetry Classifier</div>
        <h1
          style={{
            fontSize: "2.5rem",
            fontWeight: 800,
            letterSpacing: "-0.035em",
            color: "#ffffff",
            margin: "0.25rem 0 0.85rem 0",
            lineHeight: 1.15,
          }}
        >
          From-Scratch Neural Network Laboratory
        </h1>
        <p
          style={{
            color: "#a1a1a1",
            fontSize: "1.0625rem",
            lineHeight: 1.6,
            maxWidth: "840px",
            letterSpacing: "-0.01em",
          }}
        >
          A pure-mathematical deep neural network engine implemented without PyTorch, TensorFlow, or autograd libraries.
          Provides complete visibility into weight parameters, layer computations, mathematical parity, and Phase 5 empirical evaluations.
        </p>
      </div>

      {/* KPI Stats Grid */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
          gap: "1.25rem",
          marginBottom: "2.5rem",
        }}
      >
        <div className="card">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem" }}>
            <span style={{ fontSize: "0.6875rem", fontFamily: "var(--font-mono)", color: "#777777", textTransform: "uppercase", letterSpacing: "0.06em" }}>
              Observation Unit
            </span>
            <Network size={15} color="#00f0ff" />
          </div>
          <div style={{ fontSize: "1.75rem", fontWeight: 700, color: "#ffffff", fontFamily: "var(--font-mono)", letterSpacing: "-0.02em" }}>
            13 Features
          </div>
          <p style={{ fontSize: "0.8125rem", color: "#888888", marginTop: "0.35rem", lineHeight: 1.4 }}>
            60-second sliding client windows capturing rate, error ratios, diversity, and packet intervals.
          </p>
        </div>

        <div className="card">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem" }}>
            <span style={{ fontSize: "0.6875rem", fontFamily: "var(--font-mono)", color: "#777777", textTransform: "uppercase", letterSpacing: "0.06em" }}>
              Engine Implementations
            </span>
            <Cpu size={15} color="#00f5a0" />
          </div>
          <div style={{ fontSize: "1.75rem", fontWeight: 700, color: "#ffffff", fontFamily: "var(--font-mono)", letterSpacing: "-0.02em" }}>
            Scalar &bull; NumPy
          </div>
          <p style={{ fontSize: "0.8125rem", color: "#888888", marginTop: "0.35rem", lineHeight: 1.4 }}>
            Audited scalar engine (<code className="font-mono">nn/</code>) and vectorized BLAS engine (<code className="font-mono">nn_numpy/</code>) with bit-level parity.
          </p>
        </div>

        <div className="card">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem" }}>
            <span style={{ fontSize: "0.6875rem", fontFamily: "var(--font-mono)", color: "#777777", textTransform: "uppercase", letterSpacing: "0.06em" }}>
              Network Architecture
            </span>
            <Layers size={15} color="#8b5cf6" />
          </div>
          <div style={{ fontSize: "1.75rem", fontWeight: 700, color: "#ffffff", fontFamily: "var(--font-mono)", letterSpacing: "-0.02em" }}>
            {summary ? summary.architecture_desc : "13 → 16 → 8 → 1"}
          </div>
          <p style={{ fontSize: "0.8125rem", color: "#888888", marginTop: "0.35rem", lineHeight: 1.4 }}>
            {summary ? `${summary.total_parameters} learnable parameters` : "369 parameters"} &bull; ReLU & Sigmoid activations.
          </p>
        </div>

        <div className="card">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem" }}>
            <span style={{ fontSize: "0.6875rem", fontFamily: "var(--font-mono)", color: "#777777", textTransform: "uppercase", letterSpacing: "0.06em" }}>
              In-Distribution F1
            </span>
            <Zap size={15} color="#00f5a0" />
          </div>
          <div style={{ fontSize: "1.75rem", fontWeight: 700, color: "#00f5a0", fontFamily: "var(--font-mono)", letterSpacing: "-0.02em" }}>
            0.9952
          </div>
          <p style={{ fontSize: "0.8125rem", color: "#888888", marginTop: "0.35rem", lineHeight: 1.4 }}>
            99.49% accuracy evaluated across 779 test windows in Experiment 1.
          </p>
        </div>
      </div>

      {/* Forensic Alert: Zero-Error Scraping Blindspot */}
      <div
        className="card"
        style={{
          borderLeft: "3px solid var(--accent-rose)",
          background: "linear-gradient(135deg, rgba(255, 51, 102, 0.06) 0%, rgba(10, 10, 10, 0.95) 100%)",
          marginBottom: "2.75rem",
          padding: "1.75rem 2rem",
        }}
      >
        <div style={{ display: "flex", alignItems: "flex-start", gap: "1.25rem" }}>
          <AlertOctagon size={28} color="#ff3366" style={{ flexShrink: 0, marginTop: "0.2rem" }} />
          <div style={{ flex: 1 }}>
            <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "0.45rem" }}>
              <h3 style={{ fontSize: "1.125rem", fontWeight: 700, color: "#ffffff", letterSpacing: "-0.02em" }}>
                Empirical Research Finding: The Zero-Error Scraping Blindspot
              </h3>
              <span className="badge badge-attack">0% RECALL ON NOVEL SCRAPING</span>
            </div>
            <p style={{ color: "#b5b5b5", fontSize: "0.875rem", lineHeight: 1.6, marginBottom: "0.85rem" }}>
              In Experiment 2 (Unseen Zero-Day Attacks), our supervised neural classifier suffered <strong>0% detection recall</strong> on{" "}
              <code className="font-mono" style={{ color: "#ff88a3" }}>attack_high_volume_scraping</code>. Because the scraping bot queries valid public URLs,
              all requests return HTTP 200 OK with zero 4xx/5xx errors and zero auth failures.
              Having learned during supervised training that attacks manifest through error spikes, the network assigned a probability of{" "}
              <strong style={{ color: "#ffffff", fontFamily: "var(--font-mono)" }}>p = 0.0001</strong> (normal).
            </p>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: "1rem" }}>
              <span style={{ fontSize: "0.8125rem", color: "#888888" }}>
                <strong>Scientific Implication:</strong> Proves the necessity of transitioning from purely supervised models to Phase 7 Unsupervised Autoencoders.
              </span>
              <button
                className="btn btn-secondary"
                onClick={() => onNavigate("traffic")}
                style={{ fontSize: "0.75rem", gap: "0.4rem" }}
              >
                Inspect Forensic Sample <ArrowUpRight size={13} />
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Pipeline Conduit Flowchart */}
      <div style={{ marginBottom: "2.75rem" }}>
        <div className="section-kicker">End-to-End Pipeline</div>
        <h3 style={{ fontSize: "1.25rem", fontWeight: 700, color: "#ffffff", marginBottom: "1rem", letterSpacing: "-0.02em" }}>
          Inference & Telemetry Flow
        </h3>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
            gap: "0.75rem",
            background: "#080808",
            padding: "1rem",
            borderRadius: "8px",
            border: "1px solid rgba(255, 255, 255, 0.07)",
          }}
        >
          <div style={{ background: "#0f0f0f", padding: "1rem", borderRadius: "6px", border: "1px solid rgba(255, 255, 255, 0.05)" }}>
            <span style={{ fontSize: "0.6875rem", color: "#666666", fontFamily: "var(--font-mono)" }}>STEP 01</span>
            <div style={{ fontWeight: 600, color: "#ededed", fontSize: "0.875rem", margin: "0.2rem 0" }}>Web Telemetry</div>
            <p style={{ fontSize: "0.75rem", color: "#888888" }}>Passive JSONL request log stream.</p>
          </div>

          <div style={{ background: "#0f0f0f", padding: "1rem", borderRadius: "6px", border: "1px solid rgba(255, 255, 255, 0.05)" }}>
            <span style={{ fontSize: "0.6875rem", color: "#666666", fontFamily: "var(--font-mono)" }}>STEP 02</span>
            <div style={{ fontWeight: 600, color: "#ededed", fontSize: "0.875rem", margin: "0.2rem 0" }}>60s Aggregation</div>
            <p style={{ fontSize: "0.75rem", color: "#888888" }}>Sliding window grouping by client hash.</p>
          </div>

          <div style={{ background: "#0f0f0f", padding: "1rem", borderRadius: "6px", border: "1px solid rgba(255, 255, 255, 0.05)" }}>
            <span style={{ fontSize: "0.6875rem", color: "#00f0ff", fontFamily: "var(--font-mono)" }}>STEP 03</span>
            <div style={{ fontWeight: 600, color: "#00f0ff", fontSize: "0.875rem", margin: "0.2rem 0" }}>13-Feature Vector</div>
            <p style={{ fontSize: "0.75rem", color: "#888888" }}>Extracted behavioural representation.</p>
          </div>

          <div style={{ background: "#0f0f0f", padding: "1rem", borderRadius: "6px", border: "1px solid rgba(255, 255, 255, 0.05)" }}>
            <span style={{ fontSize: "0.6875rem", color: "#666666", fontFamily: "var(--font-mono)" }}>STEP 04</span>
            <div style={{ fontWeight: 600, color: "#ededed", fontSize: "0.875rem", margin: "0.2rem 0" }}>StandardScaler</div>
            <p style={{ fontSize: "0.75rem", color: "#888888" }}>Z-score normalization (train-fitted).</p>
          </div>

          <div style={{ background: "#0f0f0f", padding: "1rem", borderRadius: "6px", border: "1px solid rgba(255, 255, 255, 0.05)" }}>
            <span style={{ fontSize: "0.6875rem", color: "#8b5cf6", fontFamily: "var(--font-mono)" }}>STEP 05</span>
            <div style={{ fontWeight: 600, color: "#8b5cf6", fontSize: "0.875rem", margin: "0.2rem 0" }}>Dense MLP [16, 8]</div>
            <p style={{ fontSize: "0.75rem", color: "#888888" }}>Vectorized forward propagation.</p>
          </div>

          <div style={{ background: "#0f0f0f", padding: "1rem", borderRadius: "6px", border: "1px solid rgba(255, 255, 255, 0.05)" }}>
            <span style={{ fontSize: "0.6875rem", color: "#00f5a0", fontFamily: "var(--font-mono)" }}>STEP 06</span>
            <div style={{ fontWeight: 600, color: "#00f5a0", fontSize: "0.875rem", margin: "0.2rem 0" }}>Sigmoid Output</div>
            <p style={{ fontSize: "0.75rem", color: "#888888" }}>Probability score &amp; classification.</p>
          </div>
        </div>
      </div>

      {/* Interactive Tool Suite Grid */}
      <div>
        <div className="section-kicker">Interactive Modules</div>
        <h3 style={{ fontSize: "1.25rem", fontWeight: 700, color: "#ffffff", marginBottom: "1rem", letterSpacing: "-0.02em" }}>
          Laboratory Exploration Tools
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
                <Network size={16} color="#00f0ff" />
                <h4 style={{ fontSize: "0.9375rem", fontWeight: 600, color: "#ffffff" }}>Network Topology</h4>
              </div>
              <ArrowUpRight size={15} color="#666666" />
            </div>
            <p style={{ fontSize: "0.8125rem", color: "#888888", lineHeight: 1.5 }}>
              Interactive SVG graph displaying all 13 input features, hidden layers (16, 8), and output sigmoid. Filter connections by weight magnitude.
            </p>
          </div>

          <div
            className="card"
            style={{ cursor: "pointer" }}
            onClick={() => onNavigate("forward")}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.5rem" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <Activity size={16} color="#00f5a0" />
                <h4 style={{ fontSize: "0.9375rem", fontWeight: 600, color: "#ffffff" }}>Forward Tracer</h4>
              </div>
              <ArrowUpRight size={15} color="#666666" />
            </div>
            <p style={{ fontSize: "0.8125rem", color: "#888888", lineHeight: 1.5 }}>
              Step through inference layer-by-layer. Observe exact pre-activation sums Z and post-activation values A with verified engine parity.
            </p>
          </div>

          <div
            className="card"
            style={{ cursor: "pointer" }}
            onClick={() => onNavigate("parameters")}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.5rem" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <Layers size={16} color="#8b5cf6" />
                <h4 style={{ fontSize: "0.9375rem", fontWeight: 600, color: "#ffffff" }}>Weight Heatmaps</h4>
              </div>
              <ArrowUpRight size={15} color="#666666" />
            </div>
            <p style={{ fontSize: "0.8125rem", color: "#888888", lineHeight: 1.5 }}>
              Inspect high-resolution weight matrices, parameter sparsity distributions, Frobenius norms, and bias vectors for every layer.
            </p>
          </div>

          <div
            className="card"
            style={{ cursor: "pointer" }}
            onClick={() => onNavigate("experiments")}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.5rem" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <BarChart3 size={16} color="#f5a623" />
                <h4 style={{ fontSize: "0.9375rem", fontWeight: 600, color: "#ffffff" }}>Phase 5 Experiments</h4>
              </div>
              <ArrowUpRight size={15} color="#666666" />
            </div>
            <p style={{ fontSize: "0.8125rem", color: "#888888", lineHeight: 1.5 }}>
              Comprehensive benchmarks covering Exp 1 (In-Distribution), Exp 2 (Unseen Attacks), and Exp 3 (Unseen Polling) with confusion matrices.
            </p>
          </div>

          <div
            className="card"
            style={{ cursor: "pointer" }}
            onClick={() => onNavigate("traffic")}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.5rem" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <ShieldAlert size={16} color="#ff3366" />
                <h4 style={{ fontSize: "0.9375rem", fontWeight: 600, color: "#ffffff" }}>Traffic Lab &amp; Blindspots</h4>
              </div>
              <ArrowUpRight size={15} color="#666666" />
            </div>
            <p style={{ fontSize: "0.8125rem", color: "#888888", lineHeight: 1.5 }}>
              Analyze 26 curated authentic behavioural windows across 13 scenarios. Investigate why novel scraping evades supervised detection.
            </p>
          </div>

          <div
            className="card"
            style={{ cursor: "pointer" }}
            onClick={() => onNavigate("compare")}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.5rem" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <Cpu size={16} color="#00f0ff" />
                <h4 style={{ fontSize: "0.9375rem", fontWeight: 600, color: "#ffffff" }}>Engine Benchmark</h4>
              </div>
              <ArrowUpRight size={15} color="#666666" />
            </div>
            <p style={{ fontSize: "0.8125rem", color: "#888888", lineHeight: 1.5 }}>
              Simultaneous 4-model evaluation: Threshold vs Logistic Regression vs Scalar MLP vs NumPy MLP. Demonstrates ~170x BLAS speedup.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
