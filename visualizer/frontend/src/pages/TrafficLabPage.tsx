import React, { useEffect, useState } from "react";
import {
  AlertTriangle,
  ArrowRight,
  CheckCircle,
  Database,
  Filter,
  Flame,
  Search,
  ShieldAlert,
} from "lucide-react";
import { TrafficSampleItem } from "../types";
import { api } from "../services/api";

export const TrafficLabPage: React.FC = () => {
  const [samples, setSamples] = useState<TrafficSampleItem[]>([]);
  const [selectedSample, setSelectedSample] = useState<TrafficSampleItem | null>(null);
  const [scenarioFilter, setScenarioFilter] = useState<string>("all");
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;
    async function loadSamples() {
      try {
        const s = await api.getTelemetrySamples();
        if (!isMounted) return;
        setSamples(s);
        // Pre-select failure case if present
        const fail = s.find((item) => item.is_failure_case);
        setSelectedSample(fail || s[0] || null);
      } catch (err: any) {
        if (!isMounted) return;
        setError("Failed to load traffic samples: " + err.message);
      } finally {
        if (isMounted) setLoading(false);
      }
    }
    loadSamples();
    return () => {
      isMounted = false;
    };
  }, []);

  const scenarios = ["all", ...Array.from(new Set(samples.map((s) => s.scenario)))];
  const filteredSamples =
    scenarioFilter === "all"
      ? samples
      : samples.filter((s) => s.scenario === scenarioFilter);

  return (
    <div style={{ padding: "1.5rem 2rem", maxWidth: "1350px", margin: "0 auto" }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.25rem" }}>
        <div>
          <h2 style={{ fontSize: "1.5rem", fontWeight: 700, color: "#f8fafc", margin: 0 }}>
            Traffic Laboratory & Failure Mode Study
          </h2>
          <p style={{ fontSize: "0.85rem", color: "#94a3b8", margin: 0 }}>
            Inspect authentic behavioural windows and investigate the novel zero-error scraping blindspot.
          </p>
        </div>

        {/* Filter Dropdown */}
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <Filter size={16} color="#94a3b8" />
          <label style={{ fontSize: "0.8rem", color: "#94a3b8" }}>Filter Scenario:</label>
          <select
            value={scenarioFilter}
            onChange={(e) => setScenarioFilter(e.target.value)}
            style={{
              background: "#1e293b",
              color: "#f8fafc",
              border: "1px solid #334155",
              borderRadius: "6px",
              padding: "0.35rem 0.75rem",
              fontSize: "0.8rem",
              cursor: "pointer",
            }}
          >
            {scenarios.map((scen) => (
              <option key={scen} value={scen}>
                {scen}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Main Grid: Sample List & Detailed Feature Inspector */}
      <div style={{ display: "grid", gridTemplateColumns: "380px 1fr", gap: "1.5rem", marginBottom: "1.5rem" }}>
        {/* Sample List Card */}
        <div className="card" style={{ maxHeight: "640px", overflowY: "auto", display: "flex", flexDirection: "column", gap: "0.5rem" }}>
          <h3 style={{ fontSize: "0.95rem", fontWeight: 700, color: "#f8fafc", marginBottom: "0.25rem" }}>
            Curated Traffic Windows ({filteredSamples.length})
          </h3>

          {filteredSamples.map((sample) => {
            const isSelected = selectedSample?.sample_id === sample.sample_id;
            const isAttack = sample.ground_truth_label === 1;

            return (
              <div
                key={sample.sample_id}
                onClick={() => setSelectedSample(sample)}
                style={{
                  background: isSelected
                    ? "rgba(56, 189, 248, 0.12)"
                    : sample.is_failure_case
                    ? "rgba(239, 68, 68, 0.08)"
                    : "#0e1420",
                  padding: "0.75rem",
                  borderRadius: "8px",
                  border: isSelected
                    ? "1px solid #38bdf8"
                    : sample.is_failure_case
                    ? "1px solid #ef4444"
                    : "1px solid #1e293b",
                  cursor: "pointer",
                  transition: "all 0.15s ease",
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.3rem" }}>
                  <span style={{ fontSize: "0.8rem", fontWeight: 600, color: "#f8fafc", fontFamily: "monospace" }}>
                    {sample.scenario}
                  </span>
                  {sample.is_failure_case ? (
                    <span className="badge badge-attack" style={{ fontSize: "0.65rem" }}>FAILURE CASE</span>
                  ) : isAttack ? (
                    <span className="badge badge-attack" style={{ fontSize: "0.65rem" }}>ATTACK</span>
                  ) : (
                    <span className="badge badge-normal" style={{ fontSize: "0.65rem" }}>NORMAL</span>
                  )}
                </div>
                <p style={{ fontSize: "0.75rem", color: "#94a3b8", margin: 0, lineHeight: 1.4 }}>
                  {sample.description}
                </p>
              </div>
            );
          })}
        </div>

        {/* Selected Sample Inspection & Failure Analysis */}
        {selectedSample && (
          <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
            {/* Failure Analysis Alert Banner (if failure case) */}
            {selectedSample.is_failure_case && (
              <div
                className="card"
                style={{
                  borderLeft: "4px solid #ef4444",
                  background: "linear-gradient(90deg, rgba(239, 68, 68, 0.12), transparent)",
                  padding: "1.25rem",
                }}
              >
                <div style={{ display: "flex", alignItems: "flex-start", gap: "0.75rem" }}>
                  <AlertTriangle size={24} color="#ef4444" style={{ flexShrink: 0, marginTop: "0.15rem" }} />
                  <div>
                    <h4 style={{ fontSize: "1rem", fontWeight: 700, color: "#f8fafc", marginBottom: "0.25rem" }}>
                      Academic Finding: Supervised Model Blindspot on High-Volume Scraping
                    </h4>
                    <p style={{ fontSize: "0.85rem", color: "#cbd5e1", lineHeight: 1.5, marginBottom: "0.5rem" }}>
                      {selectedSample.failure_explanation}
                    </p>
                    <div style={{ fontSize: "0.75rem", color: "#94a3b8" }}>
                      <strong>Failed Model:</strong> {selectedSample.failure_model} &bull;{" "}
                      <strong>Ground Truth:</strong> 1 (Attack) &bull; <strong>Predicted:</strong> 0 (Normal, p &lt; 0.01)
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Feature Values Breakdown */}
            <div className="card">
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
                <div>
                  <h3 style={{ fontSize: "1rem", fontWeight: 700, color: "#f8fafc", margin: 0 }}>
                    13-Feature Window Representation ({selectedSample.scenario})
                  </h3>
                  <span style={{ fontSize: "0.75rem", color: "#94a3b8" }}>
                    Client IP Hash: <code className="font-mono">{selectedSample.client_ip_hash}</code> &bull; Window: {selectedSample.window_start_utc}
                  </span>
                </div>
                <span className={selectedSample.ground_truth_label === 1 ? "badge badge-attack" : "badge badge-normal"}>
                  {selectedSample.source.toUpperCase()}
                </span>
              </div>

              {/* Features Grid */}
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "0.75rem" }}>
                {selectedSample.feature_names.map((name, idx) => {
                  const rawVal = selectedSample.raw_features[idx];
                  const zVal = selectedSample.scaled_features[idx];
                  const isHighZ = Math.abs(zVal) > 1.5;

                  return (
                    <div
                      key={`feat-box-${idx}`}
                      style={{
                        background: "#0e1420",
                        padding: "0.6rem 0.85rem",
                        borderRadius: "6px",
                        border: isHighZ ? "1px solid #38bdf8" : "1px solid #1e293b",
                      }}
                    >
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.2rem" }}>
                        <span style={{ fontSize: "0.75rem", color: "#cbd5e1", fontFamily: "monospace" }}>
                          {name}
                        </span>
                      </div>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                        <span style={{ fontSize: "0.7rem", color: "#64748b" }}>
                          Raw: <code className="font-mono" style={{ color: "#94a3b8" }}>{rawVal.toFixed(2)}</code>
                        </span>
                        <span
                          style={{
                            fontSize: "0.85rem",
                            fontWeight: 700,
                            fontFamily: "monospace",
                            color: isHighZ ? "#38bdf8" : "#94a3b8",
                          }}
                        >
                          z={zVal.toFixed(3)}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
