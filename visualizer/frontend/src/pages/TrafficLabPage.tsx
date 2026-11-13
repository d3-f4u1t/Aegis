import React, { useEffect, useState } from "react";
import {
  AlertOctagon,
  ArrowRight,
  Database,
  Filter,
  Flame,
  Search,
  ShieldAlert,
} from "lucide-react";
import type { TrafficSampleItem } from "../types";
import { api } from "../services/api";

export const TrafficLabPage: React.FC = () => {
  const [samples, setSamples] = useState<TrafficSampleItem[]>([]);
  const [selectedSample, setSelectedSample] = useState<TrafficSampleItem | null>(null);
  const [scenarioFilter, setScenarioFilter] = useState<string>("all");

  useEffect(() => {
    let isMounted = true;
    async function loadSamples() {
      try {
        const s = await api.getTelemetrySamples();
        if (!isMounted) return;
        setSamples(s);
        const fail = s.find((item) => item.is_failure_case);
        setSelectedSample(fail || s[0] || null);
      } catch (err: any) {
        // non-blocking
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
    <div style={{ padding: "2.5rem", maxWidth: "1440px", margin: "0 auto" }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", marginBottom: "1.75rem", flexWrap: "wrap", gap: "1rem" }}>
        <div>
          <div className="section-kicker">Telemetry Laboratory</div>
          <h2 style={{ fontSize: "1.75rem", fontWeight: 800, color: "#ffffff", letterSpacing: "-0.025em", margin: 0 }}>
            Traffic Windows &amp; Failure Mode Study
          </h2>
          <p style={{ fontSize: "0.875rem", color: "#888888", marginTop: "0.25rem" }}>
            Inspect authentic behavioural windows and investigate the novel zero-error scraping blindspot.
          </p>
        </div>

        {/* Filter Dropdown */}
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <Filter size={14} color="#777777" />
          <span style={{ fontSize: "0.75rem", color: "#777777", fontFamily: "var(--font-mono)", textTransform: "uppercase" }}>
            Filter Scenario:
          </span>
          <select
            value={scenarioFilter}
            onChange={(e) => setScenarioFilter(e.target.value)}
            style={{
              background: "#111111",
              color: "#ededed",
              border: "1px solid rgba(255, 255, 255, 0.12)",
              borderRadius: "6px",
              padding: "0.4rem 0.85rem",
              fontSize: "0.75rem",
              fontFamily: "var(--font-mono)",
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

      {/* Main Grid: Sample Stream & Detail Panel */}
      <div style={{ display: "grid", gridTemplateColumns: "380px 1fr", gap: "1.5rem" }}>
        {/* Sample List */}
        <div
          className="card"
          style={{
            maxHeight: "680px",
            overflowY: "auto",
            display: "flex",
            flexDirection: "column",
            gap: "0.6rem",
            padding: "1rem",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.25rem", padding: "0 0.25rem" }}>
            <span style={{ fontSize: "0.6875rem", fontFamily: "var(--font-mono)", color: "#777777", textTransform: "uppercase" }}>
              Curated Windows ({filteredSamples.length})
            </span>
          </div>

          {filteredSamples.map((sample) => {
            const isSelected = selectedSample?.sample_id === sample.sample_id;
            const isAttack = sample.ground_truth_label === 1;

            return (
              <div
                key={sample.sample_id}
                onClick={() => setSelectedSample(sample)}
                style={{
                  background: isSelected
                    ? "rgba(0, 240, 255, 0.08)"
                    : sample.is_failure_case
                    ? "rgba(255, 51, 102, 0.06)"
                    : "#080808",
                  padding: "0.85rem",
                  borderRadius: "6px",
                  border: isSelected
                    ? "1px solid #00f0ff"
                    : sample.is_failure_case
                    ? "1px solid #ff3366"
                    : "1px solid rgba(255, 255, 255, 0.06)",
                  cursor: "pointer",
                  transition: "all 0.15s ease",
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.35rem" }}>
                  <span style={{ fontSize: "0.8125rem", fontWeight: 600, color: "#ffffff", fontFamily: "var(--font-mono)" }}>
                    {sample.scenario}
                  </span>
                  {sample.is_failure_case ? (
                    <span className="badge badge-attack" style={{ fontSize: "0.6rem" }}>BLINDSPOT</span>
                  ) : isAttack ? (
                    <span className="badge badge-attack" style={{ fontSize: "0.6rem" }}>ATTACK</span>
                  ) : (
                    <span className="badge badge-normal" style={{ fontSize: "0.6rem" }}>NORMAL</span>
                  )}
                </div>
                <p style={{ fontSize: "0.75rem", color: "#888888", margin: 0, lineHeight: 1.4 }}>
                  {sample.description}
                </p>
              </div>
            );
          })}
        </div>

        {/* Selected Sample Detail */}
        {selectedSample && (
          <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
            {/* Failure Mode Case Study Alert */}
            {selectedSample.is_failure_case && (
              <div
                className="card"
                style={{
                  borderLeft: "3px solid #ff3366",
                  background: "linear-gradient(135deg, rgba(255, 51, 102, 0.08) 0%, rgba(10, 10, 10, 0.95) 100%)",
                  padding: "1.5rem",
                }}
              >
                <div style={{ display: "flex", alignItems: "flex-start", gap: "1rem" }}>
                  <AlertOctagon size={24} color="#ff3366" style={{ flexShrink: 0, marginTop: "0.15rem" }} />
                  <div>
                    <h4 style={{ fontSize: "1.0625rem", fontWeight: 700, color: "#ffffff", marginBottom: "0.35rem", letterSpacing: "-0.01em" }}>
                      Academic Finding: Supervised Neural Network Scraping Blindspot
                    </h4>
                    <p style={{ fontSize: "0.8125rem", color: "#b5b5b5", lineHeight: 1.6, marginBottom: "0.65rem" }}>
                      {selectedSample.failure_explanation}
                    </p>
                    <div style={{ fontSize: "0.75rem", color: "#777777", fontFamily: "var(--font-mono)" }}>
                      MODEL: <span style={{ color: "#ededed" }}>{selectedSample.failure_model}</span> &bull;{" "}
                      GROUND TRUTH: <span style={{ color: "#ff3366" }}>1 (ATTACK)</span> &bull;{" "}
                      PREDICTED: <span style={{ color: "#00f5a0" }}>0 (NORMAL, p &lt; 0.01)</span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* 13 Feature Breakdown Card */}
            <div className="card">
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "1.25rem", flexWrap: "wrap", gap: "1rem" }}>
                <div>
                  <div className="section-kicker">Window Features</div>
                  <h3 style={{ fontSize: "1.125rem", fontWeight: 700, color: "#ffffff", margin: "0.2rem 0", letterSpacing: "-0.01em" }}>
                    13-Feature Behavioural Profile ({selectedSample.scenario})
                  </h3>
                  <span style={{ fontSize: "0.75rem", color: "#777777", fontFamily: "var(--font-mono)" }}>
                    CLIENT: {selectedSample.client_ip_hash} &bull; TIMESTAMP: {selectedSample.window_start_utc}
                  </span>
                </div>
                <span className={selectedSample.ground_truth_label === 1 ? "badge badge-attack" : "badge badge-normal"}>
                  {selectedSample.source.toUpperCase()}
                </span>
              </div>

              {/* Feature Grid */}
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "0.85rem" }}>
                {selectedSample.feature_names.map((name, idx) => {
                  const rawVal = selectedSample.raw_features[idx];
                  const zVal = selectedSample.scaled_features[idx];
                  const isElevated = Math.abs(zVal) > 1.5;

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
                          Raw: <span style={{ color: "#888888" }}>{rawVal.toFixed(2)}</span>
                        </span>
                        <span
                          style={{
                            fontSize: "0.875rem",
                            fontWeight: 700,
                            fontFamily: "var(--font-mono)",
                            color: isElevated ? "#00f0ff" : "#ededed",
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
