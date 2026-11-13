import React, { useEffect, useState } from "react";
import { BarChart3, Grid, Layers, Sparkles } from "lucide-react";
import type { LayerParameters, ModelArchitecture } from "../types";
import { api } from "../services/api";

interface ParametersPageProps {
  modelId: string;
}

export const ParametersPage: React.FC<ParametersPageProps> = ({ modelId }) => {
  const [arch, setArch] = useState<ModelArchitecture | null>(null);
  const [selectedLayerIdx, setSelectedLayerIdx] = useState<number>(0);
  const [layerParams, setLayerParams] = useState<LayerParameters | null>(null);
  const [hoveredCell, setHoveredCell] = useState<{ i: number; j: number; val: number } | null>(null);

  useEffect(() => {
    let isMounted = true;
    async function loadArch() {
      try {
        const a = await api.getModelArchitecture(modelId);
        if (!isMounted) return;
        setArch(a);
        setSelectedLayerIdx(0);
      } catch (err: any) {
        // non-blocking
      }
    }
    loadArch();
    return () => {
      isMounted = false;
    };
  }, [modelId]);

  useEffect(() => {
    let isMounted = true;
    async function loadLayer() {
      if (!arch || arch.sizes.length < 2) return;
      try {
        const lp = await api.getLayerParameters(modelId, selectedLayerIdx);
        if (!isMounted) return;
        setLayerParams(lp);
      } catch (err: any) {
        // non-blocking
      }
    }
    loadLayer();
    return () => {
      isMounted = false;
    };
  }, [modelId, selectedLayerIdx, arch]);

  if (!arch) {
    return (
      <div style={{ padding: "4rem 2rem", textAlign: "center", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
        LOADING PARAMETERS...
      </div>
    );
  }

  const numLayers = Math.max(1, arch.sizes.length - 1);

  // Statistics Calculation
  let minW = 0, maxW = 0, meanW = 0, stdW = 0, l2Norm = 0, nearZeroCount = 0, totalW = 0;
  if (layerParams) {
    const flatW: number[] = [];
    layerParams.weights.forEach((row) => row.forEach((w) => flatW.push(w)));
    totalW = flatW.length;

    if (totalW > 0) {
      minW = Math.min(...flatW);
      maxW = Math.max(...flatW);
      const sumW = flatW.reduce((acc, v) => acc + v, 0);
      meanW = sumW / totalW;
      const variance = flatW.reduce((acc, v) => acc + Math.pow(v - meanW, 2), 0) / totalW;
      stdW = Math.sqrt(variance);
      l2Norm = Math.sqrt(flatW.reduce((acc, v) => acc + v * v, 0));
      nearZeroCount = flatW.filter((w) => Math.abs(w) < 0.05).length;
    }
  }

  const absMax = Math.max(Math.abs(minW), Math.abs(maxW), 0.1);
  const getCellColor = (w: number) => {
    const normalized = Math.min(1, Math.abs(w) / absMax);
    if (w >= 0) {
      return `rgba(0, 240, 255, ${0.08 + normalized * 0.82})`;
    } else {
      return `rgba(255, 51, 102, ${0.08 + normalized * 0.82})`;
    }
  };

  return (
    <div style={{ padding: "2.5rem", maxWidth: "1440px", margin: "0 auto" }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", marginBottom: "1.75rem", flexWrap: "wrap", gap: "1rem" }}>
        <div>
          <div className="section-kicker">Matrix Inspection</div>
          <h2 style={{ fontSize: "1.75rem", fontWeight: 800, color: "#ffffff", letterSpacing: "-0.025em", margin: 0 }}>
            Weight Matrices &amp; Bias Vectors
          </h2>
          <p style={{ fontSize: "0.875rem", color: "#888888", marginTop: "0.25rem" }}>
            High-resolution matrix heatmaps displaying authentic parameter distributions across layers.
          </p>
        </div>

        {/* Layer Segmented Control */}
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
          {Array.from({ length: numLayers }).map((_, idx) => {
            const isSelected = selectedLayerIdx === idx;
            const inDim = arch.sizes[idx];
            const outDim = arch.sizes[idx + 1];
            return (
              <button
                key={`layer-${idx}`}
                onClick={() => setSelectedLayerIdx(idx)}
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
                Layer {idx} ({inDim} &rarr; {outDim})
              </button>
            );
          })}
        </div>
      </div>

      {/* Layer Statistics Cards */}
      {layerParams && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
            gap: "1.25rem",
            marginBottom: "1.75rem",
          }}
        >
          <div className="card">
            <span style={{ fontSize: "0.6875rem", color: "#777777", textTransform: "uppercase", fontFamily: "var(--font-mono)" }}>
              Matrix Dimensions
            </span>
            <div style={{ fontSize: "1.5rem", fontWeight: 700, color: "#ffffff", fontFamily: "var(--font-mono)", margin: "0.25rem 0" }}>
              {layerParams.n_inputs} &times; {layerParams.n_outputs}
            </div>
            <span style={{ fontSize: "0.75rem", color: "#666666" }}>
              {layerParams.n_inputs * layerParams.n_outputs} weights &bull; {layerParams.n_outputs} biases
            </span>
          </div>

          <div className="card">
            <span style={{ fontSize: "0.6875rem", color: "#777777", textTransform: "uppercase", fontFamily: "var(--font-mono)" }}>
              Weight Bounds [Min, Max]
            </span>
            <div style={{ fontSize: "1.5rem", fontWeight: 700, color: "#00f0ff", fontFamily: "var(--font-mono)", margin: "0.25rem 0" }}>
              [{minW.toFixed(3)}, {maxW.toFixed(3)}]
            </div>
            <span style={{ fontSize: "0.75rem", color: "#666666" }}>Peak magnitude: {absMax.toFixed(3)}</span>
          </div>

          <div className="card">
            <span style={{ fontSize: "0.6875rem", color: "#777777", textTransform: "uppercase", fontFamily: "var(--font-mono)" }}>
              Moments (&mu;, &sigma;)
            </span>
            <div style={{ fontSize: "1.5rem", fontWeight: 700, color: "#00f5a0", fontFamily: "var(--font-mono)", margin: "0.25rem 0" }}>
              &mu;={meanW.toFixed(3)}, &sigma;={stdW.toFixed(3)}
            </div>
            <span style={{ fontSize: "0.75rem", color: "#666666" }}>Frobenius norm: {l2Norm.toFixed(3)}</span>
          </div>

          <div className="card">
            <span style={{ fontSize: "0.6875rem", color: "#777777", textTransform: "uppercase", fontFamily: "var(--font-mono)" }}>
              Sparsity (|w| &lt; 0.05)
            </span>
            <div style={{ fontSize: "1.5rem", fontWeight: 700, color: "#f5a623", fontFamily: "var(--font-mono)", margin: "0.25rem 0" }}>
              {totalW > 0 ? ((nearZeroCount / totalW) * 100).toFixed(1) : 0}%
            </div>
            <span style={{ fontSize: "0.75rem", color: "#666666" }}>{nearZeroCount} of {totalW} near zero</span>
          </div>
        </div>
      )}

      {/* Main Heatmap and Bias Grid */}
      {layerParams && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 340px", gap: "1.5rem" }}>
          {/* Heatmap Card */}
          <div className="card">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <Grid size={16} color="#00f0ff" />
                <h3 style={{ fontSize: "0.9375rem", fontWeight: 600, color: "#ffffff", margin: 0 }}>
                  Weight Matrix W (Rows: Input Dimension, Columns: Output Dimension)
                </h3>
              </div>

              {hoveredCell ? (
                <div style={{ fontSize: "0.75rem", fontFamily: "var(--font-mono)", color: "#ffffff", background: "#111111", padding: "0.25rem 0.65rem", borderRadius: "4px", border: "1px solid rgba(255, 255, 255, 0.15)" }}>
                  w[{hoveredCell.i}][{hoveredCell.j}] ={" "}
                  <strong style={{ color: hoveredCell.val >= 0 ? "#00f0ff" : "#ff3366" }}>
                    {hoveredCell.val.toFixed(6)}
                  </strong>
                </div>
              ) : (
                <span style={{ fontSize: "0.75rem", color: "#666666", fontFamily: "var(--font-mono)" }}>
                  HOVER TO INSPECT CELL
                </span>
              )}
            </div>

            {/* Matrix Table */}
            <div style={{ overflowX: "auto", border: "1px solid rgba(255, 255, 255, 0.08)", borderRadius: "6px", maxHeight: "560px", overflowY: "auto" }}>
              <table style={{ borderCollapse: "collapse", width: "100%", fontSize: "0.75rem" }}>
                <thead>
                  <tr style={{ background: "#080808", borderBottom: "1px solid rgba(255, 255, 255, 0.08)" }}>
                    <th style={{ padding: "0.5rem 0.75rem", textAlign: "left", color: "#777777", position: "sticky", top: 0, background: "#080808", zIndex: 3, fontFamily: "var(--font-mono)" }}>
                      INPUT NODE
                    </th>
                    {Array.from({ length: layerParams.n_outputs }).map((_, j) => (
                      <th
                        key={`th-${j}`}
                        style={{
                          padding: "0.5rem 0.35rem",
                          textAlign: "center",
                          color: "#777777",
                          fontFamily: "var(--font-mono)",
                          minWidth: "46px",
                          position: "sticky",
                          top: 0,
                          background: "#080808",
                          zIndex: 3,
                        }}
                      >
                        #{j}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {layerParams.weights.map((row, i) => {
                    const rowLabel =
                      selectedLayerIdx === 0
                        ? arch.feature_names[i] || `x[${i}]`
                        : `h[${selectedLayerIdx - 1}][${i}]`;

                    return (
                      <tr key={`r-${i}`} style={{ borderBottom: "1px solid rgba(255, 255, 255, 0.03)" }}>
                        <td
                          style={{
                            padding: "0.4rem 0.75rem",
                            color: "#ededed",
                            fontFamily: "var(--font-mono)",
                            whiteSpace: "nowrap",
                            background: "#080808",
                            position: "sticky",
                            left: 0,
                            zIndex: 2,
                            borderRight: "1px solid rgba(255, 255, 255, 0.08)",
                          }}
                        >
                          {rowLabel}
                        </td>
                        {row.map((val, j) => {
                          const bg = getCellColor(val);
                          return (
                            <td
                              key={`c-${i}-${j}`}
                              onMouseEnter={() => setHoveredCell({ i, j, val })}
                              onMouseLeave={() => setHoveredCell(null)}
                              style={{
                                padding: "0.4rem 0.3rem",
                                textAlign: "center",
                                fontFamily: "var(--font-mono)",
                                background: bg,
                                color: "#ffffff",
                                cursor: "crosshair",
                              }}
                            >
                              {val.toFixed(2)}
                            </td>
                          );
                        })}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* Matrix Legend */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: "1rem", fontSize: "0.75rem", fontFamily: "var(--font-mono)", color: "#777777" }}>
              <span style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
                <span style={{ width: "10px", height: "10px", borderRadius: "2px", background: "rgba(255, 51, 102, 0.9)" }} />
                Negative (-{absMax.toFixed(2)})
              </span>
              <span style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
                <span style={{ width: "10px", height: "10px", borderRadius: "2px", background: "rgba(0, 240, 255, 0.1)" }} />
                Near Zero (0.00)
              </span>
              <span style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
                <span style={{ width: "10px", height: "10px", borderRadius: "2px", background: "rgba(0, 240, 255, 0.9)" }} />
                Positive (+{absMax.toFixed(2)})
              </span>
            </div>
          </div>

          {/* Bias Vector Card */}
          <div className="card">
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "1rem" }}>
              <BarChart3 size={16} color="#8b5cf6" />
              <h3 style={{ fontSize: "0.9375rem", fontWeight: 600, color: "#ffffff", margin: 0 }}>
                Bias Vector b ({layerParams.n_outputs})
              </h3>
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: "0.6rem", maxHeight: "560px", overflowY: "auto" }}>
              {layerParams.biases.map((bVal, j) => {
                const maxBiasAbs = Math.max(...layerParams.biases.map((b) => Math.abs(b)), 0.1);
                const pct = Math.min(100, (Math.abs(bVal) / maxBiasAbs) * 100);
                const isPos = bVal >= 0;

                return (
                  <div
                    key={`b-${j}`}
                    style={{
                      background: "#080808",
                      padding: "0.6rem 0.85rem",
                      borderRadius: "6px",
                      border: "1px solid rgba(255, 255, 255, 0.06)",
                    }}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.35rem", fontSize: "0.75rem", fontFamily: "var(--font-mono)" }}>
                      <span style={{ color: "#a1a1a1" }}>Neuron #{j}</span>
                      <span style={{ color: isPos ? "#00f0ff" : "#ff3366", fontWeight: 700 }}>
                        {isPos ? `+${bVal.toFixed(6)}` : bVal.toFixed(6)}
                      </span>
                    </div>
                    <div style={{ width: "100%", height: "4px", background: "#1a1a1a", borderRadius: "2px", overflow: "hidden" }}>
                      <div
                        style={{
                          height: "100%",
                          width: `${pct}%`,
                          background: isPos ? "#00f0ff" : "#ff3366",
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
