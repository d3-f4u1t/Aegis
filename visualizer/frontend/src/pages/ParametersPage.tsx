import React, { useEffect, useState } from "react";
import { BarChart2, Grid, Layers } from "lucide-react";
import { LayerParameters, ModelArchitecture } from "../types";
import { api } from "../services/api";

interface ParametersPageProps {
  modelId: string;
}

export const ParametersPage: React.FC<ParametersPageProps> = ({ modelId }) => {
  const [arch, setArch] = useState<ModelArchitecture | null>(null);
  const [selectedLayerIdx, setSelectedLayerIdx] = useState<number>(0);
  const [layerParams, setLayerParams] = useState<LayerParameters | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [hoveredCell, setHoveredCell] = useState<{ i: number; j: number; val: number } | null>(null);

  // Load architecture
  useEffect(() => {
    let isMounted = true;
    async function loadArch() {
      try {
        const a = await api.getModelArchitecture(modelId);
        if (!isMounted) return;
        setArch(a);
        setSelectedLayerIdx(0);
      } catch (err: any) {
        if (!isMounted) return;
        setError("Failed to load architecture: " + err.message);
      }
    }
    loadArch();
    return () => {
      isMounted = false;
    };
  }, [modelId]);

  // Load layer parameters whenever selected layer changes
  useEffect(() => {
    let isMounted = true;
    async function loadLayer() {
      if (!arch || arch.sizes.length < 2) return;
      setLoading(true);
      setError(null);
      try {
        const lp = await api.getLayerParameters(modelId, selectedLayerIdx);
        if (!isMounted) return;
        setLayerParams(lp);
      } catch (err: any) {
        if (!isMounted) return;
        setError("Failed to load layer parameters: " + err.message);
      } finally {
        if (isMounted) setLoading(false);
      }
    }
    loadLayer();
    return () => {
      isMounted = false;
    };
  }, [modelId, selectedLayerIdx, arch]);

  if (!arch) {
    return (
      <div style={{ padding: "3rem", textAlign: "center", color: "#94a3b8" }}>
        Loading model parameters...
      </div>
    );
  }

  const numLayers = Math.max(1, arch.sizes.length - 1);

  // Calculate statistics for current layer
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

  // Color mapping function
  const absMax = Math.max(Math.abs(minW), Math.abs(maxW), 0.1);
  const getCellColor = (w: number) => {
    const normalized = Math.min(1, Math.abs(w) / absMax);
    if (w >= 0) {
      return `rgba(56, 189, 248, ${0.1 + normalized * 0.85})`;
    } else {
      return `rgba(244, 63, 94, ${0.1 + normalized * 0.85})`;
    }
  };

  return (
    <div style={{ padding: "1.5rem 2rem", maxWidth: "1350px", margin: "0 auto" }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.25rem" }}>
        <div>
          <h2 style={{ fontSize: "1.5rem", fontWeight: 700, color: "#f8fafc", margin: 0 }}>
            Weight Matrices & Parameter Heatmaps
          </h2>
          <p style={{ fontSize: "0.85rem", color: "#94a3b8", margin: 0 }}>
            Inspect exact trained weight values W ∈ ℝ^(n_in × n_out) and bias vectors b ∈ ℝ^(n_out).
          </p>
        </div>

        {/* Layer Selector Tabs */}
        <div style={{ display: "flex", gap: "0.35rem", background: "#0e1420", padding: "0.25rem", borderRadius: "8px", border: "1px solid #1e293b" }}>
          {Array.from({ length: numLayers }).map((_, idx) => {
            const isSelected = selectedLayerIdx === idx;
            const inDim = arch.sizes[idx];
            const outDim = arch.sizes[idx + 1];
            return (
              <button
                key={`layer-tab-${idx}`}
                onClick={() => setSelectedLayerIdx(idx)}
                style={{
                  padding: "0.4rem 0.85rem",
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
                Layer {idx}: {inDim} → {outDim}
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
            gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
            gap: "1rem",
            marginBottom: "1.5rem",
          }}
        >
          <div className="card">
            <span style={{ fontSize: "0.75rem", color: "#94a3b8" }}>Dimensions</span>
            <p style={{ fontSize: "1.25rem", fontWeight: 700, color: "#f8fafc", margin: "0.2rem 0", fontFamily: "monospace" }}>
              {layerParams.n_inputs} × {layerParams.n_outputs}
            </p>
            <span style={{ fontSize: "0.7rem", color: "#64748b" }}>
              {layerParams.n_inputs * layerParams.n_outputs} weights + {layerParams.n_outputs} biases
            </span>
          </div>

          <div className="card">
            <span style={{ fontSize: "0.75rem", color: "#94a3b8" }}>Weight Range [Min, Max]</span>
            <p style={{ fontSize: "1.25rem", fontWeight: 700, color: "#38bdf8", margin: "0.2rem 0", fontFamily: "monospace" }}>
              [{minW.toFixed(3)}, {maxW.toFixed(3)}]
            </p>
            <span style={{ fontSize: "0.7rem", color: "#64748b" }}>Peak magnitude: {absMax.toFixed(3)}</span>
          </div>

          <div className="card">
            <span style={{ fontSize: "0.75rem", color: "#94a3b8" }}>Mean &bull; Std Dev</span>
            <p style={{ fontSize: "1.25rem", fontWeight: 700, color: "#10b981", margin: "0.2rem 0", fontFamily: "monospace" }}>
              μ={meanW.toFixed(3)}, σ={stdW.toFixed(3)}
            </p>
            <span style={{ fontSize: "0.7rem", color: "#64748b" }}>Frobenius norm: {l2Norm.toFixed(3)}</span>
          </div>

          <div className="card">
            <span style={{ fontSize: "0.75rem", color: "#94a3b8" }}>Sparsity (|w| &lt; 0.05)</span>
            <p style={{ fontSize: "1.25rem", fontWeight: 700, color: "#f59e0b", margin: "0.2rem 0", fontFamily: "monospace" }}>
              {totalW > 0 ? ((nearZeroCount / totalW) * 100).toFixed(1) : 0}%
            </p>
            <span style={{ fontSize: "0.7rem", color: "#64748b" }}>{nearZeroCount} of {totalW} near zero</span>
          </div>
        </div>
      )}

      {/* Main Heatmap and Bias Visualization Layout */}
      {layerParams && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 320px", gap: "1.5rem" }}>
          {/* Weight Matrix Heatmap */}
          <div className="card">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <Grid size={18} color="#38bdf8" />
                <h3 style={{ fontSize: "1rem", fontWeight: 600, color: "#f8fafc", margin: 0 }}>
                  Weight Matrix W (Rows: Input Nodes, Columns: Output Neurons)
                </h3>
              </div>

              {/* Hover Tooltip Indicator */}
              {hoveredCell ? (
                <div style={{ fontSize: "0.75rem", fontFamily: "monospace", color: "#f8fafc", background: "#0e1420", padding: "0.25rem 0.65rem", borderRadius: "4px", border: "1px solid #334155" }}>
                  w[{hoveredCell.i}][{hoveredCell.j}] ={" "}
                  <strong style={{ color: hoveredCell.val >= 0 ? "#38bdf8" : "#f43f5e" }}>
                    {hoveredCell.val.toFixed(6)}
                  </strong>
                </div>
              ) : (
                <span style={{ fontSize: "0.75rem", color: "#64748b" }}>Hover over cells to inspect exact values</span>
              )}
            </div>

            {/* Matrix Table */}
            <div style={{ overflowX: "auto", border: "1px solid #243042", borderRadius: "8px", maxHeight: "560px", overflowY: "auto" }}>
              <table style={{ borderCollapse: "collapse", width: "100%", fontSize: "0.75rem" }}>
                <thead>
                  <tr style={{ background: "#0e1420", borderBottom: "1px solid #243042" }}>
                    <th style={{ padding: "0.4rem 0.6rem", textAlign: "left", color: "#94a3b8", position: "sticky", top: 0, background: "#0e1420", zIndex: 3 }}>
                      Input Node
                    </th>
                    {Array.from({ length: layerParams.n_outputs }).map((_, j) => (
                      <th
                        key={`col-header-${j}`}
                        style={{
                          padding: "0.4rem 0.35rem",
                          textAlign: "center",
                          color: "#94a3b8",
                          fontFamily: "monospace",
                          minWidth: "42px",
                          position: "sticky",
                          top: 0,
                          background: "#0e1420",
                          zIndex: 3,
                        }}
                      >
                        N#{j}
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
                      <tr key={`row-${i}`} style={{ borderBottom: "1px solid #1a2332" }}>
                        <td
                          style={{
                            padding: "0.35rem 0.6rem",
                            color: "#cbd5e1",
                            fontFamily: "monospace",
                            whiteSpace: "nowrap",
                            background: "#0f1622",
                            position: "sticky",
                            left: 0,
                            zIndex: 2,
                            borderRight: "1px solid #243042",
                          }}
                        >
                          {rowLabel}
                        </td>
                        {row.map((val, j) => {
                          const bg = getCellColor(val);
                          return (
                            <td
                              key={`cell-${i}-${j}`}
                              onMouseEnter={() => setHoveredCell({ i, j, val })}
                              onMouseLeave={() => setHoveredCell(null)}
                              style={{
                                padding: "0.35rem 0.25rem",
                                textAlign: "center",
                                fontFamily: "monospace",
                                background: bg,
                                color: "#f8fafc",
                                cursor: "crosshair",
                                transition: "transform 0.1s ease",
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

            {/* Color Legend */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: "0.75rem", fontSize: "0.75rem", color: "#94a3b8" }}>
              <span style={{ display: "flex", alignItems: "center", gap: "0.35rem" }}>
                <span style={{ width: "12px", height: "12px", borderRadius: "3px", background: "rgba(244, 63, 94, 0.85)" }} />
                Negative Weight (-{absMax.toFixed(2)})
              </span>
              <span style={{ display: "flex", alignItems: "center", gap: "0.35rem" }}>
                <span style={{ width: "12px", height: "12px", borderRadius: "3px", background: "rgba(56, 189, 248, 0.1)" }} />
                Zero (0.00)
              </span>
              <span style={{ display: "flex", alignItems: "center", gap: "0.35rem" }}>
                <span style={{ width: "12px", height: "12px", borderRadius: "3px", background: "rgba(56, 189, 248, 0.85)" }} />
                Positive Weight (+{absMax.toFixed(2)})
              </span>
            </div>
          </div>

          {/* Bias Vectors */}
          <div className="card">
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "1rem" }}>
              <BarChart2 size={18} color="#a855f7" />
              <h3 style={{ fontSize: "1rem", fontWeight: 600, color: "#f8fafc", margin: 0 }}>
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
                    key={`bias-${j}`}
                    style={{
                      background: "#0e1420",
                      padding: "0.5rem 0.75rem",
                      borderRadius: "6px",
                      border: "1px solid #1e293b",
                    }}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.3rem", fontSize: "0.75rem" }}>
                      <span style={{ color: "#cbd5e1", fontWeight: 600 }}>Neuron #{j}</span>
                      <code className="font-mono" style={{ color: isPos ? "#38bdf8" : "#f43f5e", fontWeight: 700 }}>
                        {isPos ? `+${bVal.toFixed(6)}` : bVal.toFixed(6)}
                      </code>
                    </div>
                    {/* Bar meter */}
                    <div style={{ width: "100%", height: "6px", background: "#1e293b", borderRadius: "3px", overflow: "hidden" }}>
                      <div
                        style={{
                          height: "100%",
                          width: `${pct}%`,
                          background: isPos ? "#38bdf8" : "#f43f5e",
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
