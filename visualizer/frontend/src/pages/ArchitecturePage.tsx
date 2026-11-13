import React, { useEffect, useState } from "react";
import { Info, Sliders, ZoomIn, ZoomOut, RotateCcw } from "lucide-react";
import { LayerParameters, ModelArchitecture } from "../types";
import { api } from "../services/api";

interface ArchitecturePageProps {
  modelId: string;
}

interface SelectedNeuron {
  layerIdx: number;
  neuronIdx: number;
  bias?: number;
  weights?: number[];
  activation: string;
  name: string;
}

export const ArchitecturePage: React.FC<ArchitecturePageProps> = ({ modelId }) => {
  const [arch, setArch] = useState<ModelArchitecture | null>(null);
  const [layersData, setLayersData] = useState<LayerParameters[]>([]);
  const [selectedNeuron, setSelectedNeuron] = useState<SelectedNeuron | null>(null);
  const [weightThreshold, setWeightThreshold] = useState<number>(0.1);
  const [zoom, setZoom] = useState<number>(1);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;
    async function loadArchAndLayers() {
      setLoading(true);
      setError(null);
      setSelectedNeuron(null);
      try {
        const a = await api.getModelArchitecture(modelId);
        if (!isMounted) return;
        setArch(a);

        // Fetch layer parameters for all layers
        const lpPromises = [];
        for (let i = 0; i < a.sizes.length - 1; i++) {
          lpPromises.push(api.getLayerParameters(modelId, i));
        }
        const layers = await Promise.all(lpPromises);
        if (!isMounted) return;
        setLayersData(layers);
      } catch (err: any) {
        if (!isMounted) return;
        setError(err.message || "Failed to load architecture parameters.");
      } finally {
        if (isMounted) setLoading(false);
      }
    }
    loadArchAndLayers();
    return () => {
      isMounted = false;
    };
  }, [modelId]);

  if (loading) {
    return (
      <div style={{ padding: "3rem", textAlign: "center", color: "#94a3b8" }}>
        Loading architecture for {modelId}...
      </div>
    );
  }

  if (error || !arch) {
    return (
      <div style={{ padding: "3rem", textAlign: "center", color: "#ef4444" }}>
        Error loading model: {error || "Model architecture unavailable."}
      </div>
    );
  }

  // Layout calculation for SVG
  const svgWidth = 950;
  const svgHeight = 620;
  const layerSizes = arch.sizes;
  const nLayers = layerSizes.length;
  const layerSpacing = (svgWidth - 240) / Math.max(1, nLayers - 1);

  // Calculate neuron coordinates
  const neuronCoords: Array<Array<{ x: number; y: number }>> = [];
  layerSizes.forEach((size, lIdx) => {
    const x = 120 + lIdx * layerSpacing;
    const coordsForLayer: Array<{ x: number; y: number }> = [];
    const maxSpacing = 36;
    const totalHeight = Math.min((size - 1) * maxSpacing, svgHeight - 120);
    const spacing = size > 1 ? totalHeight / (size - 1) : 0;
    const startY = (svgHeight - totalHeight) / 2;

    for (let nIdx = 0; nIdx < size; nIdx++) {
      const y = size === 1 ? svgHeight / 2 : startY + nIdx * spacing;
      coordsForLayer.push({ x, y });
    }
    neuronCoords.push(coordsForLayer);
  });

  const handleNeuronClick = (lIdx: number, nIdx: number) => {
    if (lIdx === 0) {
      setSelectedNeuron({
        layerIdx: 0,
        neuronIdx: nIdx,
        activation: "Identity",
        name: arch.feature_names[nIdx] || `Feature ${nIdx}`,
      });
      return;
    }

    const lp = layersData[lIdx - 1];
    if (!lp) return;

    const neuronWeights = lp.neurons[nIdx]?.weights || [];
    const neuronBias = lp.biases[nIdx];
    const act = lIdx === nLayers - 1 ? (arch.output_activation || "sigmoid") : (arch.hidden_activation || "relu");

    setSelectedNeuron({
      layerIdx: lIdx,
      neuronIdx: nIdx,
      bias: neuronBias,
      weights: neuronWeights,
      activation: act,
      name: `Neuron ${lIdx}-${nIdx}`,
    });
  };

  return (
    <div style={{ padding: "1.5rem 2rem", maxWidth: "1400px", margin: "0 auto" }}>
      {/* Page Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.25rem" }}>
        <div>
          <h2 style={{ fontSize: "1.5rem", fontWeight: 700, color: "#f8fafc", margin: 0 }}>
            Network Architecture Explorer
          </h2>
          <p style={{ fontSize: "0.85rem", color: "#94a3b8", margin: 0 }}>
            Model: <code className="font-mono">{arch.name}</code> &bull; Total Parameters: {arch.total_parameters} &bull; Topology: {arch.sizes.join(" → ")}
          </p>
        </div>

        {/* Controls Toolbar */}
        <div style={{ display: "flex", alignItems: "center", gap: "1.25rem" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <Sliders size={16} color="#94a3b8" />
            <label style={{ fontSize: "0.8rem", color: "#cbd5e1" }}>Weight Filter (|w| ≥ {weightThreshold.toFixed(2)}):</label>
            <input
              type="range"
              min="0"
              max="1.5"
              step="0.05"
              value={weightThreshold}
              onChange={(e) => setWeightThreshold(parseFloat(e.target.value))}
              style={{ width: "90px", cursor: "pointer" }}
            />
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "0.35rem" }}>
            <button className="btn btn-secondary" style={{ padding: "0.35rem 0.5rem" }} onClick={() => setZoom(Math.max(0.7, zoom - 0.1))}>
              <ZoomOut size={14} />
            </button>
            <span style={{ fontSize: "0.8rem", color: "#94a3b8", minWidth: "42px", textAlign: "center" }}>
              {Math.round(zoom * 100)}%
            </span>
            <button className="btn btn-secondary" style={{ padding: "0.35rem 0.5rem" }} onClick={() => setZoom(Math.min(1.5, zoom + 0.1))}>
              <ZoomIn size={14} />
            </button>
            <button className="btn btn-secondary" style={{ padding: "0.35rem 0.5rem" }} onClick={() => setZoom(1)}>
              <RotateCcw size={14} />
            </button>
          </div>
        </div>
      </div>

      {/* Main Canvas & Inspection Layout */}
      <div style={{ display: "grid", gridTemplateColumns: selectedNeuron ? "1fr 340px" : "1fr", gap: "1.25rem" }}>
        {/* SVG Network Visualizer Card */}
        <div className="card" style={{ overflow: "hidden", display: "flex", flexDirection: "column", alignItems: "center", padding: "1rem" }}>
          <div style={{ overflow: "auto", width: "100%", textAlign: "center" }}>
            <svg
              width={svgWidth * zoom}
              height={svgHeight * zoom}
              viewBox={`0 0 ${svgWidth} ${svgHeight}`}
              style={{ background: "#0e1420", borderRadius: "8px", border: "1px solid #1e293b" }}
            >
              {/* Layer Title Headers */}
              {arch.sizes.map((size, lIdx) => {
                const x = 120 + lIdx * layerSpacing;
                const isInput = lIdx === 0;
                const isOutput = lIdx === nLayers - 1;
                const layerTitle = isInput
                  ? `Input (${size})`
                  : isOutput
                  ? `Output (${size})`
                  : `Hidden ${lIdx} (${size})`;
                const actTitle = isInput
                  ? "Raw / Scaled"
                  : isOutput
                  ? arch.output_activation || "sigmoid"
                  : arch.hidden_activation || "relu";

                return (
                  <g key={`header-${lIdx}`}>
                    <text
                      x={x}
                      y={30}
                      textAnchor="middle"
                      fill="#f8fafc"
                      fontSize={13}
                      fontWeight={600}
                    >
                      {layerTitle}
                    </text>
                    <text
                      x={x}
                      y={48}
                      textAnchor="middle"
                      fill="#38bdf8"
                      fontSize={10}
                      fontFamily="monospace"
                    >
                      act: {actTitle}
                    </text>
                  </g>
                );
              })}

              {/* Connection Lines */}
              {layersData.map((lp, lIdx) => {
                const srcCoords = neuronCoords[lIdx];
                const dstCoords = neuronCoords[lIdx + 1];
                if (!srcCoords || !dstCoords) return null;

                const lines: React.ReactNode[] = [];
                for (let i = 0; i < lp.n_inputs; i++) {
                  for (let j = 0; j < lp.n_outputs; j++) {
                    const w = lp.weights[i]?.[j] ?? 0;
                    if (Math.abs(w) < weightThreshold) continue;

                    const isSrcSelected =
                      selectedNeuron?.layerIdx === lIdx && selectedNeuron?.neuronIdx === i;
                    const isDstSelected =
                      selectedNeuron?.layerIdx === lIdx + 1 && selectedNeuron?.neuronIdx === j;
                    const isHighlighted = isSrcSelected || isDstSelected;

                    const strokeColor = w >= 0 ? "#38bdf8" : "#f43f5e";
                    const strokeWidth = isHighlighted
                      ? Math.min(4, Math.max(1.5, Math.abs(w) * 2))
                      : Math.min(2.5, Math.max(0.4, Math.abs(w) * 1.2));
                    const strokeOpacity = isHighlighted ? 0.95 : Math.min(0.65, Math.max(0.12, Math.abs(w) * 0.4));

                    lines.push(
                      <line
                        key={`w-${lIdx}-${i}-${j}`}
                        x1={srcCoords[i].x}
                        y1={srcCoords[i].y}
                        x2={dstCoords[j].x}
                        y2={dstCoords[j].y}
                        stroke={strokeColor}
                        strokeWidth={strokeWidth}
                        strokeOpacity={strokeOpacity}
                      />
                    );
                  }
                }
                return <g key={`connections-${lIdx}`}>{lines}</g>;
              })}

              {/* Neurons */}
              {neuronCoords.map((layerNodes, lIdx) => {
                return (
                  <g key={`nodes-layer-${lIdx}`}>
                    {layerNodes.map((pos, nIdx) => {
                      const isSelected =
                        selectedNeuron?.layerIdx === lIdx && selectedNeuron?.neuronIdx === nIdx;
                      const isInput = lIdx === 0;
                      const isOutput = lIdx === nLayers - 1;
                      const nodeColor = isInput ? "#10b981" : isOutput ? "#ec4899" : "#3b82f6";

                      return (
                        <g
                          key={`node-${lIdx}-${nIdx}`}
                          onClick={() => handleNeuronClick(lIdx, nIdx)}
                          style={{ cursor: "pointer" }}
                        >
                          <circle
                            cx={pos.x}
                            cy={pos.y}
                            r={isSelected ? 11 : 8}
                            fill={isSelected ? "#f8fafc" : nodeColor}
                            stroke={isSelected ? "#38bdf8" : "#0e1420"}
                            strokeWidth={isSelected ? 3 : 1.5}
                          />
                          {/* Label for input nodes */}
                          {isInput && (
                            <text
                              x={pos.x - 14}
                              y={pos.y + 3.5}
                              textAnchor="end"
                              fill="#cbd5e1"
                              fontSize={9.5}
                              fontFamily="monospace"
                            >
                              {arch.feature_names[nIdx] || `x[${nIdx}]`}
                            </text>
                          )}
                          {/* Label for output node */}
                          {isOutput && (
                            <text
                              x={pos.x + 14}
                              y={pos.y + 3.5}
                              textAnchor="start"
                              fill="#f472b6"
                              fontSize={10}
                              fontWeight={600}
                            >
                              P(attack)
                            </text>
                          )}
                        </g>
                      );
                    })}
                  </g>
                );
              })}
            </svg>
          </div>

          <div style={{ display: "flex", gap: "1.5rem", marginTop: "0.75rem", fontSize: "0.75rem", color: "#94a3b8" }}>
            <span style={{ display: "flex", alignItems: "center", gap: "0.35rem" }}>
              <span style={{ width: "10px", height: "10px", borderRadius: "50%", background: "#10b981" }} />
              Input Nodes (13)
            </span>
            <span style={{ display: "flex", alignItems: "center", gap: "0.35rem" }}>
              <span style={{ width: "10px", height: "10px", borderRadius: "50%", background: "#3b82f6" }} />
              Hidden Nodes (ReLU)
            </span>
            <span style={{ display: "flex", alignItems: "center", gap: "0.35rem" }}>
              <span style={{ width: "10px", height: "10px", borderRadius: "50%", background: "#ec4899" }} />
              Output Node (Sigmoid)
            </span>
            <span style={{ display: "flex", alignItems: "center", gap: "0.35rem" }}>
              <span style={{ width: "16px", height: "2px", background: "#38bdf8" }} />
              Positive Weight
            </span>
            <span style={{ display: "flex", alignItems: "center", gap: "0.35rem" }}>
              <span style={{ width: "16px", height: "2px", background: "#f43f5e" }} />
              Negative Weight
            </span>
          </div>
        </div>

        {/* Neuron Details Side Panel */}
        {selectedNeuron && (
          <div className="card" style={{ height: "fit-content" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem" }}>
              <h3 style={{ fontSize: "1rem", fontWeight: 700, color: "#f8fafc" }}>
                {selectedNeuron.name}
              </h3>
              <button
                className="btn btn-secondary"
                style={{ padding: "0.2rem 0.5rem", fontSize: "0.7rem" }}
                onClick={() => setSelectedNeuron(null)}
              >
                Close
              </button>
            </div>

            <div style={{ marginBottom: "1rem", fontSize: "0.8rem", display: "grid", gap: "0.4rem" }}>
              <div>
                <span style={{ color: "#94a3b8" }}>Layer: </span>
                <span style={{ color: "#f1f5f9", fontWeight: 600 }}>
                  {selectedNeuron.layerIdx === 0
                    ? "Input Feature Vector"
                    : `Layer ${selectedNeuron.layerIdx} (${selectedNeuron.activation})`}
                </span>
              </div>
              {selectedNeuron.bias !== undefined && (
                <div>
                  <span style={{ color: "#94a3b8" }}>Bias (b): </span>
                  <code className="font-mono" style={{ color: "#38bdf8", fontWeight: 700 }}>
                    {selectedNeuron.bias.toFixed(6)}
                  </code>
                </div>
              )}
              <div>
                <span style={{ color: "#94a3b8" }}>Activation: </span>
                <span className="badge badge-info">{selectedNeuron.activation}</span>
              </div>
            </div>

            {/* Weights Breakdown */}
            {selectedNeuron.weights && selectedNeuron.weights.length > 0 && (
              <div>
                <h4 style={{ fontSize: "0.85rem", fontWeight: 600, color: "#cbd5e1", marginBottom: "0.5rem" }}>
                  Incoming Weights ({selectedNeuron.weights.length}):
                </h4>
                <div
                  style={{
                    maxHeight: "300px",
                    overflowY: "auto",
                    border: "1px solid #243042",
                    borderRadius: "6px",
                  }}
                >
                  <table style={{ width: "100%", fontSize: "0.75rem", borderCollapse: "collapse" }}>
                    <thead>
                      <tr style={{ background: "#1e293b", color: "#94a3b8", textAlign: "left" }}>
                        <th style={{ padding: "0.35rem 0.5rem" }}>From Node</th>
                        <th style={{ padding: "0.35rem 0.5rem", textAlign: "right" }}>Weight (w)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {selectedNeuron.weights.map((w, idx) => {
                        const fromLabel =
                          selectedNeuron.layerIdx === 1
                            ? arch.feature_names[idx] || `x[${idx}]`
                            : `h[${selectedNeuron.layerIdx - 1}][${idx}]`;

                        return (
                          <tr key={`w-row-${idx}`} style={{ borderBottom: "1px solid #1e293b" }}>
                            <td style={{ padding: "0.35rem 0.5rem", color: "#cbd5e1", fontFamily: "monospace" }}>
                              {fromLabel}
                            </td>
                            <td
                              style={{
                                padding: "0.35rem 0.5rem",
                                textAlign: "right",
                                fontFamily: "monospace",
                                fontWeight: 600,
                                color: w >= 0 ? "#38bdf8" : "#f43f5e",
                              }}
                            >
                              {w >= 0 ? `+${w.toFixed(5)}` : w.toFixed(5)}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
