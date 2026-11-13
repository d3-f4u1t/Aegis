import React, { useEffect, useState } from "react";
import {
  Activity,
  Layers,
  Maximize2,
  Minimize2,
  Network,
  RotateCcw,
  Sliders,
  Sparkles,
  X,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import type { LayerParameters, ModelArchitecture } from "../types";
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
  const [hoveredNeuron, setHoveredNeuron] = useState<{ layerIdx: number; neuronIdx: number } | null>(null);
  const [weightThreshold, setWeightThreshold] = useState<number>(0.12);
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
      <div style={{ padding: "4rem 2rem", textAlign: "center", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
        INITIALIZING TOPOLOGY EXPLORER...
      </div>
    );
  }

  if (error || !arch) {
    return (
      <div style={{ padding: "4rem 2rem", textAlign: "center", color: "var(--accent-rose)", fontFamily: "var(--font-mono)" }}>
        FAILED TO LOAD ARCHITECTURE: {error}
      </div>
    );
  }

  // Canvas Layout Dimensions
  const svgWidth = 1040;
  const svgHeight = 680;
  const layerSizes = arch.sizes;
  const nLayers = layerSizes.length;
  const layerSpacing = (svgWidth - 280) / Math.max(1, nLayers - 1);

  // Layer Color Palettes (Bespoke Artistic Neural Scheme)
  const layerTheme = [
    { color: "#f59e0b", glow: "rgba(245, 158, 11, 0.4)", name: "Input Features", bg: "rgba(245, 158, 11, 0.08)" },
    { color: "#8b5cf6", glow: "rgba(139, 92, 246, 0.4)", name: "Hidden 1 (ReLU)", bg: "rgba(139, 92, 246, 0.08)" },
    { color: "#00f0ff", glow: "rgba(0, 240, 255, 0.4)", name: "Hidden 2 (ReLU)", bg: "rgba(0, 240, 255, 0.08)" },
    { color: "#ff3366", glow: "rgba(255, 51, 102, 0.4)", name: "Output (Identity Logit)", bg: "rgba(255, 51, 102, 0.08)" },
  ];

  // Calculate Neuron Coordinates
  const neuronCoords: Array<Array<{ x: number; y: number }>> = [];
  layerSizes.forEach((size, lIdx) => {
    const x = 140 + lIdx * layerSpacing;
    const coordsForLayer: Array<{ x: number; y: number }> = [];
    const maxSpacing = 40;
    const totalHeight = Math.min((size - 1) * maxSpacing, svgHeight - 160);
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
        activation: "Normalized Input",
        name: arch.feature_names[nIdx] || `Feature ${nIdx}`,
      });
      return;
    }

    const lp = layersData[lIdx - 1];
    if (!lp) return;

    const neuronWeights = lp.neurons[nIdx]?.weights || [];
    const neuronBias = lp.biases[nIdx];
    const isOutput = lIdx === nLayers - 1;
    const act = isOutput ? (arch.output_activation || "identity") : (arch.hidden_activation || "relu");

    setSelectedNeuron({
      layerIdx: lIdx,
      neuronIdx: nIdx,
      bias: neuronBias,
      weights: neuronWeights,
      activation: act.toUpperCase(),
      name: isOutput ? "Output Node (Logit z)" : `Neuron L${lIdx}:#${nIdx}`,
    });
  };

  const activeFocus = selectedNeuron || hoveredNeuron;

  return (
    <div style={{ padding: "2.5rem", maxWidth: "1440px", margin: "0 auto" }}>
      {/* Top Section Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", marginBottom: "1.75rem", flexWrap: "wrap", gap: "1rem" }}>
        <div>
          <div className="section-kicker">Interactive Topology Explorer</div>
          <h2 style={{ fontSize: "1.75rem", fontWeight: 800, color: "#ffffff", letterSpacing: "-0.025em", margin: 0 }}>
            {arch.name}
          </h2>
          <p style={{ fontSize: "0.875rem", color: "#888888", marginTop: "0.25rem" }}>
            Topology: <span style={{ color: "#ededed", fontFamily: "var(--font-mono)" }}>{arch.sizes.join(" → ")}</span> &bull; Learnable Parameters: <span style={{ color: "#00f5a0", fontFamily: "var(--font-mono)" }}>{arch.total_parameters}</span>
          </p>
        </div>

        {/* Toolbar Controls */}
        <div style={{ display: "flex", alignItems: "center", gap: "1rem", flexWrap: "wrap" }}>
          {/* Weight Threshold Pill */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "0.6rem",
              background: "#0c0c0c",
              border: "1px solid rgba(255, 255, 255, 0.08)",
              padding: "0.35rem 0.85rem",
              borderRadius: "6px",
              fontSize: "0.75rem",
              fontFamily: "var(--font-mono)",
            }}
          >
            <Sliders size={13} color="#888888" />
            <span style={{ color: "#888888" }}>FILTER |w| ≥</span>
            <input
              type="range"
              min="0"
              max="1.5"
              step="0.02"
              value={weightThreshold}
              onChange={(e) => setWeightThreshold(parseFloat(e.target.value))}
              style={{ width: "80px", cursor: "pointer", accentColor: "#00f0ff" }}
            />
            <span style={{ color: "#00f0ff", minWidth: "35px" }}>{weightThreshold.toFixed(2)}</span>
          </div>

          {/* Zoom Buttons */}
          <div style={{ display: "flex", alignItems: "center", gap: "0.25rem" }}>
            <button className="btn btn-secondary" style={{ padding: "0.35rem 0.55rem" }} onClick={() => setZoom(Math.max(0.65, zoom - 0.1))}>
              <ZoomOut size={13} />
            </button>
            <span style={{ fontSize: "0.75rem", color: "#888888", fontFamily: "var(--font-mono)", minWidth: "40px", textAlign: "center" }}>
              {Math.round(zoom * 100)}%
            </span>
            <button className="btn btn-secondary" style={{ padding: "0.35rem 0.55rem" }} onClick={() => setZoom(Math.min(1.4, zoom + 0.1))}>
              <ZoomIn size={13} />
            </button>
            <button className="btn btn-secondary" style={{ padding: "0.35rem 0.55rem" }} onClick={() => setZoom(1)}>
              <RotateCcw size={13} />
            </button>
          </div>
        </div>
      </div>

      {/* Main Canvas & Inspection Drawer */}
      <div style={{ display: "grid", gridTemplateColumns: selectedNeuron ? "1fr 360px" : "1fr", gap: "1.5rem" }}>
        {/* SVG Canvas Container */}
        <div
          className="card"
          style={{
            padding: "1rem",
            background: "#040508",
            border: "1px solid rgba(255, 255, 255, 0.08)",
            borderRadius: "10px",
            overflow: "hidden",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            boxShadow: "inset 0 0 40px rgba(0, 0, 0, 0.8)",
          }}
        >
          <div style={{ width: "100%", overflow: "auto", textAlign: "center" }}>
            <svg
              width={svgWidth * zoom}
              height={svgHeight * zoom}
              viewBox={`0 0 ${svgWidth} ${svgHeight}`}
              style={{
                borderRadius: "8px",
                backgroundImage: "radial-gradient(rgba(255, 255, 255, 0.04) 1px, transparent 1px)",
                backgroundSize: "24px 24px",
              }}
            >
              <defs>
                {/* Glow Filter for Active Nodes */}
                <filter id="neonGlow" x="-50%" y="-50%" width="200%" height="200%">
                  <feGaussianBlur stdDeviation="3.5" result="blur" />
                  <feMerge>
                    <feMergeNode in="blur" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>

                {/* Gradients for Conduits */}
                <linearGradient id="posConduit" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#00f0ff" stopOpacity="0.8" />
                  <stop offset="100%" stopColor="#8b5cf6" stopOpacity="0.8" />
                </linearGradient>
                <linearGradient id="negConduit" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#f5a623" stopOpacity="0.8" />
                  <stop offset="100%" stopColor="#ff3366" stopOpacity="0.8" />
                </linearGradient>
              </defs>

              {/* Layer Header Column Banners */}
              {arch.sizes.map((size, lIdx) => {
                const theme = layerTheme[lIdx] || layerTheme[1];
                const x = 140 + lIdx * layerSpacing;
                const isInput = lIdx === 0;
                const isOutput = lIdx === nLayers - 1;
                const layerTitle = isInput
                  ? "Input Layer"
                  : isOutput
                  ? "Output (Identity)"
                  : `Hidden L${lIdx}`;

                return (
                  <g key={`col-header-${lIdx}`}>
                    <text
                      x={x}
                      y={34}
                      textAnchor="middle"
                      fill="#ffffff"
                      fontSize={13}
                      fontWeight={700}
                      fontFamily="var(--font-sans)"
                      letterSpacing="-0.01em"
                    >
                      {layerTitle}
                    </text>
                    <rect
                      x={x - 42}
                      y={44}
                      width={84}
                      height={18}
                      rx={9}
                      fill={theme.bg}
                      stroke={theme.color}
                      strokeWidth={0.8}
                    />
                    <text
                      x={x}
                      y={56}
                      textAnchor="middle"
                      fill={theme.color}
                      fontSize={9.5}
                      fontFamily="var(--font-mono)"
                      fontWeight={600}
                    >
                      {size} NODES
                    </text>
                  </g>
                );
              })}

              {/* Connection Conduits (Graceful Cubic Bezier Splines) */}
              {layersData.map((lp, lIdx) => {
                const srcCoords = neuronCoords[lIdx];
                const dstCoords = neuronCoords[lIdx + 1];
                if (!srcCoords || !dstCoords) return null;

                const dx = dstCoords[0].x - srcCoords[0].x;
                const paths: React.ReactNode[] = [];

                for (let i = 0; i < lp.n_inputs; i++) {
                  for (let j = 0; j < lp.n_outputs; j++) {
                    const w = lp.weights[i]?.[j] ?? 0;
                    if (Math.abs(w) < weightThreshold) continue;

                    const isSrcFocused =
                      activeFocus?.layerIdx === lIdx && activeFocus?.neuronIdx === i;
                    const isDstFocused =
                      activeFocus?.layerIdx === lIdx + 1 && activeFocus?.neuronIdx === j;
                    const isHighlighted = isSrcFocused || isDstFocused;
                    const hasActiveFocus = activeFocus !== null;

                    const p1 = srcCoords[i];
                    const p2 = dstCoords[j];
                    const pathD = `M ${p1.x} ${p1.y} C ${p1.x + dx * 0.42} ${p1.y}, ${p2.x - dx * 0.42} ${p2.y}, ${p2.x} ${p2.y}`;

                    const stroke = w >= 0 ? "url(#posConduit)" : "url(#negConduit)";
                    const strokeWidth = isHighlighted
                      ? Math.min(3.5, Math.max(1.8, Math.abs(w) * 2.2))
                      : Math.min(2.0, Math.max(0.4, Math.abs(w) * 1.1));

                    const strokeOpacity = isHighlighted
                      ? 1.0
                      : hasActiveFocus
                      ? 0.04
                      : Math.min(0.5, Math.max(0.08, Math.abs(w) * 0.35));

                    paths.push(
                      <path
                        key={`conduit-${lIdx}-${i}-${j}`}
                        d={pathD}
                        fill="none"
                        stroke={stroke}
                        strokeWidth={strokeWidth}
                        strokeOpacity={strokeOpacity}
                        filter={isHighlighted ? "url(#neonGlow)" : undefined}
                      />
                    );
                  }
                }
                return <g key={`conduits-${lIdx}`}>{paths}</g>;
              })}

              {/* Neurons */}
              {neuronCoords.map((layerNodes, lIdx) => {
                const theme = layerTheme[lIdx] || layerTheme[1];
                const isInput = lIdx === 0;
                const isOutput = lIdx === nLayers - 1;

                return (
                  <g key={`nodes-${lIdx}`}>
                    {layerNodes.map((pos, nIdx) => {
                      const isSelected =
                        selectedNeuron?.layerIdx === lIdx && selectedNeuron?.neuronIdx === nIdx;
                      const isHovered =
                        hoveredNeuron?.layerIdx === lIdx && hoveredNeuron?.neuronIdx === nIdx;
                      const isHighlighted = isSelected || isHovered;

                      return (
                        <g
                          key={`neuron-${lIdx}-${nIdx}`}
                          onClick={() => handleNeuronClick(lIdx, nIdx)}
                          onMouseEnter={() => setHoveredNeuron({ layerIdx: lIdx, neuronIdx: nIdx })}
                          onMouseLeave={() => setHoveredNeuron(null)}
                          style={{ cursor: "pointer" }}
                        >
                          {/* Outer Aura Halo */}
                          <circle
                            cx={pos.x}
                            cy={pos.y}
                            r={isHighlighted ? 15 : 10}
                            fill={theme.glow}
                            opacity={isHighlighted ? 0.9 : 0.25}
                            filter="url(#neonGlow)"
                            style={{ transition: "all 0.15s ease" }}
                          />

                          {/* Middle Core Ring */}
                          <circle
                            cx={pos.x}
                            cy={pos.y}
                            r={isHighlighted ? 9 : 6.5}
                            fill="#06080d"
                            stroke={isHighlighted ? "#ffffff" : theme.color}
                            strokeWidth={isHighlighted ? 2.5 : 1.8}
                          />

                          {/* Center Nucleus Pip */}
                          <circle
                            cx={pos.x}
                            cy={pos.y}
                            r={2.2}
                            fill={isHighlighted ? "#ffffff" : theme.color}
                          />

                          {/* Left Input Feature Name Labels */}
                          {isInput && (
                            <text
                              x={pos.x - 16}
                              y={pos.y + 3.5}
                              textAnchor="end"
                              fill={isHighlighted ? "#ffffff" : "#999999"}
                              fontSize={10}
                              fontFamily="var(--font-mono)"
                              fontWeight={isHighlighted ? 600 : 400}
                            >
                              {arch.feature_names[nIdx] || `f[${nIdx}]`}
                            </text>
                          )}

                          {/* Right Output: Identity Logit -> Sigmoid Probability Conversion */}
                          {isOutput && (
                            <g>
                              <line
                                x1={pos.x + 12}
                                y1={pos.y}
                                x2={pos.x + 32}
                                y2={pos.y}
                                stroke="#ff3366"
                                strokeWidth={1.5}
                                strokeDasharray="3 3"
                              />
                              <rect
                                x={pos.x + 36}
                                y={pos.y - 12}
                                width={52}
                                height={24}
                                rx={4}
                                fill="#110508"
                                stroke="#ff3366"
                                strokeWidth={1}
                              />
                              <text
                                x={pos.x + 62}
                                y={pos.y + 4}
                                textAnchor="middle"
                                fill="#ff3366"
                                fontSize={10}
                                fontFamily="var(--font-mono)"
                                fontWeight={700}
                              >
                                &sigma;(z)
                              </text>
                              <text
                                x={pos.x + 98}
                                y={pos.y + 4}
                                textAnchor="start"
                                fill="#ffffff"
                                fontSize={11}
                                fontFamily="var(--font-mono)"
                                fontWeight={700}
                                letterSpacing="0.04em"
                              >
                                P(ATTACK)
                              </text>
                            </g>
                          )}
                        </g>
                      );
                    })}
                  </g>
                );
              })}
            </svg>
          </div>

          {/* Aesthetic Color Legend Footer */}
          <div
            style={{
              display: "flex",
              justifyContent: "center",
              gap: "2rem",
              marginTop: "0.85rem",
              paddingTop: "0.85rem",
              borderTop: "1px solid rgba(255, 255, 255, 0.05)",
              fontSize: "0.75rem",
              fontFamily: "var(--font-mono)",
              color: "#777777",
              flexWrap: "wrap",
            }}
          >
            <span style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
              <span style={{ width: "8px", height: "8px", borderRadius: "50%", background: "#f59e0b", boxShadow: "0 0 6px #f59e0b" }} />
              Input Features (13)
            </span>
            <span style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
              <span style={{ width: "8px", height: "8px", borderRadius: "50%", background: "#8b5cf6", boxShadow: "0 0 6px #8b5cf6" }} />
              Hidden L1 ReLU (16)
            </span>
            <span style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
              <span style={{ width: "8px", height: "8px", borderRadius: "50%", background: "#00f0ff", boxShadow: "0 0 6px #00f0ff" }} />
              Hidden L2 ReLU (8)
            </span>
            <span style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
              <span style={{ width: "8px", height: "8px", borderRadius: "50%", background: "#ff3366", boxShadow: "0 0 6px #ff3366" }} />
              Output Logit + Sigmoid (1)
            </span>
            <span style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
              <span style={{ width: "16px", height: "2px", background: "linear-gradient(90deg, #00f0ff, #8b5cf6)" }} />
              Positive Weights
            </span>
            <span style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
              <span style={{ width: "16px", height: "2px", background: "linear-gradient(90deg, #f5a623, #ff3366)" }} />
              Negative Weights
            </span>
          </div>
        </div>

        {/* Selected Neuron Inspector Drawer */}
        {selectedNeuron && (
          <div
            className="card"
            style={{
              height: "fit-content",
              background: "#0a0a0a",
              border: "1px solid rgba(255, 255, 255, 0.12)",
              borderRadius: "8px",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
              <div>
                <span className="section-kicker">Inspector</span>
                <h3 style={{ fontSize: "1.1rem", fontWeight: 700, color: "#ffffff", margin: 0, fontFamily: "var(--font-mono)" }}>
                  {selectedNeuron.name}
                </h3>
              </div>
              <button
                className="btn btn-secondary"
                style={{ padding: "0.3rem", borderRadius: "50%" }}
                onClick={() => setSelectedNeuron(null)}
              >
                <X size={14} />
              </button>
            </div>

            <div style={{ display: "grid", gap: "0.6rem", marginBottom: "1.25rem", fontSize: "0.8125rem", fontFamily: "var(--font-mono)" }}>
              <div style={{ display: "flex", justifyContent: "space-between", padding: "0.4rem 0", borderBottom: "1px solid rgba(255, 255, 255, 0.05)" }}>
                <span style={{ color: "#777777" }}>Layer Position</span>
                <span style={{ color: "#ededed", fontWeight: 600 }}>
                  {selectedNeuron.layerIdx === 0 ? "Input Dimension" : `Layer ${selectedNeuron.layerIdx}`}
                </span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", padding: "0.4rem 0", borderBottom: "1px solid rgba(255, 255, 255, 0.05)" }}>
                <span style={{ color: "#777777" }}>Activation Function</span>
                <span className="badge badge-info">{selectedNeuron.activation}</span>
              </div>
              {selectedNeuron.bias !== undefined && (
                <div style={{ display: "flex", justifyContent: "space-between", padding: "0.4rem 0", borderBottom: "1px solid rgba(255, 255, 255, 0.05)" }}>
                  <span style={{ color: "#777777" }}>Neuron Bias (b)</span>
                  <span style={{ color: selectedNeuron.bias >= 0 ? "#00f0ff" : "#ff3366", fontWeight: 700 }}>
                    {selectedNeuron.bias.toFixed(6)}
                  </span>
                </div>
              )}
            </div>

            {/* Formula Callout */}
            <div style={{ background: "#050505", padding: "0.75rem", borderRadius: "6px", border: "1px solid rgba(255, 255, 255, 0.06)", marginBottom: "1rem" }}>
              <span style={{ fontSize: "0.6875rem", color: "#666666", textTransform: "uppercase", fontFamily: "var(--font-mono)" }}>
                Mathematical Formulation
              </span>
              <p style={{ fontSize: "0.75rem", color: "#00f0ff", fontFamily: "var(--font-mono)", margin: "0.25rem 0 0 0" }}>
                z = b + &Sigma; (x_i &bull; w_i) &rarr; a = {selectedNeuron.activation.toLowerCase()}(z)
              </p>
            </div>

            {/* Weights Breakdown Table */}
            {selectedNeuron.weights && selectedNeuron.weights.length > 0 && (
              <div>
                <span style={{ fontSize: "0.6875rem", color: "#666666", textTransform: "uppercase", fontFamily: "var(--font-mono)", letterSpacing: "0.06em" }}>
                  Incoming Connection Weights ({selectedNeuron.weights.length})
                </span>
                <div
                  style={{
                    maxHeight: "260px",
                    overflowY: "auto",
                    marginTop: "0.5rem",
                    border: "1px solid rgba(255, 255, 255, 0.06)",
                    borderRadius: "6px",
                  }}
                >
                  <table className="clean-table" style={{ fontSize: "0.75rem" }}>
                    <thead>
                      <tr>
                        <th>Source Node</th>
                        <th style={{ textAlign: "right" }}>Weight (w)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {selectedNeuron.weights.map((w, idx) => {
                        const fromLabel =
                          selectedNeuron.layerIdx === 1
                            ? arch.feature_names[idx] || `x[${idx}]`
                            : `h[${selectedNeuron.layerIdx - 1}][${idx}]`;

                        return (
                          <tr key={`w-${idx}`}>
                            <td style={{ color: "#a1a1a1" }}>{fromLabel}</td>
                            <td
                              style={{
                                textAlign: "right",
                                fontWeight: 600,
                                color: w >= 0 ? "#00f0ff" : "#ff3366",
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
