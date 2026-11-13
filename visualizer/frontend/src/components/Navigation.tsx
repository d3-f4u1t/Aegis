import React from "react";
import {
  Activity,
  BarChart2,
  Box,
  Cpu,
  Layers,
  Network,
  ShieldAlert,
} from "lucide-react";
import { ModelSummary } from "../types";

export type NavTab =
  | "overview"
  | "architecture"
  | "forward"
  | "parameters"
  | "experiments"
  | "traffic"
  | "compare";

interface NavigationProps {
  currentTab: NavTab;
  onSelectTab: (tab: NavTab) => void;
  models: ModelSummary[];
  selectedModelId: string;
  onSelectModel: (modelId: string) => void;
}

export const Navigation: React.FC<NavigationProps> = ({
  currentTab,
  onSelectTab,
  models,
  selectedModelId,
  onSelectModel,
}) => {
  const currentModel = models.find((m) => m.model_id === selectedModelId);

  const tabs: Array<{ id: NavTab; label: string; icon: React.ReactNode }> = [
    { id: "overview", label: "Overview", icon: <Box size={16} /> },
    { id: "architecture", label: "Architecture", icon: <Network size={16} /> },
    { id: "forward", label: "Forward Pass", icon: <Activity size={16} /> },
    { id: "parameters", label: "Parameters", icon: <Layers size={16} /> },
    { id: "experiments", label: "Experiments", icon: <BarChart2 size={16} /> },
    { id: "traffic", label: "Traffic Lab", icon: <ShieldAlert size={16} /> },
    { id: "compare", label: "Model Compare", icon: <Cpu size={16} /> },
  ];

  return (
    <header style={{ borderBottom: "1px solid #243042", background: "#0e1420", position: "sticky", top: 0, zIndex: 50 }}>
      {/* Top Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "0.75rem 1.5rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
          <div
            style={{
              width: "36px",
              height: "36px",
              borderRadius: "8px",
              background: "linear-gradient(135deg, #2563eb, #38bdf8)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontWeight: 800,
              fontSize: "1.1rem",
              color: "white",
            }}
          >
            NN
          </div>
          <div>
            <h1 style={{ fontSize: "1.1rem", fontWeight: 700, color: "#f8fafc", margin: 0, lineHeight: 1.2 }}>
              Neural Network Visualization Laboratory
            </h1>
            <p style={{ fontSize: "0.75rem", color: "#94a3b8", margin: 0 }}>
              Website Traffic Anomaly Detection &bull; Pure Mathematical Implementation
            </p>
          </div>
        </div>

        {/* Model Selector & Status */}
        <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
          {currentModel && (
            <span
              className={
                currentModel.engine === "numpy"
                  ? "badge badge-info"
                  : currentModel.engine === "scalar"
                  ? "badge badge-warning"
                  : "badge badge-normal"
              }
            >
              Engine: {currentModel.engine}
            </span>
          )}

          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <label style={{ fontSize: "0.8rem", color: "#94a3b8" }}>Active Model:</label>
            <select
              value={selectedModelId}
              onChange={(e) => onSelectModel(e.target.value)}
              style={{
                background: "#1e293b",
                color: "#f1f5f9",
                border: "1px solid #334155",
                borderRadius: "6px",
                padding: "0.35rem 0.75rem",
                fontSize: "0.8rem",
                cursor: "pointer",
                maxWidth: "280px",
              }}
            >
              {models.map((m) => (
                <option key={m.model_id} value={m.model_id}>
                  {m.name} ({m.engine})
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <nav style={{ display: "flex", gap: "0.25rem", padding: "0 1.5rem", overflowX: "auto" }}>
        {tabs.map((tab) => {
          const isActive = currentTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => onSelectTab(tab.id)}
              style={{
                display: "flex",
                alignItems: "center",
                gap: "0.45rem",
                padding: "0.6rem 1rem",
                border: "none",
                background: "transparent",
                color: isActive ? "#38bdf8" : "#94a3b8",
                borderBottom: isActive ? "2px solid #38bdf8" : "2px solid transparent",
                fontWeight: isActive ? 600 : 500,
                fontSize: "0.85rem",
                cursor: "pointer",
                transition: "all 0.15s ease",
                whiteSpace: "nowrap",
              }}
            >
              {tab.icon}
              {tab.label}
            </button>
          );
        })}
      </nav>
    </header>
  );
};
