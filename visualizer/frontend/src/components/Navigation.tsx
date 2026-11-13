import React from "react";
import {
  Activity,
  BarChart3,
  Box,
  ChevronDown,
  Cpu,
  GitBranch,
  Layers,
  Network,
  Radio,
  ShieldAlert,
} from "lucide-react";
import type { ModelSummary } from "../types";

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
    { id: "overview", label: "Overview", icon: <Box size={14} /> },
    { id: "architecture", label: "Network Topology", icon: <Network size={14} /> },
    { id: "forward", label: "Forward Tracer", icon: <Activity size={14} /> },
    { id: "parameters", label: "Weight Heatmaps", icon: <Layers size={14} /> },
    { id: "experiments", label: "Phase 5 Experiments", icon: <BarChart3 size={14} /> },
    { id: "traffic", label: "Traffic Lab & Blindspots", icon: <ShieldAlert size={14} /> },
    { id: "compare", label: "Engine Benchmark", icon: <Cpu size={14} /> },
  ];

  return (
    <header
      style={{
        position: "sticky",
        top: 0,
        zIndex: 50,
        background: "rgba(0, 0, 0, 0.8)",
        backdropFilter: "blur(16px)",
        WebkitBackdropFilter: "blur(16px)",
        borderBottom: "1px solid rgba(255, 255, 255, 0.08)",
      }}
    >
      {/* Top Utility Bar (Vercel Style) */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          padding: "0.6rem 2rem",
          borderBottom: "1px solid rgba(255, 255, 255, 0.05)",
          height: "50px",
        }}
      >
        {/* Brand & Breadcrumbs */}
        <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
          {/* Vercel Geometric Glyph */}
          <div
            style={{
              width: "22px",
              height: "22px",
              background: "#ffffff",
              clipPath: "polygon(50% 0%, 0% 100%, 100% 100%)",
            }}
          />

          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", fontSize: "0.8125rem" }}>
            <span style={{ color: "#777777", fontWeight: 500 }}>deepmind</span>
            <span style={{ color: "#333333" }}>/</span>
            <span style={{ color: "#ededed", fontWeight: 600, letterSpacing: "-0.01em" }}>
              neural-traffic-detector
            </span>
          </div>

          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "0.35rem",
              background: "rgba(255, 255, 255, 0.04)",
              border: "1px solid rgba(255, 255, 255, 0.08)",
              padding: "0.15rem 0.55rem",
              borderRadius: "4px",
              fontSize: "0.6875rem",
              color: "#888888",
              fontFamily: "var(--font-mono)",
            }}
          >
            <GitBranch size={11} color="#666666" />
            <span>main</span>
            <span style={{ color: "#444444" }}>&bull;</span>
            <span style={{ color: "#aaaaaa" }}>42379bb</span>
          </div>
        </div>

        {/* Right Status & Model Selector */}
        <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
          {/* Live Port Indicator */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "0.4rem",
              fontSize: "0.6875rem",
              fontFamily: "var(--font-mono)",
              color: "#888888",
            }}
          >
            <span
              style={{
                width: "6px",
                height: "6px",
                borderRadius: "50%",
                background: "#00f5a0",
                boxShadow: "0 0 8px #00f5a0",
              }}
            />
            <span>PORT 8080</span>
          </div>

          {/* Active Engine Badge */}
          {currentModel && (
            <div
              className={
                currentModel.engine === "numpy"
                  ? "badge badge-info"
                  : currentModel.engine === "scalar"
                  ? "badge badge-warning"
                  : "badge badge-normal"
              }
              style={{ fontSize: "0.6875rem" }}
            >
              ENGINE: {currentModel.engine.toUpperCase()}
            </div>
          )}

          {/* Model Selector Dropdown Pill */}
          <div style={{ position: "relative" }}>
            <select
              value={selectedModelId}
              onChange={(e) => onSelectModel(e.target.value)}
              style={{
                appearance: "none",
                background: "#0f0f0f",
                color: "#ededed",
                border: "1px solid rgba(255, 255, 255, 0.12)",
                borderRadius: "6px",
                padding: "0.35rem 2rem 0.35rem 0.85rem",
                fontSize: "0.75rem",
                fontFamily: "var(--font-mono)",
                cursor: "pointer",
                outline: "none",
                minWidth: "250px",
              }}
            >
              {models.map((m) => (
                <option key={m.model_id} value={m.model_id}>
                  {m.name} [{m.engine}]
                </option>
              ))}
            </select>
            <ChevronDown
              size={12}
              color="#888888"
              style={{ position: "absolute", right: "0.65rem", top: "50%", transform: "translateY(-50%)", pointerEvents: "none" }}
            />
          </div>
        </div>
      </div>

      {/* Primary Navigation Tabs */}
      <nav
        style={{
          display: "flex",
          gap: "0.25rem",
          padding: "0 2rem",
          overflowX: "auto",
        }}
      >
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
                padding: "0.65rem 0.95rem",
                border: "none",
                background: "transparent",
                color: isActive ? "#ffffff" : "#777777",
                borderBottom: isActive ? "2px solid #ffffff" : "2px solid transparent",
                fontWeight: isActive ? 600 : 400,
                fontSize: "0.8125rem",
                cursor: "pointer",
                transition: "all 0.15s ease",
                whiteSpace: "nowrap",
              }}
              onMouseEnter={(e) => {
                if (!isActive) e.currentTarget.style.color = "#bbbbbb";
              }}
              onMouseLeave={(e) => {
                if (!isActive) e.currentTarget.style.color = "#777777";
              }}
            >
              <span style={{ color: isActive ? "#00f0ff" : "inherit" }}>{tab.icon}</span>
              {tab.label}
            </button>
          );
        })}
      </nav>
    </header>
  );
};
