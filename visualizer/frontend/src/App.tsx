import React, { useEffect, useState } from "react";
import { Navigation, NavTab } from "./components/Navigation";
import { OverviewPage } from "./pages/OverviewPage";
import { ArchitecturePage } from "./pages/ArchitecturePage";
import { ForwardPassPage } from "./pages/ForwardPassPage";
import { ParametersPage } from "./pages/ParametersPage";
import { ExperimentsPage } from "./pages/ExperimentsPage";
import { TrafficLabPage } from "./pages/TrafficLabPage";
import { ModelComparisonPage } from "./pages/ModelComparisonPage";
import { ModelArchitecture, ModelSummary } from "./types";
import { api } from "./services/api";

export const App: React.FC = () => {
  const [currentTab, setCurrentTab] = useState<NavTab>("overview");
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string>("phase5_exp1_indistribution_numpy_mlp");
  const [architecture, setArchitecture] = useState<ModelArchitecture | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [backendError, setBackendError] = useState<string | null>(null);

  // Load models on startup
  useEffect(() => {
    let isMounted = true;
    async function init() {
      try {
        const mList = await api.listModels();
        if (!isMounted) return;
        setModels(mList);

        // Prefer exp1 numpy mlp if present
        const preferred = mList.find((m) => m.model_id === "phase5_exp1_indistribution_numpy_mlp");
        if (preferred) {
          setSelectedModelId(preferred.model_id);
        } else if (mList.length > 0) {
          setSelectedModelId(mList[0].model_id);
        }
      } catch (err: any) {
        if (!isMounted) return;
        setBackendError(
          "Could not connect to FastAPI visualizer backend at http://localhost:8000. Please start the backend service."
        );
      } finally {
        if (isMounted) setLoading(false);
      }
    }
    init();
    return () => {
      isMounted = false;
    };
  }, []);

  // Load architecture when selectedModelId changes
  useEffect(() => {
    let isMounted = true;
    async function loadArch() {
      if (!selectedModelId) return;
      try {
        const arch = await api.getModelArchitecture(selectedModelId);
        if (!isMounted) return;
        setArchitecture(arch);
      } catch (err: any) {
        // non-blocking
      }
    }
    loadArch();
    return () => {
      isMounted = false;
    };
  }, [selectedModelId]);

  const currentSummary = models.find((m) => m.model_id === selectedModelId) || null;

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      <Navigation
        currentTab={currentTab}
        onSelectTab={setCurrentTab}
        models={models}
        selectedModelId={selectedModelId}
        onSelectModel={setSelectedModelId}
      />

      {backendError && (
        <div
          style={{
            background: "#210f13",
            color: "#fca5a5",
            padding: "0.75rem 1.5rem",
            fontSize: "0.85rem",
            borderBottom: "1px solid #b91c1c",
            textAlign: "center",
          }}
        >
          {backendError}
        </div>
      )}

      <main style={{ flex: 1 }}>
        {currentTab === "overview" && (
          <OverviewPage
            summary={currentSummary}
            architecture={architecture}
            onNavigate={setCurrentTab}
          />
        )}
        {currentTab === "architecture" && (
          <ArchitecturePage modelId={selectedModelId} />
        )}
        {currentTab === "forward" && (
          <ForwardPassPage modelId={selectedModelId} />
        )}
        {currentTab === "parameters" && (
          <ParametersPage modelId={selectedModelId} />
        )}
        {currentTab === "experiments" && (
          <ExperimentsPage />
        )}
        {currentTab === "traffic" && (
          <TrafficLabPage />
        )}
        {currentTab === "compare" && (
          <ModelComparisonPage />
        )}
      </main>

      <footer
        style={{
          borderTop: "1px solid #1e293b",
          padding: "1rem 2rem",
          fontSize: "0.75rem",
          color: "#64748b",
          textAlign: "center",
          background: "#0b0f17",
        }}
      >
        Neural Network Traffic Anomaly Detection Visualization Laboratory &bull; Phase 5 Deliverable &bull; From-Scratch Mathematical Implementation
      </footer>
    </div>
  );
};
export default App;
