# Integrated Neural Network Visualization Laboratory

## 1. Executive Summary & Purpose

The **Neural Network Visualization Laboratory** is an interactive, academic inspection application developed specifically for the From-Scratch Website Traffic Anomaly Detection Neural Network project.

Rather than providing a generic, toy neural network toy sandbox, the Visualization Laboratory is directly integrated with our actual model checkpoints, serialization formats, telemetry preprocessing artifacts, and experimental evaluation results.

Its core objectives are:
1. **Explain and Demonstrate Implementation:** Make internal operations observable ($Z = XW + b$, $A = \text{activation}(Z)$) down to each neuron.
2. **Verify Mathematical Parity:** Visually and numerically verify that the audited scalar reference engine (`nn/`) and the vectorized NumPy engine (`nn_numpy/`) produce mathematically identical outputs ($< 10^{-11}$ error).
3. **Inspect Learned Representations:** Render parameter heatmaps, weight matrices, and bias vectors directly from authentic trained weights without hardcoding or inventing parameters.
4. **Communicate Empirical Results:** Provide an interactive dashboard for Phase 5 experiment logs (In-Distribution, Unseen Attacks, Unseen Polling), confusion matrices, and the critical novel zero-error scraping failure case.
5. **Strict Architectural Separation:** Maintain complete separation between visualization code (`visualizer/`) and the core neural network engines (`nn/`, `nn_numpy/`), adhering strictly to read-only access principles.

---

## 2. System Architecture

The application is structured into a clean backend service and an optimized React/TypeScript frontend:

```
visualizer/
├── backend/
│   ├── adapters/               # Unified model adapter layer
│   │   ├── base.py             # Abstract ModelAdapter interface
│   │   ├── scalar_adapter.py   # Adapter for Format Version 1 (nn.model.MLP)
│   │   ├── numpy_adapter.py    # Adapter for Format Version 2 (nn_numpy.model.MLP)
│   │   ├── baseline_adapter.py # Adapters for Logistic Regression and Threshold
│   │   └── registry.py         # Dynamic model discovery & preprocessing loader
│   ├── artifact_readers/       # Read-only loaders for datasets and logs
│   │   ├── experiment_reader.py# Parses experiments/results/phase5_summary.json
│   │   └── dataset_reader.py   # Extracts curated samples from synthetic_phase5.jsonl
│   ├── tracing/
│   │   └── forward_tracer.py   # Observable step-by-step forward propagation
│   ├── schemas/
│   │   └── api_models.py       # Pydantic request/response schemas
│   ├── tests/                  # Pytest test suite for visualizer
│   │   ├── test_adapters.py    # Adapter loading, shape, and index bounds tests
│   │   ├── test_tracer.py      # Numerical parity verification tests (< 10^-11)
│   │   └── test_api.py         # FastAPI TestClient endpoint integration tests
│   └── main.py                 # FastAPI application & static distribution server
└── frontend/
    ├── src/
    │   ├── components/
    │   │   └── Navigation.tsx   # Header, model selector, tab navigation
    │   ├── pages/
    │   │   ├── OverviewPage.tsx        # System specs & research overview
    │   │   ├── ArchitecturePage.tsx    # Interactive SVG layered network graph
    │   │   ├── ForwardPassPage.tsx     # Step-by-step playback & formula cards
    │   │   ├── ParametersPage.tsx      # Diverging weight heatmaps & bias bars
    │   │   ├── ExperimentsPage.tsx     # Phase 5 evaluation, confusion matrix, PR-AUC
    │   │   ├── TrafficLabPage.tsx      # 13-scenario inspector & scraping study
    │   │   └── ModelComparisonPage.tsx # Simultaneous 4-model inference benchmark
    │   ├── services/
    │   │   └── api.ts          # Typed API client with auto-fallback
    │   ├── types/
    │   │   └── index.ts        # TypeScript schemas matching Pydantic models
    │   ├── App.tsx             # Root application coordinator
    │   └── index.css           # Academic dark-theme design system
    ├── dist/                   # Production pre-compiled static assets
    ├── package.json
    ├── tsconfig.json
    └── vite.config.ts
```

---

## 3. Unified Model Adapter Contract

All models—whether scalar reference, vectorized NumPy, logistic regression, or threshold rule—conform to the `ModelAdapter` interface defined in [`visualizer/backend/adapters/base.py`](file:///C:/Users/hkpan/NN/visualizer/backend/adapters/base.py):

| Method | Output | Description |
|---|---|---|
| `get_summary()` | `ModelSummary` | High-level topology, parameter count, schema version |
| `get_architecture()` | `ModelArchitecture` | Layer dimensions, activation functions, feature names |
| `get_layer_parameters(idx)` | `LayerParameters` | Weight matrix $W \in \mathbb{R}^{n_{in} \times n_{out}}$ and bias $b \in \mathbb{R}^{n_{out}}$ |
| `predict_proba(features)` | `float` | Direct attack probability $P(\text{attack} \mid x)$ |
| `predict(features, threshold)` | `int` | Binary decision ($0 = \text{normal}, 1 = \text{attack}$) |
| `trace_forward(...)` | `ForwardTraceResult` | Observable layer-by-layer computation with exact intermediate $Z$ and $A$ |

### Engine Parity Guarantee
The forward tracer in [`visualizer/backend/tracing/forward_tracer.py`](file:///C:/Users/hkpan/NN/visualizer/backend/tracing/forward_tracer.py) computes exact intermediate numbers without modifying engine code. Upon completing every forward trace, it asserts:

$$\left| P_{\text{traced}}(\text{attack}) - P_{\text{engine}}(\text{attack}) \right| < 10^{-11}$$

If parity holds, `parity_verified: true` is set on the response payload.

---

## 4. Key Laboratory Modules

### 4.1. Network Architecture Explorer
- Renders an interactive SVG topological graph representing the exact $13 \to 16 \to 8 \to 1$ multilayer perceptron.
- Edge thickness and coloring reflect actual weight parameters (blue for positive, red for negative).
- Adjustable weight threshold slider filters weak weights ($|w| < \tau$) to prevent visual clutter.
- Clicking any neuron displays incoming connection weights, bias, activation function, and algebraic formula.

### 4.2. Step-by-Step Forward Propagation Tracer
- Step-by-step playback through input feature vector, hidden layer 1, hidden layer 2, output layer, and classification decision.
- Displays mathematical equations with exact numbers substituted:
  $$z_j = b_j + \sum_{i=1}^{n_{in}} x_i \cdot w_{ij}$$
  $$a_j = \max(0, z_j) \quad \text{or} \quad \sigma(z_j)$$
- Distinguishes active ReLU neurons ($a > 0$) from inactive/dead neurons ($a = 0$).
- Real-time classification margin meter displaying $|p - \tau|$.

### 4.3. Weight Matrices & Parameter Heatmaps
- Renders diverging heatmaps of layer weight matrices ($W^{(1)} \in \mathbb{R}^{13 \times 16}$, $W^{(2)} \in \mathbb{R}^{16 \times 8}$, $W^{(3)} \in \mathbb{R}^{8 \times 1}$).
- Visualizes bias vectors with positive/negative bar indicators.
- Displays layer statistics: min/max bounds, mean, standard deviation, Frobenius $L_2$ norm, and sparsity percentage ($|w| < 0.05$).

### 4.4. Empirical Experiments Dashboard (Phase 5)
- Explores results from **Exp 1 (In-Distribution)**, **Exp 2 (Unseen Attacks)**, and **Exp 3 (Unseen Polling)**.
- Side-by-side performance table displaying Accuracy, Precision, Recall, F1, FPR, FNR, PR-AUC, and Average Precision.
- Interactive $2 \times 2$ Confusion Matrix (TP, FP, TN, FN).
- Per-scenario detection recall breakdown across all 13 scenarios.

### 4.5. Traffic Laboratory & The Zero-Error Scraping Blindspot
- Curated library of 26 authentic telemetry windows across 13 traffic scenarios.
- Detailed feature inspection comparing raw measurements against normalized z-scores.
- **Academic Deep-Dive:** Analysis of `sample_attack_high_volume_scraping_1`. Explains why the supervised MLP suffered 0% recall ($p = 0.0001$, false negative) because the scraping bot retrieved valid resources with HTTP 200 OK responses, avoiding 4xx/5xx and auth errors. This empirical proof demonstrates the limits of supervised classification and provides the academic rationale for Phase 7 (Unsupervised Autoencoder Anomaly Detection).

### 4.6. Multi-Model Engine Comparison
- Executes simultaneous inference on the same sample across all 4 project model families:
  1. Threshold Baseline (`requests_per_second >= 0.4968`)
  2. Logistic Regression ($13 \to 1$)
  3. Scalar Reference MLP ($13 \to 16 \to 8 \to 1$)
  4. Vectorized NumPy MLP ($13 \to 16 \to 8 \to 1$)
- Confirms bit-level numerical parity between Scalar and NumPy MLPs while highlighting the **~170× latency speedup** of vectorized BLAS matrix multiplication over pure Python iteration.

---

## 5. How to Run the Visualization Laboratory

### Option A: Unified Standalone Server (Production Mode)
The FastAPI backend serves the pre-compiled frontend static bundle on port `8000`:

```powershell
python -m uvicorn visualizer.backend.main:app --port 8000
```
Open your browser and navigate to:
```
http://localhost:8000
```

### Option B: Development Mode (With Hot Reloading)
For live development with instant frontend hot-reloading:

**Terminal 1 (Backend API):**
```powershell
python -m uvicorn visualizer.backend.main:app --reload --port 8000
```

**Terminal 2 (Vite Frontend):**
```powershell
cd visualizer/frontend
npm run dev
```
Open your browser and navigate to:
```
http://localhost:3000
```

---

## 6. Verification and Test Results

The test suite thoroughly validates core neural network engines, telemetry pipelines, evaluation metrics, and the visualization backend:

```powershell
pytest
```

**Test Execution Summary:**
- **Core Neural Network Suite:** 187 passed
  - Scalar Engine (`tests/test_forward.py`, `tests/test_gradients.py`, `tests/test_activations.py`, etc.)
  - Vectorized NumPy Engine (`tests/test_numpy_engine.py`, `tests/test_numpy_parity.py`)
  - Telemetry & Preprocessing (`tests/test_telemetry/`)
  - Phase 5 Experiments & Evaluators (`tests/test_harness/`, `tests/test_traffic_generator.py`)
- **Visualization Laboratory Backend Suite:** 18 passed
  - Model Adapters (`visualizer/backend/tests/test_adapters.py`)
  - Forward Tracer Parity (`visualizer/backend/tests/test_tracer.py`)
  - FastAPI API Endpoints (`visualizer/backend/tests/test_api.py`)
- **Total Passing Tests:** **205 passed in 8.86s** (0 failures).
