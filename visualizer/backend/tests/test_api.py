"""Integration tests for FastAPI endpoints."""

from fastapi.testclient import TestClient
from visualizer.backend.main import app

client = TestClient(app)


def test_health():
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["models_count"] >= 10


def test_list_models():
    res = client.get("/api/v1/models")
    assert res.status_code == 200
    models = res.json()
    assert isinstance(models, list)
    assert any(m["model_id"] == "phase5_exp1_indistribution_numpy_mlp" for m in models)


def test_get_model_architecture():
    res = client.get("/api/v1/models/phase5_exp1_indistribution_numpy_mlp")
    assert res.status_code == 200
    arch = res.json()
    assert arch["sizes"] == [13, 16, 8, 1]
    assert len(arch["feature_names"]) == 13

    # Unknown model -> 404
    bad = client.get("/api/v1/models/nonexistent_model")
    assert bad.status_code == 404


def test_get_layer_parameters():
    res = client.get("/api/v1/models/phase5_exp1_indistribution_numpy_mlp/layers/0")
    assert res.status_code == 200
    layer = res.json()
    assert layer["n_inputs"] == 13
    assert layer["n_outputs"] == 16
    assert len(layer["weights"]) == 13

    # Invalid layer index -> 400
    bad = client.get("/api/v1/models/phase5_exp1_indistribution_numpy_mlp/layers/99")
    assert bad.status_code == 400


def test_trace_endpoint():
    payload = {
        "features": [0.0] * 13,
        "is_scaled": True,
        "threshold": 0.5,
    }
    res = client.post("/api/v1/models/phase5_exp1_indistribution_numpy_mlp/trace", json=payload)
    assert res.status_code == 200
    trace = res.json()
    assert trace["parity_verified"] is True
    assert 0.0 <= trace["probability"] <= 1.0
    assert len(trace["layers"]) == 3


def test_compare_endpoint():
    payload = {
        "features": [0.0] * 13,
        "is_scaled": True,
        "threshold": 0.5,
    }
    res = client.post("/api/v1/compare", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert len(data["results"]) == 4
    engines = {r["engine"] for r in data["results"]}
    assert "numpy" in engines
    assert "scalar" in engines
    assert "logistic_regression" in engines
    assert "threshold" in engines


def test_experiments_endpoints():
    res = client.get("/api/v1/experiments")
    assert res.status_code == 200
    exps = res.json()
    assert len(exps) == 3
    exp_ids = {e["experiment_id"] for e in exps}
    assert "exp1_indistribution" in exp_ids
    assert "exp2_unseen_attacks" in exp_ids
    assert "exp3_unseen_polling" in exp_ids

    # Detail
    rep = client.get("/api/v1/experiments/exp1_indistribution")
    assert rep.status_code == 200
    data = rep.json()
    assert data["experiment_id"] == "exp1_indistribution"
    assert "numpy_mlp" in data["models"]
    assert data["models"]["numpy_mlp"]["accuracy"] > 0.95


def test_telemetry_samples_endpoint():
    res = client.get("/api/v1/telemetry/samples")
    assert res.status_code == 200
    samples = res.json()
    assert len(samples) >= 13
    assert any(s["is_failure_case"] for s in samples)

    # Filter
    filt = client.get("/api/v1/telemetry/samples?scenario=attack_high_volume_scraping")
    assert filt.status_code == 200
    items = filt.json()
    assert len(items) > 0
    assert all(i["scenario"] == "attack_high_volume_scraping" for i in items)
