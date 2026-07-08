"""FastAPI app smoke tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.server.main import app, mount_static


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health(client: TestClient) -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_graph_endpoint(client: TestClient) -> None:
    r = client.get("/api/graph")
    assert r.status_code == 200
    data = r.json()
    assert "nodes" in data
    assert "edges" in data
    assert "revision" in data.get("meta", {})


def test_rescan_endpoint(client: TestClient) -> None:
    r = client.post("/api/state/rescan")
    assert r.status_code == 200
    assert "revision" in r.json()


def test_pipelines(client: TestClient) -> None:
    r = client.get("/api/pipelines")
    assert r.status_code == 200
    assert "graph.prepare" in r.json()["order"]


def test_branch_fields_endpoint(client: TestClient) -> None:
    graph = client.get("/api/graph").json()
    edges_node = next(
        (n for n in graph["nodes"] if n.get("step_id") == "edges" and n.get("run_id")),
        None,
    )
    if edges_node is None:
        pytest.skip("no edges run step in graph")
    r = client.get(f"/api/nodes/{edges_node['id']}/branch-fields")
    assert r.status_code == 200
    data = r.json()
    assert "fields_by_step" in data
    assert len(data["fields_by_step"]) >= 1
    assert "graph.prepare" in data["pipelines_from_here"]
    step_ids = {s["step_id"] for s in data["fields_by_step"] if s["pipeline"] == "graph.prepare"}
    assert "edges" in step_ids
    assert "feature_dates" not in step_ids
    assert "universe" not in step_ids


def test_spa_fallback_serves_index_for_client_routes(client: TestClient) -> None:
    dist = ROOT / "app" / "web" / "dist"
    if not (dist / "index.html").is_file():
        pytest.skip("frontend not built")
    mount_static(dist)
    for path in ("/graph", "/jobs", "/compare", "/config"):
        r = client.get(path)
        assert r.status_code == 200
        assert "CEMSPIMS Experiments" in r.text
