"""Tests for the FastAPI wrapper."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch


@pytest.fixture
def client():
    """Create a test client for the API."""
    from src.api import app
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_not_ready(self, client):
        """Test health endpoint when not initialized."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["not_ready", "error"]


class TestQueryEndpoint:
    """Tests for the query endpoint."""

    @patch("src.api.graph")
    def test_query_success(self, mock_graph, client):
        """Test successful query."""
        mock_graph.invoke.return_value = {
            "answer": "Test answer",
            "execution_trace": [],
            "total_token_usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            "end_time": "2024-01-01T00:00:00",
            "status": "completed",
        }
        
        response = client.post("/query", json={"query": "Test query"})
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Test answer"
        assert data["status"] == "completed"

    @patch("src.api.graph")
    def test_query_with_trace(self, mock_graph, client):
        """Test query with reasoning trace."""
        mock_graph.invoke.return_value = {
            "answer": "Test answer",
            "execution_trace": [
                {
                    "node_name": "router",
                    "timestamp": "2024-01-01T00:00:00",
                    "input_summary": "Test",
                    "output_summary": "Test",
                    "token_usage": {"total_tokens": 10},
                }
            ],
            "total_token_usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            "end_time": "2024-01-01T00:00:00",
            "status": "completed",
        }
        
        response = client.post(
            "/query",
            json={"query": "Test query", "show_reasoning_trace": True}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["reasoning_trace"] is not None
        assert len(data["reasoning_trace"]) == 1

    @patch("src.api.graph")
    def test_query_graph_not_initialized(self, mock_graph, client):
        """Test query when graph is not initialized."""
        from src.api import graph
        # Temporarily set graph to None
        original_graph = graph
        graph = None
        
        try:
            response = client.post("/query", json={"query": "Test query"})
            assert response.status_code == 503
        finally:
            # Restore graph
            from src.api import graph as api_graph
            api_graph.graph = original_graph


class TestIngestEndpoint:
    """Tests for the ingest endpoint."""

    @patch("src.api.ingestion_pipeline")
    def test_ingest_success(self, mock_pipeline, client):
        """Test successful document ingestion."""
        mock_pipeline.ingest.return_value = {
            "status": "success",
            "documents_loaded": 5,
            "chunks_created": 20,
        }
        
        response = client.post("/ingest", params={"clear_existing": False})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
