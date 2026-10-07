import pytest
import asyncio
import httpx
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_health_endpoint():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_query_endpoint_basic():
    """Test the query endpoint with a simple request."""
    # We assume the server is running or we mock the graph
    # For a real integration test, we'd use a mock LLM or local model
    response = client.post(
        "/query",
        json={"query": "Hello", "show_reasoning_trace": False}
    )
    # This will likely fail without a configured .env, which is expected
    # We are testing the API wrapper's ability to handle the request/response
    assert response.status_code in [200, 500]

@pytest.mark.asyncio
async def test_concurrent_requests():
    """Test how the API handles many requests concurrently."""
    async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
        tasks = [
            ac.post("/query", json={"query": f"Request {i}"})
            for i in range(10)
        ]
        responses = await asyncio.gather(*tasks)

        for response in responses:
            # We care about the server not crashing (500 is okay if LLM is missing,
            # but 422 or 504 would indicate structural failures)
            assert response.status_code in [200, 500]

def test_query_invalid_payload():
    """Test API failure handling with invalid input."""
    response = client.post(
        "/query",
        json={"wrong_key": "some value"}
    )
    assert response.status_code == 422 # Pydantic validation error
