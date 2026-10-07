import pytest
from unittest.mock import Mock, patch
from src.llm_client import LLMClient

def test_llm_client_api_failure():
    """Test how LLMClient handles API failures (e.g., timeout or 500)."""
    with patch("langchain_openai.ChatOpenAI.invoke") as mock_invoke:
        mock_invoke.side_effect = Exception("API Connection Error")

        client = LLMClient(provider="openai", model="gpt-4o")

        with pytest.raises(Exception) as excinfo:
            client.invoke("Hello")

        assert "API Connection Error" in str(excinfo.value)

def test_llm_client_invalid_key():
    """Test LLMClient behavior with missing API key."""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError) as excinfo:
            LLMClient(provider="openai")

        assert "OPENAI_API_KEY not found" in str(excinfo.value)
