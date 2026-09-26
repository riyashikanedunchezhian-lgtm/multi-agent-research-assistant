"""Tests for the LangGraph workflow."""

import pytest
import os
from unittest.mock import Mock, patch

from src.graph import ResearchAssistantGraph
from src.vector_store import VectorStore
from src.graph_state import create_initial_state


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store for testing."""
    store = Mock(spec=VectorStore)
    store.similarity_search.return_value = [
        {"content": "Machine learning is a subset of AI", "metadata": {"source": "doc_0"}},
        {"content": "Deep learning uses neural networks", "metadata": {"source": "doc_1"}},
    ]
    return store


@pytest.fixture
def graph(mock_vector_store):
    """Create a ResearchAssistantGraph instance for testing."""
    with patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-key",
        "ROUTER_MODEL": "gpt-4o-mini",
        "SYNTHESIS_MODEL": "gpt-4o",
    }):
        return ResearchAssistantGraph(
            vector_store=mock_vector_store,
            router_model="gpt-4o-mini",
            synthesis_model="gpt-4o",
        )


class TestRouterNode:
    """Tests for the router node."""

    @patch("src.graph.LLMClient")
    def test_router_classifies_retrieval_only(self, mock_llm_client, graph):
        """Test that router correctly classifies retrieval-only queries."""
        mock_llm_instance = Mock()
        mock_llm_instance.invoke.return_value = (
            "CATEGORY: retrieval_only\nCONFIDENCE: 0.95",
            {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}
        )
        mock_llm_client.return_value = mock_llm_instance
        
        state = create_initial_state("What is machine learning?")
        result = graph.router_node(state)
        
        assert result["route_decision"] == "retrieval_only"
        assert result["route_confidence"] == 0.95
        assert len(result["execution_trace"]) == 1
        assert result["execution_trace"][0]["node_name"] == "router"

    @patch("src.graph.LLMClient")
    def test_router_classifies_tool_only(self, mock_llm_client, graph):
        """Test that router correctly classifies tool-only queries."""
        mock_llm_instance = Mock()
        mock_llm_instance.invoke.return_value = (
            "CATEGORY: tool_only\nCONFIDENCE: 0.99",
            {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}
        )
        mock_llm_client.return_value = mock_llm_instance
        
        state = create_initial_state("Calculate 25 * 17")
        result = graph.router_node(state)
        
        assert result["route_decision"] == "tool_only"
        assert result["route_confidence"] == 0.99

    @patch("src.graph.LLMClient")
    def test_router_classifies_both(self, mock_llm_client, graph):
        """Test that router correctly classifies queries needing both."""
        mock_llm_instance = Mock()
        mock_llm_instance.invoke.return_value = (
            "CATEGORY: both\nCONFIDENCE: 0.85",
            {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}
        )
        mock_llm_client.return_value = mock_llm_instance
        
        state = create_initial_state("Compare the financial ratios with current market data")
        result = graph.router_node(state)
        
        assert result["route_decision"] == "both"
        assert result["route_confidence"] == 0.85

    @patch("src.graph.LLMClient")
    def test_router_handles_uncertainty(self, mock_llm_client, graph):
        """Test that router handles uncertain queries gracefully."""
        mock_llm_instance = Mock()
        mock_llm_instance.invoke.return_value = (
            "CATEGORY: uncertain\nCONFIDENCE: 0.30",
            {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}
        )
        mock_llm_client.return_value = mock_llm_instance
        
        state = create_initial_state("I'm feeling lucky")
        result = graph.router_node(state)
        
        assert result["route_decision"] == "uncertain"
        assert result["route_confidence"] == 0.30


class TestRetrievalNode:
    """Tests for the retrieval node."""

    def test_retrieval_fetches_documents(self, graph, mock_vector_store):
        """Test that retrieval node fetches documents from vector store."""
        state = create_initial_state("What is machine learning?")
        state["route_decision"] = "retrieval_only"
        
        result = graph.retrieval_node(state)
        
        assert result["retrieved_docs"] is not None
        assert len(result["retrieved_docs"]) == 2
        assert mock_vector_store.similarity_search.called
        assert len(result["execution_trace"]) == 1
        assert result["execution_trace"][0]["node_name"] == "retrieval"


class TestToolCallingNode:
    """Tests for the tool-calling node."""

    @patch("src.graph.LLMClient")
    @patch("src.graph.TOOL_MAP")
    def test_tool_calling_calculator(self, mock_tool_map, mock_llm_client, graph):
        """Test that LLM-based tool selection chooses calculator for math queries."""
        # Mock LLM tool selection
        mock_selection_llm = Mock()
        mock_selection_llm.invoke.return_value = (
            "TOOL: calculator\nINPUT: 25 * 17",
            {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}
        )
        mock_llm_client.return_value = mock_selection_llm
        
        # Mock calculator tool
        mock_tool = Mock()
        mock_tool.invoke.return_value = "425"
        mock_tool_map.__getitem__.return_value = mock_tool
        
        state = create_initial_state("Calculate 25 * 17")
        result = graph.tool_calling_node(state)
        
        assert result["tool_name"] == "calculator"
        assert result["tool_input"] == "25 * 17"
        assert result["tool_output"] == "425"
        assert result["tool_error"] is None

    @patch("src.graph.LLMClient")
    @patch("src.graph.TOOL_MAP")
    def test_tool_calling_web_search(self, mock_tool_map, mock_llm_client, graph):
        """Test that LLM-based tool selection chooses web search."""
        # Mock LLM tool selection
        mock_selection_llm = Mock()
        mock_selection_llm.invoke.return_value = (
            "TOOL: web_search\nINPUT: Apple stock price",
            {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}
        )
        mock_llm_client.return_value = mock_selection_llm
        
        # Mock web search tool
        mock_tool = Mock()
        mock_tool.invoke.return_value = "Search results for Apple stock price"
        mock_tool_map.__getitem__.return_value = mock_tool
        
        state = create_initial_state("Search for Apple stock price")
        result = graph.tool_calling_node(state)
        
        assert result["tool_name"] == "web_search"
        assert result["tool_input"] == "Apple stock price"
        assert result["tool_output"] == "Search results for Apple stock price"

    @patch("src.graph.LLMClient")
    @patch("src.graph.TOOL_MAP")
    def test_tool_calling_handles_failure(self, mock_tool_map, mock_llm_client, graph):
        """Test that tool-calling node handles tool failures gracefully."""
        # Mock LLM tool selection
        mock_selection_llm = Mock()
        mock_selection_llm.invoke.return_value = (
            "TOOL: calculator\nINPUT: 25 * 17",
            {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}
        )
        mock_llm_client.return_value = mock_selection_llm
        
        # Mock tool that fails
        mock_tool = Mock()
        mock_tool.invoke.side_effect = Exception("Tool failed")
        mock_tool_map.__getitem__.return_value = mock_tool
        
        state = create_initial_state("Calculate something")
        result = graph.tool_calling_node(state)
        
        assert result["tool_error"] is not None
        assert "Tool execution failed" in result["tool_error"]
        assert result["tool_output"] is None

    @patch("src.graph.LLMClient")
    @patch("src.graph.TOOL_MAP")
    def test_tool_calling_llm_selection_fallback(self, mock_tool_map, mock_llm_client, graph):
        """Test that tool-calling falls back to web search if LLM selection fails."""
        # Mock LLM that fails
        mock_selection_llm = Mock()
        mock_selection_llm.invoke.side_effect = Exception("LLM failed")
        mock_llm_client.return_value = mock_selection_llm
        
        # Mock web search tool (fallback)
        mock_tool = Mock()
        mock_tool.invoke.return_value = "Search results"
        mock_tool_map.__getitem__.return_value = mock_tool
        
        state = create_initial_state("Some query")
        result = graph.tool_calling_node(state)
        
        # Should fall back to web search
        assert result["tool_name"] == "web_search"
        assert result["tool_input"] == "Some query"


class TestSynthesisNode:
    """Tests for the synthesis node."""

    @patch("src.graph.LLMClient")
    def test_synthesis_combines_retrieval_only(self, mock_llm_client, graph):
        """Test synthesis with retrieval-only context."""
        mock_llm_instance = Mock()
        mock_llm_instance.invoke.return_value = (
            "Machine learning is a subset of artificial intelligence...",
            {"prompt_tokens": 200, "completion_tokens": 50, "total_tokens": 250}
        )
        mock_llm_client.return_value = mock_llm_instance
        
        state = create_initial_state("What is machine learning?")
        state["route_decision"] = "retrieval_only"
        state["retrieved_docs"] = [
            {"content": "Machine learning is a subset of AI", "metadata": {"source": "doc_0"}},
        ]
        
        result = graph.synthesis_node(state)
        
        assert result["answer"] is not None
        assert result["status"] == "completed"
        assert result["synthesis_token_usage"] is not None

    @patch("src.graph.LLMClient")
    def test_synthesis_handles_tool_error(self, mock_llm_client, graph):
        """Test synthesis degrades gracefully when tool fails."""
        mock_llm_instance = Mock()
        mock_llm_instance.invoke.return_value = (
            "Based on the documents, here's what I found...",
            {"prompt_tokens": 200, "completion_tokens": 50, "total_tokens": 250}
        )
        mock_llm_client.return_value = mock_llm_instance
        
        state = create_initial_state("Calculate and explain")
        state["route_decision"] = "both"
        state["retrieved_docs"] = [
            {"content": "Some document content", "metadata": {"source": "doc_0"}},
        ]
        state["tool_error"] = "Tool execution failed"
        
        result = graph.synthesis_node(state)
        
        assert result["answer"] is not None
        assert result["status"] == "completed"


class TestEfficiency:
    """Tests for token efficiency."""

    @patch("src.graph.LLMClient")
    def test_rag_only_no_tool_call(self, mock_llm_client, graph, mock_vector_store):
        """Test that RAG-only queries don't trigger tool calls (efficiency check)."""
        # Mock router to return retrieval_only
        mock_router_llm = Mock()
        mock_router_llm.invoke.return_value = (
            "CATEGORY: retrieval_only\nCONFIDENCE: 0.95",
            {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}
        )
        
        # Mock synthesis
        mock_synthesis_llm = Mock()
        mock_synthesis_llm.invoke.return_value = (
            "Answer based on documents",
            {"prompt_tokens": 200, "completion_tokens": 50, "total_tokens": 250}
        )
        
        mock_llm_client.side_effect = [mock_router_llm, mock_synthesis_llm]
        
        result = graph.invoke("What is machine learning?", show_reasoning_trace=True)
        
        # Check that tool node was not called
        node_names = [entry["node_name"] for entry in result["execution_trace"]]
        assert "tool_calling" not in node_names
        assert "router" in node_names
        assert "retrieval" in node_names
        assert "synthesis" in node_names

    @patch("src.graph.LLMClient")
    @patch("src.graph.TOOL_MAP")
    def test_tool_query_invokes_tool(self, mock_tool_map, mock_llm_client, graph):
        """Test that tool-needed queries actually invoke the tool."""
        # Mock router to return tool_only
        mock_router_llm = Mock()
        mock_router_llm.invoke.return_value = (
            "CATEGORY: tool_only\nCONFIDENCE: 0.99",
            {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}
        )
        
        # Mock tool selection LLM
        mock_tool_selection_llm = Mock()
        mock_tool_selection_llm.invoke.return_value = (
            "TOOL: calculator\nINPUT: 25 * 17",
            {"prompt_tokens": 30, "completion_tokens": 8, "total_tokens": 38}
        )
        
        # Mock synthesis
        mock_synthesis_llm = Mock()
        mock_synthesis_llm.invoke.return_value = (
            "The result is 425",
            {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
        )
        
        mock_llm_client.side_effect = [mock_router_llm, mock_tool_selection_llm, mock_synthesis_llm]
        
        # Mock tool
        mock_tool = Mock()
        mock_tool.invoke.return_value = "425"
        mock_tool_map.__getitem__.return_value = mock_tool
        
        result = graph.invoke("Calculate 25 * 17", show_reasoning_trace=True)
        
        # Check that tool was called
        node_names = [entry["node_name"] for entry in result["execution_trace"]]
        assert "tool_calling" in node_names
        assert result["tool_output"] == "425"
        assert "425" in result["answer"]


class TestTraceAccuracy:
    """Tests for reasoning trace accuracy."""

    @patch("src.graph.LLMClient")
    def test_trace_reflects_actual_nodes(self, mock_llm_client, graph, mock_vector_store):
        """Test that reasoning trace accurately reflects nodes that fired."""
        mock_router_llm = Mock()
        mock_router_llm.invoke.return_value = (
            "CATEGORY: retrieval_only\nCONFIDENCE: 0.95",
            {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}
        )
        
        mock_synthesis_llm = Mock()
        mock_synthesis_llm.invoke.return_value = (
            "Answer",
            {"prompt_tokens": 200, "completion_tokens": 50, "total_tokens": 250}
        )
        
        mock_llm_client.side_effect = [mock_router_llm, mock_synthesis_llm]
        
        result = graph.invoke("What is machine learning?", show_reasoning_trace=True)
        
        # Verify trace structure
        assert len(result["execution_trace"]) == 3  # router, retrieval, synthesis
        assert result["execution_trace"][0]["node_name"] == "router"
        assert result["execution_trace"][1]["node_name"] == "retrieval"
        assert result["execution_trace"][2]["node_name"] == "synthesis"
        
        # Verify each entry has required fields
        for entry in result["execution_trace"]:
            assert "node_name" in entry
            assert "timestamp" in entry
            assert "input_summary" in entry
            assert "output_summary" in entry
            assert "token_usage" in entry

    @patch("src.graph.LLMClient")
    def test_token_usage_per_node(self, mock_llm_client, graph, mock_vector_store):
        """Test that token usage is tracked per node."""
        mock_router_llm = Mock()
        mock_router_llm.invoke.return_value = (
            "CATEGORY: retrieval_only\nCONFIDENCE: 0.95",
            {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}
        )
        
        mock_synthesis_llm = Mock()
        mock_synthesis_llm.invoke.return_value = (
            "Answer",
            {"prompt_tokens": 200, "completion_tokens": 50, "total_tokens": 250}
        )
        
        mock_llm_client.side_effect = [mock_router_llm, mock_synthesis_llm]
        
        result = graph.invoke("What is machine learning?", show_reasoning_trace=True)
        
        # Check per-node token usage
        router_tokens = result["execution_trace"][0]["token_usage"]
        assert router_tokens["total_tokens"] == 60
        
        synthesis_tokens = result["execution_trace"][2]["token_usage"]
        assert synthesis_tokens["total_tokens"] == 250
        
        # Check total
        assert result["total_token_usage"]["total_tokens"] > 0
