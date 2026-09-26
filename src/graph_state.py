"""State definition for the LangGraph workflow."""

from typing import TypedDict, List, Optional, Literal
from datetime import datetime


class NodeExecution(TypedDict):
    """Track execution of a single node."""
    node_name: str
    timestamp: str
    input_summary: str
    output_summary: str
    token_usage: dict
    confidence: Optional[float]
    error: Optional[str]


class GraphState(TypedDict):
    """State for the multi-agent research assistant graph."""
    
    # Input
    query: str
    show_reasoning_trace: bool
    
    # Routing decision
    route_decision: Optional[Literal["retrieval_only", "tool_only", "both", "uncertain"]]
    route_confidence: Optional[float]
    
    # Retrieval results
    retrieved_docs: Optional[List[dict]]
    retrieval_token_usage: Optional[dict]
    
    # Tool results
    tool_name: Optional[str]
    tool_input: Optional[str]
    tool_output: Optional[str]
    tool_token_usage: Optional[dict]
    tool_error: Optional[str]
    
    # Final answer
    answer: Optional[str]
    synthesis_token_usage: Optional[dict]
    
    # Reasoning trace
    execution_trace: List[NodeExecution]
    
    # Metadata
    total_token_usage: dict
    start_time: str
    end_time: Optional[str]
    status: Literal["pending", "running", "completed", "failed"]


def create_initial_state(query: str, show_reasoning_trace: bool = False) -> GraphState:
    """Create initial state for a new query."""
    return GraphState(
        query=query,
        show_reasoning_trace=show_reasoning_trace,
        route_decision=None,
        route_confidence=None,
        retrieved_docs=None,
        retrieval_token_usage=None,
        tool_name=None,
        tool_input=None,
        tool_output=None,
        tool_token_usage=None,
        tool_error=None,
        answer=None,
        synthesis_token_usage=None,
        execution_trace=[],
        total_token_usage={
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        start_time=datetime.now().isoformat(),
        end_time=None,
        status="pending",
    )


def add_trace_entry(
    state: GraphState,
    node_name: str,
    input_summary: str,
    output_summary: str,
    token_usage: dict,
    confidence: Optional[float] = None,
    error: Optional[str] = None,
) -> GraphState:
    """Add an entry to the execution trace."""
    entry = NodeExecution(
        node_name=node_name,
        timestamp=datetime.now().isoformat(),
        input_summary=input_summary,
        output_summary=output_summary,
        token_usage=token_usage,
        confidence=confidence,
        error=error,
    )
    
    state["execution_trace"].append(entry)
    
    # Update total token usage
    if token_usage:
        state["total_token_usage"]["prompt_tokens"] += token_usage.get("prompt_tokens", 0)
        state["total_token_usage"]["completion_tokens"] += token_usage.get("completion_tokens", 0)
        state["total_token_usage"]["total_tokens"] += token_usage.get("total_tokens", 0)
    
    return state
