"""LangGraph multi-node workflow for the research assistant."""

import os
from typing import Literal
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain.prompts import ChatPromptTemplate

from src.graph_state import GraphState, create_initial_state, add_trace_entry
from src.llm_client import LLMClient
from src.vector_store import VectorStore
from src.tools import TOOL_MAP, list_tools

load_dotenv()


class ResearchAssistantGraph:
    """Multi-agent research assistant using LangGraph."""

    def __init__(
        self,
        vector_store: VectorStore,
        router_model: str = "gpt-4o-mini",
        synthesis_model: str = "gpt-4o",
        router_provider: str = "openai",
        synthesis_provider: str = "openai",
    ):
        self.vector_store = vector_store
        self.router_model = router_model
        self.synthesis_model = synthesis_model
        self.router_provider = router_provider
        self.synthesis_provider = synthesis_provider

        # Initialize LLM clients
        self.router_llm = LLMClient(provider=router_provider, model=router_model)
        self.synthesis_llm = LLMClient(provider=synthesis_provider, model=synthesis_model)

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(GraphState)

        # Add nodes
        workflow.add_node("router", self.router_node)
        workflow.add_node("retrieval", self.retrieval_node)
        workflow.add_node("tool_calling", self.tool_calling_node)
        workflow.add_node("synthesis", self.synthesis_node)

        # Set entry point
        workflow.set_entry_point("router")

        # Add conditional edges from router
        workflow.add_conditional_edges(
            "router",
            self.route_decision,
            {
                "retrieval_only": "retrieval",
                "tool_only": "tool_calling",
                "both": "tool_calling",  # Do tool first, then retrieval
                "uncertain": "retrieval",  # Default to retrieval if uncertain
            },
        )

        # From tool_calling, if route was "both", go to retrieval, else go to synthesis
        workflow.add_conditional_edges(
            "tool_calling",
            self.after_tool_decision,
            {
                "both": "retrieval",
                "tool_only": "synthesis",
            },
        )

        # From retrieval, go to synthesis
        workflow.add_edge("retrieval", "synthesis")

        # From synthesis, end
        workflow.add_edge("synthesis", END)

        return workflow.compile()

    def router_node(self, state: GraphState) -> GraphState:
        """
        Router node: Decide whether the query needs retrieval, tools, or both.
        """
        state["status"] = "running"
        
        query = state["query"]
        available_tools = ", ".join(list_tools())
        
        router_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a router for a research assistant. Your job is to classify queries into one of these categories:

1. "retrieval_only" - The query can be answered using the document corpus (e.g., factual questions about the documents)
2. "tool_only" - The query requires external tools (e.g., calculations, current date, web search)
3. "both" - The query needs both document retrieval AND external tools
4. "uncertain" - You cannot confidently classify the query

Available tools: {tools}

Respond with ONLY the category name and a confidence score (0.0 to 1.0) in this format:
CATEGORY: [category_name]
CONFIDENCE: [confidence_score]

Examples:
- "What is machine learning?" -> CATEGORY: retrieval_only, CONFIDENCE: 0.95
- "Calculate 25 * 17" -> CATEGORY: tool_only, CONFIDENCE: 0.99
- "What's the current stock price of Apple?" -> CATEGORY: tool_only, CONFIDENCE: 0.90
- "Compare the financial ratios mentioned in the documents with industry averages" -> CATEGORY: both, CONFIDENCE: 0.85
- "I'm feeling lucky" -> CATEGORY: uncertain, CONFIDENCE: 0.30"""),
            ("user", "Query: {query}")
        ])
        
        prompt = router_prompt.format(query=query, tools=available_tools)
        response, token_usage = self.router_llm.invoke(prompt)
        
        # Parse the response
        category = "uncertain"
        confidence = 0.0
        
        for line in response.split("\n"):
            if line.startswith("CATEGORY:"):
                category = line.split(":", 1)[1].strip().lower()
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                except ValueError:
                    confidence = 0.0
        
        # Validate category
        valid_categories = ["retrieval_only", "tool_only", "both", "uncertain"]
        if category not in valid_categories:
            category = "uncertain"
            confidence = 0.0
        
        state["route_decision"] = category
        state["route_confidence"] = confidence
        
        # Add to trace
        state = add_trace_entry(
            state,
            node_name="router",
            input_summary=f"Query: {query[:100]}...",
            output_summary=f"Route: {category}, Confidence: {confidence}",
            token_usage=token_usage,
            confidence=confidence,
        )
        
        return state

    def route_decision(self, state: GraphState) -> str:
        """Determine the next node based on router decision."""
        return state["route_decision"]

    def retrieval_node(self, state: GraphState) -> GraphState:
        """
        Retrieval node: Search the vector store for relevant documents.
        """
        query = state["query"]
        
        # Perform similarity search
        results = self.vector_store.similarity_search(query, k=4)
        
        state["retrieved_docs"] = results
        
        # Estimate token usage (no LLM call here)
        token_usage = {
            "prompt_tokens": len(query.split()),
            "completion_tokens": sum(len(doc["content"].split()) for doc in results),
            "total_tokens": len(query.split()) + sum(len(doc["content"].split()) for doc in results),
        }
        
        state["retrieval_token_usage"] = token_usage
        
        # Add to trace
        state = add_trace_entry(
            state,
            node_name="retrieval",
            input_summary=f"Query: {query[:100]}...",
            output_summary=f"Retrieved {len(results)} documents",
            token_usage=token_usage,
        )
        
        return state

    def tool_calling_node(self, state: GraphState) -> GraphState:
        """
        Tool-calling node: Use LLM to intelligently select and execute the appropriate tool.
        """
        query = state["query"]
        available_tools = list_tools()
        
        # Use LLM to select the tool and extract input
        tool_selection_llm = LLMClient(
            provider=self.router_provider,
            model=self.router_model,
        )
        
        # Build tool descriptions for the LLM
        tool_descriptions = """
Available tools:
1. calculator - Evaluate mathematical expressions (e.g., "25 * 17", "3.14 * 2^2")
2. web_search - Search the web for current information (e.g., "Apple stock price", "latest AI news")
3. get_current_date - Get the current date and time
4. code_interpreter - Execute Python code safely (e.g., statistical calculations, data processing)
"""
        
        tool_selection_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a tool selection agent. Your job is to:
1. Select the most appropriate tool for the user's query
2. Extract the exact input needed for that tool

{tool_descriptions}

Respond in this format:
TOOL: [tool_name]
INPUT: [exact_input_for_tool]

Examples:
- "Calculate 25 * 17" -> TOOL: calculator, INPUT: 25 * 17
- "What's the current stock price of Apple?" -> TOOL: web_search, INPUT: Apple stock price
- "What time is it?" -> TOOL: get_current_date, INPUT: (empty string)
- "Calculate the standard deviation of [1,2,3,4,5]" -> TOOL: code_interpreter, INPUT: import statistics; print(statistics.stdev([1,2,3,4,5]))

Respond ONLY with the tool name and input, nothing else."""),
            ("user", "Query: {query}")
        ])
        
        try:
            prompt = tool_selection_prompt.format(
                tool_descriptions=tool_descriptions,
                query=query
            )
            response, selection_token_usage = tool_selection_llm.invoke(prompt)
            
            # Parse the response
            tool_name = None
            tool_input = query  # Default to full query if parsing fails
            
            for line in response.split("\n"):
                if line.startswith("TOOL:"):
                    tool_name = line.split(":", 1)[1].strip().lower()
                elif line.startswith("INPUT:"):
                    tool_input = line.split(":", 1)[1].strip()
            
            # Validate tool name
            if tool_name not in available_tools:
                tool_name = "web_search"  # Fallback to web search
                tool_input = query
            
        except Exception as e:
            # Fallback to web search if LLM selection fails
            tool_name = "web_search"
            tool_input = query
            selection_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        # Execute the tool
        tool = TOOL_MAP.get(tool_name)
        tool_output = None
        tool_error = None
        
        try:
            if tool:
                result = tool.invoke(tool_input)
                tool_output = str(result)
            else:
                tool_error = f"Tool '{tool_name}' not found"
        except Exception as e:
            tool_error = f"Tool execution failed: {str(e)}"
        
        state["tool_name"] = tool_name
        state["tool_input"] = tool_input
        state["tool_output"] = tool_output
        state["tool_error"] = tool_error
        
        # Calculate token usage (including tool selection)
        token_usage = {
            "prompt_tokens": selection_token_usage.get("prompt_tokens", 0) + len(tool_input.split()),
            "completion_tokens": selection_token_usage.get("completion_tokens", 0) + (len(tool_output.split()) if tool_output else 0),
            "total_tokens": selection_token_usage.get("total_tokens", 0) + len(tool_input.split()) + (len(tool_output.split()) if tool_output else 0),
        }
        
        state["tool_token_usage"] = token_usage
        
        # Add to trace
        state = add_trace_entry(
            state,
            node_name="tool_calling",
            input_summary=f"LLM-selected tool: {tool_name}, Input: {tool_input[:100]}...",
            output_summary=f"Output: {tool_output[:100] if tool_output else tool_error[:100]}...",
            token_usage=token_usage,
            error=tool_error,
        )
        
        return state

    def after_tool_decision(self, state: GraphState) -> str:
        """Determine next node after tool calling."""
        if state["route_decision"] == "both":
            return "both"
        return "tool_only"

    def synthesis_node(self, state: GraphState) -> GraphState:
        """
        Synthesis node: Combine retrieved context and tool results into a final answer.
        """
        query = state["query"]
        
        # Build context from retrieval and tool results
        context_parts = []
        
        if state.get("retrieved_docs"):
            context_parts.append("Retrieved Documents:")
            for i, doc in enumerate(state["retrieved_docs"], 1):
                context_parts.append(f"Document {i}: {doc['content']}")
        
        if state.get("tool_output"):
            context_parts.append(f"\nTool Result ({state['tool_name']}):")
            context_parts.append(state["tool_output"])
        
        if state.get("tool_error"):
            context_parts.append(f"\nTool Error:")
            context_parts.append(state["tool_error"])
        
        context = "\n".join(context_parts)
        
        # Build synthesis prompt
        if state["route_decision"] == "retrieval_only":
            synthesis_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful assistant. Answer the user's question based ONLY on the retrieved documents. If the documents don't contain the answer, say so clearly."""),
                ("user", "Question: {query}\n\nRetrieved Documents:\n{context}")
            ])
        elif state["route_decision"] == "tool_only":
            synthesis_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful assistant. Answer the user's question based on the tool results. If the tool failed, explain what happened and provide a helpful response."""),
                ("user", "Question: {query}\n\nTool Results:\n{context}")
            ])
        else:  # both or uncertain
            synthesis_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful assistant. Answer the user's question using both the retrieved documents and tool results. Synthesize information from both sources. If a tool failed, note this but still use the document information if available."""),
                ("user", "Question: {query}\n\nContext:\n{context}")
            ])
        
        prompt = synthesis_prompt.format(query=query, context=context)
        response, token_usage = self.synthesis_llm.invoke(prompt)
        
        state["answer"] = response
        state["synthesis_token_usage"] = token_usage
        state["status"] = "completed"
        state["end_time"] = state["execution_trace"][-1]["timestamp"] if state["execution_trace"] else None
        
        # Add to trace
        state = add_trace_entry(
            state,
            node_name="synthesis",
            input_summary=f"Context from {len(state['retrieved_docs']) if state.get('retrieved_docs') else 0} docs + tool results",
            output_summary=f"Answer: {response[:100]}...",
            token_usage=token_usage,
        )
        
        return state

    def invoke(self, query: str, show_reasoning_trace: bool = False) -> GraphState:
        """
        Invoke the graph with a query.
        
        Args:
            query: The user's question
            show_reasoning_trace: Whether to include the reasoning trace in the output
        
        Returns:
            The final state with the answer and execution trace
        """
        initial_state = create_initial_state(query, show_reasoning_trace)
        final_state = self.graph.invoke(initial_state)
        return final_state
