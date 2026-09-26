# Multi-Agent Research Assistant

A sophisticated research assistant built with LangGraph that combines RAG (Retrieval-Augmented Generation) with agentic tool-calling in an explicit multi-node graph architecture. This project demonstrates advanced AI engineering concepts including intelligent routing, state management, token efficiency, and graceful degradation.

## 🏗️ Architecture

### Graph Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     Research Assistant Graph                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   Router Node   │
                    │  (GPT-4o-mini)  │
                    │                 │
                    │ - Classify query│
                    │ - Route decision│
                    │ - Confidence   │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
           ▼                 ▼                 ▼
   ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
   │  Retrieval    │  │ Tool-Calling  │  │  Uncertain    │
   │     Node      │  │     Node      │  │   (fallback)  │
   │               │  │               │  │               │
   │ - ChromaDB    │  │ - LLM Tool    │  │ - Default to  │
   │ - Similarity  │  │   Selection   │  │   retrieval   │
   │   Search      │  │ - Calculator  │  │               │
   │               │  │ - Web Search  │  │               │
   │               │  │ - Code Exec   │  │               │
   └───────┬───────┘  └───────┬───────┘  └───────┬───────┘
           │                  │                  │
           │                  │                  │
           └──────────────────┼──────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Synthesis Node │
                    │   (GPT-4o)      │
                    │                 │
                    │ - Combine       │
                    │   context      │
                    │ - Generate     │
                    │   answer        │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Final Answer  │
                    │ + Reasoning     │
                    │   Trace         │
                    └─────────────────┘
```

### Node Descriptions

1. **Router Node** (GPT-4o-mini)
   - Classifies queries into: `retrieval_only`, `tool_only`, `both`, or `uncertain`
   - Provides confidence score for each classification
   - Uses a cheaper/faster model for efficiency
   - Falls back to retrieval if uncertain

2. **Retrieval Node**
   - Searches ChromaDB vector store for relevant documents
   - Uses real embeddings (sentence-transformers/all-MiniLM-L6-v2)
   - Returns top-k most similar documents
   - No LLM call (efficient)

3. **Tool-Calling Node**
   - Uses LLM-based intelligent tool selection (not just keyword matching)
   - Analyzes query to choose the most appropriate tool
   - Extracts precise input parameters for the selected tool
   - Available tools: Calculator, Web Search, Current Date, Code Interpreter
   - Handles tool failures gracefully with error messages
   - Continues to synthesis even if tool fails

4. **Synthesis Node** (GPT-4o)
   - Combines retrieved documents and tool results
   - Uses a more powerful model for high-quality answers
   - Only invoked when needed (efficiency optimization)
   - Synthesizes information from multiple sources

## 🚀 Why LangGraph?

LangGraph was chosen over simpler LangChain chains for several critical reasons:

### 1. **Conditional Branching**
- **Problem**: Chains execute linearly; they cannot conditionally route to different paths based on query analysis
- **Solution**: LangGraph's conditional edges allow the router to dynamically route queries to retrieval, tools, or both based on intelligent classification

### 2. **State Management**
- **Problem**: Chains have limited state visibility between steps
- **Solution**: LangGraph maintains explicit, typed state throughout the workflow, enabling comprehensive reasoning traces and per-node token tracking

### 3. **Complex Workflows**
- **Problem**: Implementing "try tool, fallback to retrieval if fails" logic in chains is cumbersome
- **Solution**: LangGraph's graph structure naturally supports complex workflows with fallbacks, loops, and parallel execution

### 4. **Observability**
- **Problem**: Chains provide limited insight into decision-making
- **Solution**: LangGraph's state graph makes every node's input, output, and decision explicitly visible in the reasoning trace

## 💰 Token Efficiency Strategy

### Model Selection Strategy

The project implements a concrete efficiency improvement by using different models for different purposes:

| Node | Model | Reason | Cost per 1K tokens |
|------|-------|--------|-------------------|
| Router | GPT-4o-mini | Simple classification task | $0.00015 |
| Tool Selection | GPT-4o-mini | Intelligent tool selection | $0.00015 |
| Synthesis | GPT-4o | Complex reasoning required | $0.005 |

### Efficiency Impact

Based on test runs with sample queries:

**RAG-only query**: "What is machine learning?"
- Router: ~60 tokens (GPT-4o-mini) = $0.000009
- Synthesis: ~250 tokens (GPT-4o) = $0.00125
- **Total**: $0.001259

**If we used GPT-4o for routing**: ~60 tokens = $0.00030
- **Savings per query**: $0.000291 (23% reduction in routing cost)

**Tool-calling query**: "Calculate 25 * 17"
- Router: ~60 tokens (GPT-4o-mini) = $0.000009
- Tool Selection: ~38 tokens (GPT-4o-mini) = $0.000006
- Synthesis: ~120 tokens (GPT-4o) = $0.00060
- **Total**: $0.000615

**If we used GPT-4o for routing + tool selection**: ~98 tokens = $0.00049
- **Savings per query**: $0.000375 (38% reduction in classification cost)

### Cumulative Impact

For 1,000 queries per day (50% RAG-only, 50% tool-calling):
- **Savings**: ~$0.33 per day, ~$9.90 per month, ~$120 per year
- **Scalability**: Savings grow linearly with query volume

## 🛡️ Guardrails

### Router Uncertainty
- If confidence < 0.5, router returns "uncertain"
- Uncertain queries default to retrieval (safe fallback)
- Prevents the system from making poor routing decisions

### Tool Failure Graceful Degradation
- Tool failures are caught and logged
- Error messages are passed to synthesis node
- Synthesis can still provide useful answers from documents
- System never crashes due to tool failures

Example degradation path:
```
Query: "Calculate and explain financial ratios"
→ Router: "both" (confidence: 0.85)
→ Tool: Calculator fails (network error)
→ Retrieval: Fetches financial ratio documents
→ Synthesis: "The calculator tool encountered an error, but based on the documents..."
```

## 📦 Installation

### Prerequisites
- Python 3.10 or higher
- OpenAI API key (or Anthropic API key)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd multiagent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. Ingest sample documents:
```bash
python -m src.ingestion
```

## 🏃 Usage

### Start the API Server

```bash
python -m src.api
```

The server will start on `http://localhost:8000`

### API Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Ingest Documents
```bash
curl -X POST http://localhost:8000/ingest?clear_existing=false
```

#### Query (Simple)
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?"}'
```

#### Query (With Reasoning Trace)
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Calculate 25 * 17", "show_reasoning_trace": true}'
```

### Response Format

```json
{
  "answer": "The result is 425",
  "reasoning_trace": [
    {
      "node_name": "router",
      "timestamp": "2024-01-01T00:00:00",
      "input_summary": "Query: Calculate 25 * 17",
      "output_summary": "Route: tool_only, Confidence: 0.99",
      "token_usage": {
        "prompt_tokens": 50,
        "completion_tokens": 10,
        "total_tokens": 60
      },
      "confidence": 0.99
    },
    {
      "node_name": "tool_calling",
      "timestamp": "2024-01-01T00:00:01",
      "input_summary": "LLM-selected tool: calculator, Input: 25 * 17",
      "output_summary": "Output: 425",
      "token_usage": {
        "prompt_tokens": 38,
        "completion_tokens": 3,
        "total_tokens": 41
      }
    },
    {
      "node_name": "synthesis",
      "timestamp": "2024-01-01T00:00:02",
      "input_summary": "Context from 0 docs + tool results",
      "output_summary": "Answer: The result is 425",
      "token_usage": {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120
      }
    }
  ],
  "token_usage": {
    "prompt_tokens": 188,
    "completion_tokens": 33,
    "total_tokens": 221
  },
  "execution_time": "2024-01-01T00:00:02",
  "status": "completed"
}
```

## 🧪 Testing

Run the test suite:

```bash
pytest
```

Run specific test categories:

```bash
# Test graph workflow
pytest tests/test_graph.py

# Test API endpoints
pytest tests/test_api.py

# Test ingestion pipeline
pytest tests/test_ingestion.py
```

### Test Coverage

The test suite covers:

1. **Router Classification**
   - Retrieval-only queries
   - Tool-only queries
   - Both retrieval and tool queries
   - Uncertain queries

2. **LLM-Based Tool Selection**
   - Accurate tool selection for different query types
   - Calculator selection for mathematical queries
   - Web search selection for information queries
   - Graceful fallback when LLM selection fails

3. **Efficiency Checks**
   - RAG-only queries don't trigger tool calls
   - Tool queries actually invoke tools
   - Token usage tracked per node

4. **Graceful Degradation**
   - Tool failures don't crash the system
   - Synthesis continues with partial information
   - LLM selection failures fall back to web search

5. **Trace Accuracy**
   - Reasoning trace reflects actual nodes fired
   - Token usage is accurate per node
   - All required fields present in trace

## 📁 Project Structure

```
multiagent/
├── src/
│   ├── __init__.py
│   ├── api.py              # FastAPI wrapper
│   ├── graph.py            # LangGraph workflow
│   ├── graph_state.py      # State definitions
│   ├── llm_client.py       # LLM client with token tracking
│   ├── tools.py            # Available tools
│   ├── vector_store.py     # ChromaDB wrapper
│   ├── document_processor.py  # Document loading and chunking
│   └── ingestion.py        # Ingestion pipeline
├── tests/
│   ├── __init__.py
│   ├── test_graph.py       # Graph workflow tests
│   ├── test_api.py         # API endpoint tests
│   └── test_ingestion.py   # Ingestion tests
├── data/
│   ├── documents/          # Source documents
│   └── chroma/             # Vector database storage
├── requirements.txt
├── pyproject.toml
├── pytest.ini
├── .env.example
└── README.md
```

## 🔧 Configuration

Key environment variables in `.env`:

```bash
# LLM Configuration
OPENAI_API_KEY=your_key_here
LLM_PROVIDER=openai
ROUTER_MODEL=gpt-4o-mini
SYNTHESIS_MODEL=gpt-4o

# Embedding Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Vector Database
CHROMA_PERSIST_DIR=./data/chroma
CHROMA_COLLECTION_NAME=documents

# Documents
DOCUMENTS_DIR=./data/documents
```

## 🎯 Key Features Demonstrated

### 1. Multi-Node LangGraph Workflow
- Explicit graph structure with conditional routing
- State passed between nodes
- Complex decision logic

### 2. RAG with Real Embeddings
- ChromaDB vector store
- sentence-transformers embeddings
- Document chunking and similarity search

### 3. Agentic Tool-Calling
- **LLM-based intelligent tool selection** (not simple keyword matching)
- Analyzes query to choose the most appropriate tool
- Extracts precise input parameters for the selected tool
- Calculator for mathematical operations
- Web search for current information
- Code interpreter for Python execution
- Current date/time retrieval

### 4. Explicit State and Traceability
- Full reasoning trace available
- Per-node token usage tracking
- Input/output summaries for each node
- Confidence scores for decisions

### 5. Token/Cost Efficiency
- Different models for different tasks
- Per-node token tracking
- Quantified cost savings
- Efficient routing (avoid unnecessary tool calls)

### 6. Guardrails
- Router uncertainty handling
- Tool failure graceful degradation
- Safe fallbacks
- Error propagation without crashes

### 7. API Wrapper
- FastAPI service
- RESTful endpoints
- Production-ready structure
- Health checks

## 🚧 Future Enhancements

- [x] Add more sophisticated tool selection using LLM (completed)
- [ ] Implement parallel execution of retrieval and tool-calling
- [ ] Add streaming responses for long-running queries
- [ ] Implement query caching
- [ ] Add more tools (weather, stock prices, etc.)
- [ ] Support for additional embedding models
- [ ] Add authentication/authorization
- [ ] Implement rate limiting
- [ ] Add monitoring and metrics collection

## 🔬 Recent Improvements

### LLM-Based Tool Selection
**Implemented**: The tool-calling node now uses an LLM to intelligently select which tool to use, rather than simple keyword matching.

**Benefits**:
- More accurate tool selection for complex queries
- Better extraction of tool input parameters
- Handles edge cases that keyword matching misses
- Scalable approach - adding new tools doesn't require updating keyword logic

**Example**:
- Old: Simple keyword matching (e.g., "calculate" → calculator)
- New: LLM understands semantic intent (e.g., "What's 25 times 17?" → calculator with input "25 * 17")

## 📄 License

This project is open source and available for educational and commercial use.

## 👤 Author

Built to demonstrate advanced AI engineering concepts including LangGraph workflows, RAG systems, and agentic tool-calling architectures.
