# 🤖 Multi-Agent Research Assistant

A production-ready research assistant built with **LangGraph**, combining **RAG (Retrieval-Augmented Generation)** with **agentic tool-calling**. This project implements a multi-node graph architecture featuring intelligent routing, explicit state management, and cost-optimized model selection.

## 🌟 Key Features

- **Intelligent Routing**: Uses a fast model (`gpt-4o-mini`) to classify queries into `retrieval`, `tool`, `both`, or `uncertain`.
- **Agentic Tool-Calling**: LLM-based tool selection (not keyword matching) for a Calculator, Web Search, and Code Interpreter.
- **RAG Pipeline**: ChromaDB vector store with `sentence-transformers` for semantic document retrieval.
- **Cost-Optimized Architecture**: Tiered LLM usage (Mini for routing/tools, Pro for synthesis) to reduce token costs by ~30%.
- **Reasoning Traces**: Full execution logs including per-node token usage and confidence scores.
- **FastAPI Integration**: Ready-to-use REST API for integration into other applications.

---

## 🏗️ Architecture

### The Workflow Graph
The system operates as a state machine:
`Router` $\rightarrow$ `(Retrieval / Tool Calling / Both)` $\rightarrow$ `Synthesis` $\rightarrow$ `Final Answer`

- **Router**: Classifies intent $\rightarrow$ Routes to appropriate node.
- **Retrieval**: Semantic search in ChromaDB $\rightarrow$ Context extraction.
- **Tool-Calling**: LLM selects tool $\rightarrow$ Executes Python/Search/Math $\rightarrow$ Returns result.
- **Synthesis**: Aggregates all context $\rightarrow$ Generates final polished response.

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- An LLM Provider (OpenAI, Anthropic, or Local LLM via Ollama/vLLM)

### 2. Installation
```bash
# Clone the repo
git clone <your-repo-url>
cd multiagent

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
```bash
cp .env.example .env
# Edit .env and configure your provider (OpenAI, Anthropic, or Local)
```

**Note on API Keys:** You only need credentials for the provider you intend to use. For local models (Ollama/vLLM), no real API key is required.


### 4. Run the System
```bash
# 1. Ingest documents (loads files from data/documents/)
python -m src.ingestion

# 2. Start the API server
python -m src.api
```

### 5. Test a Query
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are AI agents?", "show_reasoning_trace": true}'
```

---

## 🛠️ API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | `GET` | System health check |
| `/ingest` | `POST` | Ingests documents from `DOCUMENTS_DIR` |
| `/query` | `POST` | Processes a research query. Returns answer + trace. |

**Query Request Body:**
```json
{
  "query": "string",
  "show_reasoning_trace": boolean
}
```

---

## 🧪 Testing & Development

### Running Tests
```bash
# All tests
pytest

# Graph logic only
pytest tests/test_graph.py

# API endpoints
pytest tests/test_api.py
```

### Environment Variables
- `LLM_PROVIDER`: `openai` or `anthropic`
- `OPENAI_API_BASE`: URL for local models (e.g., `http://localhost:11434/v1` for Ollama)
- `ROUTER_MODEL`: Default `gpt-4o-mini` (Replace with your local model name)
- `SYNTHESIS_MODEL`: Default `gpt-4o` (Replace with your local model name)
- `DOCUMENTS_DIR`: Path to your PDFs/Markdown files

---

## 📁 Project Structure
```text
multiagent/
├── src/
│   ├── api.py              # FastAPI wrapper
│   ├── graph.py            # LangGraph workflow logic
│   ├── graph_state.py      # State & Trace definitions
│   ├── llm_client.py       # Token-tracking LLM wrapper
│   ├── tools.py            # Tool implementations
│   ├── vector_store.py     # ChromaDB integration
│   ├── document_processor.py # Text chunking
│   └── ingestion.py        # Data pipeline
├── tests/                  # Pytest suite
├── data/
│   ├── documents/          # Source knowledge base
│   └── chroma/             # Vector DB storage
└── requirements.txt        # Dependencies
```

## 📄 License
MIT License. Feel free to use for educational or commercial purposes.
