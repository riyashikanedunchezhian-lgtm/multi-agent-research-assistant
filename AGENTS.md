# Project-Specific Information

## Build Commands

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Tests
```bash
pytest
```

### Run Test with Coverage
```bash
pytest --cov=src --cov-report=html
```

### Run Specific Test File
```bash
pytest tests/test_graph.py
```

### Run Tests with Verbose Output
```bash
pytest -v
```

## Development Commands

### Ingest Documents
```bash
python -m src.ingestion
```

### Start API Server
```bash
python -m src.api
```

### Run Interactive CLI
```bash
python run.py
```

### Start API Server with Auto-reload
```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

## Project Structure Notes

- `src/` - Main application code
  - `graph.py` - LangGraph workflow with router, retrieval, tool-calling, and synthesis nodes
  - `api.py` - FastAPI wrapper exposing /query, /ingest, /health endpoints
  - `vector_store.py` - ChromaDB wrapper with real embeddings
  - `tools.py` - Calculator, web search, code interpreter tools
  - `graph_state.py` - Typed state definitions and trace management
  - `llm_client.py` - LLM client with per-node token tracking

- `tests/` - Comprehensive test suite
  - `test_graph.py` - Tests for router, retrieval, tool-calling, synthesis nodes
  - `test_api.py` - API endpoint tests
  - `test_ingestion.py` - Document ingestion tests

- `data/` - Data directory (gitignored)
  - `documents/` - Source PDF/markdown documents
  - `chroma/` - Vector database storage

## Key Features Implemented

1. **Multi-node LangGraph workflow** with conditional routing
2. **RAG** with ChromaDB and sentence-transformers embeddings
3. **Agentic tool-calling** (calculator, web search, code interpreter)
4. **Explicit state tracking** with full reasoning traces
5. **Token efficiency** using different models for routing vs synthesis
6. **Guardrails** for router uncertainty and tool failure graceful degradation
7. **FastAPI wrapper** with /query endpoint returning answer + trace

## Environment Variables Required

- `OPENAI_API_KEY` - OpenAI API key (required)
- `ROUTER_MODEL` - Model for routing (default: gpt-4o-mini)
- `SYNTHESIS_MODEL` - Model for synthesis (default: gpt-4o)
- `LLM_PROVIDER` - Provider: openai or anthropic (default: openai)
- `EMBEDDING_MODEL` - Embedding model (default: sentence-transformers/all-MiniLM-L6-v2)
- `CHROMA_PERSIST_DIR` - Vector DB storage (default: ./data/chroma)
- `DOCUMENTS_DIR` - Source documents (default: ./data/documents)
