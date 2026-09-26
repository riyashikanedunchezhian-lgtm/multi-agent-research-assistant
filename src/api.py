"""FastAPI wrapper for the research assistant."""

import os
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from src.graph import ResearchAssistantGraph
from src.vector_store import VectorStore
from src.ingestion import DocumentIngestionPipeline

load_dotenv()


app = FastAPI(
    title="Multi-Agent Research Assistant",
    description="A LangGraph-powered research assistant with RAG and tool-calling capabilities",
    version="0.1.0",
)


# Global instances
vector_store: Optional[VectorStore] = None
graph: Optional[ResearchAssistantGraph] = None
ingestion_pipeline: Optional[DocumentIngestionPipeline] = None


class QueryRequest(BaseModel):
    query: str
    show_reasoning_trace: bool = False


class QueryResponse(BaseModel):
    answer: str
    reasoning_trace: Optional[list[dict]] = None
    token_usage: dict
    execution_time: Optional[str] = None
    status: str


class HealthResponse(BaseModel):
    status: str
    vector_store_loaded: bool
    collection_stats: Optional[dict] = None


@app.on_event("startup")
async def startup_event():
    """Initialize the vector store and graph on startup."""
    global vector_store, graph, ingestion_pipeline
    
    print("Initializing vector store...")
    vector_store = VectorStore(
        persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./data/chroma"),
        collection_name=os.getenv("CHROMA_COLLECTION_NAME", "documents"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        use_openai_embeddings=False,
    )
    
    print("Initializing LangGraph...")
    graph = ResearchAssistantGraph(
        vector_store=vector_store,
        router_model=os.getenv("ROUTER_MODEL", "gpt-4o-mini"),
        synthesis_model=os.getenv("SYNTHESIS_MODEL", "gpt-4o"),
        router_provider=os.getenv("LLM_PROVIDER", "openai"),
        synthesis_provider=os.getenv("LLM_PROVIDER", "openai"),
    )
    
    print("Initializing ingestion pipeline...")
    ingestion_pipeline = DocumentIngestionPipeline(
        documents_dir=os.getenv("DOCUMENTS_DIR", "./data/documents"),
        persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./data/chroma"),
        collection_name=os.getenv("CHROMA_COLLECTION_NAME", "documents"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        use_openai_embeddings=False,
    )
    
    print("Startup complete!")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if vector_store is None:
        return HealthResponse(
            status="not_ready",
            vector_store_loaded=False,
            collection_stats=None,
        )
    
    try:
        stats = vector_store.get_collection_stats()
        return HealthResponse(
            status="ready",
            vector_store_loaded=True,
            collection_stats=stats,
        )
    except Exception as e:
        return HealthResponse(
            status="error",
            vector_store_loaded=False,
            collection_stats={"error": str(e)},
        )


@app.post("/ingest")
async def ingest_documents(clear_existing: bool = False):
    """Ingest documents into the vector store."""
    if ingestion_pipeline is None:
        raise HTTPException(status_code=503, detail="Ingestion pipeline not initialized")
    
    try:
        result = ingestion_pipeline.ingest(clear_existing=clear_existing)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the research assistant."""
    if graph is None:
        raise HTTPException(status_code=503, detail="Graph not initialized")
    
    try:
        result = graph.invoke(request.query, request.show_reasoning_trace)
        
        response = QueryResponse(
            answer=result["answer"] or "No answer generated",
            reasoning_trace=result["execution_trace"] if request.show_reasoning_trace else None,
            token_usage=result["total_token_usage"],
            execution_time=result["end_time"],
            status=result["status"],
        )
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Multi-Agent Research Assistant",
        "version": "0.1.0",
        "endpoints": {
            "health": "/health",
            "query": "/query",
            "ingest": "/ingest",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
