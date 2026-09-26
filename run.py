"""Simple script to run the research assistant."""

import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

from src.graph import ResearchAssistantGraph
from src.vector_store import VectorStore
from src.ingestion import DocumentIngestionPipeline


def main():
    """Run the research assistant interactively."""
    print("Initializing Multi-Agent Research Assistant...")
    
    # Initialize vector store
    print("Loading vector store...")
    vector_store = VectorStore(
        persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./data/chroma"),
        collection_name=os.getenv("CHROMA_COLLECTION_NAME", "documents"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        use_openai_embeddings=False,
    )
    
    # Check if vector store has documents
    stats = vector_store.get_collection_stats()
    print(f"Vector store has {stats['count']} documents")
    
    if stats["count"] == 0:
        print("No documents found. Ingesting sample documents...")
        pipeline = DocumentIngestionPipeline(
            documents_dir=os.getenv("DOCUMENTS_DIR", "./data/documents"),
            persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./data/chroma"),
            collection_name=os.getenv("CHROMA_COLLECTION_NAME", "documents"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            use_openai_embeddings=False,
        )
        
        # Create sample documents if needed
        if not os.path.exists(pipeline.documents_dir) or not os.listdir(pipeline.documents_dir):
            pipeline.create_sample_documents()
        
        # Ingest
        pipeline.ingest(clear_existing=False)
        print("Ingestion complete!")
    
    # Initialize graph
    print("Initializing LangGraph...")
    graph = ResearchAssistantGraph(
        vector_store=vector_store,
        router_model=os.getenv("ROUTER_MODEL", "gpt-4o-mini"),
        synthesis_model=os.getenv("SYNTHESIS_MODEL", "gpt-4o"),
        router_provider=os.getenv("LLM_PROVIDER", "openai"),
        synthesis_provider=os.getenv("LLM_PROVIDER", "openai"),
    )
    
    print("\n" + "="*60)
    print("Multi-Agent Research Assistant Ready!")
    print("="*60)
    print("\nType your questions below (or 'quit' to exit)")
    print("Add '--trace' to see the reasoning trace\n")
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Check if user wants to see trace
            show_trace = "--trace" in user_input
            query = user_input.replace("--trace", "").strip()
            
            print(f"\nProcessing: {query}")
            print("-" * 60)
            
            result = graph.invoke(query, show_reasoning_trace=show_trace)
            
            print(f"\nAnswer: {result['answer']}")
            
            if show_trace:
                print("\n" + "="*60)
                print("REASONING TRACE")
                print("="*60)
                for i, entry in enumerate(result["execution_trace"], 1):
                    print(f"\n{i}. {entry['node_name'].upper()}")
                    print(f"   Time: {entry['timestamp']}")
                    print(f"   Input: {entry['input_summary']}")
                    print(f"   Output: {entry['output_summary']}")
                    print(f"   Tokens: {entry['token_usage']['total_tokens']}")
                    if entry.get("confidence"):
                        print(f"   Confidence: {entry['confidence']}")
                    if entry.get("error"):
                        print(f"   Error: {entry['error']}")
                
                print("\n" + "="*60)
                print("TOTAL TOKEN USAGE")
                print("="*60)
                print(f"Prompt tokens: {result['total_token_usage']['prompt_tokens']}")
                print(f"Completion tokens: {result['total_token_usage']['completion_tokens']}")
                print(f"Total tokens: {result['total_token_usage']['total_tokens']}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
