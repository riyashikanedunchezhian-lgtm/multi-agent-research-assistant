"""Document ingestion pipeline for populating the vector store."""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore

load_dotenv()


class DocumentIngestionPipeline:
    """Pipeline for ingesting documents into the vector store."""

    def __init__(
        self,
        documents_dir: str = "./data/documents",
        persist_directory: str = "./data/chroma",
        collection_name: str = "documents",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_openai_embeddings: bool = False,
    ):
        self.documents_dir = documents_dir
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.use_openai_embeddings = use_openai_embeddings

        self.processor = DocumentProcessor()
        self.vector_store = VectorStore(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding_model=embedding_model,
            use_openai_embeddings=use_openai_embeddings,
        )

    def ingest(self, clear_existing: bool = False) -> dict:
        """Ingest documents from the documents directory."""
        if clear_existing:
            print("Clearing existing collection...")
            self.vector_store.delete_collection()
            # Reinitialize after deletion
            self.vector_store = VectorStore(
                persist_directory=self.persist_directory,
                collection_name=self.collection_name,
                embedding_model=self.embedding_model,
                use_openai_embeddings=self.use_openai_embeddings,
            )

        print(f"Loading documents from {self.documents_dir}...")
        documents = self.processor.load_documents_from_directory(self.documents_dir)

        if not documents:
            print("No documents found to ingest.")
            return {"status": "no_documents", "count": 0}

        print(f"Found {len(documents)} document pages. Chunking...")
        chunks = self.processor.chunk_documents(documents)

        print(f"Created {len(chunks)} chunks. Adding to vector store...")
        self.vector_store.add_documents(chunks)

        stats = self.vector_store.get_collection_stats()
        print(f"Ingestion complete. Collection now has {stats['count']} documents.")

        return {
            "status": "success",
            "documents_loaded": len(documents),
            "chunks_created": len(chunks),
            "total_in_collection": stats["count"],
        }

    def create_sample_documents(self):
        """Create sample documents for testing if none exist."""
        documents_dir = Path(self.documents_dir)
        documents_dir.mkdir(parents=True, exist_ok=True)

        # Create a sample markdown document about AI/ML
        sample_md = """# Machine Learning Fundamentals

## Introduction
Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.

## Key Concepts

### Supervised Learning
In supervised learning, the model is trained on labeled data. Common algorithms include:
- Linear Regression
- Decision Trees
- Random Forests
- Neural Networks

### Unsupervised Learning
Unsupervised learning deals with unlabeled data. Techniques include:
- Clustering (K-means, DBSCAN)
- Dimensionality Reduction (PCA, t-SNE)
- Association Rule Learning

### Deep Learning
Deep learning uses neural networks with many layers. Applications include:
- Image Recognition
- Natural Language Processing
- Speech Recognition
- Reinforcement Learning

## Evaluation Metrics
Common metrics for evaluating ML models:
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC
- Mean Squared Error (MSE)
- R-squared

## Best Practices
- Cross-validation for robust evaluation
- Feature engineering and selection
- Hyperparameter tuning
- Regularization to prevent overfitting
- Data preprocessing and normalization
"""

        sample_md_path = documents_dir / "ml_fundamentals.md"
        with open(sample_md_path, "w", encoding="utf-8") as f:
            f.write(sample_md)

        # Create a sample markdown document about finance
        finance_md = """# Financial Analysis Guide

## Introduction
Financial analysis involves evaluating businesses, projects, budgets, and other finance-related transactions to determine their performance and suitability.

## Key Financial Ratios

### Liquidity Ratios
- Current Ratio: Current Assets / Current Liabilities
- Quick Ratio: (Current Assets - Inventory) / Current Liabilities
- Cash Ratio: Cash / Current Liabilities

### Profitability Ratios
- Gross Margin: (Revenue - COGS) / Revenue
- Operating Margin: Operating Income / Revenue
- Net Profit Margin: Net Income / Revenue
- Return on Assets (ROA): Net Income / Total Assets
- Return on Equity (ROE): Net Income / Shareholder Equity

### Valuation Ratios
- Price-to-Earnings (P/E): Market Price per Share / Earnings per Share
- Price-to-Book (P/B): Market Price per Share / Book Value per Share
- Enterprise Value to EBITDA

## Investment Strategies

### Value Investing
Focus on undervalued companies with strong fundamentals. Key indicators:
- Low P/E ratios relative to industry
- High dividend yields
- Strong balance sheets

### Growth Investing
Focus on companies with high growth potential. Characteristics:
- High revenue growth rates
- Expanding market share
- Innovative products/services

### Technical Analysis
Uses price patterns and trading volume to predict future movements.
"""

        finance_md_path = documents_dir / "financial_analysis.md"
        with open(finance_md_path, "w", encoding="utf-8") as f:
            f.write(finance_md)

        print(f"Created sample documents in {self.documents_dir}")
        return {"status": "samples_created", "files": [str(sample_md_path), str(finance_md_path)]}


if __name__ == "__main__":
    pipeline = DocumentIngestionPipeline()
    
    # Create sample documents if directory is empty
    if not os.path.exists(pipeline.documents_dir) or not os.listdir(pipeline.documents_dir):
        pipeline.create_sample_documents()
    
    # Ingest documents
    result = pipeline.ingest(clear_existing=True)
    print(f"\nIngestion result: {result}")
