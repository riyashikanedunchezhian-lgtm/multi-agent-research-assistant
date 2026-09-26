"""Vector store module using ChromaDB with real embeddings."""

import os
from typing import List, Optional

import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings


class VectorStore:
    """Manage ChromaDB vector store with real embeddings."""

    def __init__(
        self,
        persist_directory: str = "./data/chroma",
        collection_name: str = "documents",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_openai_embeddings: bool = False,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.use_openai_embeddings = use_openai_embeddings

        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)

        # Initialize embeddings
        if use_openai_embeddings:
            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        else:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

        # Initialize vector store
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
        )

    def add_documents(self, texts: List[str], metadatas: Optional[List[dict]] = None):
        """Add documents to the vector store."""
        if metadatas is None:
            metadatas = [{"source": f"doc_{i}"} for i in range(len(texts))]

        self.vector_store.add_texts(texts=texts, metadatas=metadatas)

    def similarity_search(
        self, query: str, k: int = 4, filter_dict: Optional[dict] = None
    ) -> List[dict]:
        """Search for similar documents."""
        if filter_dict:
            results = self.vector_store.similarity_search(
                query=query, k=k, filter=filter_dict
            )
        else:
            results = self.vector_store.similarity_search(query=query, k=k)

        return [
            {"content": doc.page_content, "metadata": doc.metadata} for doc in results
        ]

    def similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> List[tuple[dict, float]]:
        """Search for similar documents with relevance scores."""
        results = self.vector_store.similarity_search_with_score(query=query, k=k)

        return [
            ({"content": doc.page_content, "metadata": doc.metadata}, score)
            for doc, score in results
        ]

    def delete_collection(self):
        """Delete the current collection."""
        self.vector_store.delete_collection()

    def get_collection_stats(self) -> dict:
        """Get statistics about the collection."""
        collection = self.vector_store._collection
        return {
            "name": collection.name,
            "count": collection.count(),
            "metadata": collection.metadata,
        }
