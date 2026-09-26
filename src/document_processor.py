"""Document processing module for ingesting PDFs and markdown files."""

import os
from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from pypdf import PdfReader


class DocumentProcessor:
    """Process documents from various formats into chunks for RAG."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def load_pdf(self, file_path: str) -> List[str]:
        """Load and extract text from a PDF file."""
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return [doc.page_content for doc in documents]

    def load_markdown(self, file_path: str) -> List[str]:
        """Load and extract text from a markdown file."""
        loader = UnstructuredMarkdownLoader(file_path)
        documents = loader.load()
        return [doc.page_content for doc in documents]

    def load_documents_from_directory(self, directory: str) -> List[str]:
        """Load all supported documents from a directory."""
        documents = []
        directory_path = Path(directory)

        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        for file_path in directory_path.rglob("*"):
            if file_path.is_file():
                if file_path.suffix.lower() == ".pdf":
                    try:
                        documents.extend(self.load_pdf(str(file_path)))
                    except Exception as e:
                        print(f"Error loading PDF {file_path}: {e}")
                elif file_path.suffix.lower() in [".md", ".markdown"]:
                    try:
                        documents.extend(self.load_markdown(str(file_path)))
                    except Exception as e:
                        print(f"Error loading markdown {file_path}: {e}")

        return documents

    def chunk_documents(self, documents: List[str]) -> List[str]:
        """Split documents into chunks for embedding."""
        all_chunks = []
        for doc in documents:
            chunks = self.text_splitter.split_text(doc)
            all_chunks.extend(chunks)
        return all_chunks
