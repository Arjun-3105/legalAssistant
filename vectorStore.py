"""
vectorStore.py — ChromaDB-backed persistent document store for legal document corpus.

Provides document ingestion (PDF), chunking, RoBERTa-based embedding, and semantic retrieval
across 300+ legal documents.
"""

import os
from dotenv import load_dotenv
load_dotenv()  # Load .env file before any os.getenv() calls
import re
import hashlib
import torch
import numpy as np
import pdfplumber
import chromadb
from chromadb.config import Settings
from transformers import AutoTokenizer, AutoModel
from nltk.tokenize import sent_tokenize
import nltk

nltk.download('punkt_tab', quiet=True)

# ─── RoBERTa Embedding Model ─────────────────────────────────────────────────
ROBERTA_MODEL_PATH = os.getenv("ROBERTA_MODEL_PATH", "roberta-base")

print(f"\n[VectorStore] Loading RoBERTa model from: {ROBERTA_MODEL_PATH}")
try:
    roberta_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL_PATH)
    roberta_model = AutoModel.from_pretrained(ROBERTA_MODEL_PATH)
    roberta_model.eval()
    print("[VectorStore] ✅ RoBERTa model loaded successfully.")
except Exception as e:
    print(f"[VectorStore] ❌ Error loading RoBERTa: {e}")
    roberta_tokenizer = None
    roberta_model = None


def get_roberta_embedding(text: str) -> list[float]:
    """Generate a dense embedding for a text string using RoBERTa."""
    inputs = roberta_tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    with torch.no_grad():
        outputs = roberta_model(**inputs)
    # Mean pooling over token embeddings
    embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze().numpy()
    return embedding.tolist()


# ─── Text Chunking ────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[dict]:
    """
    Split text into overlapping word-level chunks with sentence awareness.
    Returns list of {text, word_count} dicts.
    """
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return []

    sentences = sent_tokenize(text)
    chunks = []
    current_chunk_words = []
    current_word_count = 0

    for sentence in sentences:
        words = sentence.split()
        if current_word_count + len(words) > chunk_size and current_chunk_words:
            chunk_text_str = ' '.join(current_chunk_words)
            chunks.append({"text": chunk_text_str, "word_count": current_word_count})
            # Keep overlap
            overlap_words = current_chunk_words[-overlap:] if overlap < len(current_chunk_words) else current_chunk_words
            current_chunk_words = list(overlap_words)
            current_word_count = len(current_chunk_words)

        current_chunk_words.extend(words)
        current_word_count += len(words)

    # Final chunk
    if current_chunk_words:
        chunk_text_str = ' '.join(current_chunk_words)
        chunks.append({"text": chunk_text_str, "word_count": current_word_count})

    return chunks


def extract_pdf_text(pdf_path: str) -> list[dict]:
    """Extract text from PDF, returning list of {page, text} dicts."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append({"page": i + 1, "text": text.strip()})
    return pages


# ─── DocumentStore ────────────────────────────────────────────────────────────

class DocumentStore:
    """
    Persistent vector store for legal documents using ChromaDB.
    Supports ingestion of individual PDFs or entire directories,
    semantic retrieval, and corpus management.
    """

    def __init__(self, persist_dir: str = "./chroma_legal_store"):
        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="legal_documents",
            metadata={"hnsw:space": "cosine"}
        )
        print(f"[VectorStore] 📂 ChromaDB store at: {persist_dir}")
        print(f"[VectorStore] 📊 Current corpus: {self.collection.count()} chunks indexed")

    def _generate_chunk_id(self, filename: str, page: int, chunk_idx: int) -> str:
        """Generate a deterministic ID for a chunk."""
        raw = f"{filename}::page{page}::chunk{chunk_idx}"
        return hashlib.md5(raw.encode()).hexdigest()

    def ingest_pdf(self, pdf_path: str) -> dict:
        """
        Ingest a single PDF into the vector store.
        Returns stats: {filename, pages, chunks_added}.
        """
        filename = os.path.basename(pdf_path)
        print(f"[VectorStore] 📄 Ingesting: {filename}")

        # Check if already ingested
        existing = self.collection.get(where={"source_file": filename})
        if existing and existing["ids"]:
            print(f"[VectorStore] ⚠️  {filename} already ingested ({len(existing['ids'])} chunks). Skipping.")
            return {"filename": filename, "pages": 0, "chunks_added": 0, "status": "already_exists"}

        pages = extract_pdf_text(pdf_path)
        if not pages:
            return {"filename": filename, "pages": 0, "chunks_added": 0, "status": "no_text"}

        all_ids = []
        all_embeddings = []
        all_documents = []
        all_metadatas = []

        for page_data in pages:
            chunks = chunk_text(page_data["text"])
            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = self._generate_chunk_id(filename, page_data["page"], chunk_idx)
                embedding = get_roberta_embedding(chunk["text"])

                all_ids.append(chunk_id)
                all_embeddings.append(embedding)
                all_documents.append(chunk["text"])
                all_metadatas.append({
                    "source_file": filename,
                    "page": page_data["page"],
                    "chunk_index": chunk_idx,
                    "word_count": chunk["word_count"]
                })

        # Batch upsert to ChromaDB
        if all_ids:
            # ChromaDB has batch size limits, so batch in groups of 100
            batch_size = 100
            for i in range(0, len(all_ids), batch_size):
                self.collection.add(
                    ids=all_ids[i:i + batch_size],
                    embeddings=all_embeddings[i:i + batch_size],
                    documents=all_documents[i:i + batch_size],
                    metadatas=all_metadatas[i:i + batch_size]
                )

        stats = {
            "filename": filename,
            "pages": len(pages),
            "chunks_added": len(all_ids),
            "status": "success"
        }
        print(f"[VectorStore] ✅ Ingested {filename}: {stats['pages']} pages, {stats['chunks_added']} chunks")
        return stats

    def ingest_directory(self, dir_path: str) -> list[dict]:
        """Batch ingest all PDFs in a directory."""
        results = []
        pdf_files = [f for f in os.listdir(dir_path) if f.lower().endswith('.pdf')]
        print(f"[VectorStore] 📁 Found {len(pdf_files)} PDFs in {dir_path}")

        for pdf_file in pdf_files:
            full_path = os.path.join(dir_path, pdf_file)
            result = self.ingest_pdf(full_path)
            results.append(result)

        return results

    def query(self, query_text: str, top_k: int = 10, filter_file: str = None) -> list[dict]:
        """
        Semantic search across the entire corpus.
        Returns list of {text, source_file, page, score, chunk_index} dicts.
        """
        query_embedding = get_roberta_embedding(query_text)

        where_filter = {"source_file": filter_file} if filter_file else None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        retrieved = []
        if results and results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                retrieved.append({
                    "text": results["documents"][0][i],
                    "source_file": results["metadatas"][0][i]["source_file"],
                    "page": results["metadatas"][0][i]["page"],
                    "chunk_index": results["metadatas"][0][i]["chunk_index"],
                    "score": 1 - results["distances"][0][i]  # ChromaDB returns distance, convert to similarity
                })

        return retrieved

    def get_stats(self) -> dict:
        """Return corpus statistics."""
        total_chunks = self.collection.count()

        # Get unique documents
        if total_chunks > 0:
            all_metadata = self.collection.get(include=["metadatas"])
            unique_files = set()
            for meta in all_metadata["metadatas"]:
                unique_files.add(meta["source_file"])
            return {
                "total_documents": len(unique_files),
                "total_chunks": total_chunks,
                "document_names": sorted(list(unique_files))
            }

        return {"total_documents": 0, "total_chunks": 0, "document_names": []}

    def delete_document(self, filename: str) -> dict:
        """Remove all chunks for a given document."""
        existing = self.collection.get(where={"source_file": filename})
        if existing and existing["ids"]:
            self.collection.delete(ids=existing["ids"])
            return {"filename": filename, "chunks_removed": len(existing["ids"]), "status": "deleted"}
        return {"filename": filename, "chunks_removed": 0, "status": "not_found"}

    def get_document_list(self) -> list[dict]:
        """Return list of documents with their chunk counts."""
        stats = self.get_stats()
        if stats["total_documents"] == 0:
            return []

        doc_list = []
        for name in stats["document_names"]:
            doc_chunks = self.collection.get(where={"source_file": name})
            doc_list.append({
                "filename": name,
                "chunks": len(doc_chunks["ids"]) if doc_chunks["ids"] else 0
            })
        return doc_list
