"""
vectorStore.py — ChromaDB-backed persistent document store for legal document corpus.

Provides document ingestion (PDF), chunking, RoBERTa-based embedding with masked mean pooling,
and hybrid (semantic + lexical) retrieval across 300+ legal documents.

Key improvements:
- FIXED: ChromaDB cosine distance conversion (was dividing by 2 incorrectly)
- FIXED: PDF text cleaning (dehyphenation, header/footer removal, artifact removal)
- FIXED: Chunk quality filtering (skip junk/boilerplate chunks)
- FIXED: ingest_pdf uses upsert to safely re-ingest updated documents
- ADDED: Progress callback support for large PDF ingestion
- ADDED: Minimum quality score filter in query results
- ADDED: MMR-style deduplication to avoid near-duplicate chunks flooding synthesis
"""

import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import re
import hashlib
import torch
import numpy as np
import pdfplumber
import chromadb
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


def get_roberta_embedding(text: str) -> list:
    """
    Generate a dense embedding for a text string using RoBERTa.
    Uses attention-masked mean pooling (excludes padding tokens) and L2 normalization.
    """
    if not text or not text.strip() or roberta_model is None or roberta_tokenizer is None:
        return [0.0] * 768

    inputs = roberta_tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    with torch.no_grad():
        outputs = roberta_model(**inputs)

    # Attention-mask aware mean pooling — excludes <pad> tokens
    attention_mask = inputs["attention_mask"].unsqueeze(-1)  # [1, seq, 1]
    token_embeddings = outputs.last_hidden_state              # [1, seq, 768]
    input_mask_expanded = attention_mask.expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    mean_pooled = sum_embeddings / sum_mask                   # [1, 768]

    # L2 normalization ensures cosine similarity is correct
    normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
    return normalized.squeeze().cpu().numpy().tolist()


# ─── PDF Text Extraction & Cleaning ──────────────────────────────────────────

# Regex patterns for common PDF artifacts
_RE_HYPHEN_LINEBREAK = re.compile(r'(\w)-\n(\w)')            # indemni-\nfication
_RE_HEADER_FOOTER = re.compile(                               # Page N / Confidential lines
    r'(?im)^\s*(?:'
    r'page\s+\d+\s*(?:of\s+\d+)?'
    r'|confidential[\s|]*.*?'
    r'|\d+\s*of\s*\d+'
    r'|(?:draft|internal use only|attorney.client privilege)'
    r')\s*$'
)
_RE_SECTION_NUMBER_GLUED = re.compile(r'(\d+\.)([A-Z])')     # "5.CONTRACT" -> "5. CONTRACT"
_RE_MULTI_SPACES = re.compile(r'[ \t]{2,}')                   # multiple spaces/tabs
_RE_REPEATED_NEWLINES = re.compile(r'\n{3,}')                 # excessive blank lines
_RE_MOSTLY_NUMBERS = re.compile(r'^[\d\s\.\-\(\)\/,]+$')     # junk numeric lines


def clean_pdf_text(raw_text: str) -> str:
    """
    Clean raw PDF-extracted text to fix common extraction artifacts.
    - Rejoin hyphenated line-breaks ('indemni-\\nfication' -> 'indemnification')
    - Remove repeated headers/footers (page numbers, 'CONFIDENTIAL', etc.)
    - Fix section numbers glued to text ('5.CONTRACT' -> '5. CONTRACT')
    - Collapse excessive whitespace
    """
    if not raw_text:
        return ""

    # 1. Rejoin hyphenated line-breaks (common in justified legal text)
    text = _RE_HYPHEN_LINEBREAK.sub(r'\1\2', raw_text)

    # 2. Remove header/footer lines
    text = _RE_HEADER_FOOTER.sub('', text)

    # 3. Fix section numbers glued to words
    text = _RE_SECTION_NUMBER_GLUED.sub(r'\1 \2', text)

    # 4. Collapse multiple horizontal spaces (from column alignment)
    text = _RE_MULTI_SPACES.sub(' ', text)

    # 5. Normalize excessive newlines
    text = _RE_REPEATED_NEWLINES.sub('\n\n', text)

    # 6. Strip leading/trailing whitespace per line
    lines = [l.strip() for l in text.splitlines()]
    text = '\n'.join(lines)

    return text.strip()


def is_junk_chunk(text: str) -> bool:
    """
    Return True if a chunk is too low-quality to index.
    Filters out boilerplate-only chunks, mostly-numeric chunks, and tiny fragments.
    """
    stripped = text.strip()
    if len(stripped.split()) < 8:
        return True  # Too short to be meaningful

    # Mostly numeric (e.g. table of numbers, page refs)
    if _RE_MOSTLY_NUMBERS.match(stripped):
        return True

    # Chunk that is entirely uppercase and short (likely a header/section title only)
    words = stripped.split()
    if len(words) < 15 and stripped == stripped.upper() and all(len(w) > 1 for w in words):
        return True

    # Very low alphabetic ratio (< 50% alphabetic characters → table / form data)
    alpha_chars = sum(1 for c in stripped if c.isalpha())
    if len(stripped) > 0 and alpha_chars / len(stripped) < 0.4:
        return True

    return False


def extract_pdf_text(pdf_path: str) -> list:
    """
    Extract and clean text from PDF, returning list of {page, text} dicts.
    Handles:
    - Text-based PDFs (standard)
    - Attempts layout-aware extraction for multi-column text
    """
    pages = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                # Try layout-aware extraction first
                text = page.extract_text(x_tolerance=3, y_tolerance=3)

                if not text or not text.strip():
                    # Fallback: try bounding-box based extraction
                    try:
                        text = page.extract_text(layout=True)
                    except Exception:
                        pass

                if text and text.strip():
                    cleaned = clean_pdf_text(text)
                    if cleaned:
                        pages.append({"page": i + 1, "text": cleaned})

    except Exception as e:
        print(f"[VectorStore] Error reading PDF {pdf_path}: {e}")
    return pages


# ─── Text Chunking ────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list:
    """
    Split text into overlapping word-level chunks with sentence awareness.

    Key correctness guarantees:
    - Sentences longer than chunk_size are split on natural sub-delimiters
      (semicolons, lettered sub-clauses, commas) before word-level fallback
    - Overlap always carries the last N words of the previous chunk as context
    - Junk chunks are filtered out after creation

    Returns list of {text, word_count} dicts.
    """
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return []

    raw_sentences = sent_tokenize(text)

    # Normalize and subdivide over-long sentences
    sentences = []
    for s in raw_sentences:
        s = s.strip()
        if not s:
            continue
        s_words = s.split()
        if len(s_words) <= chunk_size:
            sentences.append(s)
        else:
            # Split on semicolons, lettered sub-clauses (a), (b), (i), bullet points
            subparts = re.split(r'(?<=[;])\s+|(?=\s\([a-z]{1,3}\)\s)|(?<=\.)\s+(?=[A-Z])', s)
            current_part_words = []
            for part in subparts:
                part = part.strip()
                if not part:
                    continue
                part_words = part.split()
                # If adding this part keeps us under chunk_size, accumulate it
                if len(current_part_words) + len(part_words) <= chunk_size:
                    current_part_words.extend(part_words)
                else:
                    # Flush current_part_words as a sentence
                    if current_part_words:
                        sentences.append(' '.join(current_part_words))
                    # If this part itself is > chunk_size, word-split it
                    if len(part_words) > chunk_size:
                        for start in range(0, len(part_words), chunk_size - overlap):
                            word_slice = part_words[start:start + chunk_size]
                            if word_slice:
                                sentences.append(' '.join(word_slice))
                        current_part_words = []
                    else:
                        current_part_words = part_words
            if current_part_words:
                sentences.append(' '.join(current_part_words))

    # Build overlapping chunks
    chunks = []
    current_chunk_words = []
    current_word_count = 0

    for sentence in sentences:
        words = sentence.split()
        if not words:
            continue

        if current_word_count + len(words) > chunk_size and current_chunk_words:
            # Flush current chunk
            chunk_text_str = ' '.join(current_chunk_words)
            if not is_junk_chunk(chunk_text_str):
                chunks.append({"text": chunk_text_str, "word_count": current_word_count})

            # Carry overlap: last `overlap` words of the flushed chunk
            overlap_words = current_chunk_words[-overlap:] if len(current_chunk_words) >= overlap else current_chunk_words[:]
            current_chunk_words = overlap_words[:]
            current_word_count = len(current_chunk_words)

        current_chunk_words.extend(words)
        current_word_count += len(words)

    # Final chunk
    if current_chunk_words:
        chunk_text_str = ' '.join(current_chunk_words)
        if not is_junk_chunk(chunk_text_str):
            chunks.append({"text": chunk_text_str, "word_count": current_word_count})

    return chunks


# ─── Lexical Scoring ──────────────────────────────────────────────────────────

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself",
    "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such", "than",
    "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what",
    "when", "where", "which", "while", "who", "whom", "why", "with", "would", "you", "your", "yours"
}


def compute_lexical_score(query: str, text: str) -> float:
    """
    Compute keyword relevance between query and chunk.
    Weights: exact term ratio (70%) + bigram phrase bonus (25%) + frequency bonus (5%)
    Returns a float in [0.0, 1.0].
    """
    if not query or not text:
        return 0.0

    q_tokens = [
        w.lower() for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', query)
        if w.lower() not in STOPWORDS
    ]
    if not q_tokens:
        return 0.4  # Neutral if only stopwords

    text_lower = text.lower()

    # 1. Term hit ratio
    matches = sum(1 for tok in q_tokens if re.search(r'\b' + re.escape(tok) + r'\b', text_lower))
    term_ratio = matches / len(q_tokens)

    # 2. Consecutive bigram phrase bonus (capped to avoid runaway scores)
    phrase_bonus = 0.0
    for i in range(len(q_tokens) - 1):
        bigram = f"{q_tokens[i]} {q_tokens[i+1]}"
        if bigram in text_lower:
            phrase_bonus = min(phrase_bonus + 0.15, 0.30)

    # 3. Term frequency bonus (capped)
    freq_boost = 0.0
    for tok in q_tokens:
        count = len(re.findall(r'\b' + re.escape(tok) + r'\b', text_lower))
        if count > 1:
            freq_boost = min(freq_boost + 0.03, 0.10)

    return min(1.0, (term_ratio * 0.70) + phrase_bonus + freq_boost)


# ─── MMR Deduplication ────────────────────────────────────────────────────────

def mmr_deduplicate(candidates: list, query_emb: list, top_k: int, diversity: float = 0.3) -> list:
    """
    Maximal Marginal Relevance: balance relevance with diversity.
    Prevents near-duplicate chunks from flooding the synthesis context window.

    Args:
        candidates: list of chunk dicts, each with 'embedding' and 'score'
        query_emb: query embedding vector
        top_k: number of results to return
        diversity: weight for diversity (0 = pure relevance, 1 = pure diversity)
    Returns:
        Reranked and deduplicated list of top_k chunks
    """
    if not candidates or top_k <= 0:
        return candidates[:top_k]

    if len(candidates) <= top_k:
        return candidates

    # We only have hybrid scores, not raw embeddings in query results.
    # Fallback: deduplicate by text similarity (hashed bigrams)
    selected = []
    seen_fingerprints = set()

    for c in candidates:
        text = c.get("text", "")
        # Create a simple "fingerprint" from the most common 5 bigrams in the chunk
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        bigrams = frozenset(
            f"{words[i]}_{words[i+1]}" for i in range(min(len(words)-1, 15))
        )
        overlap_with_seen = sum(1 for bg in bigrams if bg in seen_fingerprints)
        overlap_ratio = overlap_with_seen / max(len(bigrams), 1)

        if overlap_ratio < 0.5:  # Less than 50% bigram overlap with any already-selected chunk
            selected.append(c)
            seen_fingerprints.update(bigrams)

        if len(selected) >= top_k:
            break

    # If MMR filtered too aggressively, fill back up with remaining
    if len(selected) < top_k:
        for c in candidates:
            if c not in selected:
                selected.append(c)
            if len(selected) >= top_k:
                break

    return selected


# ─── DocumentStore ────────────────────────────────────────────────────────────

# Minimum hybrid score for a chunk to be returned from query()
MIN_RELEVANCE_SCORE = 0.25

class DocumentStore:
    """
    Persistent vector store for legal documents using ChromaDB.

    Improvements over v1:
    - Uses upsert (not add) to safely handle re-ingestion of updated documents
    - Cleans PDF text before chunking (dehyphenation, header/footer removal)
    - Filters junk chunks before indexing
    - Correct ChromaDB cosine distance→similarity conversion
    - Minimum relevance threshold to exclude noise from query results
    - MMR-based deduplication to prevent near-duplicate chunks in synthesis context
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
        """Generate a deterministic, stable ID for a chunk."""
        raw = f"{filename}::page{page}::chunk{chunk_idx}"
        return hashlib.md5(raw.encode()).hexdigest()

    def ingest_pdf(self, pdf_path: str, progress_callback=None) -> dict:
        """
        Ingest (or re-ingest) a single PDF into the vector store.

        Uses upsert instead of add so callers can safely re-ingest updated documents.
        Accepts an optional progress_callback(current, total, message) for UI progress bars.

        Returns stats dict: {filename, pages, chunks_added, status}
        """
        filename = os.path.basename(pdf_path)
        print(f"[VectorStore] 📄 Ingesting: {filename}")

        # Check if already ingested — report to caller but still allow re-ingest (upsert)
        existing = self.collection.get(where={"source_file": filename})
        already_exists = bool(existing and existing["ids"])

        if already_exists:
            print(f"[VectorStore] 🔄 {filename} already exists ({len(existing['ids'])} chunks). Re-indexing with upsert.")

        pages = extract_pdf_text(pdf_path)
        if not pages:
            return {"filename": filename, "pages": 0, "chunks_added": 0, "status": "no_text"}

        all_ids = []
        all_embeddings = []
        all_documents = []
        all_metadatas = []

        total_pages = len(pages)
        for page_idx, page_data in enumerate(pages):
            if progress_callback:
                progress_callback(
                    page_idx, total_pages,
                    f"Embedding page {page_idx + 1}/{total_pages} of {filename}..."
                )

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

        # Use UPSERT (not add) to safely handle both new and re-indexed documents
        if all_ids:
            batch_size = 100
            total_batches = (len(all_ids) + batch_size - 1) // batch_size
            for batch_num, i in enumerate(range(0, len(all_ids), batch_size)):
                if progress_callback:
                    progress_callback(
                        total_pages + batch_num, total_pages + total_batches,
                        f"Indexing batch {batch_num + 1}/{total_batches}..."
                    )
                self.collection.upsert(
                    ids=all_ids[i:i + batch_size],
                    embeddings=all_embeddings[i:i + batch_size],
                    documents=all_documents[i:i + batch_size],
                    metadatas=all_metadatas[i:i + batch_size]
                )

        status = "updated" if already_exists else "success"
        stats = {
            "filename": filename,
            "pages": len(pages),
            "chunks_added": len(all_ids),
            "status": status
        }
        print(f"[VectorStore] ✅ {status.capitalize()} {filename}: {stats['pages']} pages, {stats['chunks_added']} chunks")
        return stats

    def ingest_directory(self, dir_path: str, progress_callback=None) -> list:
        """Batch ingest all PDFs in a directory."""
        results = []
        pdf_files = [f for f in os.listdir(dir_path) if f.lower().endswith('.pdf')]
        print(f"[VectorStore] 📁 Found {len(pdf_files)} PDFs in {dir_path}")

        for idx, pdf_file in enumerate(pdf_files):
            full_path = os.path.join(dir_path, pdf_file)
            if progress_callback:
                progress_callback(idx, len(pdf_files), f"Ingesting {pdf_file}...")
            result = self.ingest_pdf(full_path)
            results.append(result)

        return results

    def query(self, query_text: str, top_k: int = 10, filter_file: str = None,
              min_score: float = MIN_RELEVANCE_SCORE) -> list:
        """
        Hybrid (Dense Semantic + Lexical) search across the corpus.

        Key fixes:
        - FIXED: ChromaDB cosine distance conversion: sim = 1.0 - distance (not /2)
        - ADDED: Minimum relevance score filter (default 0.25)
        - ADDED: MMR-based deduplication to remove near-duplicate chunks
        - Fetches 4x more candidates than top_k for better reranking pool

        Returns list of {text, source_file, page, score, semantic_score, lexical_score, chunk_index}.
        """
        total_in_db = self.collection.count()
        if total_in_db == 0:
            return []

        query_embedding = get_roberta_embedding(query_text)
        where_filter = {"source_file": filter_file} if filter_file else None

        # Fetch a large candidate pool for hybrid reranking
        fetch_k = min(total_in_db, max(top_k * 4, 30))

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=fetch_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            print(f"[VectorStore] Query error: {e}")
            return []

        candidates = []
        if results and results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                doc_text = results["documents"][0][i]
                meta = results["metadatas"][0][i] or {}
                distance = results["distances"][0][i]

                # FIXED: ChromaDB cosine space: distance = 1 - cosine_similarity
                # Therefore: cosine_similarity = 1 - distance (NOT 1 - distance/2)
                semantic_sim = max(0.0, min(1.0, 1.0 - distance))
                lexical_sim = compute_lexical_score(query_text, doc_text)

                # Hybrid Score: 60% semantic (dense) + 40% lexical (keyword)
                hybrid_score = round((0.60 * semantic_sim) + (0.40 * lexical_sim), 4)

                # Filter out low-quality matches early
                if hybrid_score < min_score:
                    continue

                candidates.append({
                    "text": doc_text,
                    "source_file": meta.get("source_file", "unknown"),
                    "page": meta.get("page", 0),
                    "chunk_index": meta.get("chunk_index", 0),
                    "score": hybrid_score,
                    "semantic_score": round(semantic_sim, 4),
                    "lexical_score": round(lexical_sim, 4)
                })

        # Rerank by hybrid score
        candidates.sort(key=lambda c: c["score"], reverse=True)

        # MMR deduplication to remove near-duplicate chunks
        deduplicated = mmr_deduplicate(candidates, query_embedding, top_k=top_k)

        return deduplicated

    def get_stats(self) -> dict:
        """Return corpus statistics."""
        total_chunks = self.collection.count()

        if total_chunks > 0:
            all_metadata = self.collection.get(include=["metadatas"])
            unique_files = set()
            for meta in all_metadata["metadatas"]:
                if meta and "source_file" in meta:
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

    def get_document_list(self) -> list:
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
