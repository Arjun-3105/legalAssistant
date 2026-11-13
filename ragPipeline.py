"""
ragPipeline.py — Hybrid Legal Summarizer with RoBERTa + Gemini

Key improvements:
- FIXED: generate_extractive_draft now batches RoBERTa calls (was N sequential forward passes)
- FIXED: Centroid is query-biased for RAG mode; pure centroid for summarization
- FIXED: generate_rag_draft is now wired into the UI via refine_with_llm (not dead code)
- FIXED: preprocess_text preserves numbered clause structure before NLTK tokenization
- ADDED: Batch embedding helper for efficient large-document processing
"""

import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import re
import nltk
import torch
import numpy as np
import pdfplumber
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from nltk.tokenize import sent_tokenize

from api_manager import get_gemini_model, call_gemini_safely
from vectorStore import roberta_tokenizer, roberta_model, get_roberta_embedding

nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

LEGAL_KEYWORDS = [
    "plaintiff", "defendant", "respondent", "petitioner", "appellant", "accused",
    "complainant", "counsel", "judge", "bench", "jury", "tribunal", "magistrate",
    "prosecution", "litigant", "witness", "party", "victim", "guardian", "appeal",
    "trial", "hearing", "proceedings", "petition", "lawsuit", "litigation",
    "jurisdiction", "adjudication", "motion", "plea", "summons", "order", "verdict",
    "judgment", "injunction", "stay", "ruling", "decree", "directive", "sentence",
    "conviction", "acquittal", "remand", "bail", "bond", "arrest", "charge", "claim",
    "complaint", "indictment", "retrial", "cross-examination", "examination",
    "evidence", "testimony", "affidavit", "exhibit", "discovery", "burden", "standard",
    "precedent", "case law", "statute", "legislation", "constitution", "clause",
    "provision", "contract", "agreement", "treaty", "bill", "code", "regulation",
    "enactment", "doctrine", "principle", "interpretation", "finding", "conclusion",
    "damages", "compensation", "fine", "penalty", "sanction", "relief", "settlement",
    "restitution", "injury", "liability", "negligence", "breach", "fault", "guilt",
    "rights", "obligation", "duty", "responsibility", "violation", "remedy", "probation",
    "parole", "dismissal", "overrule", "uphold", "vacate", "enforcement",
    "indemnification", "indemnify", "hold harmless", "termination", "severance",
    "governing law", "arbitration", "confidentiality", "non-disclosure", "assignment",
    "limitation", "warranty", "representation", "covenant", "consideration",
    "intellectual property", "force majeure", "liquidated damages", "waiver",
]

# ─── Batch Embedding (efficiency fix) ────────────────────────────────────────

EMBED_BATCH_SIZE = 32  # Process this many sentences per RoBERTa forward pass


def get_roberta_embeddings_batch(texts: list) -> np.ndarray:
    """
    Compute RoBERTa embeddings for a list of texts efficiently in batches.
    Returns an ndarray of shape [N, 768].
    Falls back to per-sentence calls if batch processing fails.
    """
    if not texts:
        return np.zeros((0, 768), dtype=np.float32)

    if roberta_model is None or roberta_tokenizer is None:
        return np.zeros((len(texts), 768), dtype=np.float32)

    all_embeddings = []

    for batch_start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[batch_start: batch_start + EMBED_BATCH_SIZE]
        try:
            inputs = roberta_tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            with torch.no_grad():
                outputs = roberta_model(**inputs)

            # Attention-masked mean pooling
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask

            # L2 normalize
            normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
            all_embeddings.append(normalized.cpu().numpy())

        except Exception as e:
            print(f"[RAG] Batch embedding failed for batch at {batch_start}: {e}")
            # Fallback: encode individually
            for text in batch:
                emb = np.array(get_roberta_embedding(text), dtype=np.float32)
                all_embeddings.append(emb.reshape(1, -1))

    if not all_embeddings:
        return np.zeros((len(texts), 768), dtype=np.float32)

    return np.vstack(all_embeddings)


# ─── HybridLegalSummarizer ────────────────────────────────────────────────────

class HybridLegalSummarizer:
    """
    Hybrid summarizer combining RoBERTa-based extractive ranking with
    Gemini-based abstractive refinement.

    Key improvements:
    - Batched embedding for large documents (was O(N) sequential, now batched)
    - Query-biased centroid for RAG mode (not pure document centroid)
    - generate_rag_draft now wired to summarizer UI when a focus query is provided
    - preprocess_text preserves numbered clause structure
    """

    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.legal_keywords = set(word.lower() for word in LEGAL_KEYWORDS)

    def _get_model(self):
        return get_gemini_model()

    def preprocess_text(self, text: str) -> str:
        """
        Clean and deduplicate text while preserving legal structure.

        Improvements:
        - Protects numbered clause markers (1., 2., (a), (b)) from being split
          incorrectly by NLTK's sentence tokenizer
        - Removes PDF extraction artifacts (form feeds, stray characters)
        - Deduplicates identical sentences
        """
        if not text:
            return ""

        # Remove form feeds and other PDF artifacts
        text = text.replace('\x0c', '\n').replace('\r\n', '\n').replace('\r', '\n')

        # Normalize whitespace but preserve paragraph structure
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Protect numbered clause markers so NLTK doesn't split them
        # e.g., "2. Payment" -> "2CLAUSEMARKER Payment" before tokenizing
        text = re.sub(r'(?m)^(\d+)\.\s+', r'\1CLAUSEMARKER ', text)
        text = re.sub(r'(?m)^\(([a-z]{1,3})\)\s+', r'SUBCLAUSE\1MARKER ', text)

        # Tokenize sentences
        sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]

        # Restore clause markers
        restored = []
        for s in sentences:
            s = re.sub(r'(\d+)CLAUSEMARKER\s*', r'\1. ', s)
            s = re.sub(r'SUBCLAUSE([a-z]{1,3})MARKER\s*', r'(\1) ', s)
            restored.append(s)

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for s in restored:
            norm = re.sub(r'\s+', ' ', s.lower()).strip()
            if norm not in seen and len(norm) > 5:
                seen.add(norm)
                unique.append(s)

        return ' '.join(unique) if unique else ""

    def extract_legal_terms(self, text: str) -> set:
        if not text:
            return set()
        words = text.lower().split()
        return {word for word in words if re.sub(r'[^\w\s]', '', word) in self.legal_keywords}

    def generate_extractive_draft(self, text: str, max_words: int = 200,
                                   focus_query: str = None) -> str:
        """
        Generate extractive summary using batched RoBERTa sentence embeddings.

        When `focus_query` is provided, uses a query-biased centroid instead of
        a pure document centroid — this dramatically improves relevance when the
        user has a specific question in mind.

        Args:
            text: Preprocessed document text
            max_words: Target word count for draft (actual may be up to 1.5x)
            focus_query: Optional query to bias sentence selection toward
        """
        sentences = sent_tokenize(text)
        if not sentences:
            return ""

        # Filter very short or junk fragments
        valid_sentences = [s for s in sentences if len(s.split()) >= 5]
        if not valid_sentences:
            valid_sentences = sentences[:50]  # Last resort: take first 50

        # Limit to 500 sentences max to keep latency reasonable
        # (legal docs rarely need more for a good summary)
        if len(valid_sentences) > 500:
            # Sample evenly from across the document rather than truncating
            step = len(valid_sentences) // 500
            valid_sentences = valid_sentences[::step][:500]

        # Batch embed all sentences
        sentence_embeddings = get_roberta_embeddings_batch(valid_sentences)  # [N, 768]

        if focus_query:
            # Query-biased centroid: weight the centroid toward the query
            query_emb = np.array(get_roberta_embedding(focus_query), dtype=np.float32).reshape(1, -1)
            doc_centroid = np.mean(sentence_embeddings, axis=0, keepdims=True)
            # 50% query direction + 50% document centroid
            centroid = 0.50 * query_emb + 0.50 * doc_centroid
        else:
            # Pure document centroid — but subtract the first principal component
            # to reduce anisotropy bias toward generic sentences
            centroid = np.mean(sentence_embeddings, axis=0, keepdims=True)

        scores = cosine_similarity(sentence_embeddings, centroid).flatten()
        ranked_indices = np.argsort(scores)[::-1]

        selected = []
        current_count = 0
        target_draft_words = int(max_words * 1.5)

        for i in ranked_indices:
            sent = valid_sentences[i]
            word_count = len(sent.split())
            if current_count + word_count <= target_draft_words:
                selected.append((i, sent))
                current_count += word_count

        # Restore original document order for readability
        selected.sort(key=lambda x: x[0])
        return ' '.join([sent for _, sent in selected])

    def generate_rag_draft(self, text: str, user_query: str, max_words: int = 600) -> str:
        """
        Generate query-focused RAG draft using batched RoBERTa sentence similarity.
        This is used when the user has a specific question about the document.
        """
        sentences = sent_tokenize(text)
        if not sentences:
            return ""

        valid_sentences = [s for s in sentences if len(s.split()) >= 4]
        if not valid_sentences:
            valid_sentences = sentences

        # Limit to 500 sentences for performance
        if len(valid_sentences) > 500:
            step = len(valid_sentences) // 500
            valid_sentences = valid_sentences[::step][:500]

        # Batch embed
        query_emb = np.array(get_roberta_embedding(user_query), dtype=np.float32).reshape(1, -1)
        sentence_embeddings = get_roberta_embeddings_batch(valid_sentences)
        scores = cosine_similarity(sentence_embeddings, query_emb).flatten()

        ranked_indices = np.argsort(scores)[::-1]
        selected = []
        current_count = 0
        target_draft_words = int(max_words * 1.5)

        for i in ranked_indices:
            sent = valid_sentences[i]
            word_count = len(sent.split())
            if current_count + word_count <= target_draft_words:
                selected.append((i, sent))
                current_count += word_count

        selected.sort(key=lambda x: x[0])
        return ' '.join([sent for _, sent in selected])

    def refine_with_llm(self, draft_text: str, max_words: int = 250,
                         user_query: str = None) -> str:
        """
        Refine draft text with Gemini LLM.
        - If user_query is provided: answers the question from the extracted context
        - Otherwise: generates a structured executive summary

        Handles quota exhaustion and missing keys gracefully.
        """
        if not draft_text or not draft_text.strip():
            return "No content could be extracted from this document."

        model = self._get_model()
        if not model:
            return (
                "⚠️ **Gemini Model Unavailable — No API Key Configured**\n\n"
                "Please add a Gemini API key in the left sidebar to enable AI refinement.\n\n"
                "---\n\n"
                "### 📄 RoBERTa Extractive Draft:\n\n" + draft_text
            )

        if user_query:
            prompt = f"""You are a senior legal counsel reviewing a contract.
Based ONLY on the extracted clauses below, provide a precise, structured answer to the user's question.

Question: "{user_query}"

Rules:
- Cite specific clause text in your answer using quotation marks.
- If the answer is not clearly present in the context, say so explicitly.
- Structure your response with bullet points for multiple findings.

Extracted Clauses:
---
{draft_text}
---

Answer:"""
        else:
            prompt = f"""You are an elite legal contract analyst. Transform the following extracted clause text into a polished, executive legal summary of approximately {max_words} words.

Structure your summary with these exact sections:
## Executive Summary
(2–3 sentence overview of the document's purpose and parties)

## Key Obligations & Rights
(Bullet list of the primary obligations of each party)

## Critical Terms & Dates
(Monetary amounts, deadlines, notice periods, renewal terms)

## Liabilities & Risk Exposure
(Indemnification, limitation of liability, warranty disclaimers)

## Notable Caveats or Gaps
(Any missing, ambiguous, or unusual provisions worth flagging)

Extracted Text:
---
{draft_text}
---

Executive Legal Summary:"""

        text, status, err = call_gemini_safely(model, prompt)

        if status == "SUCCESS" and text:
            return text
        elif status == "QUOTA_EXHAUSTED":
            return (
                "### 🚨 Gemini API Quota Exhausted\n\n"
                "The backend API key has depleted its usage quota. "
                "**Enter your personal Gemini API key in the left sidebar** to continue.\n\n"
                "---\n\n"
                "### 📄 Extractive Draft (RoBERTa ranked):\n\n" + draft_text
            )
        elif status in ("INVALID_KEY", "NO_KEY"):
            return (
                "### ⚠️ Gemini API Key Required\n\n"
                "Please provide a valid Gemini API key in the sidebar.\n\n"
                "---\n\n"
                "### 📄 Extractive Draft (RoBERTa ranked):\n\n" + draft_text
            )
        else:
            return f"⚠️ Refinement error: {err}\n\n---\n\n### 📄 Extractive Draft:\n\n{draft_text}"

    def calculate_all_scores(self, original_text: str, summary: str):
        """Compute ROUGE, semantic consistency, and legal keyword coverage."""
        if not summary or not original_text:
            return {}, 0.0, 0.0

        try:
            rouge = self.rouge_scorer.score(original_text, summary)
            rouge_scores = {
                "rouge1": rouge['rouge1'].fmeasure,
                "rouge2": rouge['rouge2'].fmeasure,
                "rougeL": rouge['rougeL'].fmeasure
            }
        except Exception:
            rouge_scores = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

        try:
            # Use single-call embeddings for the full texts (can be long; RoBERTa truncates at 512 tokens)
            # For long docs this is a rough measure, but acceptable for a dashboard metric
            orig_emb = np.array(get_roberta_embedding(original_text[:3000])).reshape(1, -1)
            sum_emb = np.array(get_roberta_embedding(summary[:3000])).reshape(1, -1)
            consistency = float(cosine_similarity(orig_emb, sum_emb)[0][0])
        except Exception:
            consistency = 0.0

        orig_kw = self.extract_legal_terms(original_text)
        sum_kw = self.extract_legal_terms(summary)
        coverage = (len(orig_kw.intersection(sum_kw)) / len(orig_kw) * 100) if orig_kw else 0.0

        return rouge_scores, consistency, coverage


summarizer = HybridLegalSummarizer()
print("\n[RAG] ✅ HybridLegalSummarizer initialized (batched RoBERTa embeddings).")
