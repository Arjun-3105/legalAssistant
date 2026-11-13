# ragPipeline.py — Hybrid Legal Summarizer with RoBERTa + Gemini

"""
Core RAG pipeline using fine-tuned/base RoBERTa for embeddings and Gemini for refinement.
Supports both extractive summarization and query-driven retrieval.
"""

import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import re
import io
import nltk
import torch
import numpy as np
import pdfplumber
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from nltk.tokenize import sent_tokenize

from api_manager import get_gemini_model, call_gemini_safely, get_active_api_key
from vectorStore import roberta_tokenizer, roberta_model, get_roberta_embedding

# Download NLTK data
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
    "parole", "dismissal", "overrule", "uphold", "vacate", "enforcement"
]


class HybridLegalSummarizer:
    """
    Hybrid summarizer combining RoBERTa-based extractive ranking with
    Gemini-based abstractive refinement.
    """

    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.legal_keywords = set(word.lower() for word in LEGAL_KEYWORDS)

    def _get_model(self):
        """Dynamically retrieve Gemini model instance with the active key."""
        return get_gemini_model()

    def preprocess_text(self, text):
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
        seen = set()
        unique_sentences = [s for s in sentences if not (s.lower() in seen or seen.add(s.lower()))]
        return ' '.join(unique_sentences) if unique_sentences else ""

    def extract_legal_terms(self, text):
        if not text:
            return set()
        words = text.lower().split()
        return {word for word in words if re.sub(r'[^\w\s]', '', word) in self.legal_keywords}

    def generate_extractive_draft(self, text, max_words=200):
        """Generate extractive summary using RoBERTa sentence embeddings + centroid ranking."""
        sentences = sent_tokenize(text)
        if not sentences:
            return ""
        
        # Filter very short fragments
        valid_sentences = [s for s in sentences if len(s.split()) >= 4]
        if not valid_sentences:
            valid_sentences = sentences

        sentence_embeddings = np.array([get_roberta_embedding(sent) for sent in valid_sentences])
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

        # Maintain original document order
        selected.sort(key=lambda x: x[0])
        return ' '.join([sent for _, sent in selected])

    def generate_rag_draft(self, text, user_query, max_words=600):
        """Generate RAG retrieval draft using RoBERTa query-sentence similarity."""
        sentences = sent_tokenize(text)
        if not sentences:
            return ""
        query_embedding = np.array(get_roberta_embedding(user_query)).reshape(1, -1)
        sentence_embeddings = np.array([get_roberta_embedding(sent) for sent in sentences])
        scores = cosine_similarity(sentence_embeddings, query_embedding).flatten()
        
        ranked_indices = np.argsort(scores)[::-1]
        selected = []
        current_count = 0
        target_draft_words = int(max_words * 1.5)
        for i in ranked_indices:
            sent = sentences[i]
            word_count = len(sent.split())
            if current_count + word_count <= target_draft_words:
                selected.append((i, sent))
                current_count += word_count
        selected.sort(key=lambda x: x[0])
        return ' '.join([sent for _, sent in selected])

    def refine_with_llm(self, draft_text, max_words=250, user_query=None):
        """Refine draft text with Gemini LLM, handling quota errors gracefully."""
        if not draft_text:
            return "No content provided to summarize."

        model = self._get_model()
        if not model:
            return (
                "⚠️ **Gemini Model Unavailable (No API Key Configured)**\n\n"
                "Please configure your Gemini API Key in the sidebar to enable AI abstractive refinement.\n\n"
                "### 📄 RoBERTa Extractive Summary Draft:\n\n" + draft_text
            )

        if user_query:
            prompt = f"""You are an expert legal counsel. Based strictly on the provided legal text, provide a concise, accurate answer to the user's question.

Question: "{user_query}"

Context:
---
{draft_text}
---

Answer:"""
        else:
            prompt = f"""You are an elite legal contract counsel. Rewrite and condense the following extractive legal draft into an executive, highly polished legal summary of approximately {max_words} words.

Requirements:
- Organize with clear headings: **Executive Summary**, **Key Obligations & Terms**, and **Important Liabilities / Caveats**.
- Retain all critical dates, monetary amounts, and legal definitions.
- Write in professional legal style.

Extractive Draft:
---
{draft_text}
---

Polished Legal Summary:"""

        text, status, err = call_gemini_safely(model, prompt)
        if status == "SUCCESS" and text:
            return text
        elif status == "QUOTA_EXHAUSTED":
            return (
                "### 🚨 Gemini API Quota Exhausted (Out of Credits)\n\n"
                "The backend Gemini API key has depleted its rate limit or free-tier quota. "
                "**Enter your personal Gemini API key in the left sidebar to enable abstractive refinement.**\n\n"
                "--- \n\n"
                "### 📄 Extractive Summary Draft (RoBERTa Centroid Ranking):\n\n" + draft_text
            )
        elif status in ("INVALID_KEY", "NO_KEY"):
            return (
                "### ⚠️ Gemini API Key Required\n\n"
                "Please provide a valid Gemini API key in the sidebar to enable AI summary refinement.\n\n"
                "--- \n\n"
                "### 📄 Extractive Summary Draft (RoBERTa Centroid Ranking):\n\n" + draft_text
            )
        else:
            return f"Refinement failed: {err}\n\n### 📄 Extractive Summary:\n\n{draft_text}"

    def calculate_all_scores(self, original_text, summary):
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
            orig_emb = np.array(get_roberta_embedding(original_text)).reshape(1, -1)
            sum_emb = np.array(get_roberta_embedding(summary)).reshape(1, -1)
            consistency = float(cosine_similarity(orig_emb, sum_emb)[0][0])
        except Exception:
            consistency = 0.0

        orig_kw = self.extract_legal_terms(original_text)
        sum_kw = self.extract_legal_terms(summary)
        coverage = (len(orig_kw.intersection(sum_kw)) / len(orig_kw) * 100) if orig_kw else 0.0
        return rouge_scores, consistency, coverage


summarizer = HybridLegalSummarizer()
print("\n[RAG] ✅ HybridLegalSummarizer initialized with RoBERTa embeddings.")
