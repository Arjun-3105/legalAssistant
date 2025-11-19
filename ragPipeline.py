# app.py

# --- 1. IMPORTS ---
import os
import re
import io
import nltk
import torch
import numpy as np
import pdfplumber
import gradio as gr
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from nltk.tokenize import sent_tokenize

# Download NLTK data during the build process
nltk.download('punkt_tab')
nltk.download('stopwords')
print(" NLTK data downloaded.")

# --- 2. MODEL LOADING & BACKEND CLASS ---

# --- IMPORTANT: LOAD API KEY FROM HF SECRETS ---
try:
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY secret not found!")
    genai.configure(api_key=GEMINI_API_KEY)
    print(" Gemini API configured.")
    GEMINI_AVAILABLE = True
except Exception as e:
    print(f" Could not configure Gemini: {e}")
    GEMINI_AVAILABLE = False

# Load LegalBERT
print("\nLoading LegalBERT model... (This may take a minute on first startup)")
try:
    legalbert_tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    legalbert_model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
    print(" LegalBERT model loaded successfully.")
except Exception as e:
    print(f" Error loading LegalBERT: {e}")

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
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.legal_keywords = set(word.lower() for word in LEGAL_KEYWORDS)
        if GEMINI_AVAILABLE:
            self.refinement_model = genai.GenerativeModel('models/gemini-2.5-flash')

    def get_legalbert_embedding(self, text):
        inputs = legalbert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = legalbert_model(**inputs)
        return torch.mean(outputs.last_hidden_state, dim=1).squeeze().numpy()

    def preprocess_text(self, text):
        text = re.sub(r'\s+', ' ', text)
        sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
        seen = set()
        unique_sentences = [s for s in sentences if not (s.lower() in seen or seen.add(s.lower()))]
        return ' '.join(unique_sentences) if unique_sentences else None

    def extract_legal_terms(self, text):
        words = text.lower().split()
        return {word for word in words if re.sub(r'[^\w\s]', '', word) in self.legal_keywords}

    def generate_extractive_draft(self, text, max_words=200):
        sentences = sent_tokenize(text)
        if not sentences: return ""
        sentence_embeddings = np.array([self.get_legalbert_embedding(sent) for sent in sentences])
        centroid = np.mean(sentence_embeddings, axis=0)
        scores = [cosine_similarity(emb.reshape(1, -1), centroid.reshape(1, -1))[0][0] for emb in sentence_embeddings]
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
        return ' '.join([sent for i, sent in selected])

    def generate_rag_draft(self, text, user_query, max_words=600):
        sentences = sent_tokenize(text)
        if not sentences: return ""
        query_embedding = self.get_legalbert_embedding(user_query)
        sentence_embeddings = np.array([self.get_legalbert_embedding(sent) for sent in sentences])
        scores = [cosine_similarity(emb.reshape(1, -1), query_embedding.reshape(1, -1))[0][0] for emb in sentence_embeddings]
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
        return ' '.join([sent for i, sent in selected])

    def refine_with_llm(self, draft_text, max_words, user_query=None):
        if not GEMINI_AVAILABLE or not draft_text: return "Skipping refinement stage."
        if user_query:
            prompt = f"""Based ONLY on the provided context, answer the user's question concisely. Question: "{user_query}" Context: --- {draft_text} --- Answer:"""
        else:
            prompt = f"""Rewrite and condense the following draft into a polished legal summary of approximately {max_words} words. Draft: --- {draft_text} --- Summary:"""
        try:
            response = self.refinement_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Refinement failed. Error: {e}"

    def calculate_all_scores(self, original_text, summary):
        if not summary or not original_text: return {}, 0.0, 0.0
        rouge = self.rouge_scorer.score(original_text, summary)
        rouge_scores = {"rouge1": rouge['rouge1'].fmeasure, "rouge2": rouge['rouge2'].fmeasure, "rougeL": rouge['rougeL'].fmeasure}
        orig_emb = self.get_legalbert_embedding(original_text).reshape(1, -1)
        sum_emb = self.get_legalbert_embedding(summary).reshape(1, -1)
        consistency = cosine_similarity(orig_emb, sum_emb)[0][0]
        orig_kw = self.extract_legal_terms(original_text)
        sum_kw = self.extract_legal_terms(summary)
        coverage = (len(orig_kw.intersection(sum_kw)) / len(orig_kw) * 100) if orig_kw else 0
        return rouge_scores, consistency, coverage

summarizer = HybridLegalSummarizer()
print("\n Backend class is defined and ready.")
