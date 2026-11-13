"""
agent.py — Agent-style task decomposition for legal question answering.

Implements a multi-step reasoning agent that:
1. Decomposes & expands queries with domain-specific legal terminology
2. Retrieves relevant context using hybrid (semantic + lexical) search
3. Cross-references and structures retrieved evidence
4. Synthesizes a grounded, citation-backed legal answer with clear analysis
5. Intelligently detects API quota exhaustion and provides actionable fallback

Uses Gemini (with dynamic user/backend key support) as the backbone LLM.
"""

import os
import time
import json
import re
from typing import Optional, List, Dict, Any

from api_manager import get_gemini_model, call_gemini_safely, get_active_api_key

# ─── Query Type Definitions ──────────────────────────────────────────────────
QUERY_TYPES = {
    "factual_lookup": "Direct factual question about specific clauses, terms, or provisions",
    "comparison": "Compare clauses, terms, or provisions across documents",
    "risk_analysis": "Identify risks, liabilities, or problematic clauses",
    "summarization": "Summarize sections, documents, or specific topics",
    "reasoning": "Complex legal reasoning requiring multi-step analysis"
}

# ─── Domain Legal Terminology Map for Query Expansion ───────────────────────
LEGAL_CONCEPT_MAP = {
    "sue": "indemnification indemnity liability damages hold harmless litigation breach",
    "sued": "indemnification indemnity liability damages hold harmless litigation breach",
    "lawsuit": "indemnification litigation dispute resolution arbitration liability",
    "fire": "termination terminate severance without cause notice period dismissal",
    "fired": "termination terminate severance without cause notice period dismissal",
    "quit": "termination resignation voluntary notice period",
    "cancel": "termination terminate expiration cancellation notice right to cancel",
    "pay": "compensation payment terms consideration fee invoice late charge interest",
    "money": "compensation remuneration consideration fees penalties liquidated damages",
    "salary": "base salary compensation bonus benefits overtime",
    "secret": "confidentiality non-disclosure proprietary information trade secrets",
    "secrets": "confidentiality non-disclosure proprietary information trade secrets",
    "leak": "confidentiality breach unauthorized disclosure remedies",
    "court": "governing law jurisdiction venue choice of law dispute resolution arbitration",
    "dispute": "dispute resolution arbitration mediation governing law venue",
    "patent": "intellectual property patent copyright ownership work made for hire assignment",
    "copyright": "intellectual property copyright ownership work made for hire assignment license",
    "software": "intellectual property software license source code ownership delivery",
    "penalty": "liquidated damages liability penalty breach late fee",
    "insurance": "insurance coverage commercial general liability policy indemnity",
    "guarantee": "warranty representation covenant guarantee disclaimer",
    "warranty": "warranties representations covenants disclaimer as-is",
    "transfer": "assignment delegation successors assigns change of control",
}


class LegalAgent:
    """
    Agent with task decomposition for legal document QA.

    Decomposes complex queries into sub-tasks (classify → expand & retrieve → analyze → synthesize),
    maintaining a transparent execution trace and robust quota recovery.
    """

    def __init__(self, document_store):
        """
        Args:
            document_store: A DocumentStore instance for retrieval.
        """
        self.store = document_store
        self.execution_trace = []

    def _get_model(self):
        """Dynamically fetch the Gemini model instance with the currently active key."""
        return get_gemini_model()

    def _log_step(self, step_name: str, detail: str, result: str = ""):
        """Log a reasoning step for transparency."""
        entry = {
            "step": step_name,
            "detail": detail,
            "result": result,
            "timestamp": time.time()
        }
        self.execution_trace.append(entry)
        return entry

    # ─── Step 1: Query Classification & Expansion ───────────────────────────

    def _classify_query(self, query: str) -> str:
        """Classify the query into a task type using LLM or rule-based fallback."""
        model = self._get_model()
        if model:
            prompt = f"""Classify the following legal question into exactly ONE of these categories:
- factual_lookup: Direct factual question about specific clauses, terms, or provisions
- comparison: Compare clauses, terms, or provisions across documents
- risk_analysis: Identify risks, liabilities, or problematic clauses
- summarization: Summarize sections, documents, or specific topics
- reasoning: Complex legal reasoning requiring multi-step analysis

Question: "{query}"

Respond with ONLY the category name, nothing else."""

            text, status, err = call_gemini_safely(model, prompt)
            if status == "SUCCESS" and text:
                category = text.strip().lower().replace(" ", "_")
                if category in QUERY_TYPES:
                    self._log_step(
                        "classify",
                        f"Determined query type for: '{query}'",
                        f"Category: {category} — {QUERY_TYPES.get(category, '')}"
                    )
                    return category

        # Rule-based fallback if LLM is unavailable or quota exhausted
        q_lower = query.lower()
        if any(w in q_lower for w in ["compare", "difference", "versus", "vs", "between"]):
            category = "comparison"
        elif any(w in q_lower for w in ["risk", "liability", "liable", "penalty", "danger", "breach"]):
            category = "risk_analysis"
        elif any(w in q_lower for w in ["summarize", "summary", "overview", "brief"]):
            category = "summarization"
        elif any(w in q_lower for w in ["why", "can they", "am i allowed", "valid", "enforceable", "consequence"]):
            category = "reasoning"
        else:
            category = "factual_lookup"

        self._log_step(
            "classify",
            f"Determined query type (heuristic) for: '{query}'",
            f"Category: {category} — {QUERY_TYPES.get(category, '')}"
        )
        return category

    def _expand_query(self, query: str) -> List[str]:
        """Expand user query with legal terminology to dramatically boost recall."""
        words = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        expansions = []
        for w in words:
            if w in LEGAL_CONCEPT_MAP:
                expansions.append(LEGAL_CONCEPT_MAP[w])

        queries = [query]
        if expansions:
            combined_legal_terms = " ".join(set(" ".join(expansions).split()))
            queries.append(f"{query} {combined_legal_terms}")
            self._log_step(
                "query_expansion",
                f"Expanded query with legal domain terms: '{query}'",
                f"Added legal concepts: {combined_legal_terms[:80]}..."
            )
        return queries

    # ─── Step 2: Hybrid Retrieval ────────────────────────────────────────────

    def _retrieve_context(self, query: str, query_type: str, filter_file: Optional[str] = None, top_k: int = 10) -> List[Dict]:
        """Retrieve relevant chunks from the vector store using hybrid retrieval and query expansion."""
        retrieval_k = {
            "factual_lookup": 6,
            "comparison": 14,
            "risk_analysis": 12,
            "summarization": 15,
            "reasoning": 10
        }.get(query_type, top_k)

        query_variants = self._expand_query(query)
        seen_ids = set()
        all_results = []

        for q_var in query_variants:
            results = self.store.query(q_var, top_k=retrieval_k, filter_file=filter_file)
            for r in results:
                chunk_key = f"{r['source_file']}::{r['page']}::{r['chunk_index']}"
                if chunk_key not in seen_ids:
                    seen_ids.add(chunk_key)
                    all_results.append(r)

        # Sort combined results by score
        all_results.sort(key=lambda x: x["score"], reverse=True)
        final_results = all_results[:retrieval_k]

        docs_count = len(set(r['source_file'] for r in final_results)) if final_results else 0
        scope_desc = f" (scoped to '{filter_file}')" if filter_file else " (across corpus)"
        
        self._log_step(
            "retrieve",
            f"Searched vector store{scope_desc} for query type '{query_type}'",
            f"Retrieved {len(final_results)} top chunks from {docs_count} document(s)"
        )
        return final_results

    # ─── Step 3: Analysis & Cross-Referencing ─────────────────────────────────

    def _analyze_context(self, query: str, query_type: str, retrieved: List[Dict]) -> Dict:
        """
        Analyze and organize retrieved chunks by document, page, and score.
        """
        if not retrieved:
            self._log_step("analyze", "No context retrieved", "Corpus search returned 0 results")
            return {"context_blocks": [], "analysis_notes": "No relevant documents or clauses found in corpus."}

        # Group by source document
        by_document = {}
        for chunk in retrieved:
            fname = chunk["source_file"]
            if fname not in by_document:
                by_document[fname] = []
            by_document[fname].append(chunk)

        # Build context blocks with explicit citations
        context_blocks = []
        for fname, chunks in by_document.items():
            chunks.sort(key=lambda c: (c["page"], c["chunk_index"]))
            for chunk in chunks:
                context_blocks.append({
                    "text": chunk["text"],
                    "citation": f"[{fname}, Page {chunk['page']}]",
                    "score": chunk["score"],
                    "source_file": fname,
                    "page": chunk["page"]
                })

        context_blocks.sort(key=lambda b: b["score"], reverse=True)

        analysis_notes = f"Analyzed {len(context_blocks)} clauses from {len(by_document)} document(s)."
        if query_type == "comparison":
            analysis_notes += f" Comparing documents: {', '.join(by_document.keys())}."
        elif query_type == "risk_analysis":
            analysis_notes += " Focus: liabilities, termination triggers, warranties, indemnities."

        self._log_step(
            "analyze",
            f"Cross-referenced {len(retrieved)} chunks across {len(by_document)} documents",
            analysis_notes
        )

        return {"context_blocks": context_blocks, "analysis_notes": analysis_notes}

    # ─── Step 4: Synthesis ───────────────────────────────────────────────────

    def _synthesize_answer(self, query: str, query_type: str, analysis: Dict) -> str:
        """
        Use Gemini to synthesize a grounded, highly structured legal answer.
        Handles API quota exhaustion gracefully by delivering structured evidence.
        """
        blocks = analysis.get("context_blocks", [])
        if not blocks:
            return (
                "ℹ️ **No Relevant Information Found**\n\n"
                "The documents in your indexed library do not appear to contain provisions matching your query. "
                "Please verify that the target document is uploaded and ingested into the corpus."
            )

        model = self._get_model()

        # Build the context string with citations
        context_parts = []
        for block in blocks[:10]:
            context_parts.append(f"Source: {block['citation']} (Relevance Score: {block['score']})\n{block['text']}")
        context_str = "\n\n---\n\n".join(context_parts)

        # Task-specific synthesis instructions
        type_instructions = {
            "factual_lookup": "Provide a direct, authoritative answer. State the specific terms, dates, dollar amounts, or provisions, and cite the document and page.",
            "comparison": "Structure your answer with a comparative breakdown between the documents. Highlight key similarities, differences, and practical consequences.",
            "risk_analysis": "Identify all potential liabilities, unfavorable terms, exposure points, and un-capped obligations. Categorize them by severity (High/Medium/Low).",
            "summarization": "Provide an executive summary of the relevant clauses, followed by bulleted key provisions and operational takeaways.",
            "reasoning": "Provide step-by-step legal analysis walking through how the cited provisions apply to the question. Clearly address any conditions or exceptions."
        }
        instruction = type_instructions.get(query_type, type_instructions["factual_lookup"])

        prompt = f"""You are an elite legal AI counsel and contract analyst.
Answer the user's legal question accurately and thoroughly based on the provided context below.

Task-Specific Guidance:
{instruction}

Strict Response Rules:
1. Ground your response firmly in the provided context.
2. Structure your answer clearly with markdown headings, bullet points, or tables.
3. Always include citations in the format [Document_Name, Page X] for every claim or quote.
4. If certain details are omitted or silent in the context, explicitly point out what is missing or ambiguous.
5. Provide actionable legal clarity without generic disclaimers.

Question:
"{query}"

Retrieved Context:
{context_str}

Structured Legal Analysis & Answer:"""

        if model:
            answer_text, status, err_detail = call_gemini_safely(model, prompt)
            if status == "SUCCESS" and answer_text:
                self._log_step(
                    "synthesize",
                    f"Generated AI answer using {len(blocks[:10])} context blocks",
                    f"Answer length: {len(answer_text.split())} words"
                )
                return answer_text

            elif status == "QUOTA_EXHAUSTED":
                self._log_step("synthesize", "API quota exhausted during synthesis", "Showing grounded fallback evidence")
                return (
                    "### 🚨 Gemini API Quota Exhausted (Out of Credits)\n\n"
                    "The backend Gemini API key has depleted its rate limits or free-tier quota. "
                    "**To enable complete AI reasoning and synthesis, please enter your personal Gemini API key in the left sidebar.**\n\n"
                    "--- \n\n"
                    "### 📑 Retrieved Document Evidence (Matched via Hybrid Search):\n"
                    + self._format_fallback_evidence(blocks[:5])
                )

            elif status in ("INVALID_KEY", "NO_KEY"):
                self._log_step("synthesize", f"API key issue: {status}", "Showing grounded fallback evidence")
                return (
                    "### ⚠️ Gemini API Key Required\n\n"
                    "No valid Gemini API key is configured. Please add your Gemini API key in the left sidebar to generate full AI answers.\n\n"
                    "--- \n\n"
                    "### 📑 Retrieved Document Evidence:\n"
                    + self._format_fallback_evidence(blocks[:5])
                )

        # Fallback if model could not be initialized
        self._log_step("synthesize", "Fallback to extracted evidence", "Model unavailable")
        return (
            "### ⚠️ AI Synthesis Unavailable\n\n"
            "Unable to connect to the Gemini LLM. Please check your API credentials in the sidebar.\n\n"
            "--- \n\n"
            "### 📑 Extracted Relevant Clauses:\n"
            + self._format_fallback_evidence(blocks[:5])
        )

    def _format_fallback_evidence(self, blocks: List[Dict]) -> str:
        """Format retrieved chunks as clean markdown cards for fallback display."""
        output = []
        for i, b in enumerate(blocks, 1):
            output.append(
                f"**{i}. {b['citation']}** *(Relevance Match: {int(b['score'] * 100)}%)*\n\n"
                f"> {b['text']}\n"
            )
        return "\n".join(output)

    # ─── Main Entry Point ────────────────────────────────────────────────────

    def query(self, user_query: str, filter_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a legal query through the full agent pipeline.

        Args:
            user_query: The user's legal question
            filter_file: Optional filename to restrict search to a single document

        Returns:
            Dict containing answer, sources, steps, query_type, and confidence.
        """
        self.execution_trace = []

        self._log_step("start", f"Received query: '{user_query}'", "Initiating agent decomposition")

        # Step 1: Classify query
        query_type = self._classify_query(user_query)

        # Step 2: Retrieve context (with hybrid search & expansion)
        retrieved = self._retrieve_context(user_query, query_type, filter_file=filter_file)

        # Step 3: Analyze context
        analysis = self._analyze_context(user_query, query_type, retrieved)

        # Step 4: Synthesize grounded answer
        answer = self._synthesize_answer(user_query, query_type, analysis)

        # Collect unique citations
        sources = list(dict.fromkeys(b["citation"] for b in analysis["context_blocks"]))

        # Confidence heuristic
        if analysis["context_blocks"]:
            avg_score = sum(b["score"] for b in analysis["context_blocks"][:5]) / min(5, len(analysis["context_blocks"]))
            confidence = min(round(avg_score * 100, 1), 99.0)
        else:
            confidence = 0.0

        self._log_step("complete", "Agent pipeline finished", f"Confidence: {confidence}%")

        return {
            "answer": answer,
            "sources": sources,
            "steps": self.execution_trace,
            "query_type": query_type,
            "confidence": confidence,
            "chunks_retrieved": len(retrieved),
            "documents_referenced": len(set(r["source_file"] for r in retrieved)) if retrieved else 0
        }

    def query_without_rag(self, user_query: str) -> str:
        """Answer a query WITHOUT retrieval (pure LLM) — used for evaluation baseline."""
        model = self._get_model()
        if not model:
            return "Gemini LLM not available or credentials missing."

        prompt = f"""You are a legal AI assistant. Answer the following legal question to the best of your knowledge without access to external documents.

Question: "{user_query}"

Answer:"""

        text, status, err = call_gemini_safely(model, prompt)
        if status == "SUCCESS" and text:
            return text
        elif status == "QUOTA_EXHAUSTED":
            return "API quota exhausted. Please provide a personal API key in the sidebar."
        return f"Synthesis failed: {err}"
