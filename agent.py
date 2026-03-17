"""
agent.py — Agent-style task decomposition for legal question answering.

Implements a multi-step reasoning agent that:
1. Classifies the query type
2. Retrieves relevant context from the vector store
3. Analyzes and cross-references retrieved evidence
4. Synthesizes a grounded, citation-backed answer

Uses Gemini as the backbone LLM for reasoning and synthesis.
"""

import os
from dotenv import load_dotenv
load_dotenv()
import time
import json
import google.generativeai as genai

# ─── Gemini Configuration ────────────────────────────────────────────────────
try:
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found")
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_AVAILABLE = True
    print("[Agent] ✅ Gemini API configured.")
except Exception as e:
    print(f"[Agent] ❌ Could not configure Gemini: {e}")
    GEMINI_AVAILABLE = False


# ─── Query Type Definitions ──────────────────────────────────────────────────
QUERY_TYPES = {
    "factual_lookup": "Direct factual question about specific clauses, terms, or provisions",
    "comparison": "Compare clauses, terms, or provisions across documents",
    "risk_analysis": "Identify risks, liabilities, or problematic clauses",
    "summarization": "Summarize sections, documents, or specific topics",
    "reasoning": "Complex legal reasoning requiring multi-step analysis"
}


class LegalAgent:
    """
    Agent with task decomposition for legal document QA.

    Decomposes complex queries into sub-tasks (classify → retrieve → analyze → synthesize),
    maintaining a transparent execution trace.
    """

    def __init__(self, document_store):
        """
        Args:
            document_store: A DocumentStore instance for retrieval.
        """
        self.store = document_store
        self.llm = genai.GenerativeModel('models/gemini-2.0-flash') if GEMINI_AVAILABLE else None
        self.execution_trace = []

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

    # ─── Step 1: Query Classification ────────────────────────────────────────

    def _classify_query(self, query: str) -> str:
        """Classify the query into a task type using the LLM."""
        if not self.llm:
            return "factual_lookup"

        prompt = f"""Classify the following legal question into exactly ONE of these categories:
- factual_lookup: Direct factual question about specific clauses, terms, or provisions
- comparison: Compare clauses, terms, or provisions across documents
- risk_analysis: Identify risks, liabilities, or problematic clauses
- summarization: Summarize sections, documents, or specific topics
- reasoning: Complex legal reasoning requiring multi-step analysis

Question: "{query}"

Respond with ONLY the category name, nothing else."""

        try:
            response = self.llm.generate_content(prompt)
            category = response.text.strip().lower().replace(" ", "_")
            if category not in QUERY_TYPES:
                category = "factual_lookup"
        except Exception:
            category = "factual_lookup"

        self._log_step(
            "classify",
            f"Determined query type for: '{query}'",
            f"Category: {category} — {QUERY_TYPES.get(category, '')}"
        )
        return category

    # ─── Step 2: Retrieval ───────────────────────────────────────────────────

    def _retrieve_context(self, query: str, query_type: str, top_k: int = 10) -> list[dict]:
        """Retrieve relevant chunks from the vector store, adapting strategy to query type."""

        # Adjust retrieval depth based on query type
        retrieval_k = {
            "factual_lookup": 5,
            "comparison": 12,
            "risk_analysis": 10,
            "summarization": 15,
            "reasoning": 10
        }.get(query_type, top_k)

        results = self.store.query(query, top_k=retrieval_k)

        self._log_step(
            "retrieve",
            f"Searched vector store with k={retrieval_k} for query type '{query_type}'",
            f"Retrieved {len(results)} chunks from {len(set(r['source_file'] for r in results))} documents"
        )
        return results

    # ─── Step 3: Analysis ────────────────────────────────────────────────────

    def _analyze_context(self, query: str, query_type: str, retrieved: list[dict]) -> dict:
        """
        Analyze retrieved chunks based on query type.
        For risk_analysis: extract risky clauses.
        For comparison: group by document.
        For all: rank by relevance.
        """
        if not retrieved:
            self._log_step("analyze", "No context retrieved", "Empty result set")
            return {"context_blocks": [], "analysis_notes": "No relevant documents found."}

        # Group by source document
        by_document = {}
        for chunk in retrieved:
            fname = chunk["source_file"]
            if fname not in by_document:
                by_document[fname] = []
            by_document[fname].append(chunk)

        # Build context blocks with citations
        context_blocks = []
        for fname, chunks in by_document.items():
            # Sort by page/chunk order within each document
            chunks.sort(key=lambda c: (c["page"], c["chunk_index"]))
            for chunk in chunks:
                context_blocks.append({
                    "text": chunk["text"],
                    "citation": f"[{fname}, Page {chunk['page']}]",
                    "score": chunk["score"]
                })

        # Sort all blocks by relevance score
        context_blocks.sort(key=lambda b: b["score"], reverse=True)

        analysis_notes = f"Analyzed {len(context_blocks)} text blocks from {len(by_document)} document(s)."

        if query_type == "comparison":
            analysis_notes += f" Documents compared: {', '.join(by_document.keys())}."
        elif query_type == "risk_analysis":
            analysis_notes += " Focus: identifying liability, indemnity, and risk clauses."

        self._log_step(
            "analyze",
            f"Cross-referenced {len(retrieved)} chunks across {len(by_document)} documents",
            analysis_notes
        )

        return {"context_blocks": context_blocks, "analysis_notes": analysis_notes}

    # ─── Step 4: Synthesis ───────────────────────────────────────────────────

    def _synthesize_answer(self, query: str, query_type: str, analysis: dict) -> str:
        """Use the LLM to synthesize a grounded answer from analyzed context."""
        if not self.llm:
            # Fallback: return concatenated context
            return "\n\n".join([b["text"] for b in analysis["context_blocks"][:5]])

        # Build the context string with citations
        context_parts = []
        for block in analysis["context_blocks"][:10]:  # Limit to top 10 for prompt size
            context_parts.append(f"{block['citation']}:\n{block['text']}")

        context_str = "\n\n---\n\n".join(context_parts)

        # Task-specific synthesis prompts
        type_instructions = {
            "factual_lookup": "Answer the question directly and concisely, citing the specific source documents.",
            "comparison": "Compare and contrast the relevant provisions across documents, highlighting key differences.",
            "risk_analysis": "Identify all potential risks, liabilities, and problematic clauses. Rate severity where possible.",
            "summarization": "Provide a comprehensive yet concise summary of the relevant sections.",
            "reasoning": "Provide a step-by-step legal analysis, citing each source that supports your reasoning."
        }

        instruction = type_instructions.get(query_type, type_instructions["factual_lookup"])

        prompt = f"""You are a legal AI assistant. Answer the following question based ONLY on the provided context.
{instruction}

Always cite your sources using the format [Document, Page X].
If the context does not contain enough information to answer, say so explicitly.

Question: "{query}"

Context:
{context_str}

Answer:"""

        try:
            response = self.llm.generate_content(prompt)
            answer = response.text.strip()
        except Exception as e:
            answer = f"Synthesis failed: {e}"

        self._log_step(
            "synthesize",
            f"Generated answer using {len(analysis['context_blocks'])} context blocks",
            f"Answer length: {len(answer.split())} words"
        )
        return answer

    # ─── Main Entry Point ────────────────────────────────────────────────────

    def query(self, user_query: str) -> dict:
        """
        Process a legal query through the full agent pipeline.

        Returns:
            {
                "answer": str,
                "sources": list of citation strings,
                "steps": list of execution trace entries,
                "query_type": str,
                "confidence": float
            }
        """
        self.execution_trace = []  # Reset trace

        self._log_step("start", f"Received query: '{user_query}'", "Initiating task decomposition")

        # Step 1: Classify
        query_type = self._classify_query(user_query)

        # Step 2: Retrieve
        retrieved = self._retrieve_context(user_query, query_type)

        # Step 3: Analyze
        analysis = self._analyze_context(user_query, query_type, retrieved)

        # Step 4: Synthesize
        answer = self._synthesize_answer(user_query, query_type, analysis)

        # Collect unique sources
        sources = list(set(b["citation"] for b in analysis["context_blocks"]))

        # Simple confidence heuristic based on retrieval scores
        if analysis["context_blocks"]:
            avg_score = sum(b["score"] for b in analysis["context_blocks"]) / len(analysis["context_blocks"])
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
        """
        Answer a query WITHOUT retrieval (pure LLM) — used for evaluation baseline.
        """
        if not self.llm:
            return "LLM not available."

        prompt = f"""You are a legal AI assistant. Answer the following legal question to the best of your knowledge.

Question: "{user_query}"

Answer:"""

        try:
            response = self.llm.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error: {e}"
