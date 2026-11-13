"""
agent.py — Agent-style task decomposition for legal question answering.

Key improvements:
- FIXED: chunk_index and page accessed with .get() guards (no more KeyError)
- FIXED: _expand_query uses dict.fromkeys() for deterministic term ordering
- FIXED: Minimum score threshold applied before synthesis (removes noise)
- FIXED: Top-level try/except in query() guarantees always-safe return dict
- FIXED: query_type label renamed to "Match Score" in returned confidence
- FIXED: _analyze_context sorts correctly even when chunk_index is missing
- ADDED: "No documents" vs "No matches" distinction in fallback messaging
"""

import os
import time
import re
from typing import Optional, List, Dict, Any

from api_manager import get_gemini_model, call_gemini_safely

# ─── Query Type Definitions ──────────────────────────────────────────────────
QUERY_TYPES = {
    "factual_lookup": "Direct factual question about specific clauses, terms, or provisions",
    "comparison": "Compare clauses, terms, or provisions across documents",
    "risk_analysis": "Identify risks, liabilities, or problematic clauses",
    "summarization": "Summarize sections, documents, or specific topics",
    "reasoning": "Complex legal reasoning requiring multi-step analysis"
}

# Minimum hybrid score for a chunk to be included in LLM context
MIN_SYNTHESIS_SCORE = 0.28

# ─── Domain Legal Terminology Map for Query Expansion ───────────────────────
LEGAL_CONCEPT_MAP = {
    # Employment & termination
    "fire": "termination terminate severance without cause notice period dismissal",
    "fired": "termination terminate severance without cause notice period dismissal",
    "quit": "termination resignation voluntary notice period",
    "resign": "termination resignation voluntary notice period",
    "layoff": "termination severance redundancy reduction workforce",
    "hire": "employment commencement onboarding probation",
    # Litigation & disputes
    "sue": "indemnification indemnity liability damages hold harmless litigation breach",
    "sued": "indemnification indemnity liability damages hold harmless litigation breach",
    "lawsuit": "indemnification litigation dispute resolution arbitration liability",
    "dispute": "dispute resolution arbitration mediation governing law venue choice of law",
    "court": "governing law jurisdiction venue choice of law dispute resolution",
    "arbitration": "arbitration dispute resolution binding final award AAA JAMS",
    # Payment & money
    "pay": "compensation payment terms consideration fee invoice late charge interest",
    "paid": "compensation payment terms consideration fee invoice late charge interest",
    "money": "compensation remuneration consideration fees penalties liquidated damages",
    "salary": "base salary compensation bonus benefits overtime commission",
    "invoice": "payment terms net-30 net-60 late fee interest invoice billing",
    "refund": "refund reimbursement return credits termination payment",
    "price": "consideration fee pricing rate compensation payment",
    # Confidentiality & IP
    "secret": "confidentiality non-disclosure proprietary information trade secrets NDA",
    "secrets": "confidentiality non-disclosure proprietary information trade secrets NDA",
    "leak": "confidentiality breach unauthorized disclosure remedies injunctive relief",
    "nda": "non-disclosure agreement confidentiality proprietary information trade secrets",
    "patent": "intellectual property patent copyright ownership work made for hire assignment",
    "copyright": "intellectual property copyright ownership work made for hire assignment license",
    "trademark": "intellectual property trademark brand ownership license",
    "software": "intellectual property software license source code ownership delivery",
    "data": "data privacy personal information security breach confidentiality GDPR",
    # Liability & warranties
    "penalty": "liquidated damages liability penalty breach late fee cure period",
    "insurance": "insurance coverage commercial general liability policy indemnity certificate",
    "guarantee": "warranty representation covenant guarantee disclaimer as-is",
    "warranty": "warranties representations covenants disclaimer as-is fitness merchantability",
    "liable": "liability indemnification damages limitation of liability cap exclusion",
    "responsible": "liability responsibility obligation duty indemnification",
    # Assignment & change
    "transfer": "assignment delegation successors assigns change of control",
    "assign": "assignment delegation successors assigns",
    "sell": "assignment change of control acquisition merger successor",
    # Contract lifecycle
    "cancel": "termination terminate expiration cancellation notice right to cancel cure",
    "renew": "renewal auto-renewal term extension notice period opt-out",
    "expire": "term expiration renewal termination notice",
    "amend": "amendment modification written consent approval",
    "change": "amendment modification change of control assignment",
    # Force majeure & misc
    "force": "force majeure act of god pandemic natural disaster unforeseeable",
    "pandemic": "force majeure act of god unforeseeable event impossibility",
    "govern": "governing law jurisdiction applicable law choice of law",
    "notice": "notice period written notice cure period notice requirements",
}


class LegalAgent:
    """
    Agent with multi-step task decomposition for legal document QA.

    Pipeline: classify → query expand → hybrid retrieve → analyze → synthesize
    """

    def __init__(self, document_store):
        self.store = document_store
        self.execution_trace = []

    def _get_model(self):
        return get_gemini_model()

    def _log_step(self, step_name: str, detail: str, result: str = ""):
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
        """Classify query type via LLM, with robust rule-based fallback."""
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

            text, status, _ = call_gemini_safely(model, prompt,
                                                  generation_config={"max_output_tokens": 20})
            if status == "SUCCESS" and text:
                category = text.strip().lower().replace(" ", "_").replace("-", "_")
                if category in QUERY_TYPES:
                    self._log_step("classify", f"LLM classified: '{query}'",
                                   f"{category} — {QUERY_TYPES[category]}")
                    return category

        # Rule-based fallback
        q_lower = query.lower()
        if any(w in q_lower for w in ["compare", "difference", "versus", "vs", "between", "both"]):
            category = "comparison"
        elif any(w in q_lower for w in ["risk", "liability", "liable", "penalty", "danger",
                                         "breach", "harm", "exposure", "unenforceable"]):
            category = "risk_analysis"
        elif any(w in q_lower for w in ["summarize", "summary", "overview", "brief", "outline",
                                         "describe", "explain the document"]):
            category = "summarization"
        elif any(w in q_lower for w in ["why", "can they", "am i allowed", "valid", "enforceable",
                                         "consequence", "what happens", "if i", "can we"]):
            category = "reasoning"
        else:
            category = "factual_lookup"

        self._log_step("classify", f"Heuristic classified: '{query}'",
                       f"{category} — {QUERY_TYPES[category]}")
        return category

    # ─── Step 2: Query Expansion ─────────────────────────────────────────────

    def _expand_query(self, query: str) -> List[str]:
        """
        Expand user query with legal domain terminology to boost recall.
        Uses dict.fromkeys() to deduplicate while preserving insertion order.
        """
        words = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        expansion_terms = []
        for w in words:
            if w in LEGAL_CONCEPT_MAP:
                expansion_terms.extend(LEGAL_CONCEPT_MAP[w].split())

        # Deduplicate preserving order
        unique_terms = list(dict.fromkeys(expansion_terms))

        queries = [query]
        if unique_terms:
            expanded = f"{query} {' '.join(unique_terms)}"
            queries.append(expanded)
            self._log_step(
                "query_expansion",
                f"Expanded '{query}' with {len(unique_terms)} legal terms",
                f"Added: {' '.join(unique_terms[:10])}{'...' if len(unique_terms) > 10 else ''}"
            )
        return queries

    # ─── Step 3: Hybrid Retrieval ─────────────────────────────────────────────

    def _retrieve_context(self, query: str, query_type: str,
                          filter_file: Optional[str] = None, top_k: int = 10) -> List[Dict]:
        """
        Retrieve chunks using hybrid search + query expansion.
        Uses .get() guards on all metadata fields to prevent KeyError.
        """
        retrieval_k = {
            "factual_lookup": 7,
            "comparison": 15,
            "risk_analysis": 12,
            "summarization": 15,
            "reasoning": 10
        }.get(query_type, top_k)

        query_variants = self._expand_query(query)
        seen_keys = set()
        all_results = []

        for q_var in query_variants:
            results = self.store.query(q_var, top_k=retrieval_k, filter_file=filter_file)
            for r in results:
                # Use .get() guards — older-indexed docs may lack chunk_index
                src = r.get("source_file", "unknown")
                pg = r.get("page", 0)
                ci = r.get("chunk_index", 0)
                chunk_key = f"{src}::{pg}::{ci}"
                if chunk_key not in seen_keys:
                    seen_keys.add(chunk_key)
                    all_results.append(r)

        # Sort by hybrid score descending
        all_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        # Apply minimum score threshold to prevent noise from reaching synthesis
        filtered = [r for r in all_results if r.get("score", 0.0) >= MIN_SYNTHESIS_SCORE]
        final_results = filtered[:retrieval_k]

        docs_count = len(set(r.get("source_file", "?") for r in final_results))
        scope_desc = f" ('{filter_file}')" if filter_file else " (corpus)"

        self._log_step(
            "retrieve",
            f"Hybrid search{scope_desc}: {len(query_variants)} query variants, type='{query_type}'",
            f"{len(final_results)} relevant chunks from {docs_count} doc(s) "
            f"(filtered {len(all_results) - len(final_results)} low-score chunks)"
        )
        return final_results

    # ─── Step 4: Analysis ────────────────────────────────────────────────────

    def _analyze_context(self, query: str, query_type: str, retrieved: List[Dict]) -> Dict:
        """
        Organize retrieved chunks by document with safe field access.
        """
        if not retrieved:
            self._log_step("analyze", "No context retrieved", "0 results after score filtering")
            return {
                "context_blocks": [],
                "analysis_notes": "No relevant clauses found above the minimum relevance threshold."
            }

        by_document: Dict[str, List] = {}
        for chunk in retrieved:
            fname = chunk.get("source_file", "unknown")
            by_document.setdefault(fname, []).append(chunk)

        context_blocks = []
        for fname, chunks in by_document.items():
            # Safe sort with .get() guards
            chunks.sort(key=lambda c: (c.get("page", 0), c.get("chunk_index", 0)))
            for chunk in chunks:
                context_blocks.append({
                    "text": chunk.get("text", ""),
                    "citation": f"[{fname}, Page {chunk.get('page', '?')}]",
                    "score": chunk.get("score", 0.0),
                    "semantic_score": chunk.get("semantic_score", 0.0),
                    "lexical_score": chunk.get("lexical_score", 0.0),
                    "source_file": fname,
                    "page": chunk.get("page", 0)
                })

        # Re-sort blocks by relevance score for prompt construction
        context_blocks.sort(key=lambda b: b["score"], reverse=True)

        analysis_notes = (
            f"Analyzed {len(context_blocks)} clause(s) from {len(by_document)} document(s). "
            f"All chunks cleared minimum relevance threshold ({MIN_SYNTHESIS_SCORE})."
        )
        if query_type == "comparison":
            analysis_notes += f" Comparing: {', '.join(by_document.keys())}."
        elif query_type == "risk_analysis":
            analysis_notes += " Focus: liabilities, terminations, warranties, indemnities, uncapped obligations."

        self._log_step(
            "analyze",
            f"Structured {len(retrieved)} chunks from {len(by_document)} document(s)",
            analysis_notes
        )
        return {"context_blocks": context_blocks, "analysis_notes": analysis_notes}

    # ─── Step 5: Synthesis ───────────────────────────────────────────────────

    def _synthesize_answer(self, query: str, query_type: str, analysis: Dict) -> str:
        """
        Synthesize a grounded legal answer using Gemini.
        Falls back to formatted evidence cards on API failure.
        """
        blocks = analysis.get("context_blocks", [])

        if not blocks:
            return (
                "### ℹ️ No Relevant Clauses Found\n\n"
                "The indexed documents do not contain provisions that match your query above the "
                f"minimum relevance threshold ({int(MIN_SYNTHESIS_SCORE * 100)}% match score).\n\n"
                "**Suggestions:**\n"
                "- Verify the document is uploaded and ingested (📥 Ingest Now button)\n"
                "- Try rephrasing your question using specific legal terms\n"
                "- Check if you're searching the right scope (single document vs entire corpus)"
            )

        model = self._get_model()

        # Build context string — top 10 most relevant blocks
        context_parts = []
        for block in blocks[:10]:
            sem = block.get("semantic_score", 0.0)
            lex = block.get("lexical_score", 0.0)
            context_parts.append(
                f"**{block['citation']}** (Semantic: {sem:.2f}, Keyword: {lex:.2f})\n"
                f"{block['text']}"
            )
        context_str = "\n\n---\n\n".join(context_parts)

        type_instructions = {
            "factual_lookup": (
                "Provide a direct, authoritative answer. "
                "Quote the exact clause text where relevant and cite [Document, Page X]. "
                "State specific terms, dates, amounts, or thresholds explicitly."
            ),
            "comparison": (
                "Create a structured comparison. Use a markdown table or side-by-side bullet format. "
                "Highlight key differences, shared terms, and practical consequences. "
                "Cite both documents for every point."
            ),
            "risk_analysis": (
                "Identify ALL risk exposures: uncapped liabilities, one-sided indemnification, "
                "unfavorable termination triggers, warranty disclaimers, and auto-renewal traps. "
                "Categorize each finding as 🔴 High / 🟡 Medium / 🟢 Low risk."
            ),
            "summarization": (
                "Provide an executive summary with: (1) Document purpose and parties, "
                "(2) Key obligations of each party, (3) Critical deadlines and monetary terms, "
                "(4) Risk and liability provisions."
            ),
            "reasoning": (
                "Walk through the legal analysis step by step. "
                "Identify the relevant provisions, apply them to the question, "
                "address any conditions or exceptions, and state your conclusion clearly."
            )
        }
        instruction = type_instructions.get(query_type, type_instructions["factual_lookup"])

        prompt = f"""You are an elite legal AI counsel with deep expertise in contract law.

TASK: {instruction}

STRICT RULES:
1. Base your answer SOLELY on the Retrieved Context below — do not use external knowledge.
2. Use markdown formatting: headings, bullets, tables, or bold for key terms.
3. Cite every factual claim with [Document_Name, Page X].
4. If the context is insufficient to answer confidently, say exactly what is missing and why.
5. Do NOT add generic legal disclaimers like "consult a lawyer" — be direct and analytical.

QUESTION: "{query}"

RETRIEVED CONTEXT:
{context_str}

LEGAL ANALYSIS:"""

        if model:
            answer_text, status, err_detail = call_gemini_safely(model, prompt)

            if status == "SUCCESS" and answer_text:
                self._log_step(
                    "synthesize",
                    f"Gemini synthesized from {min(len(blocks), 10)} blocks",
                    f"{len(answer_text.split())} words"
                )
                return answer_text

            elif status == "QUOTA_EXHAUSTED":
                self._log_step("synthesize", "API quota exhausted", "Showing evidence fallback")
                return (
                    "### 🚨 Gemini API Quota Exhausted\n\n"
                    "The backend API has depleted its usage quota. "
                    "**Add your personal Gemini API key in the left sidebar** to restore full AI synthesis.\n\n"
                    "---\n\n"
                    "### 📑 Top Retrieved Clauses (Hybrid Search Results):\n\n"
                    + self._format_fallback_evidence(blocks[:5])
                )

            elif status in ("INVALID_KEY", "NO_KEY"):
                self._log_step("synthesize", f"API key issue: {status}", "Showing evidence fallback")
                return (
                    "### ⚠️ Gemini API Key Required\n\n"
                    "No valid API key is configured. Add your Gemini API key in the left sidebar.\n\n"
                    "---\n\n"
                    "### 📑 Retrieved Document Evidence:\n\n"
                    + self._format_fallback_evidence(blocks[:5])
                )

        # Model could not be initialized
        self._log_step("synthesize", "Model unavailable", "Fallback to evidence display")
        return (
            "### ⚠️ AI Synthesis Unavailable\n\n"
            "Could not connect to Gemini. Check API credentials in the sidebar.\n\n"
            "---\n\n"
            "### 📑 Extracted Relevant Clauses:\n\n"
            + self._format_fallback_evidence(blocks[:5])
        )

    def _format_fallback_evidence(self, blocks: List[Dict]) -> str:
        """Format retrieved chunks as readable markdown evidence cards."""
        output = []
        for i, b in enumerate(blocks, 1):
            score_pct = int(b.get("score", 0.0) * 100)
            output.append(
                f"**{i}. {b['citation']}** — Match Score: {score_pct}%\n\n"
                f"> {b.get('text', '').strip()}\n"
            )
        return "\n".join(output) if output else "_No evidence chunks available._"

    # ─── Main Entry Point ────────────────────────────────────────────────────

    def query(self, user_query: str, filter_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the full agent pipeline with top-level error protection.

        Always returns a dict with keys: answer, sources, steps, query_type,
        match_score (renamed from confidence), chunks_retrieved, documents_referenced.
        """
        self.execution_trace = []
        # Sanitize and limit query length
        user_query = user_query.strip()[:2000]

        self._log_step("start", f"Query: '{user_query[:80]}...'", "Starting agent decomposition")

        try:
            query_type = self._classify_query(user_query)
            retrieved = self._retrieve_context(user_query, query_type, filter_file=filter_file)
            analysis = self._analyze_context(user_query, query_type, retrieved)
            answer = self._synthesize_answer(user_query, query_type, analysis)

            sources = list(dict.fromkeys(
                b["citation"] for b in analysis["context_blocks"]
            ))

            if analysis["context_blocks"]:
                top_scores = [b["score"] for b in analysis["context_blocks"][:5]]
                match_score = round(sum(top_scores) / len(top_scores) * 100, 1)
            else:
                match_score = 0.0

            self._log_step("complete", "Pipeline complete", f"Match score: {match_score}%")

            return {
                "answer": answer,
                "sources": sources,
                "steps": self.execution_trace,
                "query_type": query_type,
                "match_score": match_score,  # renamed from confidence (more accurate label)
                "confidence": match_score,   # keep old key for backwards compat
                "chunks_retrieved": len(retrieved),
                "documents_referenced": len(set(
                    r.get("source_file", "?") for r in retrieved
                )) if retrieved else 0
            }

        except Exception as e:
            # Guarantee the UI never crashes from an unhandled agent error
            self._log_step("error", f"Agent pipeline error: {type(e).__name__}", str(e))
            return {
                "answer": f"### ⚠️ Agent Error\n\nAn internal error occurred: `{e}`\n\nPlease try rephrasing your query.",
                "sources": [],
                "steps": self.execution_trace,
                "query_type": "error",
                "match_score": 0.0,
                "confidence": 0.0,
                "chunks_retrieved": 0,
                "documents_referenced": 0
            }

    def query_without_rag(self, user_query: str) -> str:
        """Pure LLM answer (no retrieval) — used for evaluation baseline."""
        model = self._get_model()
        if not model:
            return "Gemini LLM not available or credentials missing."

        prompt = f"""You are a legal AI assistant. Answer the following legal question to the best of your knowledge.

Question: "{user_query[:2000]}"

Answer:"""

        text, status, err = call_gemini_safely(model, prompt)
        if status == "SUCCESS" and text:
            return text
        elif status == "QUOTA_EXHAUSTED":
            return "⚠️ API quota exhausted. Please add a personal API key in the sidebar."
        return f"Synthesis failed: {err}"
