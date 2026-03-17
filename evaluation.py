"""
evaluation.py — Factual accuracy evaluation framework.

Compares RAG-augmented responses vs. pure LLM (no retrieval) responses
to measure the improvement from Retrieval-Augmented Generation.
"""

import numpy as np
import torch
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity

# Re-use the shared RoBERTa model from vectorStore
from vectorStore import get_roberta_embedding


# ─── Legal Keywords for Coverage Metric ───────────────────────────────────────
LEGAL_KEYWORDS = [
    "plaintiff", "defendant", "respondent", "petitioner", "appellant", "accused",
    "complainant", "counsel", "judge", "tribunal", "prosecution", "witness",
    "appeal", "trial", "hearing", "proceedings", "petition", "lawsuit",
    "jurisdiction", "adjudication", "motion", "plea", "summons", "order",
    "verdict", "judgment", "injunction", "ruling", "decree", "sentence",
    "conviction", "acquittal", "bail", "arrest", "charge", "claim",
    "complaint", "indictment", "evidence", "testimony", "affidavit",
    "precedent", "statute", "legislation", "constitution", "clause",
    "provision", "contract", "agreement", "regulation", "doctrine",
    "damages", "compensation", "penalty", "sanction", "relief", "settlement",
    "liability", "negligence", "breach", "fault", "guilt",
    "rights", "obligation", "duty", "violation", "remedy", "enforcement"
]


def compute_rouge(reference: str, hypothesis: str) -> dict:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1": round(scores['rouge1'].fmeasure, 4),
        "rouge2": round(scores['rouge2'].fmeasure, 4),
        "rougeL": round(scores['rougeL'].fmeasure, 4)
    }


def compute_semantic_similarity(text_a: str, text_b: str) -> float:
    """Compute cosine similarity between RoBERTa embeddings of two texts."""
    emb_a = np.array(get_roberta_embedding(text_a)).reshape(1, -1)
    emb_b = np.array(get_roberta_embedding(text_b)).reshape(1, -1)
    return float(cosine_similarity(emb_a, emb_b)[0][0])


def compute_keyword_coverage(reference: str, hypothesis: str) -> float:
    """Compute what fraction of legal keywords in the reference appear in the hypothesis."""
    ref_words = set(reference.lower().split())
    hyp_words = set(hypothesis.lower().split())

    ref_keywords = ref_words.intersection(set(LEGAL_KEYWORDS))
    if not ref_keywords:
        return 0.0

    covered = ref_keywords.intersection(hyp_words)
    return round(len(covered) / len(ref_keywords) * 100, 2)


def evaluate_single(ground_truth: str, rag_answer: str, no_rag_answer: str) -> dict:
    """
    Evaluate a single question: compare RAG vs no-RAG against ground truth.

    Returns metrics for both approaches plus the delta.
    """
    rag_rouge = compute_rouge(ground_truth, rag_answer)
    no_rag_rouge = compute_rouge(ground_truth, no_rag_answer)

    rag_sim = compute_semantic_similarity(ground_truth, rag_answer)
    no_rag_sim = compute_semantic_similarity(ground_truth, no_rag_answer)

    rag_coverage = compute_keyword_coverage(ground_truth, rag_answer)
    no_rag_coverage = compute_keyword_coverage(ground_truth, no_rag_answer)

    # Composite accuracy score (weighted average of metrics)
    rag_composite = (rag_rouge["rougeL"] * 0.3 + rag_sim * 0.4 + (rag_coverage / 100) * 0.3)
    no_rag_composite = (no_rag_rouge["rougeL"] * 0.3 + no_rag_sim * 0.4 + (no_rag_coverage / 100) * 0.3)

    improvement = ((rag_composite - no_rag_composite) / no_rag_composite * 100) if no_rag_composite > 0 else 0

    return {
        "rag": {
            "rouge": rag_rouge,
            "semantic_similarity": round(rag_sim, 4),
            "keyword_coverage": rag_coverage,
            "composite_score": round(rag_composite, 4)
        },
        "no_rag": {
            "rouge": no_rag_rouge,
            "semantic_similarity": round(no_rag_sim, 4),
            "keyword_coverage": no_rag_coverage,
            "composite_score": round(no_rag_composite, 4)
        },
        "improvement_pct": round(improvement, 2)
    }


def evaluate_batch(agent, questions: list[dict]) -> dict:
    """
    Run full evaluation over a batch of questions.

    Args:
        agent: LegalAgent instance
        questions: list of {"question": str, "ground_truth": str}

    Returns:
        {
            "results": list of per-question results,
            "summary": aggregate metrics
        }
    """
    results = []

    for item in questions:
        question = item["question"]
        ground_truth = item["ground_truth"]

        # Get RAG-augmented answer
        rag_result = agent.query(question)
        rag_answer = rag_result["answer"]

        # Get pure LLM answer (no retrieval)
        no_rag_answer = agent.query_without_rag(question)

        # Evaluate
        metrics = evaluate_single(ground_truth, rag_answer, no_rag_answer)
        metrics["question"] = question
        results.append(metrics)

    # Aggregate
    avg_rag_composite = np.mean([r["rag"]["composite_score"] for r in results])
    avg_no_rag_composite = np.mean([r["no_rag"]["composite_score"] for r in results])
    avg_improvement = np.mean([r["improvement_pct"] for r in results])

    avg_rag_rouge = np.mean([r["rag"]["rouge"]["rougeL"] for r in results])
    avg_no_rag_rouge = np.mean([r["no_rag"]["rouge"]["rougeL"] for r in results])

    summary = {
        "total_questions": len(questions),
        "avg_rag_composite": round(float(avg_rag_composite), 4),
        "avg_no_rag_composite": round(float(avg_no_rag_composite), 4),
        "avg_improvement_pct": round(float(avg_improvement), 2),
        "avg_rag_rougeL": round(float(avg_rag_rouge), 4),
        "avg_no_rag_rougeL": round(float(avg_no_rag_rouge), 4),
    }

    return {"results": results, "summary": summary}
