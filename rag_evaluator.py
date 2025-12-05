"""
Evaluation utilities for RAG vs Baseline models.
"""
import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class RAGEvaluator:
    """Computes completeness & faithfulness metrics."""

    def __init__(
        self,
        dataset_path: str = "quiz_data.csv",
        similarity_model: str = "all-MiniLM-L6-v2",
    ):
        self.dataset_path = dataset_path
        self.dataset = self._load_dataset()
        self.similarity_model = SentenceTransformer(similarity_model)

    def _load_dataset(self) -> pd.DataFrame:
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")
        df = pd.read_csv(self.dataset_path)
        if "question" in df.columns:
            df = df.dropna(subset=["question"])
            df["question"] = df["question"].astype(str)
        return df

    # ----------------------------
    # Completeness
    # ----------------------------
    def evaluate_completeness(
        self,
        generated_question: str,
        subject: str,
        topic: str,
        difficulty: str,
        question_type: str,
    ) -> Dict:
        score = 0.0
        max_score = 4.0
        details = {}

        q_lower = generated_question.lower()
        subject_lower = subject.lower()
        topic_lower = topic.lower()

        # Topic mention (0.5)
        topic_hit = topic_lower in q_lower or any(
            word in q_lower for word in topic_lower.split()
        )
        details["topic_mentioned"] = topic_hit
        score += 0.5 if topic_hit else 0.0

        # Subject mention (0.5)
        subject_hit = subject_lower in q_lower or any(
            word in q_lower for word in subject_lower.split()
        )
        details["subject_mentioned"] = subject_hit
        score += 0.5 if subject_hit else 0.0

        # Difficulty alignment (1.0)
        difficulty_keywords = {
            "easy": ["define", "what is", "identify", "list"],
            "medium": ["describe", "explain", "compare"],
            "hard": ["evaluate", "prove", "derive", "design"],
        }
        diff_hit = False
        if difficulty.lower() in difficulty_keywords:
            diff_hit = any(
                keyword in q_lower for keyword in difficulty_keywords[difficulty.lower()]
            )
            score += 1.0 if diff_hit else 0.0
        details["difficulty_appropriate"] = diff_hit

        # Question type format (1.0)
        qtype_lower = question_type.lower()
        qtype_hit: Optional[bool] = None
        if qtype_lower == "mcq":
            qtype_hit = any(marker in q_lower for marker in ["a)", "b)", "option"])
        elif qtype_lower in {"short", "short answer"}:
            qtype_hit = len(generated_question.split()) < 50
        elif qtype_lower in {"long", "long answer"}:
            qtype_hit = len(generated_question.split()) > 30
        if qtype_hit is not None:
            score += 1.0 if qtype_hit else 0.0
        details["question_type_format"] = qtype_hit

        # Completeness (1.0)
        complete = "?" in generated_question or len(generated_question.split()) > 5
        details["is_complete"] = complete
        score += 1.0 if complete else 0.0

        normalized = score / max_score
        return {
            "completeness_score": normalized,
            "raw_score": score,
            "max_score": max_score,
            "details": details,
        }

    # ----------------------------
    # Faithfulness
    # ----------------------------
    def evaluate_faithfulness(
        self,
        generated_question: str,
        retrieved_contexts: List[Dict],
        threshold: float = 0.5,
    ) -> Dict:
        if not retrieved_contexts:
            return {
                "faithfulness_score": 0.0,
                "is_grounded": False,
                "details": {"error": "No retrieved contexts"},
            }

        gen_embedding = self.similarity_model.encode(
            [generated_question],
            convert_to_numpy=True,
        )
        ctx_questions = [ctx["question"] for ctx in retrieved_contexts]
        ctx_embeddings = self.similarity_model.encode(
            ctx_questions,
            convert_to_numpy=True,
        )

        sims = cosine_similarity(gen_embedding, ctx_embeddings)[0]
        max_sim = float(np.max(sims))
        avg_sim = float(np.mean(sims))
        is_grounded = max_sim >= threshold

        gen_words = set(generated_question.lower().split())
        ctx_words = set()
        for ctx in retrieved_contexts:
            ctx_words.update(ctx["question"].lower().split())
        overlap = len(gen_words & ctx_words) / len(gen_words) if gen_words else 0.0

        faithfulness_score = (max_sim * 0.7) + (overlap * 0.3)

        return {
            "faithfulness_score": float(faithfulness_score),
            "is_grounded": is_grounded,
            "max_similarity": max_sim,
            "avg_similarity": avg_sim,
            "word_overlap": float(overlap),
            "details": {"similarities": [float(s) for s in sims], "threshold": threshold},
        }

    # ----------------------------
    # Aggregated evaluation
    # ----------------------------
    def compare_models(
        self,
        results_path: str,
        rag_results: List[Dict],
        baseline_results: List[Dict],
    ) -> Dict:
        rag_completeness = [
            r["completeness"]["completeness_score"] for r in rag_results
        ]
        rag_faithfulness = [r["faithfulness"]["faithfulness_score"] for r in rag_results]
        baseline_completeness = [
            r["completeness"]["completeness_score"] for r in baseline_results
        ]

        summary = {
            "rag": {
                "avg_completeness": float(np.mean(rag_completeness)),
                "avg_faithfulness": float(np.mean(rag_faithfulness)),
            },
            "baseline": {
                "avg_completeness": float(np.mean(baseline_completeness)),
                "avg_faithfulness": 0.0,
            },
            "improvement": {
                "completeness_delta": float(
                    np.mean(rag_completeness) - np.mean(baseline_completeness)
                ),
                "faithfulness_advantage": float(np.mean(rag_faithfulness)),
            },
        }

        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(
                {"rag": rag_results, "baseline": baseline_results, "summary": summary},
                f,
                indent=2,
            )

        return summary

