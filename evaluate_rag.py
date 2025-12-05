"""
CLI to evaluate RAG vs Baseline models.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from rag_evaluator import RAGEvaluator
from rag_inference import RAGQuizGenerator


def build_test_cases(dataset_path: str, limit: int) -> list:
    df = pd.read_csv(dataset_path)
    required_cols = ["subject", "topic", "difficulty", "question_type"]
    available = all(col in df.columns for col in required_cols)
    if not available:
        raise ValueError(
            "Dataset must include subject, topic, difficulty, question_type columns."
        )
    cases = (
        df[required_cols]
        .dropna()
        .drop_duplicates()
        .head(limit)
        .to_dict("records")
    )
    if not cases:
        raise ValueError("No valid test cases found.")
    return cases


def evaluate(args):
    rag_generator = RAGQuizGenerator(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        top_k=args.top_k,
    )
    evaluator = RAGEvaluator(dataset_path=args.dataset_path)

    test_cases = build_test_cases(args.dataset_path, args.num_cases)
    rag_results = []
    baseline_results = []

    for idx, case in enumerate(test_cases, start=1):
        print(
            f"[{idx}/{len(test_cases)}] {case['subject']} - "
            f"{case['topic']} ({case['difficulty']} / {case['question_type']})"
        )

        rag_question, contexts = rag_generator.generate_question_with_rag(**case)
        baseline_question = rag_generator.generate_question_baseline(**case)

        rag_metrics = {
            "test_case": case,
            "question": rag_question,
            "contexts": contexts,
            "completeness": evaluator.evaluate_completeness(
                rag_question, **case
            ),
            "faithfulness": evaluator.evaluate_faithfulness(
                rag_question, contexts
            ),
        }
        baseline_metrics = {
            "test_case": case,
            "question": baseline_question,
            "completeness": evaluator.evaluate_completeness(
                baseline_question, **case
            ),
            "faithfulness": {
                "faithfulness_score": 0.0,
                "is_grounded": False,
                "details": {"note": "Baseline has no retrieved context"},
            },
        }

        rag_results.append(rag_metrics)
        baseline_results.append(baseline_metrics)

    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary = evaluator.compare_models(
        str(results_dir / "rag_vs_baseline.json"), rag_results, baseline_results
    )
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG vs baseline.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./results/t5-quiz-generator",
        help="Path to fine-tuned T5 model.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="quiz_data.csv",
        help="Path to dataset CSV.",
    )
    parser.add_argument(
        "--num_cases", type=int, default=20, help="Number of test cases."
    )
    parser.add_argument(
        "--top_k", type=int, default=3, help="Number of retrieved contexts."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/rag_evaluation",
        help="Directory for evaluation artifacts.",
    )
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()

