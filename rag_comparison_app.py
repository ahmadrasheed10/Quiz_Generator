"""
Streamlit dashboard to compare Baseline vs RAG models.
"""
import streamlit as st

from rag_evaluator import RAGEvaluator
from rag_inference import RAGQuizGenerator


st.set_page_config(
    page_title="Baseline vs RAG Comparison",
    page_icon="🧠",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading RAG model and embeddings (first time only)...")
def load_rag(model_path: str, dataset_path: str, top_k: int = 1) -> RAGQuizGenerator:
    """Load RAG generator with cached embeddings."""
    return RAGQuizGenerator(
        model_path=model_path,
        dataset_path=dataset_path,
        top_k=top_k,
    )


@st.cache_resource(show_spinner=False)
def load_evaluator(dataset_path: str) -> RAGEvaluator:
    return RAGEvaluator(dataset_path=dataset_path)


def run_single_inference(
    rag_generator: RAGQuizGenerator,
    evaluator: RAGEvaluator,
    inputs: dict,
) -> dict:
    rag_question, contexts = rag_generator.generate_question_with_rag(**inputs)
    baseline_question = rag_generator.generate_question_baseline(**inputs)

    rag_metrics = {
        "completeness": evaluator.evaluate_completeness(rag_question, **inputs),
        "faithfulness": evaluator.evaluate_faithfulness(rag_question, contexts),
    }
    baseline_metrics = {
        "completeness": evaluator.evaluate_completeness(
            baseline_question,
            **inputs,
        ),
        "faithfulness": {
            "faithfulness_score": 0.0,
            "is_grounded": False,
            "details": {"note": "Baseline has no retrieved contexts"},
        },
    }

    return {
        "rag": {"question": rag_question, "contexts": contexts, "metrics": rag_metrics},
        "baseline": {
            "question": baseline_question,
            "metrics": baseline_metrics,
        },
    }




def render_metrics_card(title: str, metrics: dict):
    st.subheader(title)
    cols = st.columns(2)
    cols[0].metric(
        "Completeness",
        f"{metrics['completeness']['completeness_score']:.2f}",
    )
    cols[1].metric(
        "Faithfulness",
        f"{metrics['faithfulness']['faithfulness_score']:.2f}",
    )

    with st.expander("Details"):
        st.json(metrics)


def main():
    st.title("🔍 Baseline vs RAG Comparison")
    st.markdown(
        """
Compare two inference strategies using completeness & faithfulness:
- **Baseline**: Original T5 without retrieval
- **RAG**: T5 + dataset retrieval (using 1 retrieved example)
        """
    )

    with st.sidebar:
        st.header("⚙️ Configuration")
        model_path = st.text_input(
            "Fine-tuned model path",
            "./results/t5-quiz-generator",
        )
        dataset_path = st.text_input("Dataset path", "quiz_data.csv")
        
        st.info("💡 Using top_k=1 (single retrieved example) for fast retrieval")

    st.subheader("Single Prompt Comparison")
    col1, col2 = st.columns(2)
    with col1:
        subject = st.text_input("Subject", "NLP")
        topic = st.text_input("Topic", "Tokenization")
    with col2:
        difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard"], index=1)
        q_type = st.selectbox(
            "Question Type",
            ["MCQ", "short", "long"],
            index=0,
        )

    if st.button("🚀 Run Comparison", use_container_width=True, type="primary"):
        with st.spinner("Running inference..."):
            # Load with top_k=1 (fixed)
            rag = load_rag(model_path, dataset_path, top_k=1)
            evaluator = load_evaluator(dataset_path)

            inputs = {
                "subject": subject,
                "topic": topic,
                "difficulty": difficulty,
                "question_type": q_type,
            }
            results = run_single_inference(rag, evaluator, inputs)

        st.success("✅ Done!")

        cols = st.columns(2)
        with cols[0]:
            st.markdown("### 📝 Baseline")
            st.info(results["baseline"]["question"])
            render_metrics_card("Baseline Metrics", results["baseline"]["metrics"])

        with cols[1]:
            st.markdown("### 🧠 RAG")
            st.info(results["rag"]["question"])
            render_metrics_card("RAG Metrics", results["rag"]["metrics"])

        with st.expander("📚 Retrieved Context (RAG)"):
            for ctx in results["rag"]["contexts"]:
                st.write(f"**Example:** {ctx['question']}")
                st.caption(f"Similarity: {ctx['similarity']:.3f} | Subject: {ctx.get('subject', 'N/A')} | Topic: {ctx.get('topic', 'N/A')}")


if __name__ == "__main__":
    main()

