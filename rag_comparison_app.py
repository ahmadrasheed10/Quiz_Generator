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
        # For baseline, evaluate faithfulness against the question itself
        # (since there's no external context, check if content is self-consistent)
        "faithfulness": evaluator.evaluate_faithfulness(
            baseline_question, 
            [{"question": baseline_question, "subject": inputs.get("subject", ""), "topic": inputs.get("topic", "")}]
        ),
    }

    return {
        "rag": {"question": rag_question, "contexts": contexts, "metrics": rag_metrics},
        "baseline": {
            "question": baseline_question,
            "metrics": baseline_metrics,
        },
    }




def render_metrics_card(title: str, metrics: dict, is_winner: bool = False):
    """Render metrics card with winner highlighting"""
    
    # Add winner badge if applicable
    if is_winner:
        st.markdown(f"### {title} 🏆")
    else:
        st.subheader(title)
    
    # Display metrics in columns
    cols = st.columns(2)
    
    completeness_score = metrics['completeness']['completeness_score']
    faithfulness_score = metrics['faithfulness']['faithfulness_score']
    
    with cols[0]:
        st.metric(
            "Completeness",
            f"{completeness_score:.2f}",
            delta=f"{(completeness_score - 0.5) * 100:+.0f}%" if completeness_score != 0 else None,
            delta_color="normal"
        )
    
    with cols[1]:
        st.metric(
            "Faithfulness",
            f"{faithfulness_score:.2f}",
            delta=f"{(faithfulness_score - 0.5) * 100:+.0f}%" if faithfulness_score != 0 else None,
            delta_color="normal"
        )
    
    # Overall score
    overall = (completeness_score + faithfulness_score) / 2
    
    if is_winner:
        st.success(f"**Overall Score: {overall:.2f}** ✅")
    else:
        st.info(f"**Overall Score: {overall:.2f}**")
    
    # Detailed breakdown
    with st.expander("📊 Detailed Breakdown"):
        col_detail1, col_detail2 = st.columns(2)
        
        with col_detail1:
            st.markdown("**Completeness Details:**")
            comp_details = metrics['completeness']
            st.write(f"- Score: {comp_details['completeness_score']:.3f}")
            st.write(f"- Key Points Covered:")
            for key, val in comp_details.get('details', {}).items():
                st.write(f"  • {key}: {val}")
        
        with col_detail2:
            st.markdown("**Faithfulness Details:**")
            faith_details = metrics['faithfulness']
            st.write(f"- Score: {faith_details['faithfulness_score']:.3f}")
            st.write(f"- Is Grounded: {'✅ Yes' if faith_details.get('is_grounded', False) else '❌ No'}")
            if 'details' in faith_details:
                for key, val in faith_details['details'].items():
                    st.write(f"  • {key}: {val}")


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
    
    # Subject options (same as app.py)
    SUBJECT_OPTIONS = [
        "Applied Physics",
        "Automata",
        "COAL",
        "Calculus",
        "Cloud Computing",
        "Computer Networks",
        "Data Structures and Algorithms",
        "Digital Image Processing",
        "ICT",
        "Machine Learning and Operations",
        "NLP",
        "Object Oriented Programming",
        "Operating System",
        "Programming Fundamentals",
        "Software Design and Analysis",
        "Software for machine and devices",
        "Theory of Computation and Automata",
    ]

    st.subheader("Single Prompt Comparison")
   
    col1, col2 = st.columns(2)
    with col1:
        # Dropdown for subject selection
        subject = st.selectbox(
            "📚 Subject",
            options=SUBJECT_OPTIONS,
            help="Select the subject for the quiz question"
        )
        topic = st.text_input("Topic", "Tokenization")
    with col2:
        difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard"], index=1)
        q_type = st.selectbox(
            "Question Type",
            ["MCQ", "short", "long"],
            index=0,
        )
    
    # Show selected subject prominently
    st.info(f"**Working with Subject:** {subject}")

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
        
        # Calculate overall scores to determine winner
        baseline_overall = (
            results["baseline"]["metrics"]["completeness"]["completeness_score"] +
            results["baseline"]["metrics"]["faithfulness"]["faithfulness_score"]
        ) / 2
        
        rag_overall = (
            results["rag"]["metrics"]["completeness"]["completeness_score"] +
            results["rag"]["metrics"]["faithfulness"]["faithfulness_score"]
        ) / 2
        
        # Show which metrics differ (removed Comparison Summary section)
        st.markdown("### 🏆 Performance Breakdown")
        
        comp_diff = (
            results["rag"]["metrics"]["completeness"]["completeness_score"] -
            results["baseline"]["metrics"]["completeness"]["completeness_score"]
        )
        faith_diff = (
            results["rag"]["metrics"]["faithfulness"]["faithfulness_score"] -
            results["baseline"]["metrics"]["faithfulness"]["faithfulness_score"]
        )
        
        col_metric1, col_metric2 = st.columns(2)
        
        with col_metric1:
            if comp_diff > 0:
                st.success(f"✅ **Completeness**: RAG wins by {comp_diff:.2f}")
            elif comp_diff < 0:
                st.error(f"❌ **Completeness**: Baseline wins by {abs(comp_diff):.2f}")
            else:
                st.info("🤝 **Completeness**: Tie")
        
        with col_metric2:
            if faith_diff > 0:
                st.success(f"✅ **Faithfulness**: RAG wins by {faith_diff:.2f}")
            elif faith_diff < 0:
                st.error(f"❌ **Faithfulness**: Baseline wins by {abs(faith_diff):.2f}")
            else:
                st.info("🤝 **Faithfulness**: Tie")
        
        st.markdown("---")
        
        # Display detailed results
        st.markdown("## 🔍 Detailed Results")

        cols = st.columns(2)
        with cols[0]:
            st.markdown("### 📝 Baseline")
            st.info(results["baseline"]["question"])
            render_metrics_card(
                "Baseline Metrics", 
                results["baseline"]["metrics"],
                is_winner=(baseline_overall > rag_overall)
            )

        with cols[1]:
            st.markdown("### 🧠 RAG")
            st.info(results["rag"]["question"])
            render_metrics_card(
                "RAG Metrics", 
                results["rag"]["metrics"],
                is_winner=(rag_overall > baseline_overall)
            )

        with st.expander("📚 Retrieved Context (RAG)"):
            for i, ctx in enumerate(results["rag"]["contexts"], 1):
                st.markdown(f"**Example {i}:** {ctx['question']}")
                st.caption(f"Similarity: {ctx['similarity']:.3f} | Subject: {ctx.get('subject', 'N/A')} | Topic: {ctx.get('topic', 'N/A')}")


if __name__ == "__main__":
    main()
