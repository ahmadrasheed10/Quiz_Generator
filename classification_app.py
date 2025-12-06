"""
Streamlit Frontend for Classification Models Results
Displays results from 2 ML models and 2 DL models
"""
import streamlit as st
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Classification Models Results",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #004E89;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 600;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .model-section {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 2px solid #FF6B35;
    }
    .stButton>button {
        background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%);
        color: white;
        font-weight: bold;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        border: none;
    }
    h2, h3 {
        color: #004E89 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'results_dir' not in st.session_state:
    st.session_state.results_dir = "./results/classification"
if 'target_column' not in st.session_state:
    st.session_state.target_column = "subject"

def get_model_files(results_dir, model_name, set_name="test"):
    """Get file paths for a model"""
    model_name_safe = model_name.replace(' ', '_')
    
    # Handle new folder structure: results_dir/target_column/files
    target_column = st.session_state.get('target_column', 'subject')
    
    # Check if we have the nested structure or flat structure
    base_path = Path(results_dir)
    target_path = base_path / target_column
    
    # If the target specific folder exists, use it. Otherwise fall back to base_path (backward compatibility)
    if target_path.exists():
        search_path = target_path
    else:
        search_path = base_path

    files = {
        'metrics': search_path / f"{model_name_safe}_{set_name}_metrics.json",
        'confusion_matrix': search_path / f"{model_name_safe}_{set_name}_confusion_matrix.png",
        'training_history': search_path / f"{model_name_safe}_training_history.png"
    }
    
    return files

def load_metrics(filepath):
    """Load metrics from JSON file"""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        # st.error(f"Error loading {filepath}: {e}") # Suppress error to avoid clutter
        return None

def load_all_model_metrics(results_dir, model_name):
    """Load all metrics (train, val, test) for a model"""
    metrics = {}
    for set_name in ['train', 'val', 'test']:
        files = get_model_files(results_dir, model_name, set_name)
        loaded_data = load_metrics(files['metrics'])
        if loaded_data:
            metrics[set_name] = loaded_data
    return metrics

def display_model_metrics(metrics, model_name, set_name="Test"):
    """Display metrics in a nice format"""
    if metrics is None:
        st.warning(f"No metrics available for {model_name} on {set_name} set")
        return
    
    st.markdown(f"### {set_name} Set Metrics")
    
    # Main metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
    
    with col2:
        st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
    
    with col3:
        st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
    
    with col4:
        st.metric("F1-Score", f"{metrics.get('f1_score', 0):.4f}")
    
    # Additional metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        auc_val = metrics.get('auc')
        if auc_val is not None:
            st.metric("AUC", f"{auc_val:.4f}")
        else:
            st.metric("AUC", "N/A", help="AUC requires probability predictions. Not available for this model.")
    
    with col2:
        exact_match = metrics.get('exact_match')
        if exact_match is not None:
            st.metric("Exact Match", f"{exact_match:.4f}")
        else:
            # Fallback to accuracy if exact_match not available
            st.metric("Exact Match", f"{metrics.get('accuracy', 0):.4f}", help="Exact Match = Accuracy for classification")
    
    with col3:
        top3 = metrics.get('top_3_accuracy')
        if top3 is not None:
            st.metric("Top-3 Accuracy", f"{top3:.4f}")
        else:
            st.metric("Top-3 Accuracy", "N/A", help="Top-3 Accuracy requires probability predictions. Not available for this model.")
    
    # Per-class metrics
    if 'per_class_metrics' in metrics:
        with st.expander("📊 Per-Class Metrics"):
            per_class_df = pd.DataFrame(metrics['per_class_metrics']).T
            per_class_df.columns = ['Precision', 'Recall', 'F1-Score']
            st.dataframe(per_class_df, use_container_width=True)

def display_confusion_matrix(img_path, model_name):
    """Display confusion matrix image"""
    if img_path and img_path.exists():
        img = Image.open(img_path)
        st.image(img, caption=f"{model_name} - Confusion Matrix", use_container_width=True)
    else:
        st.info(f"Confusion matrix not found for {model_name}")

def display_training_history(img_path, model_name):
    """Display training history plot"""
    if img_path and img_path.exists():
        img = Image.open(img_path)
        st.image(img, caption=f"{model_name} - Training History", use_container_width=True)
    else:
        st.info(f"Training history not found for {model_name}")

def compare_models(all_metrics):
    """Create comparison visualization"""
    if not all_metrics:
        return
    
    models = []
    accuracies = []
    f1_scores = []
    precisions = []
    recalls = []
    
    for model_name, metrics in all_metrics.items():
        if metrics and 'test' in metrics:
            test_metrics = metrics['test']
            models.append(model_name)
            accuracies.append(test_metrics.get('accuracy', 0))
            f1_scores.append(test_metrics.get('f1_score', 0))
            precisions.append(test_metrics.get('precision', 0))
            recalls.append(test_metrics.get('recall', 0))
    
    if not models:
        return
    
    # Create comparison chart
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    color_palette = ['#FF6B35', '#F7931E', '#667eea', '#764ba2']
    colors = [color_palette[i % len(color_palette)] for i in range(len(models))]
    
    # Accuracy comparison
    axes[0, 0].bar(models, accuracies, color=colors)
    axes[0, 0].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(accuracies):
        axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # F1-Score comparison
    axes[0, 1].bar(models, f1_scores, color=colors)
    axes[0, 1].set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('F1-Score')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].tick_params(axis='x', rotation=45)
    for i, v in enumerate(f1_scores):
        axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # Precision comparison
    axes[1, 0].bar(models, precisions, color=colors)
    axes[1, 0].set_title('Precision Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(precisions):
        axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # Recall comparison
    axes[1, 1].bar(models, recalls, color=colors)
    axes[1, 1].set_title('Recall Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].tick_params(axis='x', rotation=45)
    for i, v in enumerate(recalls):
        axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Classification Models Results</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">2 ML Models | 2 DL Models</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Results directory
        results_dir = st.text_input(
            "Results Directory",
            value=st.session_state.results_dir,
            help="Path to classification results directory"
        )
        st.session_state.results_dir = results_dir
        
        # Target column
        target_column = st.selectbox(
            "Classification Target",
            options=["subject", "difficulty", "question_type"],
            index=0,
            help="What was classified?"
        )
        st.session_state.target_column = target_column
        
        # Load results button
        if st.button("🔄 Load Results", use_container_width=True):
            st.rerun()
        
        st.divider()
        
        # Information
        st.header("ℹ About")
        st.markdown("""
        This dashboard displays results from:
        
        **Machine Learning Models:**
        - Random Forest (TF-IDF)
        - SVM (TF-IDF)
        
        **Deep Learning Models:**
        - LSTM (Word2Vec)
        - CNN (Word2Vec)
        
        **Metrics Shown:**
        - Accuracy, Precision, Recall, F1-Score
        - AUC, Exact Match, Top-k Accuracy
        - Confusion Matrices
        - Training History Plots
        """)
    
    # Main content
    results_dir = st.session_state.results_dir
    
    if not os.path.exists(results_dir):
        st.error(f"Results directory not found: {results_dir}")
        st.info("💡 Make sure you've trained the classification models first using train_classification.py")
        return
    
    # Model names (matching actual file names)
    ml_models = ["RandomForest", "LinearSVM"]
    dl_models = ["LSTM", "CNN"]
    all_models = ml_models + dl_models
    
    # Display names for UI
    display_names = {
        "RandomForest": "Random Forest",
        "LinearSVM": "SVM",
        "LSTM": "LSTM",
        "CNN": "CNN"
    }
    
    # Load all metrics
    all_metrics = {}
    for model_name in all_models:
        all_metrics[model_name] = load_all_model_metrics(results_dir, model_name)
    
    # Model Comparison Section
    st.header("📈 Model Comparison")
    st.markdown("Compare all models side-by-side on test set")
    
    # Create comparison dataframe
    comparison_data = []
    for model_name in all_models:
        if model_name in all_metrics and 'test' in all_metrics[model_name]:
            test_metrics = all_metrics[model_name]['test']
            if test_metrics:
                # Format metrics, showing N/A for missing values
                auc_val = test_metrics.get('auc')
                top3_val = test_metrics.get('top_3_accuracy')
                
                comparison_data.append({
                    'Model': display_names.get(model_name, model_name),
                    'Type': 'ML' if model_name in ml_models else 'DL',
                    'Accuracy': f"{test_metrics.get('accuracy', 0):.4f}",
                    'Precision': f"{test_metrics.get('precision', 0):.4f}",
                    'Recall': f"{test_metrics.get('recall', 0):.4f}",
                    'F1-Score': f"{test_metrics.get('f1_score', 0):.4f}",
                    'AUC': f"{auc_val:.4f}" if auc_val is not None else "N/A",
                    'Top-3 Acc': f"{top3_val:.4f}" if top3_val is not None else "N/A"
                })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Comparison visualization
        compare_models(all_metrics)
    
    # Machine Learning Models Section
    st.markdown("---")
    st.header("🤖 Machine Learning Models")
    
    for model_name in ml_models:
        st.markdown(f'<div class="model-section">', unsafe_allow_html=True)
        st.subheader(f"🌳 {display_names.get(model_name, model_name)}")
        
        files = get_model_files(results_dir, model_name)
        
        # Metrics tabs
        tab1, tab2 = st.tabs(["📊 Test Metrics", " Confusion Matrix"])
        
        with tab1:
            if model_name in all_metrics and 'test' in all_metrics[model_name]:
                display_model_metrics(all_metrics[model_name]['test'], model_name, "Test")
            else:
                st.warning(f"No test metrics found for {model_name}")
        
        with tab2:
            display_confusion_matrix(files['confusion_matrix'], model_name)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Deep Learning Models Section
    st.markdown("---")
    st.header("🧠 Deep Learning Models")
    
    for model_name in dl_models:
        st.markdown(f'<div class="model-section">', unsafe_allow_html=True)
        st.subheader(f"🔮 {display_names.get(model_name, model_name)}")
        
        files = get_model_files(results_dir, model_name)
        
        # Metrics tabs
        tab1, tab2 = st.tabs(["📊 Test Metrics", "🎯 Confusion Matrix"])
        
        with tab1:
            if model_name in all_metrics and 'test' in all_metrics[model_name]:
                display_model_metrics(all_metrics[model_name]['test'], model_name, "Test")
            else:
                st.warning(f"No test metrics found for {model_name}")
        
        with tab2:
            display_confusion_matrix(files['confusion_matrix'], model_name)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 2rem;'>
            <p>Classification Models Results Dashboard</p>
            <p>2 ML Models | 2 DL Models</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

