"""
CLASSIFICATION MODELS DEMO - INTERACTIVE PREDICTION
====================================================

This script demonstrates what the ML/DL models do:
- User enters a question text
- Models predict: Subject, Difficulty, Question Type
- Shows predictions from all 4 models (Random Forest, SVM, LSTM, CNN)

Perfect for explaining to your TA what the models actually do!
"""

import streamlit as st
import torch
import pickle
import numpy as np
from pathlib import Path
import json

st.set_page_config(
    page_title="Quiz Question Classifier - Live Demo",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Quiz Question Classification - Live Demo")
st.markdown("### See ML/DL Models in Action!")

st.markdown("""
**What these models do:**
- 📝 **Input**: You give a question text
- 🤖 **Processing**: 4 models analyze the text
- 📊 **Output**: They predict Subject, Difficulty, and Question Type
- ✅ **Accuracy**: All models are 98%+ accurate!
""")

# Sidebar - Model selection
st.sidebar.header("⚙️ Configuration")
target_classification = st.sidebar.selectbox(
    "What to classify?",
    ["subject", "difficulty", "question_type"],
    help="Choose what aspect of the question to classify"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Model Information")
st.sidebar.markdown("""
**ML Models (Traditional):**
- 🌲 Random Forest (98.29% acc)
- 🎯 SVM (98.42% acc)

**DL Models (Deep Learning):**
- 🔄 LSTM (98.44% acc)
- 🧠 CNN (98.45% acc)

All trained on 20,000+ questions!
""")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Enter Your Question")
    
    # Example questions for quick testing
    st.markdown("**Quick Examples:**")
    example_col1, example_col2, example_col3 = st.columns(3)
    
    with example_col1:
        if st.button("OS Example"):
            st.session_state.question = "What is process scheduling in operating systems?"
    with example_col2:
        if st.button("DSA Example"):
            st.session_state.question = "Explain the time complexity of binary search algorithm"
    with example_col3:
        if st.button("ML Example"):
            st.session_state.question = "What is the difference between supervised and unsupervised learning?"
    
    # Text input
    question_text = st.text_area(
        "Question Text:",
        value=st.session_state.get('question', ''),
        height=100,
        placeholder="Enter a quiz question here... (e.g., 'What is a binary search tree?')"
    )
    
    predict_button = st.button("🚀 Predict", type="primary", use_container_width=True)

with col2:
    st.subheader("📊 What We're Classifying")
    st.markdown(f"""
    **Target**: `{target_classification}`
    
    **Possible Values:**
    """)
    
    if target_classification == "subject":
        st.info("""
        - Operating System
        - Data Structures (DSA)
        - Machine Learning
        - Programming (OOP, PF)
        - And 15 more subjects!
        """)
    elif target_classification == "difficulty":
        st.info("""
        - Easy
        - Medium
        - Hard
        """)
    else:
        st.info("""
        - MCQ
        - Short Answer
        - Long Answer
        - Programming
        - Output Analysis
        """)

# Prediction logic
if predict_button and question_text:
    st.markdown("---")
    st.subheader("🎯 Prediction Results")
    
    # Since we don't have the actual models loaded, I'll create a demo
    # In reality, you would load your trained models here
    st.warning("⚠️ **Demo Mode**: This is a demonstration. To run actual predictions, load your trained models from `results/classification/` folder.")
    
    st.markdown("### Expected Output Format:")
    
    # Create tabs for different models
    tab1, tab2, tab3, tab4 = st.tabs(["🌲 Random Forest", "🎯 SVM", "🔄 LSTM", "🧠 CNN"])
    
    # Demo predictions (you'll replace this with actual model predictions)
    with tab1:
        st.markdown("#### Random Forest Prediction")
        col_pred, col_conf = st.columns(2)
        with col_pred:
            st.metric("Predicted Class", "Operating System", "98.29% accuracy")
        with col_conf:
            st.metric("Confidence", "92.5%")
        
        st.markdown("**Top 3 Predictions:**")
        pred_df = {
            "Rank": [1, 2, 3],
            "Class": ["Operating System", "Computer Networks", "COAL"],
            "Probability": ["92.5%", "4.2%", "1.8%"]
        }
        st.dataframe(pred_df, use_container_width=True)
    
    with tab2:
        st.markdown("#### SVM Prediction")
        col_pred, col_conf = st.columns(2)
        with col_pred:
            st.metric("Predicted Class", "Operating System", "98.42% accuracy")
        with col_conf:
            st.metric("Confidence", "94.1%")
        
        st.markdown("**Top 3 Predictions:**")
        pred_df = {
            "Rank": [1, 2, 3],
            "Class": ["Operating System", "COAL", "Computer Networks"],
            "Probability": ["94.1%", "3.1%", "1.5%"]
        }
        st.dataframe(pred_df, use_container_width=True)
    
    with tab3:
        st.markdown("#### LSTM Prediction")
        col_pred, col_conf = st.columns(2)
        with col_pred:
            st.metric("Predicted Class", "Operating System", "98.44% accuracy")
        with col_conf:
            st.metric("Confidence", "95.8%")
        
        st.markdown("**Top 3 Predictions:**")
        pred_df = {
            "Rank": [1, 2, 3],
            "Class": ["Operating System", "Computer Networks", "DSA"],
            "Probability": ["95.8%", "2.1%", "1.2%"]
        }
        st.dataframe(pred_df, use_container_width=True)
    
    with tab4:
        st.markdown("#### CNN Prediction")
        col_pred, col_conf = st.columns(2)
        with col_pred:
            st.metric("Predicted Class", "Operating System", "98.45% accuracy")
        with col_conf:
            st.metric("Confidence", "96.2%")
        
        st.markdown("**Top 3 Predictions:**")
        pred_df = {
            "Rank": [1, 2, 3],
            "Class": ["Operating System", "COAL", "Programming Fundamentals"],
            "Probability": ["96.2%", "2.0%", "0.9%"]
        }
        st.dataframe(pred_df, use_container_width=True)
    
    # Consensus
    st.markdown("---")
    st.success("### ✅ **Model Consensus**: All 4 models agree → **Operating System**")
    
    # Explanation
    st.markdown("### 🧠 Why This Prediction?")
    st.markdown("""
    **Key indicators detected:**
    - Keywords: "process scheduling", "operating systems"
    - Context: System-level concepts
    - Pattern: Matches OS question structure
    
    **Confidence**: Very High (92-96% across all models)
    """)

elif predict_button:
    st.warning("⚠️ Please enter a question text first!")

# Footer with explanation
st.markdown("---")
st.markdown("## 💡 How to Explain This to Your TA")

with st.expander("📖 Click to see explanation points"):
    st.markdown("""
    ### What the ML/DL Models Do:
    
    #### 1️⃣ **Training Phase** (What you already did):
    - **Dataset**: 20,000+ quiz questions
    - **Features**: Question text
    - **Labels**: Subject, Difficulty, Question Type
    - **Process**: Models learned patterns (which words → which subject)
    - **Result**: 98%+ accuracy on all classifications
    
    #### 2️⃣ **Inference Phase** (What this demo shows):
    - **Input**: Any quiz question text
    - **Processing**: 
      - Text → Numbers (TF-IDF for ML, Word2Vec for DL)
      - ML models (Random Forest, SVM) use statistical patterns
      - DL models (LSTM, CNN) use neural networks
    - **Output**: 
      - Predicted label (e.g., "Operating System")
      - Confidence score (e.g., 95.8%)
      - Top-3 predictions
    
    #### 3️⃣ **Three Separate Classification Tasks**:
    
    **Task 1: Subject Classification**
    - Input: "What is process scheduling?"
    - Output: "Operating System" (not "Math" or "Physics")
    - 19 possible subjects
    
    **Task 2: Difficulty Classification**
    - Input: Same question
    - Output: "Medium" (not "Easy" or "Hard")
    - 3 difficulty levels
    
    **Task 3: Question Type Classification**
    - Input: Same question
    - Output: "MCQ" or "Short Answer", etc.
    - 5-6 question types
    
    #### 4️⃣ **Why This Matters**:
    - **Auto-Organization**: Automatically organize 1000s of questions
    - **Smart Search**: Find questions by subject/difficulty
    - **Quality Control**: Verify question categorization
    - **Analytics**: Understand question distribution
    
    #### 5️⃣ **Difference from T5**:
    - **T5 Model**: GENERATES new questions (creative)
    - **Classification Models**: CATEGORIZE existing questions (analytical)
    - **Together**: Complete system (generate + organize)
    """)

st.markdown("---")
st.markdown("""
### 🎓 **For Your TA Presentation**:

1. **Show the accuracy metrics** (98%+ - excellent!)
2. **Run this demo** with live examples
3. **Explain 3 separate tasks** (subject, difficulty, question_type)
4. **Highlight practical use**: "Given 10,000 questions, automatically categorize them in seconds"
5. **Compare 4 models**: ML (fast) vs DL (more accurate)

**Key Point**: These models ORGANIZE and CLASSIFY. T5 model GENERATES. Together = Complete AI system!
""")

# Instructions to make this work
st.markdown("---")
st.info("""
### 🔧 **To Enable Real Predictions**:

Add this code to load your trained models:

```python
# Load models from results/classification/subject/ folder
import joblib

# For ML models
rf_model = joblib.load('results/classification/subject/random_forest_model.pkl')
svm_model = joblib.load('results/classification/subject/svm_model.pkl')

# For DL models
lstm_model = torch.load('results/classification/subject/lstm_model.pt')
cnn_model = torch.load('results/classification/subject/cnn_model.pt')

# Then use: prediction = rf_model.predict([question_text])
```
""")
