# 🧠 AI-Powered Quiz Generator & Classifier

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20HuggingFace-orange)
![Model](https://img.shields.io/badge/Model-T5%20Transformer-green)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)

A comprehensive AI system that combines **Generative AI** (T5 Transformer), **Deep Learning** (LSTM, CNN), and **Machine Learning** (Random Forest, SVM) to generate, classify, and evaluate university-level quiz questions. Features 4 specialized Streamlit frontends for different use cases.

---

## 📋 Table of Contents

- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Frontend Applications](#-frontend-applications)
- [Installation](#️-installation)
- [Quick Start](#-quick-start)
- [Model Training](#-model-training)
- [Project Structure](#-project-structure)
- [Technical Details](#-technical-details)
- [Usage Guide](#-usage-guide)
- [Troubleshooting](#-troubleshooting)

---

## 🚀 Key Features

### 1. 📝 Generative AI (T5 Transformer)
- **Powered by T5-Small:** Fine-tuned Google T5 model for question generation
- **Context-Aware Generation:** Creates questions based on:
  - **Subject:** 17 subjects (CS, Math, Physics, etc.)
  - **Topic:** Specific topics within each subject
  - **Difficulty:** Easy, Medium, Hard
  - **Question Type:** MCQ, Short Answer, Long Answer, Programming
- **Creative Sampling:** Nucleus sampling and temperature scaling for diverse outputs
- **RAG Enhancement:** Retrieval-Augmented Generation for improved quality

### 2. 🎯 Classification Models (98%+ Accuracy)
- **4 Trained Models:**
  - **Random Forest:** 98.29% accuracy (TF-IDF features)
  - **SVM:** 98.42% accuracy (TF-IDF features)
  - **LSTM:** 98.44% accuracy (Word embeddings)
  - **CNN:** 98.45% accuracy (Word embeddings)
- **Subject Classification:** 19 subject categories
- **Real-time Predictions:** Instant classification with confidence scores

### 3. 📊 RAG (Retrieval-Augmented Generation)
- **Dual-Mode Generation:** Baseline vs RAG comparison
- **Semantic Search:** Uses sentence-transformers for context retrieval
- **Quality Metrics:**
  - **Completeness:** Measures topic coverage
  - **Faithfulness:** Measures factual grounding
- **Visual Analysis:** Shows retrieved examples and similarity scores

### 4. 💻 Modern Web Interface
- **4 Specialized Frontends:** Each optimized for different tasks
- **Beautiful UI:** Gradient themes, animations, responsive design
- **PDF Export:** Download formatted quiz papers
- **Real-time Feedback:** Live predictions and metrics

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    USER INTERFACE LAYER                  │
├─────────┬─────────────┬───────────────┬─────────────────┤
│ app.py  │ prediction  │ classification│ rag_comparison  │
│ (Main)  │ _app.py     │ _app.py       │ _app.py         │
└─────────┴─────────────┴───────────────┴─────────────────┘
           ↓              ↓               ↓                ↓
┌─────────────────────────────────────────────────────────┐
│                     MODELS LAYER                         │
├──────────────────┬──────────────────┬───────────────────┤
│  T5 Generator    │  Classification  │   RAG System      │
│  (Seq2Seq)       │  (4 Models)      │   (Retrieval)     │
└──────────────────┴──────────────────┴───────────────────┘
           ↓              ↓                      ↓
┌─────────────────────────────────────────────────────────┐
│                      DATA LAYER                          │
│  quiz_data.csv  |  saved_models/  |  rag_cache/         │
└─────────────────────────────────────────────────────────┘
```

---

## 🖥️ Frontend Applications

### 1. 📝 **Main Quiz Generator** (`app.py`)

**Purpose:** Production quiz question generation

**Features:**
- Select from 17 subjects
- Specify topic, difficulty, and question type
- Generate single or multiple questions
- Download as formatted PDF
- Fast inference (2-5 seconds per question)

**When to Use:**
> "I need to create 10 MCQs about Operating Systems for an exam"

**Run:**
```bash
streamlit run app.py
```

**Screenshot:**
```
┌────────────────────────────────────────────┐
│       📝 Quiz Generator                    │
├────────────────────────────────────────────┤
│ Subject:     [Operating System ▼]         │
│ Topic:       Process Scheduling            │
│ Difficulty:  [Medium ▼]                    │
│ Type:        [MCQ ▼]                       │
│                                            │
│        [🚀 Generate Question]              │
└────────────────────────────────────────────┘
```

---

### 2. 🎯 **Prediction App** (`prediction_app.py`)

**Purpose:** Live subject classification with AI models

**Features:**
- Real-time classification of question text
- Shows predictions from all 4 models simultaneously
- Confidence scores and top-3 predictions
- Model consensus detection
- Comparison table showing model agreement
- Visual accuracy charts

**When to Use:**
> "I have a question and want to know what subject it belongs to"

**Run:**
```bash
streamlit run prediction_app.py
```

**Screenshot:**
```
┌────────────────────────────────────────────┐
│     🎯 Quiz Question Classifier            │
│  Subject Classification | 98%+ Accuracy    │
├────────────────────────────────────────────┤
│ Enter Question:                            │
│ [What is process scheduling in OS?    ]   │
│                                            │
│         [🚀 Predict with All Models]       │
├────────────────────────────────────────────┤
│  ✅ Model Consensus                        │
│     Operating System                       │
│     4/4 models agree (97.5% confidence)    │
├────────────────────────────────────────────┤
│  Individual Predictions:                   │
│  🌲 Random Forest:  OS (94.2%)            │
│  🎯 SVM:            OS (98.1%)            │
│  🔄 LSTM:           OS (96.8%)            │
│  🧠 CNN:            OS (100%)  ⭐         │
└────────────────────────────────────────────┘
```

**Key Sections:**
- **Model Accuracies:** Shows bar chart with 98%+ scores
- **Model Comparison:** Side-by-side predictions
- **Confidence Metrics:** Visual confidence bars
- **Subject Classes:** 19 different categories

---

### 3. 📊 **Classification Dashboard** (`classification_app.py`)

**Purpose:** View training results and model performance

**Features:**
- Comprehensive metrics for all 4 models
- Confusion matrices visualization
- Per-class precision/recall/F1-score
- Model accuracy comparison charts
- Training history plots (for DL models)
- Detailed performance breakdown

**When to Use:**
> "I want to analyze how well my models performed during training"

**Run:**
```bash
streamlit run classification_app.py
```

**Screenshot:**
```
┌────────────────────────────────────────────┐
│   📊 Classification Models Results         │
│      2 ML Models | 2 DL Models            │
├────────────────────────────────────────────┤
│  📈 Model Comparison                       │
│                                            │
│  Model         │ Accuracy │ F1-Score      │
│  ─────────────────────────────────────────│
│  Random Forest │ 98.29%   │ 98.21%       │
│  SVM           │ 98.42%   │ 98.35%  ⭐   │
│  LSTM          │ 98.44%   │ 98.38%       │
│  CNN           │ 98.45%   │ 98.40%       │
│                                            │
│  [📊 Accuracy Comparison Chart]            │
└────────────────────────────────────────────┘
```

**Displays:**
- **Overall Metrics:** Accuracy, Precision, Recall, F1-Score, AUC
- **Top-3 Accuracy:** How often correct answer is in top-3 predictions
- **Confusion Matrix:** Visual heatmap of predictions vs actual
- **Per-Class Metrics:** Performance for each of 19 subjects

---

### 4. 🧠 **RAG Comparison** (`rag_comparison_app.py`)

**Purpose:** Compare baseline T5 vs RAG-enhanced generation

**Features:**
- Side-by-side question comparison
- Completeness score (topic coverage)
- Faithfulness score (factual accuracy)
- Performance breakdown (metric-by-metric)
- Retrieved context display with similarity scores
- Winner highlighting

**When to Use:**
> "I want to see if RAG actually improves question quality"

**Run:**
```bash
streamlit run rag_comparison_app.py
```

**Screenshot:**
```
┌────────────────────────────────────────────┐
│   🔍 Baseline vs RAG Comparison            │
├────────────────────────────────────────────┤
│ Subject:     [NLP ▼]                       │
│ Topic:       Tokenization                  │
│ Difficulty:  [Medium ▼]                    │
│                                            │
│ ℹ️ Working with Subject: NLP               │
│                                            │
│        [🚀 Run Comparison]                 │
├────────────────────────────────────────────┤
│  🏆 Performance Breakdown                  │
│  ✅ Completeness: RAG wins by 0.25         │
│  ✅ Faithfulness: RAG wins by 0.15         │
├────────────────────────────────────────────┤
│  📝 Baseline          │  🧠 RAG 🏆         │
│  ─────────────────────────────────────────│
│  [Generated Question] │ [Generated Ques.]  │
│  Completeness: 0.75   │ Completeness: 1.00 │
│  Faithfulness: 0.60   │ Faithfulness: 0.75 │
│  Overall: 0.68        │ Overall: 0.88 ✅   │
└────────────────────────────────────────────┘
```

**Metrics Explained:**
- **Completeness:** Does the question cover the topic thoroughly?
- **Faithfulness:** Is the question factually accurate and grounded?
- **Overall Score:** Average of both metrics

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional, recommended for training)
- 8GB+ RAM

### Step-by-Step Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/quiz-generator.git
   cd quiz-generator
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data:**
   ```bash
   python -c "import nltk; nltk.download('punkt')"
   ```

5. **Verify installation:**
   ```bash
   python -c "import torch; import transformers; print('✅ All dependencies installed')"
   ```

---

## 🏎️ Quick Start

### Option 1: Use Pre-trained Models (Fastest)

If you have pre-trained models in `results/` and `saved_models/`:

```bash
# 1. Main Quiz Generator
streamlit run app.py

# 2. Live Classification
streamlit run prediction_app.py

# 3. View Training Results
streamlit run classification_app.py

# 4. Compare RAG vs Baseline
streamlit run rag_comparison_app.py
```

### Option 2: Train from Scratch

If starting fresh:

```bash
# 1. Train T5 Generator (takes 2-4 hours on GPU)
python train.py

# 2. Train Classification Models (takes 30-60 minutes)
python train_classification.py --target_column subject

# 3. Run any frontend
streamlit run app.py
```

---

## 🧠 Model Training

### A. Train T5 Question Generator

**What it does:** Fine-tunes T5-small to generate questions

**Command:**
```bash
python train.py
```

**Configuration** (`config.py`):
```python
ModelConfig(
    model_name="t5-small",           # Base model
    max_length=128,                  # Max question length
    num_train_epochs=3,              # Training epochs
    learning_rate=5e-5,              # Learning rate
    per_device_train_batch_size=8    # Batch size
)
```

**Output:**
```
results/
└── t5-quiz-generator/
    ├── config.json
    ├── pytorch_model.bin
    ├── tokenizer_config.json
    └── training_history.png
```

**Time:** 2-4 hours on GPU, 8-12 hours on CPU

---

### B. Train Classification Models

**What it does:** Trains Random Forest, SVM, LSTM, CNN for subject classification

**Command:**
```bash
python train_classification.py --target_column subject
```

**Options:**
- `--target_column`: What to classify (`subject`, `difficulty`, `question_type`)
- `--test_size`: Test split ratio (default: 0.2)

**Output:**
```
results/
└── classification/
    └── subject/
        ├── RandomForest_test_metrics.json
        ├── LinearSVM_test_metrics.json
        ├── LSTM_test_metrics.json
        ├── CNN_test_metrics.json
        └── *_confusion_matrix.png
```

**For Prediction App:**
Also saves to `saved_models/subject/` for real-time inference:
```
saved_models/
└── subject/
    ├── random_forest.pkl
    ├── svm.pkl
    ├── lstm_model.pt
    ├── cnn_model.pt
    ├── tfidf_vectorizer.pkl
    ├── vocab.pkl
    ├── label_encoder.pkl
    └── model_comparison.png
```

**Time:** 30-60 minutes on GPU, 1-2 hours on CPU

---

### C. Alternative: Train on Google Colab

**For Classification Models:**

1. Upload `Train_Models_Save_For_Prediction.ipynb` to Colab
2. Upload `quiz_data.csv` when prompted
3. Run all cells
4. Download `saved_models.zip`
5. Extract in your project folder

**Benefits:**
- Free GPU access
- Faster training
- No local setup needed

---

## 📂 Project Structure

```
Quiz Generator/
│
├── 🎨 Frontend Applications
│   ├── app.py                          # Main quiz generator UI
│   ├── prediction_app.py               # Live classification UI
│   ├── classification_app.py           # Results dashboard UI
│   └── rag_comparison_app.py           # RAG evaluation UI
│
├── 🧠 Model Training
│   ├── train.py                        # T5 model training
│   ├── train_classification.py         # Classification models training
│   └── Train_Models_Save_For_Prediction.ipynb  # Colab notebook
│
├── ⚙️ Core Logic
│   ├── inference.py                    # T5 generation logic
│   ├── rag_inference.py                # RAG generation logic
│   ├── rag_evaluator.py                # Completeness/faithfulness metrics
│   ├── data_loader.py                  # Data preprocessing
│   └── config.py                       # Configuration settings
│
├── 📊 Data & Models
│   ├── quiz_data.csv                   # Training dataset (20K+ questions)
│   ├── results/                        # Training outputs
│   │   ├── t5-quiz-generator/          # Fine-tuned T5 model
│   │   └── classification/             # Classification results
│   │       └── subject/
│   │           ├── *_test_metrics.json
│   │           └── *_confusion_matrix.png
│   └── saved_models/                   # Models for prediction app
│       └── subject/
│           ├── random_forest.pkl
│           ├── svm.pkl
│           ├── lstm_model.pt
│           ├── cnn_model.pt
│           └── model_comparison.png
│
├── 📝 Documentation
│   ├── README.md                       # This file
│   ├── CLASSIFICATION_README.md        # Classification details
│   └── verify_models.py                # Model verification script
│
└── 🔧 Configuration
    └── requirements.txt                # Python dependencies
```

---

## 🔬 Technical Details

### T5 Model Architecture

**Encoder-Decoder Transformer:**
```
Input Prompt: "Generate: Subject=NLP, Topic=Tokenization, Difficulty=medium, Type=MCQ"
                           ↓
                    ┌─────────────┐
                    │   Encoder   │  (Understanding)
                    └─────────────┘
                           ↓
                    ┌─────────────┐
                    │   Decoder   │  (Generation)
                    └─────────────┘
                           ↓
Output: "What is the purpose of tokenization in NLP? A) Split text..."
```

**Fine-tuning Process:**
1. Load pre-trained T5-small (60M parameters)
2. Train on quiz_data.csv (20K+ examples)
3. Update weights to match academic question style
4. Save fine-tuned model

---

### Classification Models

**1. Random Forest (ML)**
- **Input:** TF-IDF vectors (5000 features)
- **Architecture:** 100 decision trees
- **Accuracy:** 98.29%

**2. SVM (ML)**
- **Input:** TF-IDF vectors
- **Architecture:** Linear SVM with calibration
- **Accuracy:** 98.42%

**3. LSTM (DL)**
- **Input:** Word embeddings (512 max length)
- **Architecture:** 
  - Embedding layer (vocab_size × 100)
  - Bidirectional LSTM (128 hidden units × 2 layers)
  - Fully connected output
- **Accuracy:** 98.44%

**4. CNN (DL)**
- **Input:** Word embeddings
- **Architecture:**
  - Embedding layer
  - Conv1D filters (3, 4, 5-grams)
  - Max pooling
  - Fully connected output
- **Accuracy:** 98.45%

---

### RAG System

**How it Works:**

1. **Index Creation:**
   ```
   quiz_data.csv → sentence-transformers → embeddings → FAISS index
   ```

2. **Retrieval:**
   ```
   User Query → Embed → Find Top-K Similar → Retrieve Context
   ```

3. **Generation:**
   ```
   Context + Prompt → T5 Model → Enhanced Question
   ```

**Metrics:**
- **Completeness:** Checks if question covers topic keywords
- **Faithfulness:** Checks if content matches retrieved context

---

## 📖 Usage Guide

### 1. Generating Quiz Questions (`app.py`)

**Step-by-step:**

1. Run: `streamlit run app.py`
2. Load model (first time only)
3. Select subject from dropdown
4. Enter topic (e.g., "Binary Search Trees")
5. Choose difficulty and question type
6. Click "Generate Question"
7. Download as PDF if needed

**Tips:**
- Start with 1 question to test
- CPU inference: 5-10 seconds per question
- GPU inference: 1-2 seconds per question
- Use temperature=0.9 for creative questions

---

### 2. Classifying Questions (`prediction_app.py`)

**Step-by-step:**

1. Run: `streamlit run prediction_app.py`
2. Enter or paste a question
3. Click  "Predict with All Models"
4. View predictions from all 4 models
5. Check model consensus

**Example:**
```
Input: "What is the time complexity of quicksort?"

Output:
✅ Model Consensus: Data Structures and Algorithms
  4/4 models agree (96.2% avg confidence)

Individual:
  Random Forest: DSA (94.5%)
  SVM: DSA (97.8%)
  LSTM: DSA (95.3%)
  CNN: DSA (97.2%)
```

---

### 3. Viewing Results (`classification_app.py`)

**Step-by-step:**

1. Run: `streamlit run classification_app.py`
2. Select classification target (subject/difficulty/type)
3. View comparison table
4. Expand individual models for details
5. Check confusion matrices

**What to Look For:**
- Overall accuracy (should be >95%)
- Per-class F1-scores (balanced performance)
- Confusion matrix (where models fail)
- AUC scores (probability calibration)

---

### 4. Comparing RAG vs Baseline (`rag_comparison_app.py`)

**Step-by-step:**

1. Run: `streamlit run rag_comparison_app.py`
2. Wait for embeddings to load (first time: 10-20 min)
3. Select subject, topic, difficulty
4. Click "Run Comparison"
5. Review metrics and questions

**Understanding Results:**
```
✅ Completeness: RAG wins by 0.25
  → RAG question covers more topic aspects

✅ Faithfulness: RAG wins by 0.15
  → RAG question is more factually grounded
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Model Not Found
```
Error: Model path not found: ./results/t5-quiz-generator
```

**Solution:**
```bash
# Train the model first
python train.py
```

---

#### 2. Classification Models Missing
```
Error: No models found in ./saved_models/subject/
```

**Solution:**
```bash
# Train classification models
python train_classification.py --target_column subject

# Or run verification script
python verify_models.py
```

---

#### 3. RAG App Slow First Time
```
Computing embeddings (this may take a while, but will be cached)...
```

**This is normal!**
- First run: 10-20 minutes (computes embeddings)
- Subsequent runs: 5-10 seconds (loads from cache)

---

#### 4. Out of Memory (GPU)
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```python
# In config.py, reduce batch size:
per_device_train_batch_size=4  # Instead of 8

# Or use CPU:
device = "cpu"
```

---

#### 5. Import Errors
```
ModuleNotFoundError: No module named 'transformers'
```

**Solution:**
```bash
pip install -r requirements.txt
```

---

### Verification Checklist

**Before TA Demo:**

```bash
# 1. Check if T5 model exists
ls results/t5-quiz-generator/

# 2. Check if classification models exist
python verify_models.py

# 3. Test main app
streamlit run app.py

# 4. Test classification
streamlit run prediction_app.py

# 5. Verify results dashboard
streamlit run classification_app.py

# 6. Test RAG (be patient on first run!)
streamlit run rag_comparison_app.py
```

---

## 🎓 For TA Presentation

### Demo Script

**1. Introduction (2 min)**
> "This project uses AI to generate and classify quiz questions using T5 transformers and machine learning."

**2. Main Generator Demo (`app.py`) - (3 min)**
- Show subject dropdown
- Generate an MCQ about "Process Scheduling" in OS
- Download as PDF
- Highlight: "This uses fine-tuned T5 model trained on 20,000 questions"

**3. Classification Demo (`prediction_app.py`) - (3 min)**
- Paste a question: "What is binary search?"
- Show all 4 models predicting "DSA" with 98%+ confidence
- Highlight: "4 different models, all achieving 98%+ accuracy"

**4. Results Dashboard (`classification_app.py`) - (2 min)**
- Show comparison table
- Display confusion matrix
- Highlight: "CNN achieved 98.45%, highest accuracy with perfect top-3 score"

**5. RAG Comparison (`rag_comparison_app.py`) - (2 min)**
- Run comparison for "Tokenization" in NLP
- Show RAG wins in both completeness and faithfulness
- Highlight: "RAG improves quality by retrieving similar examples"

**6. Questions (3 min)**
- Be ready to explain T5 architecture
- Discuss why 4 different models
- Explain completeness vs faithfulness metrics

---

## 🤝 Contributing

Feel free to fork this project and submit Pull Requests!

**Areas for Contribution:**
- Additional question types (True/False, Fill-in-blank)
- More subjects and topics
- Improved RAG retrieval strategies
- Better evaluation metrics

---

## 📄 License

[MIT License](https://choosealicense.com/licenses/mit/)

---

## 📞 Contact

For questions or issues, please open a GitHub issue or contact the maintainers.

---

## 🙏 Acknowledgments

- **T5 Model:** Google Research
- **Sentence Transformers:** UKPLab
- **Streamlit:** Streamlit Inc.
- **Dataset:** Custom curated from university materials

---

**Made with ❤️ using Transformers, PyTorch, and Streamlit**
