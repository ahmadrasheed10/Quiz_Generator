# 🧠 AI-Powered Quiz Generator & Classifier

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20HuggingFace-orange)
![Model](https://img.shields.io/badge/Model-T5%20Transformer-green)

A state-of-the-art system that uses **Generative AI** to create university-level quiz questions and **Machine Learning** to classify them. At its core, this project leverages **Google's T5 (Text-to-Text Transfer Transformer)**, fine-tuned to understand granular educational contexts like Subject, Topic, and Difficulty.

---

## 🚀 Key Features

### 1. 📝 Generative AI (The Core)
*   **Powered by T5:** Uses a fine-tuned T5-small transformer model. Unlike simple text predictors, T5 treats every problem as a text-to-text transformation, making it perfect for structured question generation.
*   **Smart Context Understanding:** Generates questions based on specific constraints:
    *   **Subject:** (e.g., "Data Structures", "Operating Systems", "MLOps")
    *   **Topic:** (e.g., "Binary Trees", "Deadlocks", "Model Deployment")
    *   **Difficulty:** Easy, Medium, Hard
*   **Diverse Question Types:** Supports MCQs, Short Answers, Long Answers, and Code Generation.
*   **Creative Sampling:** Uses Nucleus Sampling (Top-p) and Temperature scaling to ensure no two generated questions are identical.

### 2. 📊 Intelligent Classification
*   **Multi-Model Analysis:** Automatically categorizes questions using a suite of traditional and deep learning models:
    *   **Machine Learning:** Random Forest & SVM (using TF-IDF vectors).
    *   **Deep Learning:** LSTM & CNN (using Word2Vec embeddings).
*   **Performance Metrics:** Real-time tracking of Accuracy, F1-Score, and Confusion Matrices.

### 3. 📚 RAG Comparison (New!)
*   **Side-by-side Analysis:** Compare results from the Base T5 Model vs. RAG (Retrieval-Augmented Generation).
*   **Contextual Grounding:** See how retrieving relevant training data improves question relevance and diversity.
*   **Visual Similarity Search:** View the exact source chunks retrieved for each generation.

### 4. 💻 Modern UI
*   **Streamlit Dashboard:** A premium, dark-mode web interface.
*   **PDF Export:** Generate and download fully formatted quiz papers instantly.
*   **Interactive Design:** Real-time feedback and smooth animations.

---

## 🛠️ Installation

### Prerequisites
*   Python 3.8+
*   CUDA-capable GPU (Recommended for training, optional for inference)

### Setup
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/quiz-generator.git
    cd quiz-generator
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK Data:**
    ```bash
    python -c "import nltk; nltk.download('punkt')"
    ```

---

## 🏎️ Quick Start

### 1. Run the Quiz Generator App
This is the main application for generating questions.
```bash
streamlit run app.py
```
*   **URL:** `http://localhost:8501`
*   **Usage:** Select your subject (e.g., "Object Oriented Programming"), type a topic (e.g., "Inheritance"), and click **Generate**.

### 2. Run Classification Dashboard
Analyze how well the models can categorize existing questions.
```bash
streamlit run classification_app.py
```

### 3. Run RAG Comparison App
Compare the Base Model against the Retrieval-Augmented pipeline.
```bash
streamlit run rag_comparison_app.py
```

---

## 🧠 Model Training

This project allows you to train your own models on custom datasets.

### A. Training the T5 Generator
We use **Seq2Seq Fine-Tuning**. The model learns to map an *Instruction* ("Generate hard MCQ for OS...") to a *Target* ("What is a semaphore?...").

**To Train Locally:**
```bash
python train.py
```
*   **Config:** Settings like `epoch` and `learning_rate` can be adjusted in `config.py`.
*   **Output:** The fine-tuned model is saved to `results/t5-quiz-generator`.

### B. Training Classifiers
Train the Discriminative models (RF, SVM, LSTM, CNN) to recognize subjects.

**To Train Locally:**
```bash
python train_classification.py --target_column subject
```

---

## 📂 Project Structure

```
.
├── app.py                      # Main Quiz Generator UI (Streamlit)
├── classification_app.py       # Classification Dashboard UI
├── rag_comparison_app.py       # RAG Comparison UI
├── train.py                    # T5 Model Training Script
├── inference.py                # T5 Inference Logic (Generation)
├── config.py                   # Central Configuration
├── data_loader.py              # Data Processing for T5
├── quiz_data.csv               # Dataset
└── results/                    # Saved Models & Checkpoints
    └── t5-quiz-generator/      # Your fine-tuned T5 model resides here
```

## 📝 How T5 Works Here
Unlike BERT (which is an "Encoder-only" architecture tailored for understanding), **T5 is an "Encoder-Decoder" architecture**.
1.  **Encoder:** Reads your prompt ("Subject: OOP, Topic: Polymorphism").
2.  **Decoder:** Autoregressively generates the question word-by-word.
3.  **Fine-Tuning:** We update the model weights so it aligns specifically with the academic style of your `quiz_data.csv` dataset, rather than generic internet text.

---

## 🤝 Contributing
Feel free to fork this project and submit Pull Requests. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License
[MIT](https://choosealicense.com/licenses/mit/)
