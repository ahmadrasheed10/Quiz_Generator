"""
Evaluation script for Quiz Generator Model
"""
import torch
import pandas as pd
import numpy as np
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import List, Dict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Import evaluate library while avoiding naming conflict
# Temporarily remove current directory from path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir in sys.path:
    sys.path.remove(current_dir)
from evaluate import load as eval_load
# Add it back
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class QuizEvaluator:
    """Class for evaluating the Quiz Generator Model"""
    
    def __init__(self, model_path: str, device: str = None):
        """Initialize evaluator"""
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading model from: {model_path}")
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize metrics
        self.rouge = eval_load("rouge")
        self.bleu = eval_load("bleu")
    
    def generate_question(self, input_text: str, max_length: int = 128) -> str:
        """Generate question from input text"""
        inputs = self.tokenizer(
            input_text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    def exact_match(self, predictions: List[str], references: List[str]) -> float:
        """Calculate Exact Match (EM) score"""
        matches = sum(1 for p, r in zip(predictions, references) if p.strip().lower() == r.strip().lower())
        return (matches / len(predictions)) * 100 if predictions else 0.0
    
    def top_k_accuracy(self, predictions: List[str], references: List[str], k: int = 1) -> float:
        """Calculate Top-k Accuracy (simplified for text generation)"""
        # For text generation, we use word overlap as proxy
        correct = 0
        for pred, ref in zip(predictions, references):
            pred_words = set(pred.lower().split())
            ref_words = set(ref.lower().split())
            overlap = len(pred_words & ref_words)
            if overlap >= k:
                correct += 1
        return (correct / len(predictions)) * 100 if predictions else 0.0
    
    def compute_rouge_scores(self, predictions: List[str], references: List[str]) -> Dict:
        """Compute ROUGE scores"""
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in predictions]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in references]
        
        result = self.rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True
        )
        
        return {key: value * 100 for key, value in result.items()}
    
    def compute_bleu_score(self, predictions: List[str], references: List[str]) -> float:
        """Compute BLEU score"""
        # Format for BLEU: predictions and references as lists of token lists
        pred_tokens = [pred.split() for pred in predictions]
        ref_tokens = [[ref.split()] for ref in references]
        
        result = self.bleu.compute(
            predictions=pred_tokens,
            references=ref_tokens
        )
        
        return result['bleu'] * 100
    
    def evaluate(self, test_data_path: str, output_dir: str = "./evaluation_results"):
        """Evaluate model on test dataset"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Load test data
        print(f"Loading test data from: {test_data_path}")
        df = pd.read_csv(test_data_path)
        
        # Generate input texts
        input_texts = []
        for _, row in df.iterrows():
            subject = str(row['subject']).strip()
            topic = str(row['topic']).strip()
            difficulty = str(row['difficulty']).strip()
            question_type = str(row['question_type']).strip() if 'question_type' in row else "MCQ"
            input_text = f"Generate {difficulty} {question_type} question for {subject} topic: {topic}"
            input_texts.append(input_text)
        
        # Get references (ground truth questions)
        references = df['question'].tolist()
        
        # Generate predictions
        print("Generating predictions...")
        predictions = []
        for i, input_text in enumerate(input_texts):
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(input_texts)} examples...")
            generated = self.generate_question(input_text)
            predictions.append(generated)
        
        print("Computing metrics...")
        
        # Compute all metrics
        metrics = {}
        
        # Exact Match
        metrics['exact_match'] = self.exact_match(predictions, references)
        
        # Top-k Accuracy
        metrics['top_1_accuracy'] = self.top_k_accuracy(predictions, references, k=1)
        metrics['top_3_accuracy'] = self.top_k_accuracy(predictions, references, k=3)
        metrics['top_5_accuracy'] = self.top_k_accuracy(predictions, references, k=5)
        
        # ROUGE scores
        rouge_scores = self.compute_rouge_scores(predictions, references)
        metrics.update(rouge_scores)
        
        # BLEU score
        metrics['bleu'] = self.compute_bleu_score(predictions, references)
        
        # Print metrics
        print("\n" + "="*60)
        print("Evaluation Results")
        print("="*60)
        print(f"Exact Match: {metrics['exact_match']:.2f}%")
        print(f"Top-1 Accuracy: {metrics['top_1_accuracy']:.2f}%")
        print(f"Top-3 Accuracy: {metrics['top_3_accuracy']:.2f}%")
        print(f"Top-5 Accuracy: {metrics['top_5_accuracy']:.2f}%")
        print(f"\nROUGE Scores:")
        print(f"  ROUGE-1: {metrics['rouge1']:.2f}%")
        print(f"  ROUGE-2: {metrics['rouge2']:.2f}%")
        print(f"  ROUGE-L: {metrics['rougeL']:.2f}%")
        print(f"\nBLEU Score: {metrics['bleu']:.2f}%")
        print("="*60)
        
        # Save metrics
        import json
        metrics_file = os.path.join(output_dir, "evaluation_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to: {metrics_file}")
        
        # Save predictions
        results_df = pd.DataFrame({
            'input': input_texts,
            'reference': references,
            'prediction': predictions
        })
        results_file = os.path.join(output_dir, "predictions.csv")
        results_df.to_csv(results_file, index=False)
        print(f"Predictions saved to: {results_file}")
        
        # Plot metrics comparison
        self.plot_metrics(metrics, output_dir)
        
        # Sample predictions
        print("\nSample Predictions:")
        print("-"*60)
        for i in range(min(5, len(predictions))):
            print(f"\nExample {i+1}:")
            print(f"Input: {input_texts[i]}")
            print(f"Reference: {references[i]}")
            print(f"Prediction: {predictions[i]}")
        
        return metrics

    def plot_metrics(self, metrics: Dict, output_dir: str):
        """Plot evaluation metrics"""
        # Prepare data for plotting
        rouge_metrics = ['rouge1', 'rouge2', 'rougeL']
        rouge_values = [metrics.get(m, 0) for m in rouge_metrics]
        
        accuracy_metrics = ['exact_match', 'top_1_accuracy', 'top_3_accuracy', 'top_5_accuracy']
        accuracy_values = [metrics.get(m, 0) for m in accuracy_metrics]
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot ROUGE scores
        axes[0].bar(rouge_metrics, rouge_values, color=['#3498db', '#2ecc71', '#e74c3c'])
        axes[0].set_ylabel('Score (%)')
        axes[0].set_title('ROUGE Scores')
        axes[0].set_ylim([0, 100])
        axes[0].grid(axis='y', alpha=0.3)
        for i, v in enumerate(rouge_values):
            axes[0].text(i, v + 1, f'{v:.2f}%', ha='center', va='bottom')
        
        # Plot Accuracy metrics
        axes[1].bar(accuracy_metrics, accuracy_values, color=['#9b59b6', '#1abc9c', '#f39c12', '#e67e22'])
        axes[1].set_ylabel('Score (%)')
        axes[1].set_title('Accuracy Metrics')
        axes[1].set_ylim([0, max(accuracy_values) * 1.2 if accuracy_values else 100])
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(axis='y', alpha=0.3)
        for i, v in enumerate(accuracy_values):
            axes[1].text(i, v + max(accuracy_values) * 0.02, f'{v:.2f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'evaluation_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Metrics plot saved to: {os.path.join(output_dir, 'evaluation_metrics.png')}")

def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Quiz Generator Model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./results/t5-quiz-generator",
        help="Path to fine-tuned model directory"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        required=True,
        help="Path to test CSV file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Output directory for evaluation results"
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model path not found: {args.model_path}")
        print("Please train the model first using train.py")
        return
    
    # Initialize evaluator
    evaluator = QuizEvaluator(args.model_path)
    
    # Evaluate
    metrics = evaluator.evaluate(args.test_data, args.output_dir)
    
    return metrics

if __name__ == "__main__":
    main()


