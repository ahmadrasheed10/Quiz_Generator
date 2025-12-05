"""
Comprehensive Evaluation Module for Classification Tasks
Includes: Accuracy, Precision, Recall, F1-Score, AUC, Exact Match, Top-k Accuracy, Confusion Matrix
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from typing import Dict, List, Tuple, Optional
import json
import os

class ClassificationEvaluator:
    """Comprehensive evaluator for classification models"""
    
    def __init__(self, class_names: List[str], output_dir: str = "./results/classification"):
        """
        Initialize evaluator
        
        Args:
            class_names: List of class names
            output_dir: Directory to save evaluation results
        """
        self.class_names = class_names
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def calculate_exact_match(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Exact Match (EM) score"""
        return accuracy_score(y_true, y_pred)
    
    def calculate_top_k_accuracy(self, y_true: np.ndarray, y_proba: np.ndarray, k: int = 3) -> float:
        """Calculate Top-k Accuracy"""
        top_k_preds = np.argsort(y_proba, axis=1)[:, -k:]
        correct = 0
        for i, true_label in enumerate(y_true):
            if true_label in top_k_preds[i]:
                correct += 1
        return correct / len(y_true)
    
    def calculate_auc(self, y_true: np.ndarray, y_proba: np.ndarray, 
                     average: str = 'macro') -> float:
        """Calculate AUC score"""
        num_classes = len(self.class_names)
        if num_classes == 2:
            # Binary classification
            return roc_auc_score(y_true, y_proba[:, 1])
        else:
            # Multi-class classification
            # Convert to one-hot encoding
            y_true_onehot = np.eye(num_classes)[y_true]
            return roc_auc_score(y_true_onehot, y_proba, average=average, multi_class='ovr')
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                y_proba: Optional[np.ndarray] = None, 
                set_name: str = "test", model_name: str = None) -> Dict:
        """
        Comprehensive evaluation
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            set_name: Name of the dataset (train/val/test)
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Exact Match (same as accuracy for classification)
        metrics['exact_match'] = self.calculate_exact_match(y_true, y_pred)
        
        # Top-k Accuracy
        if y_proba is not None:
            metrics['top_1_accuracy'] = metrics['accuracy']
            metrics['top_3_accuracy'] = self.calculate_top_k_accuracy(y_true, y_proba, k=3)
            metrics['top_5_accuracy'] = self.calculate_top_k_accuracy(y_true, y_proba, k=5)
            
            # AUC
            try:
                metrics['auc'] = self.calculate_auc(y_true, y_proba)
            except Exception as e:
                print(f"Warning: Could not calculate AUC: {e}")
                metrics['auc'] = None
        else:
            metrics['top_1_accuracy'] = metrics['accuracy']
            metrics['top_3_accuracy'] = None
            metrics['top_5_accuracy'] = None
            metrics['auc'] = None
          
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        metrics['per_class_metrics'] = {
            class_name: {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1_score': float(f1_per_class[i])
            }
            for i, class_name in enumerate(self.class_names)
        }
        
        # Save metrics
        self._save_metrics(metrics, set_name, model_name)
        
        return metrics
    
    def _save_metrics(self, metrics: Dict, set_name: str, model_name: str = None):
        """Save metrics to JSON file"""
        # Create a copy without confusion matrix for JSON
        metrics_to_save = {k: v for k, v in metrics.items() if k != 'confusion_matrix'}
        metrics_to_save['confusion_matrix_shape'] = np.array(metrics['confusion_matrix']).shape
        
        # Ensure all expected metrics are present (even if None)
        # This helps identify which metrics are missing vs. not computed
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 
                           'exact_match', 'auc', 'top_1_accuracy', 
                           'top_3_accuracy', 'top_5_accuracy']
        for metric in expected_metrics:
            if metric not in metrics_to_save:
                metrics_to_save[metric] = None
        
        if model_name:
            filename = f"{model_name.replace(' ', '_')}_{set_name}_metrics.json"
        else:
            filename = f"{set_name}_metrics.json"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
    
    def plot_confusion_matrix(self, cm: np.ndarray, model_name: str, 
                            set_name: str = "test", save: bool = True):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names)
        plt.title(f'Confusion Matrix - {model_name} ({set_name})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save:
            filename = f"{model_name.replace(' ', '_')}_{set_name}_confusion_matrix.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {filepath}")
        
        plt.close()
    
    def plot_training_history(self, history: Dict, model_name: str, save: bool = True):
        """
        Plot training and validation accuracy and loss
        
        Args:
            history: Dictionary with keys 'train_acc', 'val_acc', 'train_loss', 'val_loss'
            model_name: Name of the model
            save: Whether to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(history['train_acc']) + 1)
        
        # Accuracy plot
        axes[0].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
        axes[0].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
        axes[0].set_title(f'Model Accuracy - {model_name}', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Loss plot
        axes[1].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        axes[1].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[1].set_title(f'Model Loss - {model_name}', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = f"{model_name.replace(' ', '_')}_training_history.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Training history saved to {filepath}")
        
        plt.close()
    
    def print_metrics(self, metrics: Dict, set_name: str = "test"):
        """Print metrics in a formatted way"""
        print(f"\n{'='*60}")
        print(f"Evaluation Metrics - {set_name.upper()}")
        print(f"{'='*60}")
        print(f"Accuracy:           {metrics['accuracy']:.4f}")
        print(f"Precision:          {metrics['precision']:.4f}")
        print(f"Recall:             {metrics['recall']:.4f}")
        print(f"F1-Score:           {metrics['f1_score']:.4f}")
        print(f"Exact Match (EM):   {metrics['exact_match']:.4f}")
        
        if metrics.get('top_3_accuracy') is not None:
            print(f"Top-3 Accuracy:     {metrics['top_3_accuracy']:.4f}")
        if metrics.get('top_5_accuracy') is not None:
            print(f"Top-5 Accuracy:     {metrics['top_5_accuracy']:.4f}")
        if metrics.get('auc') is not None:
            print(f"AUC:                {metrics['auc']:.4f}")
        
        print(f"\nPer-Class Metrics:")
        for class_name, class_metrics in metrics['per_class_metrics'].items():
            print(f"  {class_name}:")
            print(f"    Precision: {class_metrics['precision']:.4f}")
            print(f"    Recall:    {class_metrics['recall']:.4f}")
            print(f"    F1-Score:  {class_metrics['f1_score']:.4f}")
        print(f"{'='*60}\n")

