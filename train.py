"""
Training script for T5 Quiz Generator Model
"""
import os
import torch
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import DatasetDict
import matplotlib.pyplot as plt
import numpy as np
from config import ModelConfig, DataConfig
from data_loader import QuizDataLoader
import json

def plot_training_history(history: dict, output_dir: str):
    """Plot training and validation curves"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot loss
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    if 'loss' in history:
        plt.plot(history['loss'], label='Train Loss', marker='o')
    if 'eval_loss' in history:
        plt.plot(history['eval_loss'], label='Validation Loss', marker='s')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot ROUGE scores (if available)
    plt.subplot(1, 2, 2)
    rouge_metrics = ['eval_rouge1', 'eval_rouge2', 'eval_rougeL']
    for metric in rouge_metrics:
        if metric in history:
            plt.plot(history[metric], label=metric.replace('eval_', '').upper(), marker='o')
    plt.xlabel('Step')
    plt.ylabel('Score')
    plt.title('Validation ROUGE Scores')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training plots saved to {output_dir}/training_history.png")

class QuizGeneratorTrainer:
    """Trainer class for Quiz Generator Model"""
    
    def __init__(self, config: ModelConfig, data_config: DataConfig):
        self.config = config
        self.data_config = data_config
        
        # Initialize tokenizer
        print(f"Loading tokenizer: {config.model_name}")
        self.tokenizer = T5Tokenizer.from_pretrained(config.model_name)
        
        # Initialize model
        print(f"Loading model: {config.model_name}")
        self.model = T5ForConditionalGeneration.from_pretrained(config.model_name)
        
        # Initialize data loader
        self.data_loader = QuizDataLoader(self.tokenizer, config, data_config)
        
        # Training history
        self.training_history = {
            'loss': [],
            'eval_loss': [],
            'eval_rouge1': [],
            'eval_rouge2': [],
            'eval_rougeL': []
        }
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        import sys
        import os
        # Temporarily remove current directory from path to avoid importing local evaluate.py
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir in sys.path:
            sys.path.remove(current_dir)
        from evaluate import load as eval_load
        # Add it back
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        import nltk
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        rouge = eval_load("rouge")
        predictions, labels = eval_pred
        
        # Decode predictions
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Replace -100 in the labels as we can't decode them
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # ROUGE expects a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        
        # Compute ROUGE scores
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        
        # Extract ROUGE scores
        result = {key: value * 100 for key, value in result.items()}
        
        # Also compute exact match
        exact_matches = sum(1 for p, l in zip(decoded_preds, decoded_labels) if p.strip() == l.strip())
        result['exact_match'] = (exact_matches / len(decoded_preds)) * 100
        
        # Store in history
        if 'eval_rouge1' not in self.training_history:
            self.training_history['eval_rouge1'] = []
        if 'eval_rouge2' not in self.training_history:
            self.training_history['eval_rouge2'] = []
        if 'eval_rougeL' not in self.training_history:
            self.training_history['eval_rougeL'] = []
        
        self.training_history['eval_rouge1'].append(result['rouge1'])
        self.training_history['eval_rouge2'].append(result['rouge2'])
        self.training_history['eval_rougeL'].append(result['rougeL'])
        
        return result
    
    def train(self):
        """Train the model"""
        print("Loading and preprocessing datasets...")
        datasets = self.data_loader.prepare_datasets(
            self.data_config.dataset_path,
            seed=self.config.seed
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            logging_steps=self.config.logging_steps,
            eval_strategy=self.config.evaluation_strategy,  # Changed from evaluation_strategy
            save_strategy=self.config.save_strategy,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            seed=self.config.seed,
            fp16=self.config.fp16,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            overwrite_output_dir=self.config.overwrite_output_dir,
            logging_dir=os.path.join(self.config.output_dir, "logs"),
            report_to="tensorboard",
            save_total_limit=3,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        # Train model
        print("Starting training...")
        train_result = trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        # Save training metrics
        metrics = train_result.metrics
        metrics_file = os.path.join(self.config.output_dir, "train_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Training completed! Metrics saved to {metrics_file}")
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_metrics = trainer.evaluate(eval_dataset=datasets["test"])
        test_metrics_file = os.path.join(self.config.output_dir, "test_metrics.json")
        with open(test_metrics_file, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        print(f"Test metrics saved to {test_metrics_file}")
        
        # Plot training history
        training_state = trainer.state
        if training_state.log_history:
            for log in training_state.log_history:
                if 'loss' in log:
                    self.training_history['loss'].append(log['loss'])
                if 'eval_loss' in log:
                    self.training_history['eval_loss'].append(log['eval_loss'])
        
        plot_training_history(self.training_history, self.config.output_dir)
        
        return trainer, test_metrics

def main():
    """Main training function"""
    # Load configurations
    model_config = ModelConfig()
    data_config = DataConfig()
    
    # Create output directory
    os.makedirs(model_config.output_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = QuizGeneratorTrainer(model_config, data_config)
    
    # Train model
    trainer, test_metrics = trainer.train()
    
    print("\n" + "="*50)
    print("Training Summary")
    print("="*50)
    print(f"Model saved to: {model_config.output_dir}")
    print("\nTest Set Metrics:")
    for key, value in test_metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")

if __name__ == "__main__":
    main()

