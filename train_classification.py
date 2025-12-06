"""
Main Training Script for Classification Tasks
Implements: 2 ML models, 2 DL models, 1 Transformer model
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional
from collections import Counter
import os
import json
from tqdm import tqdm

from classification_data_loader import ClassificationDataLoader, ClassificationDataset
from classification_models import (
    RandomForestModel, SVMModel, LSTMClassifier, 
    CNNClassifier, BERTClassifier, ModelWrapper
)
from classification_evaluator import ClassificationEvaluator

class ClassificationTrainer:
    """Main trainer class for all classification models"""
    
    def __init__(self, csv_path: str, target_column: str = 'subject', 
                output_dir: str = "./results/classification", device: Optional[str] = None):
        """
        Initialize trainer
        
        Args:
            csv_path: Path to CSV file
            target_column: Column to classify (subject, difficulty, question_type)
            output_dir: Output directory for results
            device: Device to use (cuda/cpu)
        """
        self.csv_path = csv_path
        self.target_column = target_column
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        print(f"Using device: {self.device}")
        
        # Will be initialized in prepare_data
        self.data_loaders = {}
        self.datasets = {}
        self.num_classes = None
        self.class_names = None
        self.evaluator = None
    
    def prepare_data(self):
        """Prepare data with different embedding techniques"""
        print("\n" + "="*60)
        print("Preparing Data with Different Embeddings")
        print("="*60)
        
        # TF-IDF for ML models
        print("\n1. Preparing TF-IDF embeddings for ML models...")
        self.data_loaders['tfidf'] = ClassificationDataLoader(
            self.csv_path, self.target_column, embedding_type='tfidf'
        )
        tfidf_data = self.data_loaders['tfidf'].prepare_datasets()
        self.num_classes = tfidf_data['num_classes']
        self.class_names = tfidf_data['class_names']
        
        # Word2Vec for LSTM/CNN
        print("\n2. Preparing Word2Vec embeddings for LSTM/CNN...")
        self.data_loaders['word2vec'] = ClassificationDataLoader(
            self.csv_path, self.target_column, embedding_type='word2vec'
        )
        word2vec_data = self.data_loaders['word2vec'].prepare_datasets()
        
        # BERT for Transformer
        print("\n3. Preparing BERT tokenization for Transformer model...")
        self.data_loaders['bert'] = ClassificationDataLoader(
            self.csv_path, self.target_column, embedding_type='bert'
        )
        bert_data = self.data_loaders['bert'].prepare_datasets()
        
        # Store datasets
        self.datasets = {
            'tfidf': tfidf_data,
            'word2vec': word2vec_data,
            'bert': bert_data
        }
        
        # Initialize evaluator
        self.evaluator = ClassificationEvaluator(self.class_names, self.output_dir)
        
        print(f"\nNumber of classes: {self.num_classes}")
        print(f"Class names: {self.class_names}")
    
    def train_ml_models(self):
        """Train Machine Learning models (Random Forest, SVM)"""
        print("\n" + "="*60)
        print("Training Machine Learning Models")
        print("="*60)
        
        data = self.datasets['tfidf']
        X_train = data['train']['embeddings']
        y_train = data['train']['labels']
        X_val = data['val']['embeddings']
        y_val = data['val']['labels']
        X_test = data['test']['embeddings']
        y_test = data['test']['labels']
        
        models = {}
        
        # 1. Random Forest
        print("\n1. Training Random Forest...")
        # Optimize for large datasets: reduce estimators and depth
        n_samples = X_train.shape[0]
        if n_samples > 50000:
            n_estimators = 50  # Reduce for large datasets
            max_depth = 15
            print(f"   Large dataset detected ({n_samples} samples). Using optimized parameters: n_estimators={n_estimators}, max_depth={max_depth}")
        else:
            n_estimators = 100
            max_depth = 20
        
        rf_model = RandomForestModel(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        print("   Training in progress... (this may take a few minutes)")
        rf_model.train(X_train, y_train)
        print("   ✅ Random Forest training complete!")
        models['RandomForest'] = rf_model
        
        # 2. SVM - Use Linear kernel for large datasets (much faster)
        print("\n2. Training SVM...")
        n_samples = X_train.shape[0]
        if n_samples > 30000:
            # Use LinearSVM for large datasets (much faster than RBF)
            from sklearn.svm import LinearSVC
            from sklearn.calibration import CalibratedClassifierCV
            
            print(f"   Large dataset detected ({n_samples} samples). Using LinearSVM (faster than RBF).")
            print("   Training LinearSVM... (this may take several minutes)")
            
            # Create LinearSVM - use smaller sample for calibration to speed up
            linear_svm = LinearSVC(C=1.0, random_state=42, max_iter=2000, dual=False, verbose=1)
            linear_svm.fit(X_train, y_train)
            print("   LinearSVM training complete. Calibrating for probabilities...")
            
            # Use smaller sample for calibration (faster)
            if n_samples > 50000:
                calib_sample_size = 20000
                calib_idx = np.random.choice(n_samples, size=calib_sample_size, replace=False)
                X_calib = X_train[calib_idx]
                y_calib = y_train[calib_idx]
                print(f"   Using {calib_sample_size} samples for calibration (faster)...")
            else:
                X_calib = X_train
                y_calib = y_train
            
            # Wrap for probability predictions (use fewer folds for speed)
            calibrated_svm = CalibratedClassifierCV(linear_svm, method='sigmoid', cv=2, n_jobs=-1)
            calibrated_svm.fit(X_calib, y_calib)
            
            # Create wrapper model
            class LinearSVMModel:
                def __init__(self, model):
                    self.model = model
                    self.name = "LinearSVM"
                def predict(self, X):
                    return self.model.predict(X)
                def predict_proba(self, X):
                    return self.model.predict_proba(X)
            
            svm_model = LinearSVMModel(calibrated_svm)
            print("   ✅ LinearSVM training complete!")
        else:
            # Use RBF for smaller datasets
            svm_model = SVMModel(kernel='rbf', C=1.0, random_state=42)
            print("   Training in progress...")
            svm_model.train(X_train, y_train)
            print("   ✅ SVM training complete!")
        
        models['LinearSVM'] = svm_model
        
        # Evaluate ML models
        for model_name, model in models.items():
            print(f"\nEvaluating {model_name}...")
            
            # For large datasets, skip train evaluation or use sample
            n_train = X_train.shape[0]
            if n_train > 50000:
                print(f"   Large training set ({n_train} samples). Evaluating on sample for train set...")
                # Sample 10k for train evaluation
                sample_idx = np.random.choice(n_train, size=min(10000, n_train), replace=False)
                X_train_sample = X_train[sample_idx]
                y_train_sample = y_train[sample_idx]
                y_train_pred = model.predict(X_train_sample)
                y_train_proba = model.predict_proba(X_train_sample)
                train_metrics = self.evaluator.evaluate(y_train_sample, y_train_pred, y_train_proba, "train", model_name)
            else:
                print("   Evaluating on train set...")
                y_train_pred = model.predict(X_train)
                y_train_proba = model.predict_proba(X_train)
                train_metrics = self.evaluator.evaluate(y_train, y_train_pred, y_train_proba, "train", model_name)
            
            # Validation
            print("   Evaluating on validation set...")
            y_val_pred = model.predict(X_val)
            y_val_proba = model.predict_proba(X_val)
            val_metrics = self.evaluator.evaluate(y_val, y_val_pred, y_val_proba, "val", model_name)
            self.evaluator.print_metrics(val_metrics, "validation")
            
            # Test
            print("   Evaluating on test set...")
            y_test_pred = model.predict(X_test)
            y_test_proba = model.predict_proba(X_test)
            test_metrics = self.evaluator.evaluate(y_test, y_test_pred, y_test_proba, "test", model_name)
            self.evaluator.print_metrics(test_metrics, "test")
            
            # Confusion Matrix
            print("   Generating confusion matrix...")
            cm = np.array(test_metrics['confusion_matrix'])
            self.evaluator.plot_confusion_matrix(cm, model_name, "test")
            print(f"   ✅ {model_name} evaluation complete!")
        
        return models
    
    def train_dl_models(self, epochs=20, batch_size=32, learning_rate=0.001):
        """Train Deep Learning models (LSTM, CNN)"""
        print("\n" + "="*60)
        print("Training Deep Learning Models")
        print("="*60)
        
        # Note: For LSTM/CNN, we need to create a vocabulary from texts
        # This is a simplified version - in practice, you'd use proper tokenization
        data = self.datasets['word2vec']
        
        # Create simple tokenization for LSTM/CNN
        # In practice, you'd use a proper tokenizer
        all_texts = list(data['train']['texts']) + list(data['val']['texts']) + list(data['test']['texts'])
        word_counts = Counter()
        for text in all_texts:
            words = str(text).lower().split()
            word_counts.update(words)
        
        vocab = {word: idx + 2 for idx, (word, count) in enumerate(word_counts.most_common(10000))}
        vocab['<PAD>'] = 0
        vocab['<UNK>'] = 1
        vocab_size = len(vocab)
        self.vocab = vocab  # Store vocab for later use
        
        def text_to_sequence(text, max_length=512):
            words = str(text).lower().split()
            seq = [vocab.get(word, vocab['<UNK>']) for word in words[:max_length]]
            seq = seq + [vocab['<PAD>']] * (max_length - len(seq))
            return seq
        
        # Create datasets
        train_texts_seq = [text_to_sequence(text) for text in data['train']['texts']]
        val_texts_seq = [text_to_sequence(text) for text in data['val']['texts']]
        test_texts_seq = [text_to_sequence(text) for text in data['test']['texts']]
        
        train_dataset = ClassificationDataset(
            data['train']['texts'], data['train']['labels'], 
            use_bert=False, sequences=train_texts_seq
        )
        val_dataset = ClassificationDataset(
            data['val']['texts'], data['val']['labels'], 
            use_bert=False, sequences=val_texts_seq
        )
        test_dataset = ClassificationDataset(
            data['test']['texts'], data['test']['labels'], 
            use_bert=False, sequences=test_texts_seq
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        models = {}
        histories = {}
        
        # 1. LSTM
        print("\n1. Training LSTM...")
        lstm_model = LSTMClassifier(
            vocab_size=vocab_size,
            embedding_dim=300,
            hidden_dim=128,
            num_layers=2,
            num_classes=self.num_classes,
            dropout=0.3
        ).to(self.device)
        
        lstm_history = self._train_dl_model(
            lstm_model, train_loader, val_loader, epochs, learning_rate, "LSTM"
        )
        models['LSTM'] = lstm_model
        histories['LSTM'] = lstm_history
        
        # 2. CNN
        print("\n2. Training CNN...")
        cnn_model = CNNClassifier(
            vocab_size=vocab_size,
            embedding_dim=300,
            num_filters=100,
            filter_sizes=[3, 4, 5],
            num_classes=self.num_classes,
            dropout=0.3
        ).to(self.device)
        
        cnn_history = self._train_dl_model(
            cnn_model, train_loader, val_loader, epochs, learning_rate, "CNN"
        )
        models['CNN'] = cnn_model
        histories['CNN'] = cnn_history
        
        # Evaluate DL models
        for model_name, model in models.items():
            print(f"\nEvaluating {model_name}...")
            
            # Train (for completeness)
            y_train_pred, y_train_proba = self._evaluate_dl_model(model, train_loader)
            train_metrics = self.evaluator.evaluate(
                data['train']['labels'], y_train_pred, y_train_proba, "train", model_name
            )
            
            # Validation
            y_val_pred, y_val_proba = self._evaluate_dl_model(model, val_loader)
            val_metrics = self.evaluator.evaluate(
                data['val']['labels'], y_val_pred, y_val_proba, "val", model_name
            )
            self.evaluator.print_metrics(val_metrics, "validation")
            
            # Test
            y_test_pred, y_test_proba = self._evaluate_dl_model(model, test_loader)
            test_metrics = self.evaluator.evaluate(
                data['test']['labels'], y_test_pred, y_test_proba, "test", model_name
            )
            self.evaluator.print_metrics(test_metrics, "test")
            
            # Confusion Matrix
            cm = np.array(test_metrics['confusion_matrix'])
            self.evaluator.plot_confusion_matrix(cm, model_name, "test")
            
            # Training History
            self.evaluator.plot_training_history(histories[model_name], model_name)
        
        return models, histories
    
    def train_transformer_model(self, epochs=10, batch_size=16, learning_rate=2e-5):
        """Train Transformer-based model (BERT)"""
        print("\n" + "="*60)
        print("Training Transformer-based Model (BERT)")
        print("="*60)
        
        data = self.datasets['bert']
        tokenizer = self.data_loaders['bert'].bert_tokenizer
        
        # Create datasets
        train_dataset = ClassificationDataset(
            data['train']['texts'], data['train']['labels'],
            tokenizer=tokenizer, use_bert=True, max_length=512
        )
        val_dataset = ClassificationDataset(
            data['val']['texts'], data['val']['labels'],
            tokenizer=tokenizer, use_bert=True, max_length=512
        )
        test_dataset = ClassificationDataset(
            data['test']['texts'], data['test']['labels'],
            tokenizer=tokenizer, use_bert=True, max_length=512
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model
        bert_model = BERTClassifier(
            model_name='distilbert-base-uncased',
            num_classes=self.num_classes,
            dropout=0.3
        ).to(self.device)
        
        # Train
        print("\nTraining BERT...")
        bert_history = self._train_dl_model(
            bert_model, train_loader, val_loader, epochs, learning_rate, "BERT"
        )
        
        # Evaluate
        print("\nEvaluating BERT...")
        
        # Train (for completeness)
        y_train_pred, y_train_proba = self._evaluate_dl_model(bert_model, train_loader, is_bert=True)
        train_metrics = self.evaluator.evaluate(
            data['train']['labels'], y_train_pred, y_train_proba, "train", "BERT"
        )
        
        # Validation
        y_val_pred, y_val_proba = self._evaluate_dl_model(bert_model, val_loader, is_bert=True)
        val_metrics = self.evaluator.evaluate(
            data['val']['labels'], y_val_pred, y_val_proba, "val", "BERT"
        )
        self.evaluator.print_metrics(val_metrics, "validation")
        
        # Test
        y_test_pred, y_test_proba = self._evaluate_dl_model(bert_model, test_loader, is_bert=True)
        test_metrics = self.evaluator.evaluate(
            data['test']['labels'], y_test_pred, y_test_proba, "test", "BERT"
        )
        self.evaluator.print_metrics(test_metrics, "test")
        
        # Confusion Matrix
        cm = np.array(test_metrics['confusion_matrix'])
        self.evaluator.plot_confusion_matrix(cm, "BERT", "test")
        
        # Training History
        self.evaluator.plot_training_history(bert_history, "BERT")
        
        return bert_model, bert_history
    
    def _train_dl_model(self, model, train_loader, val_loader, epochs, learning_rate, model_name):
        """Train a deep learning model"""
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        history = {
            'train_acc': [],
            'val_acc': [],
            'train_loss': [],
            'val_loss': []
        }
        
        is_bert = model_name == "BERT"
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                if is_bert:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    outputs = model(input_ids, attention_mask)
                else:
                    inputs = batch['input_ids'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_acc = train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    if is_bert:
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['labels'].to(self.device)
                        outputs = model(input_ids, attention_mask)
                    else:
                        inputs = batch['input_ids'].to(self.device)
                        labels = batch['labels'].to(self.device)
                        outputs = model(inputs)
                    
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_acc = val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)
            
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            
            print(f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        return history
    
    def _evaluate_dl_model(self, model, data_loader, is_bert=False):
        """Evaluate a deep learning model"""
        model.eval()
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for batch in data_loader:
                if is_bert:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    outputs = model(input_ids, attention_mask)
                else:
                    inputs = batch['input_ids'].to(self.device)
                    outputs = model(inputs)
                
                probs = torch.nn.functional.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
        
        return np.array(predictions), np.array(probabilities)
    
    def train_all(self, dl_epochs=20, dl_batch_size=32, dl_lr=0.001,
                transformer_epochs=10, transformer_batch_size=16, transformer_lr=2e-5):
        """Train all models"""
        print("\n" + "="*80)
        print("COMPREHENSIVE CLASSIFICATION MODEL TRAINING")
        print("="*80)
        
        # Prepare data
        self.prepare_data()
        
        # Train ML models
        ml_models = self.train_ml_models()
        
        # Train DL models
        dl_models, dl_histories = self.train_dl_models(
            epochs=dl_epochs, batch_size=dl_batch_size, learning_rate=dl_lr
        )
        
        # Train Transformer model
        transformer_model, transformer_history = self.train_transformer_model(
            epochs=transformer_epochs, batch_size=transformer_batch_size, learning_rate=transformer_lr
        )
        
        print("\n" + "="*80)
        print("ALL MODELS TRAINED SUCCESSFULLY!")
        print("="*80)
        print(f"\nResults saved to: {self.output_dir}")
        
        return {
            'ml_models': ml_models,
            'dl_models': dl_models,
            'transformer_model': transformer_model,
            'dl_histories': dl_histories,
            'transformer_history': transformer_history
        }

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Classification Models')
    parser.add_argument('--csv_path', type=str, default='quiz_data.csv',
                    help='Path to CSV file')
    parser.add_argument('--target_column', type=str, default='subject',
                    choices=['subject', 'difficulty', 'question_type'],
                    help='Column to classify')
    parser.add_argument('--output_dir', type=str, default='./results/classification',
                    help='Output directory')
    parser.add_argument('--dl_epochs', type=int, default=20,
                    help='Epochs for DL models')
    parser.add_argument('--transformer_epochs', type=int, default=10,
                    help='Epochs for Transformer model')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = ClassificationTrainer(
        csv_path=args.csv_path,
        target_column=args.target_column,
        output_dir=args.output_dir
    )
    
    # Train all models
    trainer.train_all(
        dl_epochs=args.dl_epochs,
        transformer_epochs=args.transformer_epochs
    )

if __name__ == "__main__":
    main()

