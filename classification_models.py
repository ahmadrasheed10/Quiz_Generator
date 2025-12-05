"""
Classification Models: ML, Deep Learning, and Transformer-based models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from transformers import AutoModel, AutoTokenizer
import numpy as np
from typing import Dict, Optional

# ==================== Machine Learning Models ====================

class RandomForestModel:
    """Random Forest Classifier"""
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.name = "Random Forest"
    
    def train(self, X_train, y_train):
        """Train the model"""
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        """Predict classes"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        return self.model.predict_proba(X)

class SVMModel:
    """Support Vector Machine Classifier"""
    def __init__(self, kernel='rbf', C=1.0, random_state=42):
        self.model = SVC(
            kernel=kernel,
            C=C,
            probability=True,
            random_state=random_state
        )
        self.name = "SVM"
    
    def train(self, X_train, y_train):
        """Train the model"""
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        """Predict classes"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        return self.model.predict_proba(X)

# ==================== Deep Learning Models ====================

class LSTMClassifier(nn.Module):
    """LSTM-based Text Classifier"""
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=128, 
                 num_layers=2, num_classes=2, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        self.name = "LSTM"
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)
        # Use the last hidden state
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        output = self.dropout(hidden)
        output = self.fc(output)
        return output

class CNNClassifier(nn.Module):
    """CNN-based Text Classifier"""
    def __init__(self, vocab_size, embedding_dim=300, num_filters=100,
                 filter_sizes=[3, 4, 5], num_classes=2, dropout=0.3):
        super(CNNClassifier, self).__init__()
        self.name = "CNN"
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        embedded = embedded.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_len)
        
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))  # (batch_size, num_filters, conv_seq_len)
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)  # (batch_size, num_filters)
            conv_outputs.append(pooled)
        
        concatenated = torch.cat(conv_outputs, dim=1)  # (batch_size, len(filter_sizes) * num_filters)
        output = self.dropout(concatenated)
        output = self.fc(output)
        return output

# ==================== Transformer-based Model ====================

class BERTClassifier(nn.Module):
    """BERT-based Text Classifier"""
    def __init__(self, model_name='distilbert-base-uncased', num_classes=2, dropout=0.3):
        super(BERTClassifier, self).__init__()
        self.name = "BERT"
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token
        output = self.dropout(pooled_output)
        output = self.classifier(output)
        return output

# ==================== Model Wrapper for Training ====================

class ModelWrapper:
    """Wrapper class to handle different model types uniformly"""
    def __init__(self, model, model_type='ml', device='cpu'):
        self.model = model
        self.model_type = model_type  # 'ml', 'dl', 'transformer'
        self.device = device
        self.name = model.name if hasattr(model, 'name') else type(model).__name__
        
        if model_type in ['dl', 'transformer']:
            self.model.to(device)
    
    def train(self, train_loader=None, X_train=None, y_train=None, 
              epochs=10, learning_rate=0.001, optimizer=None, criterion=None):
        """Train the model"""
        if self.model_type == 'ml':
            # ML models
            if X_train is None or y_train is None:
                raise ValueError("X_train and y_train required for ML models")
            self.model.train(X_train, y_train)
        else:
            # Deep learning models
            if train_loader is None:
                raise ValueError("train_loader required for DL models")
            if optimizer is None:
                optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            if criterion is None:
                criterion = nn.CrossEntropyLoss()
            
            self.model.train()
            for epoch in range(epochs):
                total_loss = 0
                for batch in train_loader:
                    if self.model_type == 'transformer':
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['labels'].to(self.device)
                        outputs = self.model(input_ids, attention_mask)
                    else:
                        # For LSTM/CNN, we need to handle tokenization differently
                        # This is a simplified version - actual implementation may vary
                        inputs = batch['input_ids'].to(self.device)
                        labels = batch['labels'].to(self.device)
                        outputs = self.model(inputs)
                    
                    loss = criterion(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(train_loader)
                if (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    def predict(self, test_loader=None, X_test=None):
        """Predict classes"""
        if self.model_type == 'ml':
            if X_test is None:
                raise ValueError("X_test required for ML models")
            return self.model.predict(X_test)
        else:
            if test_loader is None:
                raise ValueError("test_loader required for DL models")
            self.model.eval()
            predictions = []
            with torch.no_grad():
                for batch in test_loader:
                    if self.model_type == 'transformer':
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        outputs = self.model(input_ids, attention_mask)
                    else:
                        inputs = batch['input_ids'].to(self.device)
                        outputs = self.model(inputs)
                    preds = torch.argmax(outputs, dim=1)
                    predictions.extend(preds.cpu().numpy())
            return np.array(predictions)
    
    def predict_proba(self, test_loader=None, X_test=None):
        """Predict class probabilities"""
        if self.model_type == 'ml':
            if X_test is None:
                raise ValueError("X_test required for ML models")
            return self.model.predict_proba(X_test)
        else:
            if test_loader is None:
                raise ValueError("test_loader required for DL models")
            self.model.eval()
            probabilities = []
            with torch.no_grad():
                for batch in test_loader:
                    if self.model_type == 'transformer':
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        outputs = self.model(input_ids, attention_mask)
                    else:
                        inputs = batch['input_ids'].to(self.device)
                        outputs = self.model(inputs)
                    probs = F.softmax(outputs, dim=1)
                    probabilities.extend(probs.cpu().numpy())
            return np.array(probabilities)

