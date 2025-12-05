"""
Data loading and preprocessing utilities for Classification Tasks
Supports multiple embedding techniques: TF-IDF, Word2Vec, BERT
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import gensim
from gensim.models import Word2Vec
from typing import Dict, List, Tuple, Optional
import os
import pickle

class ClassificationDataset(Dataset):
    """PyTorch Dataset for classification tasks"""
    def __init__(self, texts, labels, tokenizer=None, max_length=512, use_bert=False, sequences=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_bert = use_bert
        self.sequences = sequences  # For LSTM/CNN with pre-tokenized sequences
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        
        if self.use_bert and self.tokenizer:
            text = str(self.texts[idx])
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }
        elif self.sequences is not None:
            # For LSTM/CNN with pre-tokenized sequences
            sequence = self.sequences[idx]
            return {
                'input_ids': torch.tensor(sequence, dtype=torch.long),
                'labels': torch.tensor(label, dtype=torch.long)
            }
        else:
            text = str(self.texts[idx])
            return {
                'text': text,
                'labels': torch.tensor(label, dtype=torch.long)
            }

class ClassificationDataLoader:
    """Class to handle dataset loading and preprocessing for classification tasks"""
    
    def __init__(self, csv_path: str, target_column: str = 'subject', 
                 embedding_type: str = 'tfidf', max_length: int = 512):
        """
        Initialize the data loader
        
        Args:
            csv_path: Path to CSV file
            target_column: Column to use as classification target (subject, difficulty, question_type)
            embedding_type: Type of embedding ('tfidf', 'word2vec', 'bert')
            max_length: Maximum sequence length for BERT
        """
        self.csv_path = csv_path
        self.target_column = target_column
        self.embedding_type = embedding_type
        self.max_length = max_length
        
        self.df = None
        self.label_encoder = LabelEncoder()
        self.vectorizer = None
        self.word2vec_model = None
        self.bert_tokenizer = None
        self.bert_model = None
        
        # Load and prepare data
        self._load_data()
        self._prepare_embeddings()
    
    def _load_data(self):
        """Load and preprocess the dataset"""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Dataset file not found: {self.csv_path}")
        
        self.df = pd.read_csv(self.csv_path)
        
        # Remove rows with missing values
        required_cols = ['question', self.target_column]
        self.df = self.df.dropna(subset=required_cols)
        
        # Clean text
        self.df['question'] = self.df['question'].astype(str).str.strip()
        
        print(f"Loaded {len(self.df)} samples")
        print(f"Target column: {self.target_column}")
        print(f"Number of classes: {self.df[self.target_column].nunique()}")
        print(f"Classes: {self.df[self.target_column].unique()}")
    
    def _prepare_embeddings(self):
        """Prepare embeddings based on embedding type"""
        if self.embedding_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
        elif self.embedding_type == 'word2vec':
            # Tokenize texts for Word2Vec
            tokenized_texts = [str(text).lower().split() for text in self.df['question']]
            self.word2vec_model = Word2Vec(
                sentences=tokenized_texts,
                vector_size=300,
                window=5,
                min_count=2,
                workers=4,
                sg=0  # CBOW
            )
        elif self.embedding_type == 'bert':
            self.bert_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            # We'll use the tokenizer for encoding, model will be loaded in the model itself
    
    def get_embeddings(self, texts: List[str], fit: bool = False):
        """Get embeddings for texts based on embedding type"""
        if self.embedding_type == 'tfidf':
            if fit:
                return self.vectorizer.fit_transform(texts)
            else:
                return self.vectorizer.transform(texts)
        elif self.embedding_type == 'word2vec':
            # Average Word2Vec embeddings
            embeddings = []
            for text in texts:
                words = str(text).lower().split()
                word_vectors = [
                    self.word2vec_model.wv[word] 
                    for word in words 
                    if word in self.word2vec_model.wv
                ]
                if word_vectors:
                    embeddings.append(np.mean(word_vectors, axis=0))
                else:
                    embeddings.append(np.zeros(300))
            return np.array(embeddings)
        elif self.embedding_type == 'bert':
            # BERT embeddings will be handled by the model
            return texts
    
    def prepare_datasets(self, test_size: float = 0.2, val_size: float = 0.1, 
                        random_state: int = 42) -> Dict:
        """Prepare train/val/test datasets"""
        # Extract texts and labels
        texts = self.df['question'].values
        labels = self.df[self.target_column].values
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        num_classes = len(self.label_encoder.classes_)
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            texts, encoded_labels, test_size=test_size, 
            random_state=random_state, stratify=encoded_labels
        )
        
        # Split train into train/val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size / (1 - test_size),
            random_state=random_state, stratify=y_train
        )
        
        print(f"\nDataset splits:")
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Get embeddings
        if self.embedding_type == 'bert':
            # For BERT, we return texts and labels (tokenization happens in Dataset)
            train_data = {
                'texts': X_train,
                'labels': y_train,
                'embeddings': None
            }
            val_data = {
                'texts': X_val,
                'labels': y_val,
                'embeddings': None
            }
            test_data = {
                'texts': X_test,
                'labels': y_test,
                'embeddings': None
            }
        else:
            # Fit vectorizer on training data
            train_embeddings = self.get_embeddings(X_train, fit=True)
            val_embeddings = self.get_embeddings(X_val, fit=False)
            test_embeddings = self.get_embeddings(X_test, fit=False)
            
            train_data = {
                'texts': X_train,
                'labels': y_train,
                'embeddings': train_embeddings
            }
            val_data = {
                'texts': X_val,
                'labels': y_val,
                'embeddings': val_embeddings
            }
            test_data = {
                'texts': X_test,
                'labels': y_test,
                'embeddings': test_embeddings
            }
        
        return {
            'train': train_data,
            'val': val_data,
            'test': test_data,
            'num_classes': num_classes,
            'class_names': self.label_encoder.classes_
        }
    
    def get_class_names(self):
        """Get class names from label encoder"""
        return self.label_encoder.classes_

