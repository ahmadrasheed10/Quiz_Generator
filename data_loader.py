"""
Data loading and preprocessing utilities for Quiz Generator
"""
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import T5Tokenizer
from typing import Dict, List, Tuple
import os

class QuizDataLoader:
    """Class to handle dataset loading and preprocessing for quiz generation"""
    
    def __init__(self, tokenizer: T5Tokenizer, config, data_config=None):
        self.tokenizer = tokenizer
        self.config = config  # ModelConfig
        self.data_config = data_config  # DataConfig
        self.max_input_length = config.max_length
        self.max_target_length = config.max_target_length
    
    def load_dataset(self, csv_path: str) -> pd.DataFrame:
        """Load dataset from CSV file"""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        required_columns = ['id', 'subject', 'topic', 'year', 'exam_type', 
                          'question_type', 'difficulty', 'question']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Remove rows with missing values
        df = df.dropna(subset=['question', 'subject', 'topic', 'difficulty'])
        
        return df
    
    def create_input_text(self, row: pd.Series) -> str:
        """Create input text from row data for T5 model"""
        subject = str(row['subject']).strip()
        topic = str(row['topic']).strip()
        difficulty = str(row['difficulty']).strip()
        question_type = str(row['question_type']).strip() if 'question_type' in row else "MCQ"
        exam_type = str(row['exam_type']).strip() if 'exam_type' in row else "Final"
        
        # Create a structured prompt for T5
        input_text = f"Generate {difficulty} {question_type} question for {subject} topic: {topic}"
        
        return input_text
    
    def preprocess_function(self, examples: Dict) -> Dict:
        """Preprocess examples for T5 model"""
        # Handle batched examples from HuggingFace Dataset
        # When batched=True, examples is a dict with lists as values
        input_texts = []
        
        # Get number of examples in batch
        first_key = list(examples.keys())[0]
        n_examples = len(examples[first_key])
        
        # Process each example in the batch
        for i in range(n_examples):
            # Create a row dictionary from the batch
            row_dict = {}
            for col in ['subject', 'topic', 'difficulty', 'question_type', 'exam_type']:
                if col in examples and i < len(examples[col]):
                    row_dict[col] = str(examples[col][i]).strip()
                else:
                    # Default values if column not present
                    defaults = {
                        'subject': 'General',
                        'topic': 'General',
                        'difficulty': 'medium',
                        'question_type': 'MCQ',
                        'exam_type': 'Final'
                    }
                    row_dict[col] = defaults.get(col, 'default')
            
            # Create pandas Series for create_input_text
            row = pd.Series(row_dict)
            input_text = self.create_input_text(row)
            input_texts.append(input_text)
        
        # Get target texts (questions)
        target_column = self.data_config.target_column if self.data_config else "question"
        if target_column in examples:
            target_texts = [str(q).strip() for q in examples[target_column]]
        else:
            target_texts = [""] * n_examples
        
        # Tokenize inputs
        model_inputs = self.tokenizer(
            input_texts,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True
        )
        
        # Tokenize targets (using text_target parameter instead of deprecated as_target_tokenizer)
        labels = self.tokenizer(
            text_target=target_texts,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True
        )
        
        # Replace padding token id's of the labels by -100 so it's ignored by the loss function
        labels_input_ids = labels["input_ids"]
        labels_input_ids = [[(token_id if token_id != self.tokenizer.pad_token_id else -100) 
                            for token_id in label_ids] for label_ids in labels_input_ids]
        
        model_inputs["labels"] = labels_input_ids
        
        return model_inputs
    
    def prepare_datasets(self, csv_path: str, seed: int = 42) -> DatasetDict:
        """Prepare train/val/test datasets"""
        # Load data
        df = self.load_dataset(csv_path)
        
        # Shuffle dataset
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        # Split dataset
        n = len(df)
        # Get split ratios from data_config if available
        if self.data_config:
            train_split = self.data_config.train_split
            val_split = self.data_config.val_split
        else:
            train_split = 0.8
            val_split = 0.1
        
        train_end = int(train_split * n)
        val_end = train_end + int(val_split * n)
        
        train_df = df[:train_end]
        val_df = df[train_end:val_end]
        test_df = df[val_end:]
        
        print(f"Dataset splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Convert to HuggingFace datasets
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        test_dataset = Dataset.from_pandas(test_df)
        
        # Preprocess datasets
        train_dataset = train_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Preprocessing train dataset"
        )
        
        val_dataset = val_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=val_dataset.column_names,
            desc="Preprocessing validation dataset"
        )
        
        test_dataset = test_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=test_dataset.column_names,
            desc="Preprocessing test dataset"
        )
        
        return DatasetDict({
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset
        })

