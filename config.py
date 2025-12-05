"""
Configuration file for Quiz Generator Model
"""
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Configuration for the T5 model"""
    model_name: str = "t5-small"
    max_length: int = 512
    max_target_length: int = 128
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    output_dir: str = "./results/t5-quiz-generator"
    overwrite_output_dir: bool = True
    do_train: bool = True
    do_eval: bool = True
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "rouge1"
    greater_is_better: bool = True
    seed: int = 42
    fp16: bool = True
    gradient_accumulation_steps: int = 4

@dataclass
class DataConfig:
    """Configuration for data processing"""
    dataset_path: str = "quiz_data.csv"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    input_columns: list = None
    target_column: str = "question"
    
    def __post_init__(self):
        if self.input_columns is None:
            self.input_columns = ["subject", "topic", "difficulty", "question_type", "exam_type"]


