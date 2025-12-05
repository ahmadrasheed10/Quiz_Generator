"""
RAG-based Quiz Generator with Retrieval-Augmented Generation
"""
import os
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer


class RAGQuizGenerator:
    """Quiz question generator that augments T5 with retrieved context."""

    def __init__(
        self,
        model_path: str,
        dataset_path: str = "quiz_data.csv",
        device: Optional[str] = None,
        retrieval_model: str = "all-MiniLM-L6-v2",
        top_k: int = 3,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Find latest checkpoint if model_path is a directory
        actual_model_path = self._find_latest_checkpoint(model_path)

        # Load T5 model + tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(actual_model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(actual_model_path)
        self.model.to(self.device)
        self.model.eval()

        # Load retrieval model
        self.retrieval_model = SentenceTransformer(retrieval_model)
        if self.device == "cuda":
            self.retrieval_model = self.retrieval_model.to(self.device)

        # Load dataset and pre-compute embeddings (with caching)
        self.dataset = self._load_dataset(dataset_path)
        self.dataset_embeddings = self._load_or_create_embeddings(dataset_path, retrieval_model)

        self.top_k = top_k

    def _find_latest_checkpoint(self, base_path: str) -> str:
        """Find the latest checkpoint folder in the model directory"""
        if not os.path.exists(base_path):
            return base_path
        
        # Check if base_path itself contains model files
        if os.path.exists(os.path.join(base_path, "config.json")):
            return base_path
        
        # Look for checkpoint folders
        checkpoints = []
        for item in os.listdir(base_path):
            checkpoint_path = os.path.join(base_path, item)
            if os.path.isdir(checkpoint_path) and item.startswith("checkpoint-"):
                # Check if it contains model files
                if os.path.exists(os.path.join(checkpoint_path, "config.json")):
                    try:
                        checkpoint_num = int(item.split("-")[1])
                        checkpoints.append((checkpoint_num, checkpoint_path))
                    except (ValueError, IndexError):
                        continue
        
        if checkpoints:
            # Return the latest checkpoint (highest number)
            checkpoints.sort(key=lambda x: x[0], reverse=True)
            return checkpoints[0][1]
        
        return base_path

    def _load_dataset(self, dataset_path: str) -> pd.DataFrame:
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")

        df = pd.read_csv(dataset_path)
        required_cols = ["question", "subject", "topic", "difficulty", "question_type"]
        available_cols = [c for c in required_cols if c in df.columns]
        df = df[available_cols].dropna(subset=["question"])
        df["question"] = df["question"].astype(str)
        return df

    def _get_embeddings_cache_path(self, dataset_path: str, retrieval_model: str) -> str:
        """Generate cache file path based on dataset and model."""
        dataset_name = Path(dataset_path).stem
        model_name = retrieval_model.replace("/", "_").replace("-", "_")
        cache_dir = Path("./results/rag_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        return str(cache_dir / f"{dataset_name}_{model_name}_embeddings.pkl")
    
    def _load_or_create_embeddings(self, dataset_path: str, retrieval_model: str) -> np.ndarray:
        """Load embeddings from cache or create and cache them."""
        cache_path = self._get_embeddings_cache_path(dataset_path, retrieval_model)
        
        # Check if cache exists and is valid
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    cached_data = pickle.load(f)
                    cached_embeddings = cached_data["embeddings"]
                    cached_dataset_hash = cached_data.get("dataset_hash")
                    
                    # Verify dataset hasn't changed (simple check: row count)
                    current_hash = len(self.dataset)
                    if cached_dataset_hash == current_hash and cached_embeddings.shape[0] == current_hash:
                        print(f"✅ Loaded cached embeddings from {cache_path}")
                        return cached_embeddings
                    else:
                        print(f"⚠️ Dataset changed, recomputing embeddings...")
            except Exception as e:
                print(f"⚠️ Error loading cache: {e}, recomputing...")
        
        # Create embeddings
        print("🔄 Computing embeddings (this may take a while, but will be cached)...")
        questions = self.dataset["question"].tolist()
        embeddings = self.retrieval_model.encode(
            questions,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        
        # Cache embeddings
        try:
            with open(cache_path, "wb") as f:
                pickle.dump({
                    "embeddings": embeddings,
                    "dataset_hash": len(self.dataset),
                }, f)
            print(f"✅ Cached embeddings to {cache_path}")
        except Exception as e:
            print(f"⚠️ Could not cache embeddings: {e}")
        
        return embeddings

    def retrieve_relevant_context(
        self,
        subject: str,
        topic: str,
        difficulty: str,
        question_type: str,
    ) -> List[Dict]:
        query = f"{subject} {topic} {difficulty} {question_type}"
        query_embedding = self.retrieval_model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        similarities = cosine_similarity(query_embedding, self.dataset_embeddings)[0]

        mask = np.ones(len(self.dataset), dtype=bool)
        if "subject" in self.dataset.columns:
            mask &= self.dataset["subject"].str.lower() == subject.lower()
        if "difficulty" in self.dataset.columns:
            mask &= self.dataset["difficulty"].str.lower() == difficulty.lower()
        if "question_type" in self.dataset.columns:
            mask &= (
                self.dataset["question_type"].str.lower() == question_type.lower()
            )
        if mask.sum() == 0:
            filtered_similarities = similarities
        else:
            filtered_similarities = similarities.copy()
            filtered_similarities[~mask] = -1

        top_indices = np.argsort(filtered_similarities)[-self.top_k :][::-1]
        top_indices = top_indices[filtered_similarities[top_indices] > -1]

        retrieved = []
        for idx in top_indices:
            row = self.dataset.iloc[idx]
            retrieved.append(
                {
                    "question": row["question"],
                    "subject": row.get("subject", ""),
                    "topic": row.get("topic", ""),
                    "difficulty": row.get("difficulty", ""),
                    "question_type": row.get("question_type", ""),
                    "similarity": float(filtered_similarities[idx]),
                }
            )
        return retrieved

    def _build_prompt_with_context(
        self,
        subject: str,
        topic: str,
        difficulty: str,
        question_type: str,
        retrieved_contexts: List[Dict],
    ) -> str:
        context_block = "\n".join(
            [
                f"Example {i+1}: {ctx['question']}"
                for i, ctx in enumerate(retrieved_contexts)
            ]
        )

        prompt = (
            f"Generate {difficulty} {question_type} question for {subject} topic: {topic}.\n"
            f"Relevant examples:\n{context_block}\n\n"
            "Generate a similar question grounded in these examples:"
        )
        return prompt

    def _generate(
        self,
        input_text: str,
        max_length: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
    ) -> str:
        inputs = self.tokenizer(
            input_text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        gen_kwargs = {
            "max_length": max_length,
            "num_return_sequences": 1,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": 1.2,
        }

        if do_sample:
            gen_kwargs.update(
                {
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "do_sample": True,
                }
            )
        else:
            gen_kwargs.update({"num_beams": 4, "do_sample": False})

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def generate_question_with_rag(
        self,
        subject: str,
        topic: str,
        difficulty: str = "medium",
        question_type: str = "MCQ",
        **kwargs,
    ) -> Tuple[str, List[Dict]]:
        contexts = self.retrieve_relevant_context(
            subject, topic, difficulty, question_type
        )
        prompt = self._build_prompt_with_context(
            subject, topic, difficulty, question_type, contexts
        )
        question = self._generate(prompt, **kwargs)
        return question, contexts

    def generate_question_baseline(
        self,
        subject: str,
        topic: str,
        difficulty: str = "medium",
        question_type: str = "MCQ",
        **kwargs,
    ) -> str:
        prompt = (
            f"Generate {difficulty} {question_type} question for {subject} topic: {topic}"
        )
        return self._generate(prompt, **kwargs)

