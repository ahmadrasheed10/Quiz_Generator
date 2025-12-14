"""
Inference script for Quiz Generator Model
"""
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import os
from typing import Optional

class QuizGenerator:
    """Class for generating quiz questions using fine-tuned T5 model"""
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize the Quiz Generator
        
        Args:
            model_path: Path to the fine-tuned model directory
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
        """
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading model from: {model_path}")
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully!")
    
    def generate_question(
        self,
        subject: str,
        topic: str,
        difficulty: str = "medium",
        question_type: str = "MCQ",
        exam_type: str = "Final",
        max_length: int = 128,
        num_return_sequences: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True
    ) -> str:
        """
        Generate a quiz question
        
        Args:
            subject: Subject name (e.g., "Computer Science", "Mathematics")
            topic: Topic name (e.g., "Machine Learning", "Calculus")
            difficulty: Difficulty level (e.g., "easy", "medium", "hard")
            question_type: Type of question (e.g., "MCQ", "Short Answer", "Long Answer")
            exam_type: Type of exam (e.g., "Midterm", "Final")
            max_length: Maximum length of generated question
            num_return_sequences: Number of questions to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling or greedy decoding
        
        Returns:
            Generated question(s) as string or list of strings
        """
        # Create input prompt
        input_text = f"Generate {difficulty} {question_type} question for {subject} topic: {topic}"
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate question
        with torch.no_grad():
            # Prepare generation kwargs
            gen_kwargs = {
                "max_length": max_length,
                "num_return_sequences": num_return_sequences,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "repetition_penalty": 1.2  # Reduce repetition
            }
            
            if do_sample:
                # Use sampling
                gen_kwargs.update({
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "do_sample": True
                })
            else:
                # Use beam search for greedy decoding
                gen_kwargs.update({
                    "num_beams": 4,
                    "do_sample": False
                })
            
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        # Decode generated text
        generated_questions = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )
        
        if num_return_sequences == 1:
            return generated_questions[0]
        else:
            return generated_questions
    
    def generate_multiple_questions(
        self,
        subject: str,
        topic: str,
        difficulty: str = "medium",
        question_type: str = "MCQ",
        num_questions: int = 5,
        **kwargs
    ) -> list:
        """
        Generate multiple unique quiz questions
        
        Args:
            subject: Subject name
            topic: Topic name
            difficulty: Difficulty level
            question_type: Type of question
            num_questions: Number of questions to generate
            **kwargs: Additional generation parameters
        
        Returns:
            List of generated questions
        """
        questions = []
        seen = set()
        max_attempts = num_questions * 10  # Prevent infinite loop
        attempts = 0
        
        print(f"Generating questions... (this may take a while on CPU)")
        
        while len(questions) < num_questions and attempts < max_attempts:
            attempts += 1
            # Show progress
            if attempts % 5 == 0:
                print(f"  Attempt {attempts}: Generated {len(questions)}/{num_questions} unique questions...")
            
            # Generate with some randomness to get diverse questions
            temp = kwargs.get('temperature', 0.9)
            question = self.generate_question(
                subject=subject,
                topic=topic,
                difficulty=difficulty,
                question_type=question_type,
                temperature=temp,
                **{k: v for k, v in kwargs.items() if k != 'temperature'}
            )
            
            # Simple deduplication (relaxed - only check first 50 chars for uniqueness)
            question_lower = question.lower().strip()
            question_key = question_lower[:50] if len(question_lower) > 50 else question_lower
            
            if question_key not in seen and len(question) > 10:
                questions.append(question)
                seen.add(question_key)
        
        if len(questions) < num_questions:
            print(f"  Warning: Only generated {len(questions)}/{num_questions} unique questions after {attempts} attempts.")
        
        return questions

def main():
    """Example usage of QuizGenerator"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate quiz questions using fine-tuned T5 model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./results/t5-quiz-generator",
        help="Path to fine-tuned model directory"
    )
    parser.add_argument(
        "--subject",
        type=str,
        required=True,
        help="Subject name (e.g., 'Computer Science')"
    )
    parser.add_argument(
        "--topic",
        type=str,
        required=True,
        help="Topic name (e.g., 'Machine Learning')"
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="medium",
        choices=["easy", "medium", "hard"],
        help="Difficulty level"
    )
    parser.add_argument(
        "--question_type",
        type=str,
        default="MCQ",
        help="Type of question (e.g., 'MCQ', 'Short Answer')"
    )
    parser.add_argument(
        "--num_questions",
        type=int,
        default=1,
        help="Number of questions to generate"
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model path not found: {args.model_path}")
        print("Please train the model first using train.py")
        return
    
    # Initialize generator
    generator = QuizGenerator(args.model_path)
    
    # Generate questions
    print("\n" + "="*60)
    print(f"Generating {args.num_questions} {args.difficulty} {args.question_type} question(s)")
    print(f"Subject: {args.subject}")
    print(f"Topic: {args.topic}")
    print("="*60 + "\n")
    
    if args.num_questions == 1:
        question = generator.generate_question(
            subject=args.subject,
            topic=args.topic,
            difficulty=args.difficulty,
            question_type=args.question_type
        )
        print(f"Generated Question:\n{question}\n")
    else:
        questions = generator.generate_multiple_questions(
            subject=args.subject,
            topic=args.topic,
            difficulty=args.difficulty,
            question_type=args.question_type,
            num_questions=args.num_questions
        )
        for i, question in enumerate(questions, 1):
            print(f"Question {i}:\n{question}\n")
            print("-"*60 + "\n")

if __name__ == "__main__":
    main()


