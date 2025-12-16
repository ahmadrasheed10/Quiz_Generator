
"""
training script for t5 quiz generator model
"""
# import system operations module
import os
# import deep learning library

# import transformers library classes
from transformers import (
    # t5 model for generation
    T5ForConditionalGeneration,
    # tokenizer for text processing
    T5Tokenizer,
    # arguments for training setup
    TrainingArguments,
    # trainer class for handling training
    Trainer,
    # collator for sequence-to-sequence data
    DataCollatorForSeq2Seq
)
#  dataset dictionary structure
from datasets import DatasetDict
#  plotting library
import matplotlib.pyplot as plt
#  numpy for math operations
import numpy as np
#  project configuration classes
from config import ModelConfig, DataConfig
#  custom data loader
from data_loader import QuizDataLoader
#  json for data serialization
import json

def plot_training_history(history: dict, output_dir: str):
    """plot training and validation curves"""
    # make output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # create figure for plots
    plt.figure(figsize=(12, 5))
    
    # create subplot for loss
    plt.subplot(1, 2, 1)
    # plot training loss if exists
    if 'loss' in history:
        plt.plot(history['loss'], label='Train Loss', marker='o')
    # plot validation loss if exists
    if 'eval_loss' in history:
        plt.plot(history['eval_loss'], label='Validation Loss', marker='s')
    # label x-axis as step
    plt.xlabel('Step')
    # label y-axis as loss
    plt.ylabel('Loss')
    # set title for loss plot
    plt.title('Training and Validation Loss')
    # show legend for lines
    plt.legend()
    # enable grid lines
    plt.grid(True)
    
    # create subplot for rouge scores
    plt.subplot(1, 2, 2)
    # define rouge metrics to plot
    rouge_metrics = ['eval_rouge1', 'eval_rouge2', 'eval_rougeL']
    # loop through each metric
    for metric in rouge_metrics:
        # check if metric exists
        if metric in history:
            # plot metric with label
            plt.plot(history[metric], label=metric.replace('eval_', '').upper(), marker='o')
    # label x-axis as step
    plt.xlabel('Step')
    # label y-axis as score
    plt.ylabel('Score')
    # set title for rouge plot
    plt.title('Validation ROUGE Scores')
    # show legend for lines
    plt.legend()
    # enable grid lines
    plt.grid(True)
    
    # adjust layout to fit
    plt.tight_layout()
    # save plot to file
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    # close the plot figure
    plt.close()
    
    # print confirmation message
    print(f"Training plots saved to {output_dir}/training_history.png")

# This class handles teaching the T5 model to make quiz questions
class QuizGeneratorTrainer:
    """trainer class for quiz generator model"""
    
    def __init__(self, config: ModelConfig, data_config: DataConfig):
        # store model configuration
        self.config = config
        # store data configuration
        self.data_config = data_config
        
        # Load the tools that convert text to numbers and vice versa
        print(f"Loading tokenizer: {config.model_name}")
        # load pretrained t5 tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(config.model_name)
        
        # Load the pretrained T5 model from Google
        print(f"Loading model: {config.model_name}")
        # load pretrained t5 model
        self.model = T5ForConditionalGeneration.from_pretrained(config.model_name)
        
        # initialize quiz data loader
        self.data_loader = QuizDataLoader(self.tokenizer, config, data_config)
        
        # Set up empty containers to track training progress
        self.training_history = {
            'loss': [],
            'eval_loss': [],
            'eval_rouge1': [],
            'eval_rouge2': [],
            'eval_rougeL': []
        }
    
    def compute_metrics(self, eval_pred):
        """compute evaluation metrics"""
        # import system library
        import sys
        # import os library
        import os
        # get current file directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # remove dir from path temporary
        if current_dir in sys.path:
            sys.path.remove(current_dir)
        # load evaluation metric
        from evaluate import load as eval_load
        # restore dir to path
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        # import nltk library
        import nltk
        
        try:
            # check for punkt tokenizer
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            # download punkt if missing
            nltk.download('punkt')
        
        # Load ROUGE - tool that checks how good generated text is
        rouge = eval_load("rouge")
        # unpack predictions and labels
        predictions, labels = eval_pred
        
        # Turn number sequences back into readable text
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # ignore pad tokens in labels
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        # decode labels to text
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # add newline for rouge format
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        # format labels for rouge score
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        
        # Measure how similar generated questions are to real ones
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        
        # convert scores to percentage
        result = {key: value * 100 for key, value in result.items()}
        
        # count exact matches found
        exact_matches = sum(1 for p, l in zip(decoded_preds, decoded_labels) if p.strip() == l.strip())
        # calculate exact match percentage
        result['exact_match'] = (exact_matches / len(decoded_preds)) * 100
        
        # check if rouge1 key exists
        if 'eval_rouge1' not in self.training_history:
            self.training_history['eval_rouge1'] = []
        # check if rouge2 key exists
        if 'eval_rouge2' not in self.training_history:
            self.training_history['eval_rouge2'] = []
        # check if rougel key exists
        if 'eval_rougeL' not in self.training_history:
            self.training_history['eval_rougeL'] = []
        
        # append rouge1 score to history
        self.training_history['eval_rouge1'].append(result['rouge1'])
        # append rouge2 score to history
        self.training_history['eval_rouge2'].append(result['rouge2'])
        # append rougel score to history
        self.training_history['eval_rougeL'].append(result['rougeL'])
        
        # return the computed results
        return result
    
    def train(self):
        """train the model"""
        # load and prepare all our quiz questions for training
        datasets = self.data_loader.prepare_datasets(
            self.data_config.dataset_path,
            seed=self.config.seed
        )
        
        # create data collator object
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True
        )
        
        # configure training arguments
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
            eval_strategy=self.config.evaluation_strategy,  
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
        
        # initialize trainer object
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        # Actually start teaching the model now
        print("Starting training...")
        # execute model training process
        train_result = trainer.train()
        
        # save trained model weights
        trainer.save_model()
        # save tokenizer files
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        # get training metrics data
        metrics = train_result.metrics
        # set path for metrics file
        metrics_file = os.path.join(self.config.output_dir, "train_metrics.json")
        # write metrics to json file
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # print completion and save path
        print(f"Training completed! Metrics saved to {metrics_file}")
        
        # print testing start message
        print("\nEvaluating on test set...")
        # evaluate model on test set
        test_metrics = trainer.evaluate(eval_dataset=datasets["test"])
        # set path for test metrics
        test_metrics_file = os.path.join(self.config.output_dir, "test_metrics.json")
        # write test metrics to file
        with open(test_metrics_file, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        # print test metrics save path
        print(f"Test metrics saved to {test_metrics_file}")
        
        # get training state object
        training_state = trainer.state
        # check if log history exists
        if training_state.log_history:
            # loop through each log entry
            for log in training_state.log_history:
                # check for loss key
                if 'loss' in log:
                    self.training_history['loss'].append(log['loss'])
                # check for eval_loss key
                if 'eval_loss' in log:
                    self.training_history['eval_loss'].append(log['eval_loss'])
        
        # plot training history graph
        plot_training_history(self.training_history, self.config.output_dir)
        
        # return trainer and metrics
        return trainer, test_metrics

def main():
    """main training function"""
    # load model configuration settings
    model_config = ModelConfig()
    # load data configuration settings
    data_config = DataConfig()
    
    # create output directory path
    os.makedirs(model_config.output_dir, exist_ok=True)
    
    # initialize the trainer class
    trainer = QuizGeneratorTrainer(model_config, data_config)
    
    # start training the model
    trainer, test_metrics = trainer.train()
    
    # print separator line
    print("\n" + "="*50)
    # print training summary header
    print("Training Summary")
    # print separator line again
    print("="*50)
    # print model save location
    print(f"Model saved to: {model_config.output_dir}")
    # print test set metrics header
    print("\nTest Set Metrics:")
    # loop through test metrics items
    for key, value in test_metrics.items():
        # check if value is number
        if isinstance(value, (int, float)):
            # print metric name and value
            print(f"  {key}: {value:.4f}")

if __name__ == "__main__":
    main()

"""
this script starts the t5 model and gets the quiz data ready
it uses a trainer to teach the model how to make questions
the code watches the training loss to see if it is learning
at the end it saves the new model and draws performance graphs
"""
