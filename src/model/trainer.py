"""
Model training for Arabic sentiment classification
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import logging
import json
import time

from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments, 
    EarlyStoppingCallback,
    set_seed
)
from datasets import Dataset

from config import SentimentLabels

logger = logging.getLogger(__name__)

class SentimentTrainer:
    """Arabic sentiment classification trainer"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.output_dir = Path(config.get('output_dir', 'outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer"""
        model_config = self.config.get('model', {})
        model_name = model_config.get('pretrained_name', 'aubmindlab/bert-base-arabertv2')
        
        logger.info(f"Loading model: {model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,
            id2label=SentimentLabels.ID_TO_LABEL,
            label2id=SentimentLabels.LABEL_TO_ID
        )
        
        logger.info(f"✅ Model loaded successfully")
        return model, tokenizer
    
    def create_datasets(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, 
                       tokenizer, text_col: str):
        """Create tokenized datasets"""
        max_length = self.config.get('model', {}).get('max_length', 512)
        
        logger.info(f"Creating datasets with max_length={max_length}")
        
        def tokenize_function(examples):
            return tokenizer(
                examples[text_col],
                truncation=True,
                padding=True,
                max_length=max_length
            )
        
        # Create datasets
        train_dataset = Dataset.from_pandas(train_df[[text_col, 'label_id']])
        valid_dataset = Dataset.from_pandas(valid_df[[text_col, 'label_id']])
        
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Valid dataset size: {len(valid_dataset)}")
        
        # Tokenize
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        valid_dataset = valid_dataset.map(tokenize_function, batched=True)
        
        # Rename label column
        train_dataset = train_dataset.rename_column('label_id', 'labels')
        valid_dataset = valid_dataset.rename_column('label_id', 'labels')
        
        return train_dataset, valid_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        f1_macro = f1_score(labels, predictions, average='macro')
        f1_weighted = f1_score(labels, predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted
        }
    
    def create_training_args(self) -> TrainingArguments:
        """Create training arguments"""
        training_config = self.config.get('training', {})
        
        return TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=training_config.get('num_epochs', 3),
            per_device_train_batch_size=training_config.get('batch_size', 16),
            per_device_eval_batch_size=training_config.get('batch_size', 16) * 2,
            learning_rate=training_config.get('learning_rate', 2e-5),
            weight_decay=training_config.get('weight_decay', 0.01),
            warmup_ratio=training_config.get('warmup_ratio', 0.1),
            
            # Evaluation and saving
            eval_strategy="steps",
            eval_steps=training_config.get('eval_steps', 500),
            save_strategy="steps", 
            save_steps=training_config.get('save_steps', 500),
            save_total_limit=3,
            
            # Logging
            logging_steps=training_config.get('logging_steps', 100),
            
            # Best model
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            
            # Performance
            dataloader_pin_memory=torch.cuda.is_available(),
            fp16=torch.cuda.is_available(),
            
            # Disable wandb
            report_to=[],
        )
    
    def find_latest_checkpoint(self) -> Optional[str]:
        """Find latest checkpoint for resume"""
        checkpoints = list(self.output_dir.glob("checkpoint-*"))
        
        if not checkpoints:
            logger.info("No previous checkpoints found")
            return None
        
        # Sort by step number
        valid_checkpoints = []
        for ckpt in checkpoints:
            try:
                step = int(ckpt.name.split("-")[-1])
                # Basic validation - check if model file exists
                model_files = [
                    ckpt / "pytorch_model.bin",
                    ckpt / "model.safetensors"
                ]
                if any(f.exists() for f in model_files):
                    valid_checkpoints.append((step, str(ckpt)))
            except (ValueError, OSError):
                continue
        
        if valid_checkpoints:
            latest = max(valid_checkpoints, key=lambda x: x[0])
            logger.info(f"Found latest checkpoint: {latest[1]} (step {latest[0]})")
            return latest[1]
        
        logger.info("No valid checkpoints found")
        return None
    
    def train(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, 
              text_col: str, resume: bool = True) -> Trainer:
        """Train the model"""
        
        logger.info("🚀 Starting training process")
        
        # Setup model and tokenizer
        model, tokenizer = self.setup_model_and_tokenizer()
        
        # Create datasets
        train_dataset, valid_dataset = self.create_datasets(
            train_df, valid_df, tokenizer, text_col
        )
        
        # Training arguments
        training_args = self.create_training_args()
        
        # Find checkpoint for resume
        resume_from_checkpoint = None
        if resume:
            resume_from_checkpoint = self.find_latest_checkpoint()
            if resume_from_checkpoint:
                logger.info(f"🔄 Will resume from: {resume_from_checkpoint}")
            else:
                logger.info("🆕 Starting new training")
        else:
            logger.info("🆕 Starting fresh training (resume disabled)")
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            processing_class=tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train
        try:
            start_time = time.time()
            
            if resume_from_checkpoint:
                logger.info(f"🔄 Resuming training from checkpoint: {resume_from_checkpoint}")
                trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            else:
                logger.info("🚀 Starting new training")
                trainer.train()
                
            training_time = time.time() - start_time
            logger.info(f"⏱️ Training completed in {training_time/3600:.2f} hours")
            
        except KeyboardInterrupt:
            logger.info("⏹️ Training interrupted by user")
            self._save_interrupted_model(trainer, tokenizer)
            raise
        except Exception as e:
            logger.error(f"💥 Training failed: {e}")
            self._save_emergency_model(trainer, tokenizer)
            raise
        
        # Save best model
        self._save_best_model(trainer, tokenizer)
        
        return trainer
    
    def _save_best_model(self, trainer: Trainer, tokenizer):
        """Save the best model"""
        best_model_dir = self.output_dir / "best_model"
        best_model_dir.mkdir(exist_ok=True)
        
        try:
            trainer.save_model(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            
            # Save model info
            model_info = {
                "best_metric": getattr(trainer.state, 'best_metric', None),
                "best_model_checkpoint": getattr(trainer.state, 'best_model_checkpoint', None),
                "training_completed": time.strftime("%Y-%m-%d %H:%M:%S"),
                "labels": SentimentLabels.LABELS,
                "label_mapping": SentimentLabels.LABEL_TO_ID,
                "model_config": self.config
            }
            
            with open(best_model_dir / "model_info.json", "w", encoding="utf-8") as f:
                json.dump(model_info, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"🏆 Best model saved to: {best_model_dir}")
            if hasattr(trainer.state, 'best_metric') and trainer.state.best_metric:
                logger.info(f"📊 Best F1-Macro: {trainer.state.best_metric:.4f}")
                
        except Exception as e:
            logger.error(f"Failed to save best model: {e}")
    
    def _save_interrupted_model(self, trainer: Trainer, tokenizer):
        """Save model when training is interrupted"""
        try:
            interrupted_dir = self.output_dir / "interrupted_checkpoint"
            trainer.save_model(interrupted_dir)
            tokenizer.save_pretrained(interrupted_dir)
            logger.info(f"💾 Interrupted model saved to: {interrupted_dir}")
        except Exception as e:
            logger.error(f"Failed to save interrupted model: {e}")
    
    def _save_emergency_model(self, trainer: Trainer, tokenizer):
        """Save model in case of emergency"""
        try:
            emergency_dir = self.output_dir / "emergency_checkpoint"
            trainer.save_model(emergency_dir)
            tokenizer.save_pretrained(emergency_dir)
            logger.info(f"🚨 Emergency model saved to: {emergency_dir}")
        except Exception as e:
            logger.error(f"Failed to save emergency model: {e}")
    
    def evaluate_model(self, trainer: Trainer, test_df: pd.DataFrame, 
                      text_col: str) -> Dict:
        """Evaluate model on test set"""
        
        logger.info("📈 Evaluating model on test set")
        
        # Create test dataset
        tokenizer = trainer.tokenizer
        max_length = self.config.get('model', {}).get('max_length', 512)
        
        def tokenize_function(examples):
            return tokenizer(
                examples[text_col],
                truncation=True,
                padding=True,
                max_length=max_length
            )
        
        test_dataset = Dataset.from_pandas(test_df[[text_col, 'label_id']])
        test_dataset = test_dataset.map(tokenize_function, batched=True)
        test_dataset = test_dataset.rename_column('label_id', 'labels')
        
        # Evaluate
        test_results = trainer.evaluate(test_dataset)
        
        # Save results
        results_file = self.output_dir / "test_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📊 Test Results saved to: {results_file}")
        return test_results

def quick_train(config: Dict = None, config_path: str = None, resume: bool = True):
    """Quick training function for easy use"""
    
    logger.info("🚀 Starting quick training")
    
    # Import here to avoid circular imports
    from data_loader import DataLoader
    from config import get_config
    
    # Load configuration
    if config is None:
        if config_path:
            config = get_config(config_path)
        else:
            config = get_config()  # Load from default location
    
    logger.info(f"⚙️ Configuration loaded")
    logger.info(f"📊 Data file: {config['data']['csv_path']}")
    logger.info(f"📁 Output dir: {config['output_dir']}")
    logger.info(f"🤖 Model: {config['model']['pretrained_name']}")
    
    # Setup
    set_seed(config.get('seed', 42))
    
    # Load data
    logger.info("📊 Loading and preparing data...")
    data_loader = DataLoader(config)
    
    try:
        train_df, valid_df, test_df = data_loader.prepare_data(config['data']['csv_path'])
    except Exception as e:
        logger.error(f"❌ Data preparation failed: {e}")
        logger.info("💡 Try running: python main.py debug --data your_file.csv")
        raise
    
    # Print data summary
    data_loader.print_data_summary(train_df, valid_df, test_df, 
                                  config['data']['text_column'], 
                                  config['data']['label_column'])
    
    # Check if we have enough data
    if len(train_df) < 10:
        raise ValueError(f"Not enough training data: {len(train_df)} samples. Need at least 10.")
    
    # Train model
    logger.info("🚀 Starting model training...")
    trainer_obj = SentimentTrainer(config)
    
    trained_model = trainer_obj.train(
        train_df, valid_df, 
        config['data']['text_column'], 
        resume=resume
    )
    
    # Evaluate
    logger.info("📈 Evaluating on test set...")
    test_results = trainer_obj.evaluate_model(
        trained_model, test_df, 
        config['data']['text_column']
    )
    
    logger.info("🎉 Training completed successfully!")
    logger.info(f"📊 Test Accuracy: {test_results.get('eval_accuracy', 0):.4f}")
    logger.info(f"📊 Test F1-Macro: {test_results.get('eval_f1_macro', 0):.4f}")
    
    return trained_model, test_results

def main():
    """Main function for standalone training"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Arabic Sentiment Model")
    parser.add_argument("--data", required=True, help="Path to CSV data file")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from checkpoint")
    
    args = parser.parse_args()
    
    try:
        quick_train(args.data, args.output, resume=not args.no_resume)
        logger.info("✅ Training completed successfully!")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()