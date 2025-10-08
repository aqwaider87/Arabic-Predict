"""
Configuration classes for Arabic Sentiment Classification
يقرأ الإعدادات من ملف config/sentiment_config.yaml
"""

import yaml
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class SentimentLabels:
    """Arabic sentiment labels configuration"""
    POSITIVE = "ايجابي"
    NEGATIVE = "سلبي"
    NEUTRAL = "محايد"
    
    LABELS = [POSITIVE, NEGATIVE, NEUTRAL]
    LABEL_TO_ID = {POSITIVE: 0, NEGATIVE: 1, NEUTRAL: 2}
    ID_TO_LABEL = {0: POSITIVE, 1: NEGATIVE, 2: NEUTRAL}
    
    # Mapping from various formats to Arabic
    MAPPING = {
        "positive": POSITIVE, "pos": POSITIVE, "1": POSITIVE, 1: POSITIVE,
        "negative": NEGATIVE, "neg": NEGATIVE, "0": NEGATIVE, 0: NEGATIVE,
        "neutral": NEUTRAL, "neu": NEUTRAL, "-1": NEUTRAL, -1: NEUTRAL,
        "good": POSITIVE, "great": POSITIVE, "excellent": POSITIVE,
        "bad": NEGATIVE, "terrible": NEGATIVE, "awful": NEGATIVE,
        "okay": NEUTRAL, "ok": NEUTRAL, "fine": NEUTRAL,
        "جيد": POSITIVE, "ممتاز": POSITIVE, "رائع": POSITIVE,
        "سيء": NEGATIVE, "رديء": NEGATIVE, "فظيع": NEGATIVE,
        "عادي": NEUTRAL, "مقبول": NEUTRAL, "متوسط": NEUTRAL,
    }

def load_config(config_path: str = "config/sentiment_config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing configuration
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        logger.warning(f"Configuration file not found: {config_path}")
        logger.info("Using default configuration")
        return get_default_config()
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Configuration loaded from: {config_path}")
        
        # Validate and complete configuration
        config = validate_and_complete_config(config)
        
        return config
        
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {config_path}: {e}")
        logger.info("Using default configuration")
        return get_default_config()
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        logger.info("Using default configuration")
        return get_default_config()

def validate_and_complete_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configuration and fill missing values with defaults
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Validated and completed configuration
    """
    default_config = get_default_config()
    
    # Recursively merge configurations
    def merge_configs(default: Dict, custom: Dict) -> Dict:
        result = default.copy()
        
        for key, value in custom.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    # Merge with defaults
    complete_config = merge_configs(default_config, config)
    
    # Validate critical settings
    required_fields = [
        ("data", "csv_path"),
        ("data", "text_column"),
        ("data", "label_column"),
        ("model", "pretrained_name"),
        ("output_dir",)
    ]
    
    for field_path in required_fields:
        current = complete_config
        field_name = ""
        
        try:
            for field in field_path:
                field_name = field
                current = current[field]
        except KeyError:
            raise ValueError(f"Missing required configuration field: {'.'.join(field_path)}")
    
    logger.info("Configuration validation passed")
    return complete_config

def get_default_config() -> Dict[str, Any]:
    """Get default configuration"""
    return {
        "run_name": "arabic_sentiment_classification",
        "seed": 42,
        "output_dir": "outputs/arabic_sentiment_model",
        
        # Data Configuration
        "data": {
            "csv_path": "data/mix_sample_arabic_data.csv",
            "text_column": "text",
            "label_column": "sentiment",
            "max_length": 512,
            "valid_size": 0.15,
            "test_size": 0.15,
            "stratify": True,
            "normalize_arabic": True,
            "remove_english": False,
            "min_text_length": 2,
            "train_split": 0.7,
            "valid_split": 0.15,
            "test_split": 0.15
        },
        
        # Model Configuration
        "model": {
            "pretrained_name": "aubmindlab/bert-base-arabertv2",
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "max_length": 512
        },
        
        # Training Configuration
        "training": {
            "num_epochs": 5,
            "batch_size": 16,
            "eval_batch_size": 32,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "eval_steps": 500,
            "save_steps": 500,
            "logging_steps": 50,
            "save_total_limit": 5,
            "resume_training": True,
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "label_smoothing": 0.1,
            "freeze_epochs": 1,
            "fp16": False,
            "bf16": False,
            "gradient_checkpointing": False
        },
        
        # Early Stopping Configuration
        "early_stopping": {
            "enabled": True,
            "patience": 3,
            "min_delta": 0.001
        },
        
        # Validation Configuration
        "validation": {
            "min_text_length": 2,
            "max_text_length": 1000,
            "min_arabic_ratio": 0.5,
            "allow_emojis": True,
            "reject_english_only": True,
            "normalize_arabic": True,
            "remove_diacritics": True,
            "handle_code_switching": True
        },
        
        # Class Balancing
        "class_balancing": {
            "enabled": True,
            "method": "class_weights"
        },
        
        # Model Calibration
        "calibration": {
            "enabled": True,
            "method": "temperature_scaling"
        },
        
        # Emoji Configuration
        "emoji_classification": {
            "enabled": True,
            "high_confidence_threshold": 0.95,
            "neutral_confidence": 0.90,
            "dominant_sentiment_threshold": 0.6
        },
        
        # Language Validation
        "language_validation": {
            "arabic_only": True,
            "min_arabic_ratio": 0.3,
            "reject_english_only": True,
            "english_rejection_message": "هذا النموذج مُصمم للغة العربية فقط. يرجى إدخال نص باللغة العربية.",
            "low_arabic_message": "هذا النموذج مُصمم للغة العربية فقط. النص يحتوي على نسبة قليلة من الأحرف العربية.",
            "allow_mixed_with_arabic": True,
            "allow_english_with_emojis": False
        },
        
        # Hardware Configuration
        "hardware": {
            "device": "auto",
            "mixed_precision": "auto"
        },
        
        # Logging Configuration
        "logging": {
            "level": "INFO",
            "save_logs": True,
            "log_file": "training.log"
        },
        
        # Export Configuration
        "export": {
            "save_onnx": False,
            "save_torchscript": False,
            "quantize_model": False
        }
    }

def create_default_config_file(config_path: str = "config/sentiment_config.yaml"):
    """
    Create a default configuration file
    
    Args:
        config_path: Path where to save the configuration file
    """
    config_file = Path(config_path)
    
    if config_file.exists():
        logger.warning(f"Configuration file already exists: {config_path}")
        response = input("Do you want to overwrite it? (y/N): ")
        if response.lower() != 'y':
            logger.info("Configuration file creation cancelled")
            return
    
    # Create the YAML content
    yaml_content = """# Arabic Sentiment Classification Configuration
# نظام تصنيف المشاعر العربية

run_name: "arabic_sentiment_classification"
seed: 42
output_dir: "outputs/arabic_sentiment_model"

# Data Configuration
data:
  csv_path: "data/mix_sample_arabic_data.csv"
  text_column: "text"
  label_column: "sentiment"
  max_length: 512
  valid_size: 0.15
  test_size: 0.15
  stratify: true
  normalize_arabic: true
  remove_english: false
  min_text_length: 2

# Model Configuration
model:
  pretrained_name: "aubmindlab/bert-base-arabertv2"
  dropout: 0.1
  attention_dropout: 0.1
  max_length: 512

# Training Configuration
training:
  num_epochs: 5
  batch_size: 16
  eval_batch_size: 32
  learning_rate: 2e-5
  weight_decay: 0.01
  warmup_ratio: 0.1
  eval_steps: 500
  save_steps: 500
  logging_steps: 50
  save_total_limit: 5
  resume_training: true
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0
  label_smoothing: 0.1
  freeze_epochs: 1
  fp16: false
  bf16: false
  gradient_checkpointing: false

# Early Stopping
early_stopping:
  enabled: true
  patience: 3
  min_delta: 0.001

# Text Validation
validation:
  min_text_length: 2
  max_text_length: 1000
  min_arabic_ratio: 0.5
  allow_emojis: true
  reject_english_only: true
  normalize_arabic: true
  remove_diacritics: true
  handle_code_switching: true

# Class Balancing
class_balancing:
  enabled: true
  method: "class_weights"

# Model Calibration
calibration:
  enabled: true
  method: "temperature_scaling"

# Emoji Classification
emoji_classification:
  enabled: true
  high_confidence_threshold: 0.95
  neutral_confidence: 0.90
  dominant_sentiment_threshold: 0.6

# Language Validation
language_validation:
  arabic_only: true
  min_arabic_ratio: 0.3
  reject_english_only: true
  english_rejection_message: "هذا النموذج مُصمم للغة العربية فقط. يرجى إدخال نص باللغة العربية."
  low_arabic_message: "هذا النموذج مُصمم للغة العربية فقط. النص يحتوي على نسبة قليلة من الأحرف العربية."
  allow_mixed_with_arabic: true
  allow_english_with_emojis: false

# Hardware Configuration
hardware:
  device: "auto"
  mixed_precision: "auto"

# Logging
logging:
  level: "INFO"
  save_logs: true
  log_file: "training.log"

# Export Options
export:
  save_onnx: false
  save_torchscript: false
  quantize_model: false
"""
    
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        
        logger.info(f"Default configuration file created: {config_path}")
        print(f"✅ Default configuration file created: {config_path}")
        print(f"💡 Edit this file to customize your settings")
        
    except Exception as e:
        logger.error(f"Failed to create configuration file: {e}")
        raise

# Global configuration instance
try:
    DEFAULT_CONFIG = load_config()
except Exception as e:
    logger.warning(f"Failed to load configuration: {e}")
    logger.info("Using built-in default configuration")
    DEFAULT_CONFIG = get_default_config()

def get_config(config_path: str = None) -> Dict[str, Any]:
    """
    Get configuration from file or return default
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if config_path:
        return load_config(config_path)
    return DEFAULT_CONFIG

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Configuration Management")
    parser.add_argument("--create", action="store_true", help="Create default config file")
    parser.add_argument("--file", default="config/sentiment_config.yaml", help="Config file path")
    parser.add_argument("--validate", action="store_true", help="Validate config file")
    
    args = parser.parse_args()
    
    if args.create:
        create_default_config_file(args.file)
    elif args.validate:
        try:
            config = load_config(args.file)
            print(f"✅ Configuration file is valid: {args.file}")
            print(f"📊 Loaded {len(config)} configuration sections")
        except Exception as e:
            print(f"❌ Configuration file is invalid: {e}")
    else:
        # Show current configuration
        config = load_config(args.file)
        print(f"📋 Configuration from {args.file}:")
        print(f"   Run name: {config.get('run_name')}")
        print(f"   Data file: {config.get('data', {}).get('csv_path')}")
        print(f"   Model: {config.get('model', {}).get('pretrained_name')}")
        print(f"   Output dir: {config.get('output_dir')}")