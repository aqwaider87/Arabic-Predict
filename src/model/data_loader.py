"""
Data loading and processing for Arabic sentiment analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
from sklearn.model_selection import train_test_split
import logging

from config import SentimentLabels
from validator import ArabicValidator
from data_augmentation import augment_dataset

logger = logging.getLogger(__name__)

class DataLoader:
    """Handles data loading and preprocessing"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.validator = ArabicValidator()
    
    def load_csv(self, file_path: str) -> pd.DataFrame:
        """Load CSV file with proper encoding detection"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Try different encodings
        encodings = ['utf-8', 'cp1256', 'iso-8859-6', 'latin1']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                logger.info(f"Successfully loaded file with {encoding} encoding")
                return df
            except UnicodeDecodeError:
                continue
        
        raise ValueError(f"Could not read file {file_path} with any supported encoding")
    
    def validate_columns(self, df: pd.DataFrame, text_col: str, label_col: str):
        """Validate that required columns exist"""
        missing_cols = []
        
        if text_col not in df.columns:
            missing_cols.append(text_col)
        
        if label_col not in df.columns:
            missing_cols.append(label_col)
        
        if missing_cols:
            available_cols = list(df.columns)
            raise ValueError(
                f"Missing columns: {missing_cols}\n"
                f"Available columns: {available_cols}"
            )
    
    def convert_labels(self, df: pd.DataFrame, label_col: str) -> pd.DataFrame:
        """Convert various label formats to Arabic labels"""
        df = df.copy()
        
        def map_label(label):
            # Handle NaN values
            if pd.isna(label):
                return SentimentLabels.NEUTRAL
            
            # Convert to string and normalize
            label_str = str(label).lower().strip()
            
            # Direct mapping
            if label_str in SentimentLabels.MAPPING:
                return SentimentLabels.MAPPING[label_str]
            
            # Already Arabic
            if label_str in SentimentLabels.LABELS:
                return label_str
            
            # Pattern matching
            if any(word in label_str for word in ['pos', 'good', 'great', 'جيد', 'ممتاز', 'رائع']):
                return SentimentLabels.POSITIVE
            elif any(word in label_str for word in ['neg', 'bad', 'سيء', 'رديء', 'فظيع']):
                return SentimentLabels.NEGATIVE
            else:
                return SentimentLabels.NEUTRAL
        
        df[label_col] = df[label_col].apply(map_label)
        
        # Log label distribution
        label_counts = df[label_col].value_counts()
        logger.info(f"Label distribution after conversion: {label_counts.to_dict()}")
        
        return df
    
    def clean_dataset(self, df: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
        """Clean and validate the dataset"""
        original_size = len(df)
        logger.info(f"Starting with {original_size} samples")
        
        # Remove missing values
        df = df.dropna(subset=[text_col, label_col])
        after_na = len(df)
        logger.info(f"After removing NaN: {after_na} samples ({original_size - after_na} removed)")
        
        # Validate and clean texts
        valid_data = []
        invalid_count = 0
        
        for idx, row in df.iterrows():
            text = str(row[text_col]) if row[text_col] is not None else ""
            label = row[label_col]
            
            is_valid, cleaned_text, error_msg = self.validator.validate_text(text)
            
            if is_valid and cleaned_text:
                valid_data.append({
                    text_col: cleaned_text,
                    label_col: label
                })
            else:
                invalid_count += 1
                if invalid_count <= 5:  # Log first 5 errors
                    logger.warning(f"Invalid text at index {idx}: {error_msg}")
        
        if not valid_data:
            raise ValueError("No valid texts found after cleaning!")
        
        # Create cleaned dataframe
        cleaned_df = pd.DataFrame(valid_data)
        logger.info(f"After text validation: {len(cleaned_df)} samples ({invalid_count} invalid)")
        
        # Remove duplicates
        before_dedup = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates(subset=[text_col]).reset_index(drop=True)
        after_dedup = len(cleaned_df)
        logger.info(f"After deduplication: {after_dedup} samples ({before_dedup - after_dedup} duplicates)")
        
        if len(cleaned_df) == 0:
            raise ValueError("No data remaining after cleaning!")
        
        return cleaned_df
    
    def split_data(self, df: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train/validation/test sets"""
        data_config = self.config.get('data', {})
        train_ratio = data_config.get('train_split', 0.7)
        valid_ratio = data_config.get('valid_split', 0.15)
        test_ratio = data_config.get('test_split', 0.15)
        
        # Ensure ratios sum to 1
        total_ratio = train_ratio + valid_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.01:
            logger.warning(f"Split ratios sum to {total_ratio}, normalizing...")
            train_ratio /= total_ratio
            valid_ratio /= total_ratio
            test_ratio /= total_ratio
        
        # First split: train vs (valid + test)
        train_df, temp_df = train_test_split(
            df,
            test_size=(valid_ratio + test_ratio),
            stratify=df[label_col],
            random_state=42
        )
        
        # Second split: valid vs test
        valid_size = valid_ratio / (valid_ratio + test_ratio)
        valid_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - valid_size),
            stratify=temp_df[label_col],
            random_state=42
        )
        
        logger.info(f"Data split - Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")
        
        return train_df, valid_df, test_df
    
    def augment_data(self, df: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
        """Apply data augmentation to training data"""
        aug_config = self.config.get('augmentation', {})
        
        if not aug_config.get('enabled', False):
            logger.info("📊 Data augmentation: DISABLED")
            return df
        
        logger.info("🔄 Applying data augmentation...")
        
        # Extract parameters
        n_aug = aug_config.get('n_aug', 2)
        balance_classes = aug_config.get('balance_classes', True)
        
        try:
            # Augment
            augmented_texts, augmented_labels = augment_dataset(
                texts=df[text_col].tolist(),
                labels=df[label_col].tolist(),
                n_aug=n_aug,
                balance_classes=balance_classes
            )
            
            # Create new dataframe
            aug_df = pd.DataFrame({
                text_col: augmented_texts,
                label_col: augmented_labels
            })
            
            logger.info(f"✅ Data augmentation: {len(df)} → {len(aug_df)} samples")
            
            # Show distribution
            label_dist = aug_df[label_col].value_counts()
            for label, count in label_dist.items():
                logger.info(f"   {label}: {count}")
            
            return aug_df
            
        except Exception as e:
            logger.error(f"❌ Data augmentation failed: {e}")
            logger.info("⚠️ Continuing without augmentation")
            return df
    
    def prepare_data(self, file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Complete data preparation pipeline"""
        data_config = self.config.get('data', {})
        text_col = data_config.get('text_column', 'text')
        label_col = data_config.get('label_column', 'sentiment')
        
        # Load data
        logger.info(f"Loading data from {file_path}")
        df = self.load_csv(file_path)
        
        # Validate columns
        self.validate_columns(df, text_col, label_col)
        
        # Convert labels to Arabic format
        df = self.convert_labels(df, label_col)
        
        # Clean dataset
        df = self.clean_dataset(df, text_col, label_col)
        
        # Apply data augmentation (before splitting)
        df = self.augment_data(df, text_col, label_col)
        
        # Split data
        train_df, valid_df, test_df = self.split_data(df, label_col)
        
        # Add label IDs
        for split_df in [train_df, valid_df, test_df]:
            split_df['label_id'] = split_df[label_col].map(SentimentLabels.LABEL_TO_ID)
        
        return train_df, valid_df, test_df
    
    def get_data_info(self, df: pd.DataFrame, text_col: str, label_col: str) -> Dict:
        """Get comprehensive data information"""
        info = {
            'total_samples': len(df),
            'text_column': text_col,
            'label_column': label_col,
            'missing_text': df[text_col].isna().sum(),
            'missing_labels': df[label_col].isna().sum(),
            'empty_text': (df[text_col].str.strip() == '').sum(),
            'duplicate_texts': df[text_col].duplicated().sum(),
            'label_distribution': df[label_col].value_counts().to_dict(),
            'text_length_stats': {
                'mean': df[text_col].str.len().mean(),
                'median': df[text_col].str.len().median(),
                'min': df[text_col].str.len().min(),
                'max': df[text_col].str.len().max(),
                'std': df[text_col].str.len().std()
            }
        }
        
        return info
    
    def print_data_summary(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, 
                          test_df: pd.DataFrame, text_col: str, label_col: str):
        """Print comprehensive data summary"""
        print("\n" + "="*60)
        print("📊 DATA SUMMARY")
        print("="*60)
        
        splits = [
            ("Training", train_df),
            ("Validation", valid_df),
            ("Test", test_df)
        ]
        
        for split_name, split_df in splits:
            print(f"\n📋 {split_name} Set:")
            print(f"   Samples: {len(split_df)}")
            
            label_dist = split_df[label_col].value_counts()
            for label, count in label_dist.items():
                percentage = (count / len(split_df)) * 100
                print(f"   {label}: {count} ({percentage:.1f}%)")
        
        # Overall statistics
        all_data = pd.concat([train_df, valid_df, test_df])
        text_lengths = all_data[text_col].str.len()
        
        print(f"\n📏 Text Length Statistics:")
        print(f"   Average: {text_lengths.mean():.1f} characters")
        print(f"   Median: {text_lengths.median():.1f} characters")
        print(f"   Min: {text_lengths.min()} characters")
        print(f"   Max: {text_lengths.max()} characters")
        
        print("="*60)

def test_data_loader():
    """Test the data loader"""
    # Create sample data
    sample_data = {
        'text': [
            'هذا المنتج ممتاز جداً',
            'الخدمة سيئة للغاية',
            'المنتج عادي',
            'This is English',  # Should be rejected
            '',  # Empty text
            'أحب هذا الفيلم 😍',
            '😊😄😃',  # Emoji only
        ],
        'sentiment': ['positive', 'negative', 'neutral', 'positive', 'neutral', 'positive', 'positive']
    }
    
    df = pd.DataFrame(sample_data)
    temp_file = Path('temp_test_data.csv')
    df.to_csv(temp_file, index=False, encoding='utf-8')
    
    try:
        # Test data loader
        from config import DEFAULT_CONFIG
        loader = DataLoader(DEFAULT_CONFIG)
        
        train_df, valid_df, test_df = loader.prepare_data(str(temp_file))
        loader.print_data_summary(train_df, valid_df, test_df, 'text', 'sentiment')
        
    finally:
        # Clean up
        if temp_file.exists():
            temp_file.unlink()

if __name__ == "__main__":
    test_data_loader()