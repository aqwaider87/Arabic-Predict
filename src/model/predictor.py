#!/usr/bin/env python3

"""
Arabic sentiment prediction with emoji support
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import logging
import json

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import SentimentLabels
from validator import ArabicValidator

logger = logging.getLogger(__name__)

class ArabicSentimentPredictor:
    """Arabic sentiment classifier with emoji support"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained model
            device: Device to use (auto, cpu, cuda, mps)
        """
        self.model_path = Path(model_path)
        self.device = self._get_device(device)
        self.validator = ArabicValidator()
        
        # Load model and tokenizer
        self._load_model()
        
        logger.info(f"Predictor loaded from {model_path} on {self.device}")
    
    def _get_device(self, device: str) -> str:
        """Determine best device"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch, "backends") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _load_model(self):
        """Load model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # Load model info if available
            info_file = self.model_path / "model_info.json"
            if info_file.exists():
                with open(info_file, 'r', encoding='utf-8') as f:
                    self.model_info = json.load(f)
            else:
                self.model_info = {}
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict_single(self, text: str, return_probabilities: bool = True) -> Dict:
        """
        Predict sentiment for single text
        
        Args:
            text: Input text (Arabic text or emojis)
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Dictionary with prediction results
        """
        # Validate text
        is_valid, cleaned_text, error_msg = self.validator.validate_text(text)
        
        if not is_valid:
            return {
                "success": False,
                "error": error_msg,
                "text": text,
                "prediction": None,
                "confidence": 0.0,
                "probabilities": None,
                "method": "validation_failed"
            }
        
        try:
            # Extract emojis
            emojis = self.validator.extract_emojis(text)
            is_emoji_only = self.validator.is_emoji_only(text)
            
            # Handle emoji-only text
            if is_emoji_only and emojis:
                return self._predict_emoji_sentiment(text, emojis, return_probabilities)
            
            # Handle text with neural model
            return self._predict_neural_sentiment(
                text, cleaned_text, emojis, return_probabilities
            )
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                "success": False,
                "error": f"خطأ في التنبؤ: {str(e)}",
                "text": text,
                "prediction": None,
                "confidence": 0.0,
                "probabilities": None,
                "method": "prediction_failed"
            }
    
    def _predict_emoji_sentiment(self, text: str, emojis: List[str], 
                                return_probabilities: bool) -> Dict:
        """Predict sentiment for emoji-only text"""
        emoji_sentiment = self.validator.classify_emoji_sentiment(emojis)
        
        # High confidence for emoji classification
        if emoji_sentiment == SentimentLabels.POSITIVE:
            confidence = 0.95
            probabilities = {
                SentimentLabels.POSITIVE: 0.95,
                SentimentLabels.NEGATIVE: 0.025,
                SentimentLabels.NEUTRAL: 0.025
            }
        elif emoji_sentiment == SentimentLabels.NEGATIVE:
            confidence = 0.95
            probabilities = {
                SentimentLabels.POSITIVE: 0.025,
                SentimentLabels.NEGATIVE: 0.95,
                SentimentLabels.NEUTRAL: 0.025
            }
        else:
            confidence = 0.90
            probabilities = {
                SentimentLabels.POSITIVE: 0.05,
                SentimentLabels.NEGATIVE: 0.05,
                SentimentLabels.NEUTRAL: 0.90
            }
        
        result = {
            "success": True,
            "error": None,
            "text": text,
            "cleaned_text": text,
            "prediction": emoji_sentiment,
            "confidence": confidence,
            "method": "emoji_analysis",
            "emojis_found": emojis,
            "emoji_count": len(emojis)
        }
        
        if return_probabilities:
            result["probabilities"] = probabilities
        
        return result
    
    def _predict_neural_sentiment(self, original_text: str, cleaned_text: str, 
                                 emojis: List[str], return_probabilities: bool) -> Dict:
        """Predict sentiment using neural model"""
        
        # Tokenize
        inputs = self.tokenizer(
            cleaned_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.cpu().numpy()[0]
        
        # Convert to probabilities
        probabilities_array = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        
        # Get prediction
        predicted_id = np.argmax(probabilities_array)
        prediction = SentimentLabels.ID_TO_LABEL[predicted_id]
        confidence = float(probabilities_array[predicted_id])
        
        # Consider emoji influence if present
        method = "neural_model"
        if emojis:
            emoji_sentiment = self.validator.classify_emoji_sentiment(emojis)
            
            # Boost confidence if emoji and text agree
            if emoji_sentiment == prediction:
                confidence = min(0.99, confidence * 1.1)
                method = "neural_model_with_emoji_boost"
            else:
                # Slight penalty if they disagree
                confidence = confidence * 0.95
                method = "neural_model_with_emoji_conflict"
        
        # Create probabilities dictionary
        probabilities_dict = {
            SentimentLabels.POSITIVE: float(probabilities_array[0]),
            SentimentLabels.NEGATIVE: float(probabilities_array[1]),
            SentimentLabels.NEUTRAL: float(probabilities_array[2])
        }
        
        result = {
            "success": True,
            "error": None,
            "text": original_text,
            "cleaned_text": cleaned_text,
            "prediction": prediction,
            "confidence": confidence,
            "method": method,
            "emojis_found": emojis,
            "emoji_count": len(emojis)
        }
        
        if emojis:
            result["emoji_sentiment"] = self.validator.classify_emoji_sentiment(emojis)
        
        if return_probabilities:
            result["probabilities"] = probabilities_dict
        
        return result
    
    def predict_batch(self, texts: List[str], return_probabilities: bool = True) -> List[Dict]:
        """Predict sentiment for multiple texts"""
        results = []
        
        for text in texts:
            result = self.predict_single(text, return_probabilities)
            results.append(result)
        
        return results
    
    def predict_dataframe(self, df: pd.DataFrame, text_column: str, 
                         output_prefix: str = "sentiment") -> pd.DataFrame:
        """Predict sentiment for DataFrame"""
        df = df.copy()
        
        predictions = []
        confidences = []
        errors = []
        methods = []
        
        logger.info(f"Predicting sentiment for {len(df)} texts...")
        
        for i, text in enumerate(df[text_column]):
            if i % 100 == 0:
                logger.info(f"Processed {i}/{len(df)} texts")
            
            result = self.predict_single(text, return_probabilities=False)
            
            if result["success"]:
                predictions.append(result["prediction"])
                confidences.append(result["confidence"])
                errors.append(None)
                methods.append(result["method"])
            else:
                predictions.append(None)
                confidences.append(0.0)
                errors.append(result["error"])
                methods.append("failed")
        
        # Add columns
        df[f"{output_prefix}_prediction"] = predictions
        df[f"{output_prefix}_confidence"] = confidences
        df[f"{output_prefix}_method"] = methods
        df[f"{output_prefix}_error"] = errors
        
        return df
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        info = {
            "model_path": str(self.model_path),
            "device": self.device,
            "labels": SentimentLabels.LABELS,
            "label_mapping": SentimentLabels.LABEL_TO_ID
        }
        info.update(self.model_info)
        return info

def demo_predictions():
    """Demo function showing various prediction examples"""
    
    # Sample texts for testing
    test_texts = [
        # Arabic text examples
        "هذا المنتج ممتاز جداً وأنصح الجميع بشرائه",
        "الخدمة سيئة جداً ولا أنصح أحداً بهذا المكان", 
        "المنتج عادي، لا شيء مميز فيه",
        
        # Arabic with emojis
        "أحب هذا الفيلم كثيراً 😍❤️",
        "هذا أسوأ طعام تذوقته في حياتي 😭💔",
        "الطقس اليوم غائم قليلاً 🌤️",
        
        # Emoji only
        "😍❤️🎉👏",
        "😭💔😞😢", 
        "😐🤔😶",
        
        # English text (should be rejected)
        "This is a great product!",
        "I hate this service",
        
        # Invalid cases
        "",
        "123456789",
    ]
    
    print("🔮 Arabic Sentiment Prediction Demo")
    print("=" * 60)
    
    # This would work if you have a trained model
    model_path = "outputs/best_model"
    
    if not Path(model_path).exists():
        print(f"❌ Model not found at {model_path}")
        print("💡 Train a model first using trainer.py")
        return
    
    try:
        predictor = ArabicSentimentPredictor(model_path)
        
        for i, text in enumerate(test_texts, 1):
            print(f"\n📝 Example {i}: {repr(text)}")
            
            result = predictor.predict_single(text)
            
            if result["success"]:
                print(f"   🎯 Prediction: {result['prediction']}")
                print(f"   📊 Confidence: {result['confidence']:.3f}")
                print(f"   🔍 Method: {result['method']}")
                
                if result.get("emojis_found"):
                    print(f"   😊 Emojis: {', '.join(result['emojis_found'])}")
                
                if result.get("probabilities"):
                    print(f"   📈 Probabilities:")
                    for label, prob in sorted(result["probabilities"].items(), 
                                            key=lambda x: x[1], reverse=True):
                        print(f"      {label}: {prob:.3f}")
            else:
                print(f"   ❌ Error: {result['error']}")
            
            print("-" * 60)
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    demo_predictions()