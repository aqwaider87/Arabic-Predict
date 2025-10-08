"""
Arabic text validation and processing
"""

import re
from typing import List, Tuple
from config import SentimentLabels

class ArabicValidator:
    """Validates and processes Arabic text"""
    
    # Arabic character ranges
    ARABIC_RANGES = [
        (0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF),
        (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)
    ]
    
    # Diacritics to remove
    DIACRITICS = [
        '\u064B', '\u064C', '\u064D', '\u064E', '\u064F', '\u0650', 
        '\u0651', '\u0652', '\u0653', '\u0654', '\u0655', '\u0656',
        '\u0657', '\u0658', '\u0670'
    ]
    
    # Emoji ranges
    EMOJI_RANGES = [
        (0x1F600, 0x1F64F), (0x1F300, 0x1F5FF), (0x1F680, 0x1F6FF),
        (0x1F1E0, 0x1F1FF), (0x2600, 0x26FF), (0x2700, 0x27BF),
        (0x1F900, 0x1F9FF)
    ]
    
    # Emoji sentiment mapping
    POSITIVE_EMOJIS = {
        '😊', '😄', '😃', '😀', '😁', '😆', '😂', '🤣', '😍', '🥰', 
        '😘', '🙂', '🤗', '🤩', '😎', '🥳', '😇', '👍', '👌', '✌️',
        '👏', '🙌', '💪', '❤️', '💕', '💖', '💗', '💘', '🔥', '⭐',
        '🌟', '✨', '🎉', '🎊', '🏆', '🥇', '👑', '💎', '🌈', '☀️'
    }
    
    NEGATIVE_EMOJIS = {
        '😢', '😭', '😞', '😔', '😟', '😕', '🙁', '☹️', '😣', '😖',
        '😫', '😩', '🥺', '😤', '😠', '😡', '🤬', '😱', '😨', '😰',
        '😥', '🤧', '🤒', '🤕', '🤢', '🤮', '💔', '💀', '☠️', '👎',
        '😈', '👿', '💩', '🤡', '👹', '👺', '❌', '🚫', '⛔'
    }
    
    NEUTRAL_EMOJIS = {
        '😐', '😑', '😶', '🙄', '😏', '😒', '🤔', '🤷', '🤭', '🤫',
        '🤐', '😴', '😌', '🧐', '🤓', '🙃', '😯', '😦', '😧', '😮',
        '🤨', '😲', '🤯', '😵', '🥴', '🤪', '😜', '😝', '🤑', '🤠'
    }
    
    @classmethod
    def is_arabic_char(cls, char: str) -> bool:
        """Check if character is Arabic"""
        if not char:
            return False
        try:
            char_code = ord(char)
            return any(start <= char_code <= end for start, end in cls.ARABIC_RANGES)
        except:
            return False
    
    @classmethod
    def is_emoji(cls, char: str) -> bool:
        """Check if character is emoji"""
        if not char:
            return False
        try:
            char_code = ord(char)
            return any(start <= char_code <= end for start, end in cls.EMOJI_RANGES)
        except:
            return False
    
    @classmethod
    def extract_emojis(cls, text: str) -> List[str]:
        """Extract all emojis from text"""
        if not isinstance(text, str):
            return []
        return [char for char in text if cls.is_emoji(char)]
    
    @classmethod
    def classify_emoji_sentiment(cls, emojis: List[str]) -> str:
        """Classify sentiment based on emojis"""
        if not emojis:
            return SentimentLabels.NEUTRAL
        
        positive_count = sum(1 for emoji in emojis if emoji in cls.POSITIVE_EMOJIS)
        negative_count = sum(1 for emoji in emojis if emoji in cls.NEGATIVE_EMOJIS)
        
        if positive_count > negative_count:
            return SentimentLabels.POSITIVE
        elif negative_count > positive_count:
            return SentimentLabels.NEGATIVE
        else:
            return SentimentLabels.NEUTRAL
    
    @classmethod
    def is_emoji_only(cls, text: str) -> bool:
        """Check if text contains only emojis"""
        if not isinstance(text, str) or not text.strip():
            return False
        
        cleaned_text = re.sub(r'\s+', '', text)
        if not cleaned_text:
            return False
        
        return all(cls.is_emoji(char) for char in cleaned_text)
    
    @classmethod
    def is_english_only(cls, text: str) -> bool:
        """Check if text is English only"""
        if not isinstance(text, str) or not text.strip():
            return False
        
        # Extract only alphabetic characters
        alpha_chars = [char for char in text if char.isalpha()]
        if not alpha_chars:
            return False
        
        arabic_count = sum(1 for char in alpha_chars if cls.is_arabic_char(char))
        english_count = sum(1 for char in alpha_chars if char.isascii())
        
        total_chars = len(alpha_chars)
        english_ratio = english_count / total_chars
        arabic_ratio = arabic_count / total_chars
        
        return english_ratio > 0.8 and arabic_ratio < 0.2
    
    @classmethod
    def normalize_arabic(cls, text: str) -> str:
        """Normalize Arabic text"""
        if not isinstance(text, str):
            return ""
        
        # Remove diacritics
        for diacritic in cls.DIACRITICS:
            text = text.replace(diacritic, '')
        
        # Normalize characters
        normalizations = [
            (r'[أإآٱ]', 'ا'),  # Alef variations
            (r'ة', 'ه'),       # Teh marbuta
            (r'[يی]', 'ى'),    # Yeh variations
        ]
        
        for pattern, replacement in normalizations:
            text = re.sub(pattern, replacement, text)
        
        # Clean spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    @classmethod
    def validate_text(cls, text: str) -> Tuple[bool, str, str]:
        """
        Validate text and return (is_valid, cleaned_text, error_message)
        """
        try:
            # Basic checks
            if not isinstance(text, str):
                return False, "", "النص يجب أن يكون نصاً صالحاً"
            
            text = text.strip()
            if not text:
                return False, "", "النص فارغ"
            
            # if len(text) < 2:
            #     return False, "", "النص قصير جداً"
            
            if len(text) > 1000:
                return False, "", "النص طويل جداً"
            
            # Check for English-only text
            if cls.is_english_only(text):
                return False, "", "هذا النموذج مُصمم للغة العربية فقط. يرجى إدخال نص باللغة العربية."
            
            # Handle emoji-only text
            if cls.is_emoji_only(text):
                emojis = cls.extract_emojis(text)
                if emojis:
                    return True, text, ""
                else:
                    return False, "", "النص يحتوي على رموز فقط بدون محتوى نصي"
            
            # Check Arabic content for mixed text
            emojis = cls.extract_emojis(text)
            text_without_emojis = text
            for emoji in emojis:
                text_without_emojis = text_without_emojis.replace(emoji, '')
            
            # Allow emoji-only content
            if emojis and not text_without_emojis.strip():
                return True, text, ""
            
            # Check Arabic ratio for remaining text
            if text_without_emojis.strip():
                alpha_chars = [char for char in text_without_emojis if char.isalpha()]
                if alpha_chars:
                    arabic_count = sum(1 for char in alpha_chars if cls.is_arabic_char(char))
                    arabic_ratio = arabic_count / len(alpha_chars)
                    
                    if arabic_ratio < 0.3:
                        return False, "", "هذا النموذج مُصمم للغة العربية فقط. النص يحتوي على نسبة قليلة من الأحرف العربية."
            
            # Normalize text
            normalized_text = cls.normalize_arabic(text)
            if not normalized_text and not emojis:
                return False, "", "النص غير صالح بعد التنظيف"
            
            return True, normalized_text if normalized_text else text, ""
            
        except Exception as e:
            return False, "", f"خطأ في معالجة النص: {str(e)}"

def test_validator():
    """Test the validator with sample texts"""
    test_cases = [
        "هذا نص عربي صحيح",
        "This is English text",
        "😊😍❤️",
        "أحب هذا المنتج 😍",
        "",
        "نص عربي مع English مختلط",
        "ا",  # Very short
        None,
    ]
    
    print("Testing Arabic Validator:")
    print("-" * 50)
    
    for i, text in enumerate(test_cases, 1):
        print(f"\nTest {i}: {repr(text)}")
        is_valid, cleaned, error = ArabicValidator.validate_text(text)
        print(f"Valid: {is_valid}")
        print(f"Cleaned: {repr(cleaned)}")
        print(f"Error: {error}")

if __name__ == "__main__":
    test_validator()