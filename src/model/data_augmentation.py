"""
Data Augmentation for Arabic Text
توسيع البيانات للنصوص العربية

Techniques:
1. Synonym replacement
2. Random deletion
3. Random swap
4. Back-translation (optional)
"""

import random
import re
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class ArabicDataAugmenter:
    """Data augmentation for Arabic text"""
    
    # Common Arabic synonyms for sentiment-bearing words
    SYNONYMS = {
        # Positive words
        "ممتاز": ["رائع", "عظيم", "مذهل", "جيد جداً"],
        "جيد": ["حسن", "لا بأس به", "مقبول", "جميل"],
        "رائع": ["ممتاز", "مذهل", "بديع", "خيالي"],
        "أحب": ["أعشق", "أفضل", "يعجبني"],
        "سعيد": ["مسرور", "فرح", "مبتهج"],
        
        # Negative words
        "سيء": ["رديء", "فظيع", "مروع", "سيئ جداً"],
        "رديء": ["سيء", "فظيع", "بشع"],
        "فظيع": ["مروع", "سيء جداً", "كارثي"],
        "أكره": ["أمقت", "لا أحب", "أستاء من"],
        "حزين": ["محزن", "مؤسف", "محبط"],
        
        # Neutral words
        "عادي": ["مقبول", "متوسط", "عادي جداً"],
        "مقبول": ["متوسط", "عادي", "لا بأس"],
    }
    
    # Common Arabic stopwords (less important for sentiment)
    STOPWORDS = {
        "في", "من", "إلى", "على", "عن", "مع", "هذا", "هذه", "ذلك", 
        "تلك", "أن", "إن", "لكن", "لكن", "كان", "يكون"
    }
    
    def __init__(self, augment_prob: float = 0.1):
        """
        Initialize augmenter
        
        Args:
            augment_prob: Probability of augmenting each word (0.0-1.0)
        """
        self.augment_prob = augment_prob
    
    def augment_text(self, text: str, n_aug: int = 1, 
                     methods: List[str] = None) -> List[str]:
        """
        Generate augmented versions of text
        
        Args:
            text: Original text
            n_aug: Number of augmented versions to generate
            methods: List of methods to use ['synonym', 'swap', 'delete']
                    If None, uses all methods
        
        Returns:
            List of augmented texts (including original)
        """
        if methods is None:
            methods = ['synonym', 'swap', 'delete']
        
        augmented = [text]  # Include original
        
        for _ in range(n_aug):
            method = random.choice(methods)
            
            if method == 'synonym':
                aug_text = self.synonym_replacement(text)
            elif method == 'swap':
                aug_text = self.random_swap(text)
            elif method == 'delete':
                aug_text = self.random_deletion(text)
            else:
                aug_text = text
            
            if aug_text != text:  # Only add if different
                augmented.append(aug_text)
        
        return augmented
    
    def synonym_replacement(self, text: str, n: int = None) -> str:
        """
        Replace words with synonyms
        
        Args:
            text: Input text
            n: Number of words to replace (None = use augment_prob)
        
        Returns:
            Augmented text
        """
        words = text.split()
        
        if len(words) == 0:
            return text
        
        # Determine how many words to replace
        if n is None:
            n = max(1, int(len(words) * self.augment_prob))
        
        # Get replaceable words
        replaceable_indices = [
            i for i, word in enumerate(words)
            if word in self.SYNONYMS and word not in self.STOPWORDS
        ]
        
        if not replaceable_indices:
            return text
        
        # Randomly select words to replace
        replace_count = min(n, len(replaceable_indices))
        indices_to_replace = random.sample(replaceable_indices, replace_count)
        
        # Replace words
        new_words = words.copy()
        for idx in indices_to_replace:
            word = words[idx]
            synonyms = self.SYNONYMS.get(word, [])
            if synonyms:
                new_words[idx] = random.choice(synonyms)
        
        return ' '.join(new_words)
    
    def random_swap(self, text: str, n: int = 1) -> str:
        """
        Randomly swap two words
        
        Args:
            text: Input text
            n: Number of swaps
        
        Returns:
            Augmented text
        """
        words = text.split()
        
        if len(words) < 2:
            return text
        
        new_words = words.copy()
        
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(new_words)), 2)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        
        return ' '.join(new_words)
    
    def random_deletion(self, text: str, p: float = None) -> str:
        """
        Randomly delete words
        
        Args:
            text: Input text
            p: Probability of deleting each word (None = use augment_prob)
        
        Returns:
            Augmented text
        """
        if p is None:
            p = self.augment_prob
        
        words = text.split()
        
        if len(words) <= 1:
            return text
        
        # Don't delete all words
        new_words = [word for word in words if random.random() > p]
        
        if len(new_words) == 0:
            # If all deleted, keep a random word
            return random.choice(words)
        
        return ' '.join(new_words)
    
    def random_insertion(self, text: str, n: int = 1) -> str:
        """
        Randomly insert synonyms of existing words
        
        Args:
            text: Input text
            n: Number of insertions
        
        Returns:
            Augmented text
        """
        words = text.split()
        
        if len(words) == 0:
            return text
        
        new_words = words.copy()
        
        for _ in range(n):
            # Find words with synonyms
            candidates = [w for w in words if w in self.SYNONYMS]
            
            if not candidates:
                break
            
            # Pick a random word and its synonym
            random_word = random.choice(candidates)
            synonym = random.choice(self.SYNONYMS[random_word])
            
            # Insert at random position
            random_idx = random.randint(0, len(new_words))
            new_words.insert(random_idx, synonym)
        
        return ' '.join(new_words)


def augment_dataset(texts: List[str], labels: List[str], 
                    n_aug: int = 2, balance_classes: bool = True) -> Tuple[List[str], List[str]]:
    """
    Augment entire dataset
    
    Args:
        texts: List of text samples
        labels: List of labels
        n_aug: Number of augmentations per sample
        balance_classes: Whether to balance classes by augmenting minority classes more
    
    Returns:
        Tuple of (augmented_texts, augmented_labels)
    """
    augmenter = ArabicDataAugmenter()
    
    augmented_texts = []
    augmented_labels = []
    
    if balance_classes:
        # Count classes
        from collections import Counter
        label_counts = Counter(labels)
        max_count = max(label_counts.values())
        
        # Calculate how much to augment each class
        augment_ratios = {
            label: max_count / count 
            for label, count in label_counts.items()
        }
        
        logger.info(f"Class distribution: {dict(label_counts)}")
        logger.info(f"Augmentation ratios: {dict(augment_ratios)}")
        
        for text, label in zip(texts, labels):
            # Original
            augmented_texts.append(text)
            augmented_labels.append(label)
            
            # Augmentations based on class
            n_aug_for_sample = int(augment_ratios[label] * n_aug)
            
            if n_aug_for_sample > 0:
                aug_versions = augmenter.augment_text(text, n_aug_for_sample)
                
                for aug_text in aug_versions[1:]:  # Skip original
                    augmented_texts.append(aug_text)
                    augmented_labels.append(label)
    else:
        # Standard augmentation
        for text, label in zip(texts, labels):
            aug_versions = augmenter.augment_text(text, n_aug)
            
            for aug_text in aug_versions:
                augmented_texts.append(aug_text)
                augmented_labels.append(label)
    
    logger.info(f"Augmented dataset size: {len(texts)} -> {len(augmented_texts)}")
    
    return augmented_texts, augmented_labels


if __name__ == "__main__":
    # Test augmentation
    augmenter = ArabicDataAugmenter(augment_prob=0.2)
    
    test_texts = [
        "هذا المنتج ممتاز جداً وأنا سعيد به",
        "الخدمة سيئة والطعام رديء",
        "المنتج عادي ولا بأس به",
    ]
    
    print("="*60)
    print("🔄 Arabic Data Augmentation Test")
    print("="*60)
    
    for text in test_texts:
        print(f"\n📝 Original: {text}")
        
        augmented = augmenter.augment_text(text, n_aug=3)
        
        for i, aug_text in enumerate(augmented[1:], 1):
            print(f"   Aug {i}: {aug_text}")
