"""
BanglaSentNet Preprocessing Pipeline
====================================
Implements the 2-phase data preprocessing methodology described in Section 3.2
of the BanglaSentNet conference paper (ICDSAIA 2025).

Phase 1: Automated linguistic cleaning & normalization
Phase 2: Dictionary-based token standardization & noise filtering
"""

import re
import unicodedata
from typing import List, Dict, Any


# Standard Bengali character range: \u0980-\u09FF
BANGLA_CHAR_PATTERN = re.compile(r'[\u0980-\u09FF]+')
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
HTML_TAG_PATTERN = re.compile(r'<.*?>')
REPEATED_CHAR_PATTERN = re.compile(r'(.)\1{2,}')
EXCESS_WHITESPACE = re.compile(r'\s+')


def clean_bangla_text(text: str) -> str:
    """
    Cleans raw review text from e-commerce platforms:
    - Normalizes Unicode (NFKC)
    - Removes HTML tags and URLs
    - Normalizes elongated characters (e.g., 'অনেককক' -> 'অনেক')
    - Strips emojis and non-Bangla symbols (preserving Bangla sentence structures)
    - Normalizes extraneous whitespaces
    
    Args:
        text (str): Raw input review string.
        
    Returns:
        str: Cleaned, normalized Bangla string.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Unicode normalization
    text = unicodedata.normalize('NFKC', text)
    
    # 2. Strip HTML tags and URLs
    text = HTML_TAG_PATTERN.sub('', text)
    text = URL_PATTERN.sub('', text)
    
    # 3. Collapse repeated character sequences (e.g. 'খুউউব' -> 'খুব')
    text = REPEATED_CHAR_PATTERN.sub(r'\1', text)
    
    # 4. Remove special noise characters, keeping Bangla characters, numbers, and core punctuation
    # Bangla Unicode: \u0980-\u09FF; Danda: \u0964; Double Danda: \u0965
    cleaned_tokens = []
    # Retain Bangla words, digits, and essential punctuation
    text = re.sub(r'[^\u0980-\u09FF0-9\s।,!?.]', ' ', text)
    
    # 5. Clean whitespace
    text = EXCESS_WHITESPACE.sub(' ', text).strip()
    return text


def tokenize_bangla(text: str) -> List[str]:
    """
    Tokenizes normalized Bangla review text into word tokens.
    
    Args:
        text (str): Cleaned Bangla text.
        
    Returns:
        List[str]: List of word tokens.
    """
    # Separate punctuation from words
    text = re.sub(r'([।,!?])', r' \1 ', text)
    tokens = text.split()
    return [t for t in tokens if t.strip()]


def get_review_length_category(text: str) -> str:
    """
    Categorizes review length into Short (< 10 words), Medium (10-20 words),
    or Long (> 20 words) for dynamic ensemble weighting (Table 8 in paper).
    
    Args:
        text (str): Review text.
        
    Returns:
        str: 'short', 'medium', or 'long'
    """
    words = [w for w in text.split() if w not in ['।', ',', '?', '!']]
    word_count = len(words)
    if word_count < 10:
        return 'short'
    elif word_count <= 20:
        return 'medium'
    else:
        return 'long'


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    sample = "প্রোডাক্টের কোয়ালিটি অনেককক ভালো!! কিন্তু ডেলিভারি পেতে ৫ দিন লাগলো... https://daraz.com.bd"
    cleaned = clean_bangla_text(sample)
    tokens = tokenize_bangla(cleaned)
    category = get_review_length_category(cleaned)
    print("Raw:      ", sample)
    print("Cleaned:  ", cleaned)
    print("Tokens:   ", tokens)
    print("Category: ", category)
