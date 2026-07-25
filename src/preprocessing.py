# =============================================================
# preprocessing.py
# CCS3356 - Natural Language Processing
# NLP_Group_10 - Fake News Detection
# Shared preprocessing utility for all members
# =============================================================

import re
import string
import nltk

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('omw-1.4', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    """
    Full text preprocessing pipeline.
    Steps:
        1. Lowercase
        2. Remove URLs
        3. Remove punctuation and special characters
        4. Tokenize
        5. Remove stopwords
        6. Lemmatize
    Args:
        text (str): Raw input text
    Returns:
        str: Cleaned and preprocessed text
    """
    # Lowercase
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove punctuation and special characters
    text = re.sub(r'[^a-z\s]', '', text)

    # Tokenize
    tokens = text.split()

    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens
              if word not in stop_words and len(word) > 2]

    return ' '.join(tokens)


def preprocess_batch(texts):
    """
    Apply preprocessing to a list or pandas Series of texts.
    Args:
        texts: list or pandas Series of raw texts
    Returns:
        list: List of cleaned texts
    """
    return [preprocess_text(text) for text in texts]


def clean_for_display(text, max_length=500):
    """
    Light cleaning for display purposes in the Streamlit app.
    Does not remove stopwords — keeps text readable.
    Args:
        text (str): Raw input text
        max_length (int): Max number of words to display
    Returns:
        str: Lightly cleaned text
    """
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()[:max_length]
    return ' '.join(words)


if __name__ == '__main__':
    # Quick test
    sample = "BREAKING NEWS: President signs new bill! Visit http://fakenews.com for details."
    print("Original:", sample)
    print("Preprocessed:", preprocess_text(sample))
    print("Display clean:", clean_for_display(sample))