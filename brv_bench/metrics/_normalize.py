"""Shared answer-normalization utilities (SQuAD / LoCoMo convention)."""

import re
import string

from nltk.stem import PorterStemmer

_stemmer = PorterStemmer()


def normalize_answer(text: str) -> str:
    """Lowercase, strip articles/punctuation, collapse whitespace."""
    text = text.lower()
    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # collapse whitespace
    text = " ".join(text.split())
    return text


def get_tokens(text: str, stem: bool = False) -> list[str]:
    """Normalize, split, and optionally stem tokens."""
    tokens = normalize_answer(text).split()
    if stem:
        tokens = [_stemmer.stem(w) for w in tokens]
    return tokens
