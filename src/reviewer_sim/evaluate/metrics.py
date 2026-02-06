import re
from typing import Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def tfidf_cosine(a: str, b: str) -> float:
    vect = TfidfVectorizer(stop_words="english")
    X = vect.fit_transform([a, b])
    sim = cosine_similarity(X[0], X[1])[0][0]
    return float(sim)


def keyword_jaccard(a: str, b: str) -> float:
    def toks(s: str) -> set[str]:
        # simple tokenization; keeps alphabetic tokens longer than 3 chars
        return {t.lower() for t in s.split() if t.isalpha() and len(t) > 3}

    A, B = toks(a), toks(b)
    if not A and not B:
        return 0.0
    return float(len(A & B) / max(1, len(A | B)))


def parse_numeric_rating(value: object) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)", value)
        if match:
            return float(match.group(1))
    return None


def evaluate(example: Dict, generated: Dict) -> Dict:
    """
    Compare generated review to human review for a single example.

    Expected schema:
      example["human_review"]["text"]  (string)
      example["human_review"]["rating"] (optional number)
      generated["text"] (string)
      generated["score"] (optional number)
    """
    human = example.get("human_review", {}) or {}
    human_text = human.get("text", "") or ""
    human_rating = human.get("rating", None)

    gen_text = generated.get("text", "") or ""
    gen_score = generated.get("score", None)

    score_abs_diff = None
    human_numeric = parse_numeric_rating(human_rating)
    gen_numeric = parse_numeric_rating(gen_score)
    if human_numeric is not None and gen_numeric is not None:
        score_abs_diff = abs(float(human_numeric) - float(gen_numeric))

    return {
        "tfidf_cosine": tfidf_cosine(human_text, gen_text) if human_text and gen_text else 0.0,
        "keyword_jaccard": keyword_jaccard(human_text, gen_text) if human_text and gen_text else 0.0,
        "score_abs_diff": score_abs_diff,
    }
