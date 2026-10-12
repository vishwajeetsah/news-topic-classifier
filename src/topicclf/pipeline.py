"""Model pipelines: TF-IDF features + classifier, with save/load helpers."""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

MODEL_NAMES = ["nb", "logreg", "svc"]


def make_pipeline(model_name: str, **vectorizer_kwargs) -> Pipeline:
    """Build a ``TfidfVectorizer -> classifier`` pipeline.

    Extra keyword arguments override the vectorizer defaults (useful for
    tiny toy datasets in tests, e.g. ``min_df=1``).
    """
    if model_name not in MODEL_NAMES:
        raise ValueError(f"unknown model {model_name!r}; choose from {MODEL_NAMES}")
    vectorizer_params = {
        "stop_words": "english",
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 0.9,
        "sublinear_tf": True,
    }
    vectorizer_params.update(vectorizer_kwargs)
    vectorizer = TfidfVectorizer(**vectorizer_params)
    classifiers = {
        "nb": MultinomialNB(),
        "logreg": LogisticRegression(max_iter=1000, C=1.0),
        "svc": LinearSVC(C=1.0),
    }
    return Pipeline([("tfidf", vectorizer), ("clf", classifiers[model_name])])


def save_model(pipeline: Pipeline, path: str | Path) -> Path:
    """Persist a fitted pipeline with joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_model(path: str | Path) -> Pipeline:
    """Load a persisted pipeline."""
    return joblib.load(path)
