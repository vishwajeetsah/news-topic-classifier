"""News topic classifier: scikit-learn text classification on 20 Newsgroups."""

from topicclf.data import CATEGORIES, load_data
from topicclf.evaluate import error_analysis, evaluate
from topicclf.pipeline import MODEL_NAMES, load_model, make_pipeline, save_model

__version__ = "0.1.0"

__all__ = [
    "CATEGORIES",
    "MODEL_NAMES",
    "error_analysis",
    "evaluate",
    "load_data",
    "load_model",
    "make_pipeline",
    "save_model",
]
