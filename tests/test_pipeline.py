"""Tests for the topic classifier.

Uses a tiny synthetic dataset (clearly labelled as such) so the suite runs
offline in milliseconds. Real-data behaviour is exercised by the CLI.
"""

import pytest

from topicclf.evaluate import error_analysis, evaluate
from topicclf.pipeline import MODEL_NAMES, load_model, make_pipeline, save_model

# Synthetic toy data — for unit tests only, NOT a real benchmark.
TOY_TEXTS = [
    "goal scored in the final match",
    "the team won the championship game",
    "pitcher threw a perfect inning",
    "graphics card rendering polygons",
    "pixel shader 3d animation",
    "doctor prescribed antibiotics",
    "patient diagnosis treatment hospital",
    "election vote parliament government",
]
TOY_LABELS = [0, 0, 0, 1, 1, 2, 2, 3]
TOY_NAMES = ["sports", "graphics", "medicine", "politics"]


def test_make_pipeline_rejects_unknown_model():
    with pytest.raises(ValueError, match="unknown model"):
        make_pipeline("nope", min_df=1)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_pipeline_fit_predict(name):
    pipe = make_pipeline(name, min_df=1)
    pipe.fit(TOY_TEXTS, TOY_LABELS)
    preds = pipe.predict(["a thrilling baseball game", "3d rendering engine"])
    assert len(preds) == 2
    assert all(p in range(4) for p in preds)


def test_evaluate_returns_expected_keys():
    pipe = make_pipeline("nb", min_df=1).fit(TOY_TEXTS, TOY_LABELS)
    results = evaluate(pipe, TOY_TEXTS, TOY_LABELS, TOY_NAMES)
    assert {"accuracy", "macro_f1", "report", "confusion_matrix", "n_test"} <= set(results)
    assert 0.0 <= results["accuracy"] <= 1.0
    assert results["n_test"] == len(TOY_TEXTS)


def test_error_analysis_shape():
    pipe = make_pipeline("nb", min_df=1).fit(TOY_TEXTS, TOY_LABELS)
    mistakes = error_analysis(pipe, TOY_TEXTS, TOY_LABELS, TOY_NAMES, n=3)
    assert isinstance(mistakes, list)
    for m in mistakes:
        assert {"true_label", "predicted_label", "text"} <= set(m)


def test_save_and_load_roundtrip(tmp_path):
    pipe = make_pipeline("logreg", min_df=1).fit(TOY_TEXTS, TOY_LABELS)
    path = tmp_path / "model.joblib"
    save_model(pipe, path)
    assert path.exists()
    loaded = load_model(path)
    assert (loaded.predict(TOY_TEXTS) == pipe.predict(TOY_TEXTS)).all()
