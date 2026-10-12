"""Evaluation: accuracy, per-class scores, confusion matrix, error analysis."""

from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def evaluate(pipeline, texts: list[str], labels, target_names: list[str]) -> dict:
    """Score a fitted pipeline; returns accuracy, macro F1, report and matrix."""
    predicted = pipeline.predict(texts)
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, average="macro")),
        "report": classification_report(
            labels, predicted, target_names=target_names, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(labels, predicted).tolist(),
        "n_test": len(labels),
    }


def error_analysis(
    pipeline,
    texts: list[str],
    labels,
    target_names: list[str],
    n: int = 5,
) -> list[dict]:
    """Return up to ``n`` misclassified examples with true vs predicted labels."""
    predicted = pipeline.predict(texts)
    mistakes = []
    for text, true, pred in zip(texts, labels, predicted):
        if true != pred and len(mistakes) < n:
            mistakes.append(
                {
                    "true_label": target_names[true],
                    "predicted_label": target_names[pred],
                    "text": text[:300] + ("…" if len(text) > 300 else ""),
                }
            )
    return mistakes


def format_report(results: dict, target_names: list[str]) -> str:
    """Render a human-readable summary of ``evaluate()`` output."""
    lines = [
        f"Accuracy : {results['accuracy']:.4f}",
        f"Macro F1 : {results['macro_f1']:.4f}",
        f"Test size: {results['n_test']}",
        "",
        "Per-class scores:",
    ]
    for name in target_names:
        scores = results["report"][name]
        lines.append(
            f"  {name:22s} precision={scores['precision']:.3f} "
            f"recall={scores['recall']:.3f} f1={scores['f1-score']:.3f}"
        )
    lines += ["", "Confusion matrix (rows=true, cols=predicted):"]
    header = " " * 24 + "".join(f"{n[:8]:>9}" for n in target_names)
    lines.append(header)
    for name, row in zip(target_names, results["confusion_matrix"]):
        lines.append(f"  {name:22s}" + "".join(f"{v:>9}" for v in row))
    return "\n".join(lines)
