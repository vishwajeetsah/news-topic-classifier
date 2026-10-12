"""Command-line interface: train, evaluate, classify."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sklearn.model_selection import cross_val_score

from topicclf.data import CATEGORIES, load_data
from topicclf.evaluate import error_analysis, evaluate, format_report
from topicclf.pipeline import MODEL_NAMES, load_model, make_pipeline, save_model

DEFAULT_MODEL_PATH = Path("models") / "best_model.joblib"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="topicclf", description="News topic classifier (20 Newsgroups)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train and compare models, save the best.")
    train_p.add_argument("--cv", type=int, default=3, help="Cross-validation folds.")
    train_p.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)

    eval_p = sub.add_parser("evaluate", help="Evaluate the saved model on the test set.")
    eval_p.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    eval_p.add_argument("--errors", type=int, default=5, help="Show N misclassified examples.")

    clf_p = sub.add_parser("classify", help="Classify a single piece of text.")
    clf_p.add_argument("text", help="The text to classify.")
    clf_p.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    return parser


def cmd_train(cv: int, model_path: Path) -> int:
    print("Loading training data (downloads ~14 MB on first run)…")
    train = load_data("train")
    print(f"  {len(train.data)} documents, {len(train.target_names)} classes.")

    best_name, best_score, best_pipe = None, -1.0, None
    for name in MODEL_NAMES:
        pipe = make_pipeline(name)
        scores = cross_val_score(pipe, train.data, train.target, cv=cv, n_jobs=-1)
        mean = float(scores.mean())
        print(f"  {name:8s} cv macro-F1: {mean:.4f}  (folds: {scores.round(4).tolist()})")
        if mean > best_score:
            best_name, best_score, best_pipe = name, mean, pipe

    print(f"\nBest model: {best_name} (cv macro-F1 {best_score:.4f}). Refitting on full train set…")
    best_pipe.fit(train.data, train.target)
    save_model(best_pipe, model_path)
    print(f"Saved to {model_path}")
    return 0


def cmd_evaluate(model_path: Path, n_errors: int) -> int:
    if not model_path.exists():
        print(f"No saved model at {model_path} — run `topicclf train` first.", file=sys.stderr)
        return 1
    pipe = load_model(model_path)
    print("Loading test data…")
    test = load_data("test")
    results = evaluate(pipe, test.data, test.target, test.target_names)
    print(format_report(results, test.target_names))
    if n_errors:
        print("\nMisclassified examples:")
        for m in error_analysis(pipe, test.data, test.target, test.target_names, n_errors):
            print(f"  true={m['true_label']}, predicted={m['predicted_label']}")
            print(f"    {m['text'][:160]}…\n")
    with open("eval_results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print("Wrote eval_results.json")
    return 0


def cmd_classify(text: str, model_path: Path) -> int:
    if not model_path.exists():
        print(f"No saved model at {model_path} — run `topicclf train` first.", file=sys.stderr)
        return 1
    pipe = load_model(model_path)
    test = load_data("test")  # for target names only
    pred = pipe.predict([text])[0]
    proba = None
    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        proba = float(pipe.named_steps["clf"].predict_proba(pipe.named_steps["tfidf"].transform([text])).max())
    label = test.target_names[pred]
    print(f"Predicted topic: {label}" + (f"  (confidence {proba:.2f})" if proba else ""))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        return cmd_train(args.cv, args.model_path)
    if args.command == "evaluate":
        return cmd_evaluate(args.model_path, args.errors)
    if args.command == "classify":
        return cmd_classify(args.text, args.model_path)
    return 1


if __name__ == "__main__":
    sys.exit(main())
