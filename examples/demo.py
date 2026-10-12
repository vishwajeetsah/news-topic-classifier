"""End-to-end demo: train on the real 20 Newsgroups subset and print a report.

Downloads ~14 MB on first run (cached afterwards in ~/scikit_learn_data).
"""

from topicclf.data import load_data
from topicclf.evaluate import error_analysis, evaluate, format_report
from topicclf.pipeline import make_pipeline


def main() -> None:
    train = load_data("train")
    test = load_data("test")
    print(f"Train: {len(train.data)} docs | Test: {len(test.data)} docs")

    pipe = make_pipeline("logreg")
    pipe.fit(train.data, train.target)

    results = evaluate(pipe, test.data, test.target, test.target_names)
    print(format_report(results, test.target_names))

    print("\nSample misclassifications:")
    for m in error_analysis(pipe, test.data, test.target, test.target_names, n=3):
        print(f"  true={m['true_label']}, predicted={m['predicted_label']}")


if __name__ == "__main__":
    main()
