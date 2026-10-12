# news-topic-classifier

A complete, reproducible **text-classification** project: given a news article, predict its topic using classic machine-learning (TF-IDF + linear models) on the public **20 Newsgroups** benchmark.

## Problem statement

Automatically routing documents to topics is one of the oldest — and still most practical — NLP tasks: spam filters, news aggregators, support-ticket triage and content moderation all rely on it. This project implements the full workflow end-to-end: data loading, feature extraction, model comparison with cross-validation, evaluation with error analysis, model persistence, and a CLI for inference.

## Key features

- **Real benchmark data** — 20 Newsgroups (4 topics, 2,240 train / 1,492 test docs), headers/footers/quotes stripped so models learn from content, not metadata
- **Model comparison** — Multinomial Naive Bayes vs Logistic Regression vs Linear SVM, ranked by 3-fold cross-validated macro-F1
- **Honest evaluation** — accuracy, per-class precision/recall/F1, confusion matrix, and misclassified examples on the held-out test set
- **Inference CLI** — `topicclf classify "some text"` with confidence scores
- **Reproducible** — fixed random seeds, pinned dependencies, CI-tested

## Architecture

```
20 Newsgroups ──▶ TfidfVectorizer ──▶ classifier ──▶ metrics / report
   (train)        (1–2 grams,          (NB / LogReg / SVM)
                   stop words,
                   sublinear TF)
```

## Technologies

Python 3.10+, scikit-learn, NumPy, joblib. No deep-learning frameworks, no API keys, no paid services.

## Installation

```bash
pip install -e ".[test]"
```

## Usage

```bash
# Train all three models, cross-validate, save the best (downloads ~14 MB once)
topicclf train

# Evaluate the saved model on the held-out test set
topicclf evaluate

# Classify your own text
topicclf classify "The new graphics card renders 3D scenes beautifully"
# → Predicted topic: comp.graphics  (confidence 0.81)
```

Or run the end-to-end demo script: `python examples/demo.py`.

## Sample output

```
Accuracy : 0.8854
Macro F1 : 0.8830
Test size: 1492

Per-class scores:
  comp.graphics          precision=0.901 recall=0.936 f1=0.918
  rec.sport.baseball     precision=0.856 recall=0.942 f1=0.897
  sci.med                precision=0.867 recall=0.859 f1=0.863
  talk.politics.misc     precision=0.938 recall=0.784 f1=0.854
```

## Testing and evaluation methodology

- **Unit tests** (`tests/`): 7 tests on a tiny synthetic dataset (clearly labelled synthetic) covering pipelines, metrics, error analysis and model save/load — run offline with `pytest -q`.
- **Model selection**: 3-fold cross-validation, macro-F1, on the training split.
- **Final evaluation**: single held-out test split (the dataset's own), never touched during training.

### Actual results (measured 2026-10-12)

| Model | CV macro-F1 |
| ----- | ----------: |
| MultinomialNB | **0.9080** |
| LinearSVC | 0.9040 |
| LogisticRegression | 0.8987 |

Best model (Naive Bayes) on the held-out test set: **accuracy 0.8854, macro-F1 0.8830** (n=1,492). Politics is the hardest class (recall 0.784 — often confused with medicine, e.g. posts about health policy).

## Limitations and known issues

- Bag-of-words features ignore word order and semantics; sarcasm and topic overlap (health *policy*) confuse the model.
- The 20 Newsgroups data is from the 1990s — vocabulary and topics are dated.
- The CLI `classify` downloads nothing but needs a trained model (`topicclf train` first).
- No hyperparameter search beyond sensible defaults; no calibration of NB probabilities.

## Future improvements

- [ ] Grid search over `C`, n-gram range and `min_df`
- [ ] Add a transformer baseline (e.g. DistilBERT fine-tune) for comparison
- [ ] Probability calibration + confidence thresholding ("I don't know" option)
- [ ] Simple web UI for interactive classification

## Dataset attribution

20 Newsgroups dataset (public benchmark, collected by Ken Lang). Loaded via scikit-learn's `fetch_20newsgroups`; see http://qwone.com/~jason/20Newsgroups/.

## What I Learned

- **TF-IDF**: why term frequency is damped (sublinear TF) and why ubiquitous words are down-weighted (IDF) — and how `min_df`/`max_df` prune noise.
- **Naive Bayes for text**: the "naive" independence assumption is wrong but the model is fast, needs little data, and often wins on sparse text — as it did here.
- **Cross-validation vs test set**: CV picks the model, the held-out test set gives the honest number. Mixing the two is the classic way to fool yourself.
- **Error analysis > accuracy**: the confusion matrix showed politics↔medicine confusion, which a single accuracy number hides.
- **Stripping metadata matters**: removing headers/footers/quotes made the task harder but the evaluation honest — models can't cheat on `From:` lines.

## License

MIT — see [LICENSE](LICENSE).
