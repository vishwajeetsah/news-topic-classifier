"""Dataset loading.

Uses scikit-learn's 20 Newsgroups downloader (public benchmark dataset,
~14 MB on first run, cached afterwards in ``~/scikit_learn_data``).
Headers, footers and quotes are stripped so the models must learn from
the message *content*, not metadata — a harder, more honest task.
"""

from __future__ import annotations

from sklearn.datasets import fetch_20newsgroups

CATEGORIES = [
    "comp.graphics",
    "rec.sport.baseball",
    "sci.med",
    "talk.politics.misc",
]

REMOVE = ("headers", "footers", "quotes")


def load_data(subset: str = "train", categories: list[str] | None = None):
    """Fetch the 20 Newsgroups subset as a sklearn Bunch.

    ``subset`` is ``"train"`` or ``"test"`` (the dataset's own fixed split).
    """
    return fetch_20newsgroups(
        subset=subset,
        categories=categories or CATEGORIES,
        remove=REMOVE,
        shuffle=True,
        random_state=42,
    )
