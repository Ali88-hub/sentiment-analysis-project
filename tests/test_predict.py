import sys
from unittest.mock import MagicMock

import numpy as np

from src.predict import (
    format_prediction_lines,
    predict_texts,
)

sys.modules["joblib"] = MagicMock()


# -----------------------
# Mock classifiers
# -----------------------


class MockClassifierWithProba:
    def predict(self, texts):
        return np.array([1 if "good" in t else 0 for t in texts])

    def predict_proba(self, texts):
        # probability of positive class
        return np.array([[0.2, 0.8] if "good" in t else [0.9, 0.1] for t in texts])


class MockClassifierNoProba:
    def predict(self, texts):
        return np.array([0] * len(texts))


# -----------------------
# Tests for predict_texts
# -----------------------


def test_predict_texts_with_predict_proba():
    clf = MockClassifierWithProba()
    texts = ["good product", "bad product"]

    preds, probs = predict_texts(clf, texts)

    assert preds == [1, 0]
    assert probs == [0.8, 0.1]


def test_predict_texts_without_predict_proba():
    clf = MockClassifierNoProba()
    texts = ["anything", "something"]

    preds, probs = predict_texts(clf, texts)

    assert preds == [0, 0]
    assert probs == [None, None]


# -----------------------
# Tests for format_prediction_lines
# -----------------------


def test_format_prediction_lines_with_probs():
    texts = ["hello", "world"]
    preds = [1, 0]
    probs = [0.98765, 0.12345]

    lines = format_prediction_lines(texts, preds, probs)

    assert lines == [
        "1\t0.988\thello",
        "0\t0.123\tworld",
    ]


def test_format_prediction_lines_without_probs():
    texts = ["hello"]
    preds = [0]
    probs = [None]

    lines = format_prediction_lines(texts, preds, probs)

    assert lines == ["0\thello"]


# -----------------------
# Edge cases
# -----------------------


def test_empty_input():
    clf = MockClassifierNoProba()
    preds, probs = predict_texts(clf, [])

    assert preds == []
    assert probs == []
