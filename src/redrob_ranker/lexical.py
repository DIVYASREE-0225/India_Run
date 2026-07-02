"""Lexical retrieval: BM25 over candidate documents.

A compact, dependency-light BM25 (Okapi) implemented on top of scikit-learn's
CountVectorizer so we avoid an extra package and keep the ranking step fast and
fully offline. Returns a normalized [0,1] relevance per candidate against the
JD query text.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75, max_features: int = 60000):
        self.k1 = k1
        self.b = b
        self.vectorizer = CountVectorizer(
            max_features=max_features, stop_words="english",
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9.+#-]{1,}\b",
        )
        self._tf = None          # term-frequency matrix (csr)
        self._idf = None
        self._doc_len = None
        self._avgdl = None

    def fit(self, documents: list[str]) -> "BM25":
        tf = self.vectorizer.fit_transform(documents)   # n_docs x vocab, counts
        self._tf = tf.tocsr()
        n_docs = tf.shape[0]
        df = np.asarray((tf > 0).sum(axis=0)).ravel()
        # BM25 idf with +1 smoothing (always positive).
        self._idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        self._doc_len = np.asarray(tf.sum(axis=1)).ravel()
        self._avgdl = float(self._doc_len.mean()) if n_docs else 0.0
        return self

    def score(self, query: str) -> np.ndarray:
        """BM25 score of every fitted document against the query."""
        q = self.vectorizer.transform([query])
        q_terms = q.indices
        if len(q_terms) == 0:
            return np.zeros(self._tf.shape[0], dtype=np.float32)

        tf = self._tf[:, q_terms].toarray().astype(np.float32)   # n_docs x |q|
        idf = self._idf[q_terms]                                  # |q|
        denom_norm = self.k1 * (1.0 - self.b + self.b * (self._doc_len / (self._avgdl or 1.0)))
        # BM25 term contribution: idf * (tf*(k1+1)) / (tf + denom_norm)
        denom = tf + denom_norm[:, None]
        contrib = (tf * (self.k1 + 1.0)) / np.maximum(denom, 1e-9)
        scores = contrib @ idf
        return scores.astype(np.float32)


def normalize01(x: np.ndarray) -> np.ndarray:
    """Min-max to [0,1]; flat input -> zeros."""
    x = np.asarray(x, dtype=np.float32)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)
