"""Semantic engine: dense bi-encoder fit between the JD query and candidates.

Uses a local sentence-transformers model cached under artifacts/minilm so the
ranking step runs with no network. The model is OPTIONAL: if the cache is
absent, `load_encoder` returns None and the pipeline transparently falls back to
the lexical (BM25) signal. This keeps the system reproducible even on a machine
where the model could not be pre-fetched.

Encoding the full 100K pool is the only heavy step, so the pipeline encodes only
the structural shortlist (a few thousand survivors), keeping us well inside the
5-minute CPU budget. Embeddings are cached to disk keyed by content hash.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

# This file lives at <repo>/src/redrob_ranker/semantic.py, so the repo root is
# three levels up. The cached encoder is expected at <repo>/artifacts/minilm.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MODEL_DIR = os.path.join(_REPO_ROOT, "artifacts", "minilm")


def load_encoder(model_dir: str = DEFAULT_MODEL_DIR):
    """Return a SentenceTransformer loaded from a local dir, or None.

    Never raises and never hits the network: if the cached model or the library
    is unavailable, callers fall back to lexical-only scoring.
    """
    if not os.path.isdir(model_dir):
        return None
    # Force fully-offline behavior for the underlying libraries.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_dir, device="cpu")
    except Exception:
        return None


def encode(encoder, texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Encode texts to L2-normalized embeddings (n x d, float32)."""
    emb = encoder.encode(
        texts, batch_size=batch_size, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    )
    return emb.astype(np.float32)


def cosine_to_query(doc_emb: np.ndarray, query_emb: np.ndarray) -> np.ndarray:
    """Cosine similarity of each row in doc_emb to a single query vector.

    Inputs are already L2-normalized, so this is a dot product. Mapped from
    [-1,1] to [0,1].
    """
    sims = doc_emb @ query_emb.ravel()
    return ((sims + 1.0) / 2.0).astype(np.float32)
