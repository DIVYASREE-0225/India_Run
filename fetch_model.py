"""One-time model fetch. Run once with network access; caches the encoder into
artifacts/minilm so the ranker can run fully offline thereafter.

Usage:
    python fetch_model.py
"""
import os

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

from sentence_transformers import SentenceTransformer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEST = os.path.join(os.path.dirname(__file__), "artifacts", "minilm")

if __name__ == "__main__":
    print(f"Downloading {MODEL} ...")
    SentenceTransformer(MODEL).save(DEST)
    print(f"SAVED to {DEST}")
