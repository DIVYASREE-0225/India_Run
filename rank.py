#!/usr/bin/env python3
"""Redrob Ranker CLI — produces the top-100 submission CSV from candidates.jsonl.

Reproduce command (matches submission_metadata.yaml):

    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Constraints honored: CPU-only, no network during ranking, < 5 min, < 16 GB.
The semantic model is loaded from a local cache (artifacts/minilm); if absent,
the ranker degrades gracefully to a lexical + structured ranking.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

# Make `src` importable whether run from repo root or installed.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from redrob_ranker.pipeline import rank_candidates  # noqa: E402


def write_submission(rows, out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in rows:
            w.writerow([r.candidate_id, r.rank, f"{r.score:.6f}", r.reasoning])


def main() -> int:
    ap = argparse.ArgumentParser(description="Redrob intelligent candidate ranker")
    ap.add_argument("--candidates", required=True,
                    help="Path to candidates.jsonl (or .jsonl.gz)")
    ap.add_argument("--out", default="submission.csv", help="Output CSV path")
    ap.add_argument("--top-n", type=int, default=100)
    ap.add_argument("--rerank-pool", type=int, default=4000,
                    help="How many top provisional candidates get dense re-ranking")
    ap.add_argument("--model-dir", default=None,
                    help="Local sentence-transformers dir (default: artifacts/minilm)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    log = (lambda *a, **k: None) if args.quiet else print
    rows = rank_candidates(
        candidates_path=args.candidates,
        top_n=args.top_n,
        rerank_pool=args.rerank_pool,
        model_dir=args.model_dir,
        log=log,
    )
    write_submission(rows, args.out)
    log(f"[write] wrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
