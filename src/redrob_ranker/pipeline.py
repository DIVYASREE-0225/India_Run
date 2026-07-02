"""End-to-end ranking pipeline.

Two stages, designed for the 5-minute / 16 GB / CPU-only / no-network budget:

  Stage A (cheap, all 100K): load + type each record, run honeypot detection and
          structured scoring, compute BM25 lexical fit. Produce a provisional
          score WITHOUT the dense encoder. This alone is a strong ranker.

  Stage B (dense re-rank, shortlist only): take the top-K survivors by the
          provisional score and the semantic candidate pool, encode just those
          with the bi-encoder, and re-fuse with the semantic component. Encoding
          a few thousand docs (not 100K) is what keeps us inside the budget.

If the encoder is unavailable, Stage B is skipped and the semantic component is
taken from the (already computed) BM25 lexical fit — a graceful, fully-offline
fallback that still produces a valid, high-quality submission.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from . import semantic as sem
from .fuse import FusedResult, build_reasoning, fuse
from .honeypot import detect_honeypot
from .jobspec import JD_QUERY_TEXT
from .lexical import BM25, normalize01
from .schema import Candidate
from .structured import StructuredScores, score_structured
from .text import candidate_document


@dataclass
class RankedRow:
    candidate_id: str
    rank: int
    score: float
    reasoning: str


def _iter_candidates(path: str) -> Iterable[dict]:
    opener = open
    if path.endswith(".gz"):
        import gzip
        opener = lambda p: gzip.open(p, "rt", encoding="utf-8")  # noqa: E731
    with opener(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def rank_candidates(
    candidates_path: str,
    top_n: int = 100,
    rerank_pool: int = 4000,
    model_dir: Optional[str] = None,
    log=print,
) -> list[RankedRow]:
    t0 = time.time()

    # ---- Stage A: load + structured + lexical over the full pool ----------
    log("[stageA] loading & scoring full pool ...")
    cands: list[Candidate] = []
    docs: list[str] = []
    structured: list[StructuredScores] = []
    honeypot_flags: list[bool] = []
    honeypot_reasons: list[tuple] = []

    for raw in _iter_candidates(candidates_path):
        c = Candidate(raw)
        cands.append(c)
        docs.append(candidate_document(c))
        structured.append(score_structured(c))
        v = detect_honeypot(c)
        honeypot_flags.append(v.is_honeypot)
        honeypot_reasons.append(v.reasons)

    n = len(cands)
    log(f"[stageA] {n} candidates loaded in {time.time()-t0:.1f}s; "
        f"{sum(honeypot_flags)} honeypots flagged")

    # BM25 lexical fit against the JD query.
    bm25 = BM25().fit(docs)
    lexical = normalize01(bm25.score(JD_QUERY_TEXT))

    # Provisional intrinsic score (no dense yet): use lexical as the semantic
    # proxy so the shortlist for Stage B is well-chosen.
    prov: list[FusedResult] = []
    for i in range(n):
        prov.append(fuse(
            cands[i].id, structured[i], float(lexical[i]),
            honeypot_flags[i], honeypot_reasons[i],
        ))
    prov_scores = np.array([p.final_score for p in prov], dtype=np.float32)

    # ---- Stage B: dense re-rank of the shortlist --------------------------
    encoder = sem.load_encoder(model_dir) if model_dir else sem.load_encoder()
    semantic_fit = lexical.copy()  # default/fallback

    if encoder is not None:
        # Shortlist: top `rerank_pool` by provisional score (non-honeypot).
        order = np.argsort(-prov_scores)
        shortlist = [int(i) for i in order[:rerank_pool]]
        log(f"[stageB] encoder loaded; dense re-ranking {len(shortlist)} candidates ...")
        q_emb = sem.encode(encoder, [JD_QUERY_TEXT])
        d_emb = sem.encode(encoder, [docs[i] for i in shortlist])
        cos = sem.cosine_to_query(d_emb, q_emb)
        for local_idx, global_idx in enumerate(shortlist):
            semantic_fit[global_idx] = cos[local_idx]
        log(f"[stageB] dense re-rank done in {time.time()-t0:.1f}s")
    else:
        log("[stageB] no local encoder found -> lexical-only semantic fallback")

    # ---- Final fusion -----------------------------------------------------
    fused: list[FusedResult] = []
    for i in range(n):
        fused.append(fuse(
            cands[i].id, structured[i], float(semantic_fit[i]),
            honeypot_flags[i], honeypot_reasons[i],
        ))

    # Sort: score desc, then candidate_id asc. Candidate IDs are CAND_ + 7
    # zero-padded digits, so lexicographic order == numeric order, matching the
    # validator's string-based tie-break exactly.
    ordered = sorted(fused, key=lambda r: (-r.final_score, r.candidate_id))
    by_id = {c.id: c for c in cands if c.id}  # drop any record missing an id

    rows: list[RankedRow] = []
    prev_score = None
    for rank, fr in enumerate(ordered[:top_n], start=1):
        c = by_id.get(fr.candidate_id)
        if c is None:
            # Should never happen (ids come from the same records), but never
            # emit reasoning sourced from the wrong candidate.
            continue
        score = round(fr.final_score, 6)
        # Guard: rounding must never break the non-increasing invariant the
        # validator checks on the score column directly. The pre-round sort is
        # already monotonic; clamp any rounding artifact to the prior score.
        if prev_score is not None and score > prev_score:
            score = prev_score
        prev_score = score
        rows.append(RankedRow(
            candidate_id=fr.candidate_id,
            rank=rank,
            score=score,
            reasoning=build_reasoning(c, fr, rank),
        ))

    log(f"[done] produced top-{top_n} in {time.time()-t0:.1f}s")
    return rows
