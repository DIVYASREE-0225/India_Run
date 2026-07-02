"""Streamlit sandbox app for the Redrob Ranker.

Satisfies the submission-spec Section 10.5 sandbox requirement: a hosted UI that
accepts a small candidate sample (<=100), runs the ranking end-to-end on CPU,
and shows the ranked CSV. Deployable free on HuggingFace Spaces / Streamlit Cloud.

Run locally:
    streamlit run app.py
"""
import io
import json
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from redrob_ranker.pipeline import rank_candidates  # noqa: E402

st.set_page_config(page_title="Redrob Ranker", page_icon="🎯", layout="wide")
st.title("🎯 Redrob Intelligent Candidate Ranker")
st.caption(
    "Hybrid ranker — dense semantic fit + BM25 + structured signal scoring — for "
    "the Senior AI Engineer JD. Upload a small JSONL sample (≤100 candidates) or "
    "use the bundled sample."
)

with st.sidebar:
    st.header("How it works")
    st.markdown(
        "- **Role gate** keeps keyword-stuffers (AI skills on off-target careers) out\n"
        "- **Honeypot filter** kills physically-impossible profiles\n"
        "- **Semantic + lexical** capture fit beyond keywords\n"
        "- **Behavioral multiplier** down-weights unavailable candidates\n"
    )
    top_n = st.slider("Top N to return", 5, 100, 25)
    rerank_pool = st.slider("Dense re-rank pool", 50, 4000, 500, step=50)

uploaded = st.file_uploader("Candidate JSONL (one JSON object per line)", type=["jsonl"])

default_sample = os.path.join(os.path.dirname(__file__), "data", "sample.jsonl")
use_default = st.checkbox("Use bundled sample (50 candidates)", value=uploaded is None)

if st.button("Rank candidates", type="primary"):
    # Resolve input to a temp path the pipeline can stream.
    tmp_path = os.path.join(os.path.dirname(__file__), "data", "_uploaded.jsonl")
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    if uploaded is not None and not use_default:
        lines = uploaded.getvalue().decode("utf-8").splitlines()
        lines = [ln for ln in lines if ln.strip()][:100]  # enforce <=100
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        src = tmp_path
    elif os.path.exists(default_sample):
        src = default_sample
    else:
        st.error("No input: upload a JSONL or ensure data/sample.jsonl exists.")
        st.stop()

    with st.spinner("Ranking on CPU…"):
        rows = rank_candidates(src, top_n=top_n, rerank_pool=rerank_pool,
                               log=lambda *a, **k: None)

    df = pd.DataFrame([{"candidate_id": r.candidate_id, "rank": r.rank,
                        "score": round(r.score, 4), "reasoning": r.reasoning}
                       for r in rows])
    st.success(f"Ranked top {len(df)} candidates.")
    st.dataframe(df, use_container_width=True, hide_index=True)

    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    st.download_button("Download submission.csv", csv_buf.getvalue(),
                       file_name="submission.csv", mime="text/csv")
