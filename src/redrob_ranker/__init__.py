"""Redrob Ranker — intelligent candidate discovery & ranking for the Redrob
hackathon.

The package is organized as a small, testable pipeline:

    schema    -> typed view over a raw candidate record
    jobspec   -> the structured interpretation of the job description
    text      -> profile/JD text construction for the lexical & semantic engines
    honeypot  -> "subtly impossible profile" detector (hard kill in ranking)
    structured-> rule-based scoring of role/experience/location/education/signals
    lexical   -> BM25 over profile text
    semantic  -> sentence-transformer dense fit (optional; degrades to lexical)
    fuse      -> combine components into a final calibrated score
    pipeline  -> end-to-end orchestration producing the top-100 submission
"""

__version__ = "1.0.0"
