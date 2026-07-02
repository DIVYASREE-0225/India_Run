"""Text construction for the lexical (BM25) and semantic (bi-encoder) engines.

Both engines need a single string per candidate. We weight the most
JD-relevant evidence by repeating it: the headline and summary carry the
candidate's own framing, while career descriptions carry what they actually
*did* — which the JD says matters more than the skills list. Skills are included
but deliberately not over-weighted, so a keyword-stuffed skills section cannot
dominate the semantic/lexical signal on its own.
"""
from __future__ import annotations

from .schema import Candidate


def candidate_document(c: Candidate) -> str:
    """Build the text representation used by both retrieval engines."""
    parts: list[str] = []

    # Identity / self-framing (weighted x2 — concise, high-signal).
    if c.current_title:
        parts.append(c.current_title)
        parts.append(c.current_title)
    if c.headline:
        parts.append(c.headline)
    if c.summary:
        parts.append(c.summary)

    # What they actually did — career titles + descriptions are the strongest
    # evidence of real, applied experience (JD: "career history shows they built
    # a recommendation system" beats a buzzword skills list).
    for j in c.career:
        if j.title:
            parts.append(j.title)
        if j.description:
            parts.append(j.description)
        if j.industry:
            parts.append(j.industry)

    # Skills (single weight) and education field of study.
    skill_str = " ".join(s.name for s in c.skills if s.name)
    if skill_str:
        parts.append(skill_str)
    for e in c.education:
        fos = e.get("field_of_study")
        if fos:
            parts.append(fos)

    return "\n".join(parts).strip()
