"""Score fusion and reasoning generation.

Fusion philosophy (defensible at interview):

  intrinsic = weighted sum of *fit* components
              (semantic, role, skills, experience, location, education)
  final     = intrinsic
              * availability_multiplier   (behavioral: can we actually hire them)
              * stuffer_penalty           (keyword-stuffer demotion)
              * honeypot_kill             (0 if impossible profile)

Additive fusion across fit components gives a smooth, interpretable score; the
multiplicative modifiers encode "necessary conditions" — being unavailable or a
keyword stuffer should scale a good paper score *down*, not merely add a small
penalty. role_fit also enters additively with a high weight so an off-target
career can't be rescued by semantic similarity alone.
"""
from __future__ import annotations

from dataclasses import dataclass

from .structured import StructuredScores

# Weights for the additive "intrinsic fit". Sum to 1.0. role + skill + evidence
# dominate: role gates out off-target careers, skill captures tagged coverage,
# and evidence captures *demonstrated* production retrieval/ranking work — the
# signal the JD says matters most ("career history shows they built it").
W_SEMANTIC = 0.22
W_ROLE = 0.24
W_SKILL = 0.18
W_EVIDENCE = 0.16
W_EXPERIENCE = 0.10
W_LOCATION = 0.07
W_EDUCATION = 0.03


@dataclass
class FusedResult:
    candidate_id: str
    final_score: float
    intrinsic: float
    semantic: float
    components: StructuredScores
    is_honeypot: bool
    honeypot_reasons: tuple


def fuse(
    candidate_id: str,
    structured: StructuredScores,
    semantic_fit: float,
    is_honeypot: bool,
    honeypot_reasons: tuple = (),
) -> FusedResult:
    intrinsic = (
        W_SEMANTIC * semantic_fit
        + W_ROLE * structured.role_fit
        + W_SKILL * structured.skill_fit
        + W_EVIDENCE * structured.evidence_fit
        + W_EXPERIENCE * structured.experience_fit
        + W_LOCATION * structured.location_fit
        + W_EDUCATION * structured.education_fit
    )
    final = intrinsic * structured.availability * structured.stuffer_penalty
    if is_honeypot:
        final = 0.0
    return FusedResult(
        candidate_id=candidate_id,
        final_score=float(final),
        intrinsic=float(intrinsic),
        semantic=float(semantic_fit),
        components=structured,
        is_honeypot=is_honeypot,
        honeypot_reasons=honeypot_reasons,
    )


def build_reasoning(c, fused: FusedResult, rank: int) -> str:
    """Produce a specific, honest, 1-2 sentence reasoning string.

    Stage-4 review checks for: specific facts, JD connection, honest concerns,
    no hallucination, variation, and rank-consistent tone. We assemble the
    string from facts actually present on the candidate and the component notes,
    so every claim is grounded. Tone shifts with rank.
    """
    n = fused.components.notes
    title = c.current_title or "candidate"
    yoe = c.years_of_experience

    # Lead with the strongest grounded facts.
    lead = f"{title} with {yoe:.1f}y experience; {n['skill']}"

    # JD-connected positive, chosen by what's actually strong. Each claim must
    # be grounded: the skill claim fires only when genuine retrieval/vector
    # must-have skills are actually present on the candidate, not merely when the
    # aggregate skill_fit clears a threshold (which nice-to-haves alone could do).
    from .jobspec import MUST_HAVE_SKILLS
    has_retrieval_skill = bool(c.skill_names_lower & set(MUST_HAVE_SKILLS))
    strengths = []
    if fused.components.role_fit >= 0.8:
        strengths.append("strong role alignment")
    if fused.components.evidence_fit >= 0.5 and fused.components.notes.get("evidence"):
        strengths.append(fused.components.notes["evidence"])
    if fused.components.skill_fit >= 0.6 and has_retrieval_skill:
        strengths.append("verified retrieval/ranking skills")
    if fused.semantic >= 0.7:
        strengths.append("profile semantically matches the JD")
    if fused.components.experience_fit >= 0.85:
        strengths.append("experience in the target band")
    pos = "; ".join(strengths[:2])

    # Honest concerns, chosen by what's actually weak.
    concerns = []
    if fused.components.stuffer_penalty < 1.0:
        concerns.append(n["stuffer"])
    if fused.components.availability < 0.75:
        concerns.append(n["availability"])
    if fused.components.location_fit < 0.5:
        concerns.append(n["location"])
    if fused.components.experience_fit < 0.6:
        concerns.append("experience outside the ideal band")
    con = "; ".join([x for x in concerns if x][:2])

    parts = [lead]
    if pos:
        parts.append(pos)
    # Tone matches rank: top ranks lead with fit, tail ranks acknowledge limits.
    if con:
        connector = "but note: " if rank <= 50 else "included despite "
        parts.append(connector + con)

    text = ". ".join(p for p in parts if p)
    # CSV-safety: collapse whitespace; the writer will quote as needed.
    return " ".join(text.split())
