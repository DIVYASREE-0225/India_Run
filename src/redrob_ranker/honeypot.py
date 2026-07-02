"""Honeypot detection — "subtly impossible profile" filter.

The spec (Section 7) seeds ~80 honeypots with internally inconsistent profiles
(e.g. "8 years at a company founded 3 years ago", "expert in 10 skills with 0
years used"). They are forced to relevance tier 0 in the ground truth, and a
top-100 honeypot rate > 10% is an automatic Stage-3 disqualification.

We do NOT special-case them by ID (we can't see ground truth). Instead we detect
the *physical impossibilities* a real profile cannot exhibit. A candidate
tripping any high-confidence rule is hard-killed in ranking. Each rule is
intentionally conservative to avoid false positives on genuine candidates.
"""
from __future__ import annotations

from dataclasses import dataclass

from .schema import TODAY, Candidate, _months_between


@dataclass(frozen=True)
class HoneypotVerdict:
    is_honeypot: bool
    reasons: tuple[str, ...]


# Slack allowed between a stated duration and the duration implied by dates,
# to tolerate rounding / overlapping months in legitimate profiles.
_DURATION_SLACK_MONTHS = 6
# How far a candidate's earliest career start may predate their stated YoE
# before we treat it as fabricated.
_CAREER_VS_YOE_SLACK_MONTHS = 36
# Total tenure may exceed YoE somewhat (parallel roles, rounding); beyond this
# it's inconsistent.
_TENURE_VS_YOE_SLACK_MONTHS = 30


def detect_honeypot(c: Candidate) -> HoneypotVerdict:
    reasons: list[str] = []
    yoe_months = c.years_of_experience * 12

    # Rule 1: a job claims more months than its own start->end window allows.
    # e.g. a "current" role started 2024 but duration_months = 166.
    for j in c.career:
        span = j.span_months
        if span is not None and j.duration_months - span > _DURATION_SLACK_MONTHS:
            reasons.append(
                f"job '{j.title}@{j.company}' claims {j.duration_months}mo "
                f"but dates span only {span}mo"
            )

    # Rule 2: expert proficiency in a skill used for 0 months — impossible.
    zero_dur_expert = [
        s.name for s in c.skills
        if s.proficiency == "expert" and s.duration_months == 0
    ]
    if zero_dur_expert:
        reasons.append(
            "expert proficiency with 0 months used: " + ", ".join(zero_dur_expert[:4])
        )

    # Rule 3: earliest career start implies far more career than stated YoE.
    # e.g. claims 6 yrs but earliest job began 14 years ago.
    if c.career_start is not None:
        career_months = _months_between(c.career_start, TODAY)
        if career_months - yoe_months > _CAREER_VS_YOE_SLACK_MONTHS:
            reasons.append(
                f"career began {career_months}mo ago but states "
                f"{c.years_of_experience:.1f}y experience"
            )

    # Rule 4: total tenure across roles greatly exceeds stated YoE.
    total_tenure = sum(j.duration_months for j in c.career)
    if total_tenure - yoe_months > _TENURE_VS_YOE_SLACK_MONTHS:
        reasons.append(
            f"tenure sums to {total_tenure}mo vs stated "
            f"{c.years_of_experience:.1f}y experience"
        )

    # Rule 5: >=3 "expert" skills each used < 6 months (instant-expert pattern).
    instant_expert = [
        s.name for s in c.skills
        if s.proficiency == "expert" and 0 < s.duration_months < 6
    ]
    if len(instant_expert) >= 3:
        reasons.append(
            "multiple expert skills with <6mo use: " + ", ".join(instant_expert[:4])
        )

    return HoneypotVerdict(is_honeypot=bool(reasons), reasons=tuple(reasons))
