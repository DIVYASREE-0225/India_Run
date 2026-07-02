"""Structured, rule-based scoring of a candidate against the JobSpec.

Each function returns a sub-score in [0, 1] (or a multiplier around 1.0 for the
behavioral modifier) plus, where useful, a short human-readable note. The notes
feed the reasoning column required at Stage 4. Everything is interpretable on
purpose: at the Stage-5 interview we want to point at exactly why a candidate
scored as they did.

The decisive design choices, straight from the JD's "for hackathon participants"
note:
  * role_fit (title + career discipline) is the gate against keyword stuffers.
    AI skills on an "HR Manager / Accountant / Marketing Manager" career are
    penalized hard.
  * skills are trust-weighted by endorsements, proficiency, duration, and
    on-platform assessment scores, so a lazily stuffed skills list earns little.
  * behavioral signals form a *multiplier*: a perfect-on-paper candidate who is
    inactive and unresponsive is, for hiring, unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import jobspec as J
from .schema import TODAY, Candidate


@dataclass
class StructuredScores:
    role_fit: float            # title + career discipline alignment [0,1]
    skill_fit: float           # trust-weighted JD-skill coverage [0,1]
    evidence_fit: float        # production retrieval/ranking evidence in prose [0,1]
    experience_fit: float      # YoE vs band, product-vs-services [0,1]
    location_fit: float        # geography / relocation [0,1]
    education_fit: float       # tier + relevant field [0,1]
    availability: float        # behavioral multiplier (~0.4..1.15)
    stuffer_penalty: float     # multiplier <=1.0 for keyword stuffing
    notes: dict                # component -> short string for reasoning


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
_PROFICIENCY_W = {"beginner": 0.4, "intermediate": 0.7, "advanced": 0.9, "expert": 1.0}


def _title_class(title: str) -> str:
    """Classify a title as 'core', 'adjacent', or 'off' (or 'other')."""
    t = title.lower()
    if any(k in t for k in J.CORE_TITLES):
        return "core"
    if any(k in t for k in J.OFF_TARGET_TITLES):
        return "off"
    if any(k in t for k in J.ADJACENT_TITLES):
        return "adjacent"
    return "other"


def _skill_trust(c: Candidate, skill_name: str) -> float:
    """How much to trust a claimed skill, in [0,1].

    Combines self-reported proficiency & tenure with platform-verified
    endorsements and assessment scores. A skill claimed at 'expert' with no
    endorsements, no tenure, and no assessment is barely trusted.
    """
    name_l = skill_name.lower()
    sk = next((s for s in c.skills if s.name.lower() == name_l), None)
    if sk is None:
        return 0.0
    prof = _PROFICIENCY_W.get(sk.proficiency, 0.5)
    dur = min(sk.duration_months / 36.0, 1.0)          # 3y use -> full credit
    end = min(sk.endorsements / 25.0, 1.0)             # 25 endorsements -> full
    # platform assessment score for this skill, if present (0-100)
    assess_raw = (c.signal("skill_assessment_scores") or {}).get(sk.name)
    assess = (assess_raw / 100.0) if isinstance(assess_raw, (int, float)) else None

    # Base on proficiency, but require corroboration from tenure/endorsements.
    trust = 0.5 * prof + 0.25 * dur + 0.25 * end
    if assess is not None:
        # blend assessment in when available — verified signal beats self-report
        trust = 0.6 * trust + 0.4 * assess
    return max(0.0, min(1.0, trust))


# ---------------------------------------------------------------------------
# component scorers
# ---------------------------------------------------------------------------
def score_role_fit(c: Candidate) -> tuple[float, str]:
    """Title + career-discipline alignment — the anti-stuffer gate.

    We look at the current title and the whole career arc. A career spent
    entirely in off-target roles caps the score low regardless of skills.
    """
    cur_class = _title_class(c.current_title)
    classes = [_title_class(j.title) for j in c.career] or [cur_class]
    n = len(classes)
    core_frac = classes.count("core") / n
    adj_frac = classes.count("adjacent") / n
    off_frac = classes.count("off") / n

    # Base from current title.
    base = {"core": 0.95, "adjacent": 0.62, "other": 0.4, "off": 0.12}[cur_class]
    # Career arc adjusts: sustained core/adjacent work lifts; off-target drags.
    arc = 0.6 * core_frac + 0.35 * adj_frac - 0.25 * off_frac
    score = max(0.0, min(1.0, 0.65 * base + 0.35 * max(0.0, arc + 0.2)))

    if cur_class == "core":
        note = f"core role '{c.current_title}'"
    elif cur_class == "adjacent":
        note = f"adjacent role '{c.current_title}' (transferable to AI/IR)"
    elif cur_class == "off":
        note = f"off-target role '{c.current_title}'"
    else:
        note = f"role '{c.current_title}'"
    return score, note


def score_skill_fit(c: Candidate) -> tuple[float, str]:
    """Trust-weighted coverage of must-have and nice-to-have JD skills.

    Coverage is computed across skills, headline, summary, and career text so a
    candidate who *describes* doing retrieval gets credit even without the exact
    skill tag. Must-haves dominate; nice-to-haves add a smaller bonus.
    """
    haystack = " ".join(
        [c.headline.lower(), c.summary.lower()]
        + [j.description.lower() for j in c.career]
        + [j.title.lower() for j in c.career]
    )

    must_total = sum(J.MUST_HAVE_SKILLS.values())
    must_got = 0.0
    matched: list[str] = []
    for kw, w in J.MUST_HAVE_SKILLS.items():
        if kw in c.skill_names_lower:
            must_got += w * (0.4 + 0.6 * _skill_trust(c, kw))  # trust-scaled
            matched.append(kw)
        elif kw in haystack:
            must_got += w * 0.5                                 # text-only credit
            matched.append(kw)
    must_cov = min(1.0, must_got / (0.45 * must_total))         # don't need all

    nice_total = sum(J.NICE_TO_HAVE_SKILLS.values())
    nice_got = 0.0
    for kw, w in J.NICE_TO_HAVE_SKILLS.items():
        if kw in c.skill_names_lower:
            nice_got += w * (0.5 + 0.5 * _skill_trust(c, kw))
        elif kw in haystack:
            nice_got += w * 0.4
    nice_cov = min(1.0, nice_got / (0.5 * nice_total))

    # Strong Python is an explicit must-have.
    py = max((_skill_trust(c, p) for p in J.PYTHON_SKILLS), default=0.0)
    if not py and "python" in haystack:
        py = 0.4

    score = max(0.0, min(1.0, 0.62 * must_cov + 0.23 * nice_cov + 0.15 * py))
    top = matched[:4]
    note = (
        f"{len([m for m in matched])} core AI/IR skills"
        + (f" ({', '.join(top)})" if top else "")
    )
    return score, note


def score_evidence_fit(c: Candidate) -> tuple[float, str]:
    """Production retrieval/ranking evidence found in career prose.

    The JD's decisive instruction for this challenge is to value demonstrated
    work over a tagged skill list: "if their career history shows they built a
    recommendation system at a product company, they're a fit." A candidate who
    *describes* shipping a ranking pipeline, quotes NDCG, or names hybrid
    retrieval at scale is a stronger signal than one who merely lists the skills.

    We scan the headline, summary, and (most importantly) career descriptions —
    NOT the skill tags, which `score_skill_fit` already covers — for weighted
    evidence phrases. The score saturates so a handful of strong, distinct
    phrases is enough; it does not reward keyword repetition.
    """
    prose = " ".join(
        [c.headline.lower(), c.summary.lower()]
        + [j.description.lower() for j in c.career]
    )
    got = 0.0
    hits: list[str] = []
    for phrase, w in J.PRODUCTION_EVIDENCE_PHRASES.items():
        if phrase in prose:
            got += w
            hits.append(phrase)
    # Saturate: ~4-5 strong phrases (sum ~4.0) reaches full credit.
    score = max(0.0, min(1.0, got / 4.0))
    # De-dupe near-identical hits for a clean note.
    note = ""
    if hits:
        shown = [h for h in ("ndcg", "hybrid retrieval", "ranking pipeline",
                             "recommendation system", "semantic search")
                 if h in hits][:2] or hits[:2]
        note = "career shows production " + "/".join(shown)
    return score, note


def score_experience_fit(c: Candidate) -> tuple[float, str]:
    """YoE vs the 5-9 band (ideal 6-8), plus product-vs-services signal."""
    y = c.years_of_experience
    if J.EXP_IDEAL_LOW <= y <= J.EXP_IDEAL_HIGH:
        band = 1.0
    elif J.EXP_OK_LOW <= y <= J.EXP_OK_HIGH:
        band = 0.85
    elif y < J.EXP_OK_LOW:
        band = max(0.25, 0.85 - (J.EXP_OK_LOW - y) * 0.18)     # juniors taper
    else:  # over the band — still useful; JD says band is "a range not a
           # requirement" and will consider strong candidates outside it.
        band = max(0.62, 0.9 - (y - J.EXP_OK_HIGH) * 0.035)

    # Product-vs-services: services-only careers are an anti-signal.
    companies = [j.company.lower() for j in c.career]
    industries = [j.industry.lower() for j in c.career]
    n = max(1, len(companies))
    services_frac = sum(
        any(s in comp for s in J.SERVICES_COMPANIES) for comp in companies
    ) / n
    product_frac = sum(
        any(p in ind for p in J.PRODUCT_INDUSTRIES) for ind in industries
    ) / n

    multiplier = 1.0 + 0.15 * product_frac - 0.3 * services_frac
    score = max(0.0, min(1.0, band * multiplier))

    note = f"{y:.1f}y experience"
    if services_frac >= 0.6:
        note += "; mostly services-firm background"
    elif product_frac >= 0.4:
        note += "; product-company background"
    return score, note


def score_location_fit(c: Candidate) -> tuple[float, str]:
    """Pune/Noida preferred; NCR/Hyd/Mumbai welcome; India ok; ex-India weak."""
    loc = c.location.lower()
    country = c.country.lower()
    relocate = bool(c.signal("willing_to_relocate", False))

    if any(p in loc for p in J.PREFERRED_LOCATIONS):
        return 1.0, f"{c.location} (preferred hub)"
    if any(w in loc for w in J.WELCOME_LOCATIONS):
        return 0.9, f"{c.location} (welcome city)"
    if country == "india":
        base = 0.7 if relocate else 0.55
        return base, f"{c.location}, India" + (", open to relocate" if relocate else "")
    # Outside India: case-by-case, no visa sponsorship -> weak unless relocating.
    base = 0.45 if relocate else 0.2
    return base, f"{c.location}, {c.country} (outside India)"


def score_education_fit(c: Candidate) -> tuple[float, str]:
    """Light touch — tier + CS/ML-relevant field. JD weights skills over creds."""
    if not c.education:
        return 0.55, "no education listed"
    tier_w = {"tier_1": 1.0, "tier_2": 0.85, "tier_3": 0.7, "tier_4": 0.55,
              "unknown": 0.6}
    best_tier = max((tier_w.get(e.get("tier", "unknown"), 0.6) for e in c.education),
                    default=0.6)
    relevant_fields = ("computer", "data", "machine learning", "artificial",
                       "statistics", "mathematics", "information", "electronics")
    field_match = any(
        any(rf in (e.get("field_of_study", "") or "").lower() for rf in relevant_fields)
        for e in c.education
    )
    score = 0.7 * best_tier + (0.3 if field_match else 0.05)
    score = max(0.0, min(1.0, score))
    note = ("relevant degree" if field_match else "degree") + (
        f" (best tier {max(c.education, key=lambda e: tier_w.get(e.get('tier','unknown'),0.6)).get('tier','?')})"
    )
    return score, note


def score_availability(c: Candidate) -> tuple[float, str]:
    """Behavioral availability MULTIPLIER (~0.4..1.15).

    The JD is explicit: a perfect-on-paper candidate who hasn't logged in for
    months with a 5% response rate is, for hiring, not available. We fold the
    23 Redrob signals into one modifier that can meaningfully demote such
    candidates without ever fully zeroing a genuine fit.
    """
    s = c.signals

    # Recency of activity (days since last_active).
    last = s.get("last_active_date")
    recency = 0.4  # missing date -> treat as moderately stale, not fresh
    if last is not None:
        try:
            import datetime as _dt
            d = _dt.date.fromisoformat(str(last)[:10])
            days = (TODAY - d).days
            recency = 1.0 if days <= 14 else 0.85 if days <= 45 else \
                0.6 if days <= 120 else 0.3 if days <= 240 else 0.12
        except (TypeError, ValueError):
            pass

    open_flag = 1.0 if s.get("open_to_work_flag") else 0.6
    resp_rate = _clamp01(s.get("recruiter_response_rate", 0.3))
    interview = _clamp01(s.get("interview_completion_rate", 0.5))
    completeness = _clamp01((s.get("profile_completeness_score", 50) or 50) / 100.0)

    # Recruiter demand signals (saved / search appearances) — mild positive.
    saved = min((s.get("saved_by_recruiters_30d", 0) or 0) / 8.0, 1.0)

    # Verification — light trust bump.
    verified = sum(bool(s.get(k)) for k in
                   ("verified_email", "verified_phone", "linkedin_connected")) / 3.0

    # Weighted blend, centered so a typical candidate lands near ~0.85-1.0.
    raw = (0.30 * recency + 0.22 * resp_rate + 0.15 * open_flag +
           0.12 * interview + 0.10 * completeness + 0.06 * saved +
           0.05 * verified)
    # Map [0,1]-ish raw into a multiplier band [0.4, 1.15].
    mult = 0.4 + 0.75 * raw
    mult = max(0.4, min(1.15, mult))

    note = f"response rate {resp_rate:.2f}"
    if recency <= 0.3:
        note += ", inactive recently"
    elif recency >= 1.0:
        note += ", active"
    if not s.get("open_to_work_flag"):
        note += ", not open-to-work"
    return mult, note


def score_stuffer_penalty(c: Candidate, role_fit: float, skill_fit: float) -> tuple[float, str]:
    """Keyword-stuffer trap: high AI-skill coverage on an off-target career.

    The classic trap in this dataset is a Marketing Manager / Accountant whose
    skills list is packed with AI keywords. role_fit already caps these, but we
    apply an extra multiplicative penalty when the *gap* between skill_fit and
    role_fit is large and the career is genuinely off-target — the signature of
    stuffing rather than a real career pivot.
    """
    cur_class = _title_class(c.current_title)
    career_classes = [_title_class(j.title) for j in c.career]
    off_frac = (career_classes.count("off") / len(career_classes)) if career_classes else 0.0

    # Only suspicious when the career is substantially off-target.
    if cur_class == "off" and off_frac >= 0.6 and skill_fit - role_fit > 0.35:
        return 0.45, "likely keyword-stuffed (AI skills on off-target career)"
    if cur_class == "off" and skill_fit > 0.6:
        return 0.7, "AI skills present but role/career off-target"
    return 1.0, ""


def _clamp01(x) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return 0.0


def score_structured(c: Candidate) -> StructuredScores:
    role, role_n = score_role_fit(c)
    skill, skill_n = score_skill_fit(c)
    evidence, evidence_n = score_evidence_fit(c)
    exp, exp_n = score_experience_fit(c)
    loc, loc_n = score_location_fit(c)
    edu, edu_n = score_education_fit(c)
    avail, avail_n = score_availability(c)
    stuff, stuff_n = score_stuffer_penalty(c, role, skill)
    return StructuredScores(
        role_fit=role, skill_fit=skill, evidence_fit=evidence,
        experience_fit=exp, location_fit=loc,
        education_fit=edu, availability=avail, stuffer_penalty=stuff,
        notes={"role": role_n, "skill": skill_n, "evidence": evidence_n,
               "experience": exp_n, "location": loc_n, "education": edu_n,
               "availability": avail_n, "stuffer": stuff_n},
    )
