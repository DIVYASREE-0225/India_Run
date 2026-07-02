"""Typed, defensive view over a raw candidate record.

The dataset is synthetic but realistic: fields can be missing, null, or
out-of-range. Every accessor here is total (never raises) so the scoring layer
can stay clean. Derived quantities used across multiple scorers (current role,
career start, AI-skill set) are computed once here.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Optional

# Anchor "today" to a fixed date so ranking is deterministic and reproducible
# regardless of when the code runs. Chosen to sit just after the latest activity
# dates in the dataset (last_active_date values run into mid-2026).
TODAY = _dt.date(2026, 6, 1)

_COMPANY_SIZE_ORDER = {
    "1-10": 0, "11-50": 1, "51-200": 2, "201-500": 3,
    "501-1000": 4, "1001-5000": 5, "5001-10000": 6, "10001+": 7,
}


def _parse_date(value: Any) -> Optional[_dt.date]:
    if not isinstance(value, str):
        return None
    try:
        return _dt.date.fromisoformat(value[:10])
    except ValueError:
        return None


def _months_between(start: _dt.date, end: _dt.date) -> int:
    """Whole-month span between two dates (>= 0)."""
    return max(0, (end.year - start.year) * 12 + (end.month - start.month))


@dataclass(frozen=True)
class Job:
    """One career-history entry, with parsed dates."""

    company: str
    title: str
    industry: str
    company_size: str
    description: str
    duration_months: int
    is_current: bool
    start_date: Optional[_dt.date]
    end_date: Optional[_dt.date]

    @property
    def company_size_rank(self) -> int:
        return _COMPANY_SIZE_ORDER.get(self.company_size, -1)

    @property
    def span_months(self) -> Optional[int]:
        """Months implied by the start/end dates (end defaults to TODAY)."""
        if self.start_date is None:
            return None
        return _months_between(self.start_date, self.end_date or TODAY)


@dataclass(frozen=True)
class Skill:
    name: str
    proficiency: str
    endorsements: int
    duration_months: int


@dataclass
class Candidate:
    """Typed wrapper over one raw JSONL record."""

    raw: dict = field(repr=False)

    # ----- identity -----
    @property
    def id(self) -> str:
        return self.raw.get("candidate_id", "")

    @property
    def profile(self) -> dict:
        return self.raw.get("profile") or {}

    @property
    def signals(self) -> dict:
        return self.raw.get("redrob_signals") or {}

    # ----- profile scalars -----
    @property
    def name(self) -> str:
        return self.profile.get("anonymized_name", "")

    @property
    def headline(self) -> str:
        return self.profile.get("headline", "") or ""

    @property
    def summary(self) -> str:
        return self.profile.get("summary", "") or ""

    @property
    def current_title(self) -> str:
        return self.profile.get("current_title", "") or ""

    @property
    def current_company(self) -> str:
        return self.profile.get("current_company", "") or ""

    @property
    def current_industry(self) -> str:
        return self.profile.get("current_industry", "") or ""

    @property
    def location(self) -> str:
        return self.profile.get("location", "") or ""

    @property
    def country(self) -> str:
        return self.profile.get("country", "") or ""

    @property
    def years_of_experience(self) -> float:
        try:
            return float(self.profile.get("years_of_experience", 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    # ----- collections -----
    @cached_property
    def career(self) -> list[Job]:
        out: list[Job] = []
        for j in self.raw.get("career_history") or []:
            out.append(
                Job(
                    company=j.get("company", "") or "",
                    title=j.get("title", "") or "",
                    industry=j.get("industry", "") or "",
                    company_size=j.get("company_size", "") or "",
                    description=j.get("description", "") or "",
                    duration_months=int(j.get("duration_months", 0) or 0),
                    is_current=bool(j.get("is_current", False)),
                    start_date=_parse_date(j.get("start_date")),
                    end_date=_parse_date(j.get("end_date")),
                )
            )
        return out

    @cached_property
    def skills(self) -> list[Skill]:
        out: list[Skill] = []
        for s in self.raw.get("skills") or []:
            out.append(
                Skill(
                    name=(s.get("name", "") or "").strip(),
                    proficiency=(s.get("proficiency", "") or "").lower(),
                    endorsements=int(s.get("endorsements", 0) or 0),
                    duration_months=int(s.get("duration_months", 0) or 0),
                )
            )
        return out

    @property
    def education(self) -> list[dict]:
        return self.raw.get("education") or []

    @property
    def certifications(self) -> list[dict]:
        return self.raw.get("certifications") or []

    # ----- derived helpers -----
    @cached_property
    def current_jobs(self) -> list[Job]:
        cur = [j for j in self.career if j.is_current]
        return cur or self.career[:1]

    @cached_property
    def career_start(self) -> Optional[_dt.date]:
        starts = [j.start_date for j in self.career if j.start_date]
        return min(starts) if starts else None

    @cached_property
    def skill_names_lower(self) -> set[str]:
        return {s.name.lower() for s in self.skills if s.name}

    def signal(self, key: str, default: Any = None) -> Any:
        return self.signals.get(key, default)
