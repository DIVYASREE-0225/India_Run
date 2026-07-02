"""Unit tests for the core ranking guarantees.

Run: python -m pytest tests/ -q   (or: python tests/test_ranker.py)

These tests pin the behaviors that matter for the submission spec and the JD's
explicit traps, so refactors can't silently regress them.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from redrob_ranker.honeypot import detect_honeypot
from redrob_ranker.schema import Candidate
from redrob_ranker.structured import score_structured, _title_class
from redrob_ranker.fuse import fuse, build_reasoning


def _base_candidate(**overrides):
    """A minimal valid candidate; override pieces per test."""
    rec = {
        "candidate_id": "CAND_0000001",
        "profile": {
            "anonymized_name": "Test User", "headline": "ML Engineer",
            "summary": "Built retrieval and ranking systems.",
            "location": "Pune", "country": "India",
            "years_of_experience": 7.0, "current_title": "Machine Learning Engineer",
            "current_company": "Acme", "current_company_size": "201-500",
            "current_industry": "Software",
        },
        "career_history": [{
            "company": "Acme", "title": "Machine Learning Engineer",
            "start_date": "2021-01-01", "end_date": None, "duration_months": 36,
            "is_current": True, "industry": "Software", "company_size": "201-500",
            "description": "Built embeddings-based retrieval with FAISS and NDCG eval.",
        }],
        "education": [{"institution": "IIT", "degree": "B.Tech",
                       "field_of_study": "Computer Science", "start_year": 2014,
                       "end_year": 2018, "tier": "tier_1"}],
        "skills": [
            {"name": "Embeddings", "proficiency": "expert", "endorsements": 30,
             "duration_months": 40},
            {"name": "FAISS", "proficiency": "advanced", "endorsements": 20,
             "duration_months": 30},
            {"name": "Python", "proficiency": "expert", "endorsements": 40,
             "duration_months": 60},
        ],
        "redrob_signals": {
            "last_active_date": "2026-05-25", "open_to_work_flag": True,
            "recruiter_response_rate": 0.8, "interview_completion_rate": 0.9,
            "profile_completeness_score": 95, "saved_by_recruiters_30d": 5,
            "verified_email": True, "verified_phone": True, "linkedin_connected": True,
            "skill_assessment_scores": {"Embeddings": 88},
        },
    }
    prof = overrides.pop("profile", {})
    rec["profile"].update(prof)
    rec.update(overrides)
    return Candidate(rec)


def test_honeypot_duration_exceeds_span():
    c = _base_candidate(career_history=[{
        "company": "X", "title": "Engineer", "start_date": "2024-01-01",
        "end_date": None, "duration_months": 166, "is_current": True,
        "industry": "Software", "company_size": "201-500", "description": "x",
    }])
    assert detect_honeypot(c).is_honeypot


def test_honeypot_expert_zero_duration():
    c = _base_candidate(skills=[
        {"name": "RAG", "proficiency": "expert", "endorsements": 5,
         "duration_months": 0}])
    assert detect_honeypot(c).is_honeypot


def test_genuine_candidate_not_honeypot():
    assert not detect_honeypot(_base_candidate()).is_honeypot


def test_keyword_stuffer_demoted():
    """An off-target career stuffed with AI skills must score below a genuine ML eng."""
    genuine = _base_candidate()
    stuffer = _base_candidate(
        profile={"current_title": "HR Manager"},
        career_history=[{
            "company": "X", "title": "HR Manager", "start_date": "2018-01-01",
            "end_date": None, "duration_months": 90, "is_current": True,
            "industry": "IT Services", "company_size": "10001+",
            "description": "Recruiting and payroll.",
        }],
    )
    sg = score_structured(genuine)
    ss = score_structured(stuffer)
    fg = fuse(genuine.id, sg, 0.8, False)
    fs = fuse(stuffer.id, ss, 0.8, False)
    assert fg.final_score > fs.final_score
    assert _title_class("HR Manager") == "off"


def test_honeypot_zeroed_in_fusion():
    c = _base_candidate()
    s = score_structured(c)
    f = fuse(c.id, s, 0.9, True, ("impossible",))
    assert f.final_score == 0.0


def test_reasoning_is_grounded_and_nonempty():
    c = _base_candidate()
    s = score_structured(c)
    f = fuse(c.id, s, 0.85, False)
    r = build_reasoning(c, f, rank=1)
    assert "Machine Learning Engineer" in r
    assert "7.0y" in r
    assert len(r) > 20


def test_missing_fields_do_not_crash():
    c = Candidate({"candidate_id": "CAND_0000002"})  # almost everything missing
    s = score_structured(c)  # must not raise
    f = fuse(c.id, s, 0.0, False)
    assert 0.0 <= f.final_score <= 2.0
    assert detect_honeypot(c) is not None


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
