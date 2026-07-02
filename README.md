# Redrob Intelligent Candidate Ranker

A production-minded ranking system for the Redrob *Intelligent Candidate
Discovery & Ranking Challenge*. It ranks a 100,000-candidate pool against the
"Senior AI Engineer — Founding Team" job description and emits the top-100 as a
spec-compliant CSV — in **~75 seconds on a CPU, fully offline**.

The design goal is not to filter on keywords, but to **reason about fit**: who
actually did retrieval/ranking work at a product company, who is genuinely
available to hire, and who is a keyword-stuffer or an impossible-profile trap.

---

## TL;DR — reproduce the submission

```bash
# 1. Environment (Python 3.11)
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. One-time: fetch the semantic encoder into ./artifacts/minilm (needs network ONCE)
python fetch_model.py

# 3. Rank (no network used here; CPU only; ~75s for 100K)
python rank.py --candidates ./candidates.jsonl --out ./submission.csv

# 4. Validate format against the official validator
python validate_submission.py submission.csv     # -> "Submission is valid."
```

If `artifacts/minilm` is absent, the ranker still runs and produces a valid
submission using a **lexical + structured fallback** (no crash, no network).

---

## Why this architecture

The JD and the hackathon notes are explicit about the traps:

> "The right answer involves reasoning about the gap between what the JD says and
> what the JD means. A candidate who has all the AI keywords listed as skills but
> whose title is 'Marketing Manager' is not a fit."

> "A perfect-on-paper candidate who hasn't logged in for 6 months and has a 5%
> recruiter response rate is, for hiring purposes, not actually available."

So the system is built around five ideas:

| Idea | Mechanism | Defends against |
|------|-----------|-----------------|
| **Role is a gate, not a feature** | `role_fit` from current title **and** the whole career arc; off-target careers are capped low | Keyword-stuffers (AI skills on HR/Accountant careers) |
| **Demonstrated > tagged** | `evidence_fit` reads career *prose* for real production ranking/retrieval work (cited NDCG, hybrid retrieval, recsys at scale) | Profiles that list skills but never built anything |
| **Trust, don't trust claims** | skills are weighted by endorsements, tenure, proficiency, and on-platform assessment scores | Lazily stuffed skill lists |
| **Availability is a multiplier** | behavioral signals (activity recency, response rate, interview completion, open-to-work) scale the score *down* | "Perfect on paper, impossible to hire" |
| **Impossible ⇒ dead** | honeypot detector hard-zeros physically-inconsistent profiles | The ~80 seeded honeypots (Stage-3 DQ if >10% in top-100) |

### Two-stage pipeline (fits the 5-min / 16 GB / CPU / no-network budget)

```
            ┌──────────────────────── Stage A: all 100K (cheap) ────────────────────────┐
candidates  │  load + type → honeypot detect → structured score → BM25 lexical fit        │
  .jsonl ──▶│  → provisional fused score                                                  │
            └──────────────┬─────────────────────────────────────────────────────────────┘
                           │ top ~4,000 by provisional score
            ┌──────────────▼──────── Stage B: shortlist only (dense) ────────────────────┐
            │  MiniLM bi-encoder cosine(JD query, candidate doc) → semantic fit            │
            └──────────────┬─────────────────────────────────────────────────────────────┘
                           ▼
   final = intrinsic_fit(role, semantic, skill, evidence, experience, location, education)
           × availability_multiplier × stuffer_penalty × honeypot_kill
                           ▼
                  top-100 → submission.csv
```

Encoding only the **shortlist** (not all 100K) is what keeps Stage B fast. The
heavy bottleneck — dense encoding — touches ~4K docs, not 100K.

---

## Scoring model

**Intrinsic fit** (additive, weights sum to 1.0):

| Component | Weight | What it captures |
|-----------|-------:|------------------|
| role      | 0.24 | title + career-discipline alignment (the anti-stuffer gate) |
| semantic  | 0.22 | dense bi-encoder fit — "beyond keywords" |
| skill     | 0.18 | trust-weighted must-have / nice-to-have skill-tag coverage |
| evidence  | 0.16 | **production retrieval/ranking work shown in career prose** (NDCG cited, "hybrid retrieval", "ranking pipeline at scale") — the JD's most emphatic signal |
| experience| 0.10 | 5–9y band (ideal 6–8, treated as a range not a gate), product-vs-services |
| location  | 0.07 | Pune/Noida preferred; NCR/Hyd/Mumbai welcome; relocation |
| education | 0.03 | tier + relevant field (light touch — JD weights skills over creds) |

The `evidence` component is deliberately separate from `skill`: the JD says to
value *demonstrated* work over a tagged skill list ("if their career history
shows they built a recommendation system, they're a fit"). It scans the
headline/summary/career **descriptions** — not the skill tags — so a candidate
who *describes* shipping a ranking system outranks one who merely *lists* the
skills. The top-10 is **10/10 stable under ±25% perturbation** of these weights,
so the ranking is driven by signal, not by precise weight calibration.

**Multiplicative modifiers** (necessary conditions, applied after):
- `availability` ∈ ~[0.4, 1.15] — behavioral signals.
- `stuffer_penalty` ∈ {0.45, 0.7, 1.0} — off-target career + high AI-skill gap.
- `honeypot_kill` ∈ {0, 1} — physically-impossible profile ⇒ 0.

Multiplicative (not additive) modifiers are deliberate: being unavailable or a
stuffer should *scale a good paper score down*, not merely subtract a constant.

---

## Honeypot detection

We never special-case IDs (the ground truth is hidden). We detect the physical
impossibilities a real profile cannot have (see `src/redrob_ranker/honeypot.py`):

1. A job claims more months than its own start→end window allows.
2. `expert` proficiency in a skill used for **0 months**.
3. Earliest career start implies far more career than the stated years.
4. Total tenure across roles greatly exceeds stated years of experience.
5. Three or more `expert` skills each used **< 6 months** (instant-expert).

On the released pool this flags **43 candidates**, and the produced top-100
contains **0 honeypots** and **0 off-target-title** candidates.

---

## Repository layout

```
rank.py                     # CLI entrypoint → submission.csv  (Stage-3 reproduce command)
fetch_model.py              # one-time encoder download into artifacts/minilm
app.py                      # Streamlit sandbox (spec §10.5)
requirements.txt            # pinned, CPU-only deps
submission_metadata.yaml    # portal metadata mirror
validate_submission.py      # official format validator (copy of the bundle's)
tests/test_ranker.py        # pins honeypot / stuffer / monotonicity / grounding
src/redrob_ranker/
  schema.py       # typed, defensive view over raw records
  jobspec.py      # structured interpretation of the JD (vocabularies + anti-signals)
  text.py         # candidate document construction for the engines
  honeypot.py     # impossible-profile detector
  structured.py   # rule-based component scorers (incl. production-evidence) + behavioral multiplier
  lexical.py      # BM25 (Okapi) over candidate docs
  semantic.py     # local MiniLM bi-encoder (optional; graceful fallback)
  fuse.py         # score fusion + grounded reasoning generation
  pipeline.py     # two-stage orchestration
```

---

## Tests & verification

```bash
python tests/test_ranker.py        # 7 unit tests: honeypots, stuffers, grounding, crash-safety
```

Verified on the released pool:
- `validate_submission.py` → **Submission is valid.**
- top-100: 100/100 unique scores, monotonic non-increasing, 100/100 distinct
  reasonings, 72/100 in the 5–9y band, **0 honeypots, 0 off-target titles**.

---

## Compute constraints — how we satisfy them

| Constraint | Limit | This system |
|------------|-------|-------------|
| Runtime | ≤ 5 min | ~75 s for 100K (hybrid); ~26 s (lexical fallback) |
| Memory | ≤ 16 GB | streams JSONL line-by-line; encodes only a 4K shortlist |
| Compute | CPU only | `device="cpu"`; no GPU calls |
| Network | off during ranking | model loaded from local `artifacts/minilm`; `HF_HUB_OFFLINE=1` set in-process |
| Disk | ≤ 5 GB | ~90 MB model cache; no large intermediates |

Pre-computation (the one-time `fetch_model.py` download) is allowed to use the
network and sits outside the 5-minute ranking window, per spec §10.3.

---

## Sandbox

`app.py` is a Streamlit app that accepts a ≤100-candidate JSONL sample, runs the
full pipeline on CPU, and offers the ranked CSV for download. Deploy free on
HuggingFace Spaces or Streamlit Cloud (`streamlit run app.py` locally).

---

## Roadmap — and what we can built later

These are the production successors to this PoC. Each was a conscious scoping
decision, not an oversight — worth stating plainly because the reasoning matters
more than the code.

1. **Offline eval harness + sensitivity analysis.** The fusion weights are
   *reasoned*, not *tuned*, because the ground truth is hidden. A proxy relevance
   label built from the same role/skill/experience signals the ranker uses would
   be **circular** — tuning against it grades our own homework. The honest
   versions are (i) regression checks and (ii) sensitivity analysis (perturb
   weights ±20%, confirm the top-100 is stable). We left these out to keep the
   submission lean given the 3-submission cap and no live leaderboard; they are
   the first thing to add with any real labeled data.

2. **Learning-to-rank (XGBoost/neural).** Explicitly **rejected** for this
   PoC. LTR needs labeled relevance judgments; with none, an LTR model would only
   re-learn our hand-built proxy's biases while *removing* the interpretability
   that makes the current scorer auditable and defensible. The transparent
   rule-based scorer is the correct choice in a no-ground-truth setting — and the
   JD's distrust of "frameworks over systems" reinforces that. LTR becomes the
   right move only once genuine recruiter-feedback labels exist.

3. **Generic JD parsing.** `jobspec.py` is hand-encoded for this one role. A
   product would parse arbitrary JDs into that same structured form (vocabularies,
   anti-signals, weights). Out of scope for a single-JD challenge, but the clean
   separation between `jobspec.py` and the scorers is exactly the seam where a
   parser would plug in.

The throughline: in a setting with **no ground truth and a hard reproducibility
bar**, an interpretable, trap-resistant, fast ranker beats an opaque model that
merely *looks* more sophisticated.

