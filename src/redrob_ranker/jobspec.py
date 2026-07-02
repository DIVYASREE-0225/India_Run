"""Structured interpretation of the Senior AI Engineer JD.

This module is the "deep job understanding" layer. Rather than treat the JD as a
bag of keywords, we encode what it *means* into typed, weighted vocabularies and
explicit anti-signals. Every entry here is traceable to a specific line in
job_description.docx — see the comments. Keeping this declarative makes the
ranker auditable and easy to defend in the Stage-5 interview.

Key reading-between-the-lines points the JD makes explicit:
  * The ideal is ~6-8 yrs, 4-5 in applied ML at PRODUCT companies (not services).
  * Must-haves: embeddings retrieval, vector/hybrid search, strong Python,
    ranking evaluation (NDCG/MRR/MAP).
  * Hard anti-signals: pure-research-only, consulting-firm-only careers,
    CV/speech/robotics without NLP/IR, title-chasers (job-hop every ~1.5y),
    "LangChain-calls-OpenAI" as the only AI experience.
  * Behavioral availability matters: a perfect-on-paper candidate who is inactive
    / unresponsive is, for hiring purposes, not available -> down-weight.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Role identity: titles that *are* the job vs adjacent vs off-target.
# Matching is substring-based on a normalized (lowercased) title.
# ---------------------------------------------------------------------------

# Core target roles — this is literally the job.
CORE_TITLES = {
    "ai engineer", "machine learning engineer", "ml engineer",
    "applied scientist", "applied ml", "nlp engineer",
    "research engineer",  # acceptable IF production exists (handled in scoring)
    "ai research engineer", "staff machine learning", "senior machine learning",
}

# Strongly adjacent — the JD explicitly says a data/backend person who shipped a
# recommendation/search/ranking system at a product company is a fit even without
# the buzzwords.
ADJACENT_TITLES = {
    "data scientist", "data engineer", "senior data engineer",
    "analytics engineer", "backend engineer", "software engineer",
    "senior software engineer", "full stack developer", "platform engineer",
    "search engineer", "data science",
}

# Off-target roles — non-technical or wrong-discipline. A profile whose entire
# career sits here is not a fit no matter how many AI skills are listed (the
# keyword-stuffer trap the JD calls out).
OFF_TARGET_TITLES = {
    "hr manager", "accountant", "sales executive", "marketing manager",
    "content writer", "graphic designer", "customer support",
    "operations manager", "project manager", "business analyst",
    "mechanical engineer", "civil engineer", "electrical engineer",
}

# ---------------------------------------------------------------------------
# Skill vocabularies, weighted by how decisive the JD makes them.
# Stored lowercase; matched against skills, headline, summary, and career text.
# ---------------------------------------------------------------------------

# "Things you absolutely need" — retrieval/embeddings/vector-search/ranking-eval.
MUST_HAVE_SKILLS = {
    # embeddings & retrieval
    "embeddings": 1.0, "sentence-transformers": 1.0, "sentence transformers": 1.0,
    "bge": 0.8, "e5": 0.7, "retrieval": 1.0, "semantic search": 1.0,
    "dense retrieval": 1.0, "hybrid search": 1.0, "rag": 0.9,
    # vector DBs / search infra
    "pinecone": 1.0, "weaviate": 1.0, "qdrant": 1.0, "milvus": 1.0,
    "faiss": 1.0, "opensearch": 0.9, "elasticsearch": 0.9, "vector database": 1.0,
    "vector search": 1.0, "bm25": 0.8,
    # ranking & evaluation
    "learning to rank": 1.0, "ranking": 0.9, "ndcg": 1.0, "mrr": 1.0,
    "map@k": 0.9, "recommendation": 0.9, "recommender": 0.9,
    "reranking": 1.0, "re-ranking": 1.0,
}

# "Nice to have" — won't reject, but positive signal.
NICE_TO_HAVE_SKILLS = {
    "lora": 0.6, "qlora": 0.6, "peft": 0.6, "fine-tuning llms": 0.6,
    "fine-tuning": 0.5, "xgboost": 0.5, "llm": 0.5, "llms": 0.5,
    "transformers": 0.5, "pytorch": 0.5, "tensorflow": 0.4,
    "nlp": 0.6, "information retrieval": 0.8, "mlops": 0.4, "mlflow": 0.4,
    "distributed systems": 0.4, "large-scale inference": 0.5,
}

# Strong Python is an explicit must-have; treated separately so it can't be
# satisfied purely by keyword stuffing — see structured scoring.
PYTHON_SKILLS = {"python", "pyspark"}

# Disciplines the JD explicitly de-prioritizes (CV/speech/robotics WITHOUT NLP/IR).
WRONG_DISCIPLINE_SKILLS = {
    "image classification", "object detection", "opencv", "computer vision",
    "speech recognition", "tts", "asr", "robotics", "gans", "image segmentation",
}

# "LangChain-to-call-OpenAI as the only AI experience" anti-signal vocabulary.
SHALLOW_LLM_SKILLS = {"langchain", "llamaindex", "openai api", "prompt engineering"}

# ---------------------------------------------------------------------------
# Location: Pune/Noida preferred; Hyderabad/Mumbai/Delhi NCR welcome; India ok;
# outside India case-by-case and no visa sponsorship.
# ---------------------------------------------------------------------------
PREFERRED_LOCATIONS = {"pune", "noida"}
WELCOME_LOCATIONS = {
    "hyderabad", "mumbai", "delhi", "new delhi", "gurgaon", "gurugram",
    "noida", "pune", "ncr", "delhi ncr",
}

# ---------------------------------------------------------------------------
# Companies: services/consulting-only careers are an anti-signal.
# ---------------------------------------------------------------------------
SERVICES_COMPANIES = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "mindtree", "mphasis", "hcl", "tech mahindra",
    "ltimindtree", "l&t infotech",
}

# Industries that read as "product company" (positive per the JD's product>services).
PRODUCT_INDUSTRIES = {
    "software", "saas", "fintech", "e-commerce", "ecommerce", "food delivery",
    "ai/ml", "adtech", "edtech", "insurance tech", "transportation",
}

# ---------------------------------------------------------------------------
# Experience band. JD: 5-9 nominal, ideal 6-8; a range not a hard gate.
# ---------------------------------------------------------------------------
EXP_IDEAL_LOW, EXP_IDEAL_HIGH = 6.0, 8.0
EXP_OK_LOW, EXP_OK_HIGH = 5.0, 9.0

# Notice period: sub-30 ideal; up to 30 buyable; 30+ bar rises.
NOTICE_IDEAL_DAYS = 30

# A canonical natural-language rendering of the JD for the semantic engine. This
# is the "query" side of the bi-encoder. Phrased the way an ideal candidate's
# profile would read, to maximize cosine alignment with genuine fits.
JD_QUERY_TEXT = (
    "Senior AI Engineer for a Series A AI-native talent platform. "
    "Owns the intelligence layer: ranking, retrieval, and matching systems. "
    "Production experience with embeddings-based retrieval (sentence-transformers, "
    "BGE, E5), vector databases and hybrid search (Pinecone, Weaviate, Qdrant, "
    "Milvus, FAISS, Elasticsearch, OpenSearch). Has shipped an end-to-end search, "
    "ranking, or recommendation system to real users at meaningful scale. "
    "Strong Python and code quality. Designs evaluation frameworks for ranking "
    "systems using NDCG, MRR, MAP, and A/B testing. Applied ML at product "
    "companies rather than pure research or pure services. Comfortable shipping "
    "fast and iterating. LLM fine-tuning with LoRA or QLoRA is a plus. "
    "Based in or willing to relocate to Pune or Noida, India."
)

# ---------------------------------------------------------------------------
# Production-evidence phrases. The JD's single most important instruction for
# this challenge: "if their career history shows they built a recommendation
# system at a product company, they're a fit" — value demonstrated work over a
# tagged skill list. These phrases, found in the headline/summary/career
# DESCRIPTIONS (prose, not skill tags), are strong evidence of having actually
# shipped retrieval/ranking/recsys to real users. Weighted by how decisive each
# is. This signal is what separates a genuine builder from a keyword profile.
# ---------------------------------------------------------------------------
PRODUCTION_EVIDENCE_PHRASES = {
    # ranking-eval metrics actually cited (very strong — you only quote NDCG if
    # you built and measured a ranker)
    "ndcg": 1.0, "mrr": 0.9, "map@k": 0.7, "recall@k": 0.7,
    # retrieval/ranking system construction
    "ranking pipeline": 1.0, "hybrid retrieval": 1.0, "dense retrieval": 1.0,
    "semantic search": 0.9, "vector search": 0.9, "embedding-based": 0.9,
    "re-rank": 0.9, "reranking": 0.9, "learning to rank": 1.0,
    "recommendation system": 1.0, "recommender": 0.9, "recommendation": 0.6,
    "retrieval": 0.6, "candidate-jd matching": 1.0, "matching pipeline": 0.8,
    # production scale & rigor
    "production ml": 0.6, "a/b test": 0.7, "ab test": 0.7,
    "queries per month": 0.6, "qps": 0.6, "at scale": 0.4, "real users": 0.5,
}
