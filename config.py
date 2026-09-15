from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # picks up ANTHROPIC_API_KEY / GENERATION_BACKEND from a local .env file, if present

# ── Paths ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_CACHE_DIR = DATA_DIR / "raw"          # cached MedlinePlus XML pulls
CORPUS_DIR = DATA_DIR / "corpus"          # the markdown document layer (source of truth for indexing)
STORAGE_DIR = ROOT_DIR / "storage"        # FAISS index + BM25 pickle + fingerprint file

for d in (RAW_CACHE_DIR, CORPUS_DIR, STORAGE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Data source (MedlinePlus) ────────────────────────────────────────────
MEDLINEPLUS_XML_URL_TEMPLATE = "https://medlineplus.gov/xml/mplus_topics_{date}.xml"

# Topic scope filter: a health-topic is kept if any of these strings appear
# (case-insensitive) in its title or "also-called" vocabulary.
TOPIC_KEYWORDS = [
    "diabetes",
    "blood sugar",
    "blood glucose",
    "insulin",
    "prediabetes",
    "hyperglycemia",
    "hypoglycemia",
    "a1c",
]

MIN_CONTENT_CHARS = 200  # below this, a topic summary is considered too thin to index

# ── Markdown corpus layer ────────────────────────────────────────────────
CORPUS_SOURCE_NAME = "MedlinePlus"

# ── Chunking ──────────────────────────────────────────────────────────────
CHUNK_SIZE = 500
CHUNK_OVERLAP = 120
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
MARKDOWN_HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

# ── Embeddings / dense retrieval ─────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 32

# ── Hybrid retrieval ──────────────────────────────────────────────────────
TOP_K = 6                # final number of chunks returned to the caller
DENSE_CANDIDATE_K = 20    # candidates pulled from FAISS before fusion
BM25_CANDIDATE_K = 20     # candidates pulled from BM25 before fusion
RRF_K = 60                # standard Reciprocal Rank Fusion smoothing constant (Cormack et al.)

# Similarity gate: below (i.e. more similar than) this L2 distance, a query
# is considered "in scope" for this corpus. Tune against EVAL_SET.
SIMILARITY_THRESHOLD = 1.3

# ── Generation ────────────────────────────────────────────────────────────
# Local backend (llama.cpp) - default, no API key required.
LOCAL_MODEL_PATH = ROOT_DIR / "models" / "qwen2.5-3b-instruct-q5_k_m.gguf"
LOCAL_MODEL_N_CTX = 4096
LOCAL_MODEL_N_GPU_LAYERS = int(os.environ.get("LOCAL_MODEL_N_GPU_LAYERS", "0"))  # 0 = CPU only

# "local" (llama.cpp / Qwen) or "anthropic" (API). Defaults to local since
# that's what you have set up; switch via .env if you add an API key later.
GENERATION_BACKEND = os.environ.get("GENERATION_BACKEND", "local")

GENERATION_MAX_TOKENS = 700
GENERATION_TEMPERATURE = 0.2

OUT_OF_SCOPE_MESSAGE = (
    "That question is outside the scope of this assistant's medical knowledge base "
    "(patient education on diabetes / blood sugar). Please consult a clinician or a "
    "general-purpose source for that question."
)

SYSTEM_PROMPT = """You are a patient-education assistant. You answer ONLY using the \
provided context, which comes from MedlinePlus (National Library of Medicine).
 
Rules:
- Only use facts present in the context below. Do not add outside medical knowledge.
- The context below may contain several chunks from different MedlinePlus pages.
  Read all of them and COMBINE the relevant pieces into one coherent answer - do not
  just restate the single chunk that looks most relevant and stop there.
- Aim for 3-6 sentences, or a short list, when the context supports it. A one-line
  answer is only acceptable if the context genuinely contains nothing more to add.
- For each claim, briefly say why it matters or what it means practically for the
  reader, if the context supports that - don't just state a fact in isolation.
- If the context does not fully answer the question, say clearly what's missing or
  say the source doesn't specify a number, rather than guessing one.
- Write for a general reader: short sentences, no jargon without explanation.
- After each claim, cite the source in brackets, e.g. [MedlinePlus: Diabetes].
- End with: "This is general information, not medical advice - talk to a healthcare provider \
about your specific situation."
"""