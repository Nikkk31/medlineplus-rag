# Medical RAG — Diabetes & Blood Sugar Assistant

A retrieval-augmented generation system that answers patient-education
questions about diabetes and blood sugar, grounded only in MedlinePlus
(National Library of Medicine) content. Runs fully locally by default
(no API key required), with an optional Anthropic API backend.

## Architecture

```
MedlinePlus XML  →  markdown corpus  →  chunks  →  FAISS + BM25 indexes  →  hybrid retrieval  →  LLM  →  answer + sources
   (ingest/)         (data/corpus/)   (retrieval/loaders.py) (retrieval/indexing.py) (retrieval/hybrid.py) (generation/)
```

1. **`ingest/fetch_medlineplus.py`** — downloads the MedlinePlus topic XML
   dump, filters it down to topics matching `TOPIC_KEYWORDS` in
   `config.py` (diabetes, blood sugar, insulin, a1c, etc.), caches results
   in `data/raw/`.
2. **`ingest/to_markdown.py`** — converts each filtered topic into one
   `.md` file under `data/corpus/`, with YAML-style frontmatter
   (title, source URL, MeSH headings). This markdown layer is the source
   of truth for everything downstream — human-readable and diffable.
3. **`retrieval/loaders.py`** — loads the markdown corpus and splits it
   into chunks: first by markdown header (so a chunk never straddles
   unrelated sections), then by size (`CHUNK_SIZE` / `CHUNK_OVERLAP`).
4. **`retrieval/indexing.py`** — builds two indexes over the chunks:
   a dense FAISS index (`sentence-transformers/all-MiniLM-L6-v2`
   embeddings) and a sparse BM25 index. Both are fingerprinted on corpus
   content, so re-running only rebuilds when the corpus actually changed.
5. **`retrieval/hybrid.py`** — fuses dense + BM25 results with
   Reciprocal Rank Fusion (RRF) at query time, and gates out-of-scope
   questions using FAISS distance against `SIMILARITY_THRESHOLD`.
6. **`generation/generate.py`** — formats retrieved chunks into a
   prompt, sends it to either a local Qwen2.5-3B-Instruct GGUF model
   (via `llama-cpp-python`) or the Anthropic API, and returns an answer
   with deduped source citations.
7. **`streamlit_app.py`** — chat UI. Keeps the model and retriever
   loaded for the life of the server process (`@st.cache_resource` /
   `@lru_cache`), so only the first question after startup pays the
   full model-load cost.

## Setup

```powershell
# create/activate your environment, then:
pip install -r requirements.txt

```

Download a GGUF model into `models/` — see `config.py`'s
`LOCAL_MODEL_PATH` for the expected filename. This project was built and
tuned against `Qwen2.5-3B-Instruct` at `Q4_K_M` quantization.

## Building the corpus and indexes

```powershell
python -m ingest.fetch_medlineplus
python -m ingest.to_markdown
```

The FAISS + BM25 indexes build automatically on first use (via
`retrieval/indexing.py`'s `build_or_load_indexes()`) and are cached in
`storage/`. They rebuild automatically when the corpus changes, based on
a content fingerprint in `storage/fingerprint.json`.

**Note:** if you rebuild the corpus or indexes while the Streamlit app is
running, restart the app (`Ctrl+C`, not just a browser refresh) —
`st.cache_resource` holds the old index in memory otherwise.

## Running

```powershell
# chat UI (recommended - keeps the model loaded across questions)
streamlit run streamlit_app.py

# one-off CLI query
python cli.py "What is type 2 diabetes?"
```

## Configuration

Key settings in `config.py`:

| Setting | Purpose |
|---|---|
| `TOPIC_KEYWORDS` | Which MedlinePlus topics get pulled into the corpus |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Chunk granularity for indexing |
| `TOP_K` | Number of chunks retrieved per question |
| `SIMILARITY_THRESHOLD` | In-scope gate cutoff (FAISS L2 distance) |
| `GENERATION_BACKEND` | `"local"` (default) or `"anthropic"` |
| `LOCAL_MODEL_PATH` | Path to the local GGUF model file |

## Evaluation

```powershell
python -m eval.eval_retrieval
```

Reports dense/BM25/hybrid retrieval accuracy against topic-level ground
truth in `eval/eval_set.py`, plus in-scope gate accuracy against
out-of-scope control questions. Questions with `expected_topics: None`
are not scored — the harness instead prints what was actually retrieved
so ground truth can be filled in from real output rather than guessed.

## Known limitations

- **No dedicated diabetic ketoacidosis or diet/nutrition content.**
  `TOPIC_KEYWORDS` doesn't currently include "ketoacidosis" or
  "diet"/"nutrition", so those MedlinePlus topics were never ingested.
  Questions on these topics retrieve scattered, low-confidence chunks.
- **Local generation is slow on CPU-only hardware** (~60-130s per
  question on a 4GB RAM laptop with no GPU), even with Q4_K_M
  quantization and `CPU_REPACK` matmul kernels engaged. The Anthropic
  backend is meaningfully faster if low latency matters more than
  running fully offline.
- This tool provides general information only and is not a substitute
  for professional medical advice.
