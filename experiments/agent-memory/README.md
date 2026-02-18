# Agent Memory — RAG + Vector DB

Persistent memory system for AI agents. From-scratch Python — no framework abstractions — to understand how extraction, embedding, retrieval, and consolidation actually work together.

## Quick Start

```bash
cd experiments/agent-memory
uv run python cli.py stats          # no API key needed — just checks the store
```

For the full pipeline (extraction via LLM):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
uv run python cli.py add "I talked to Alice today. She's switching from Python to Rust for backend."
uv run python cli.py search "What language does Alice use?"
uv run python main.py               # http://localhost:8001
```

`uv run` creates the venv and installs deps on first invocation. The embedding model (~90MB) downloads on first use.

## CLI

```
cli.py add <text>               Extract facts from text via LLM, embed, dedup, store
cli.py search <query> [-k 10]   Time-weighted search with MMR diversity
cli.py stats                    Memory counts by type
cli.py consolidate              Merge old memories into summaries
cli.py evaluate [data_file]     Recall@k, Precision@k, MRR on labeled dataset
cli.py clear [--yes]            Delete all memories
```

## API

### POST /api/add

```json
{ "text": "Conversation text to extract memories from..." }
```

Response: `{ "ids": ["a1b2c3d4e5f6"], "count": 1 }`

### POST /api/search

```json
{ "query": "What language does Alice use?", "k": 10 }
```

Response: array of results with `text`, `memory_type`, `importance`, `topic`, `entities`, `combined_score`, `raw_similarity`, `recency_score`.

### GET /api/stats

Returns `{ "total": 42, "by_type": {"fact": 30, "preference": 8, "event": 4} }`.

### POST /api/consolidate

Triggers consolidation. Returns `{ "summaries_created": 3 }`.

## Configuration

All via environment variables with sane defaults:

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | (required for add/consolidate) | Claude API key |
| `LLM_MODEL` | `claude-sonnet-4-20250514` | Model for extraction/consolidation |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model name |
| `MEMORY_DB_PATH` | `./data/memory.lance` | LanceDB storage directory |
| `PORT` | `8001` | FastAPI server port |

---

## Architecture Walkthrough

This section explains the design decisions, the alternatives considered, and why each component works the way it does. Read this if you want to understand the system deeply or adapt it.

### The Core Idea

Most "memory" systems for AI agents just dump raw conversation text into a vector store and do similarity search. That works for toy demos but breaks down because:

1. **Raw text is noisy.** A 500-word conversation might contain 2 useful facts and 498 words of filler.
2. **Similarity alone is a bad ranking signal.** A memory from 6 months ago and a memory from today might have the same cosine similarity, but the fresh one is almost certainly more relevant.
3. **Dense retrieval returns redundant results.** The top-10 nearest neighbors are often paraphrases of each other, wasting context window.
4. **Memory grows unbounded.** Without consolidation, the store fills with overlapping micro-facts.

This system addresses all four through an **extraction-first write pipeline**, **time-weighted scoring**, **MMR diversity**, and **periodic consolidation**.

### Write Pipeline: Why Extraction-First?

```
User text --> LLM extracts discrete facts --> Embed each fact --> Dedup --> Store
```

**Why not store raw text?**

Raw conversation chunks are terrible retrieval units. Consider: "Hey, so I was talking to Alice yesterday, and she mentioned she's been doing a lot of Rust lately — apparently she's moving her whole backend off Python. Wild, right?" The useful signal is one sentence: "Alice is switching from Python to Rust for backend development."

The LLM extracts self-contained, third-person facts that are:
- Individually meaningful without surrounding context
- Uniform in granularity (one fact = one memory)
- Classified by type (fact, preference, event, relationship)
- Scored by importance (0.0–1.0)

**Why tool_use for extraction?**

Structured output from LLMs is unreliable with plain text prompting. You get inconsistent JSON, missing fields, creative formatting. Claude's `tool_use` forces the output through a schema-validated path — every extraction returns a proper `memories` array with all required fields. No parsing, no retries.

**Why dedup at write time?**

If someone mentions "Alice uses Rust" three times across conversations, we don't want three near-identical memories. At write time, we compute cosine similarity between each new fact and all existing memories. If any existing memory exceeds 0.92 similarity, the new one is skipped.

Why 0.92? Below ~0.90, you start catching legitimately different facts that share vocabulary. Above ~0.95, you miss obvious paraphrases. 0.92 is empirically the sweet spot for this embedding model. You can tune it via `DEDUP_COSINE_THRESHOLD`.

**Alternatives considered:**
- **Chunking strategies** (fixed-size, sentence-level, recursive): These are the standard RAG approach. They preserve more context but produce worse retrieval units. Good for document QA, wrong for memory.
- **No extraction, just embed raw text**: Simpler, no LLM cost on write. But retrieval quality suffers badly because you're matching queries against noisy text.
- **Client-side extraction with regex/NLP**: Cheaper, but can't handle the nuance of what's "worth remembering." An LLM judgment call is genuinely needed here.

### Embedding Model: Why all-MiniLM-L6-v2?

| Model | Dimensions | Size | Speed (CPU) | Quality (MTEB) |
|---|---|---|---|---|
| **all-MiniLM-L6-v2** | 384 | 90MB | ~3ms/query | 63.0 |
| all-mpnet-base-v2 | 768 | 420MB | ~10ms/query | 65.0 |
| BGE-M3 | 1024 | 2.2GB | ~30ms/query | 68.2 |
| text-embedding-3-small (OpenAI) | 1536 | API | ~100ms/query | 62.3 |
| Cohere embed-v3 | 1024 | API | ~80ms/query | 66.5 |

MiniLM is the right choice for a local experiment because:

1. **CPU-friendly.** 3ms per query on an M-series Mac. No GPU needed.
2. **Small.** 90MB download, 384-dim vectors = tiny storage footprint.
3. **Good enough.** 63.0 MTEB is not SOTA, but for short factual sentences (which is what our extraction pipeline produces), the quality gap vs. larger models is minimal. The bottleneck is retrieval ranking, not embedding quality.
4. **L2-normalized by default.** Cosine similarity = dot product. Simplifies math everywhere.

**When to upgrade:**

If you're storing memories in multiple languages, switch to **BGE-M3** — it's the best multilingual model and supports hybrid sparse+dense retrieval. If you need absolute best quality in English and don't mind API costs, **text-embedding-3-large** from OpenAI (3072-dim, 67.8 MTEB) is hard to beat. Both are drop-in replacements — change `EMBEDDING_MODEL` and `EMBEDDING_DIM` in config.

### Vector Store: Why LanceDB?

| Store | Type | Setup | ANN Index | Filtering | Disk Format |
|---|---|---|---|---|---|
| **LanceDB** | Embedded | Zero config | IVF-PQ, flat | SQL WHERE | Lance (columnar) |
| ChromaDB | Embedded | Zero config | HNSW | Metadata filter | SQLite + files |
| Qdrant | Server or embedded | Docker/binary | HNSW | Rich filter API | Custom |
| Pinecone | Cloud | API key | Proprietary | Metadata filter | Managed |
| pgvector | Extension | PostgreSQL | IVF, HNSW | Full SQL | PostgreSQL |
| FAISS | Library | In-process | Many (IVF, PQ, HNSW) | None (pre-filter) | Memory/mmap |

LanceDB wins for this use case because:

1. **Truly embedded.** It's a library, not a server. `pip install lancedb` and you're done. No Docker, no ports, no config files. The data lives in `./data/memory.lance/` as regular files.
2. **Lance columnar format.** Unlike ChromaDB (SQLite blob store), Lance is a proper columnar format designed for ML workloads. Fast full scans, efficient appends, no compaction overhead.
3. **SQL WHERE filtering.** You can filter by metadata (`memory_type = 'fact'`, `importance > 0.7`) during vector search. This is critical for consolidation queries.
4. **No index needed for <100K vectors.** At experiment scale, brute-force scan is fast enough. LanceDB supports IVF-PQ indexing if you need to scale.

**Why not ChromaDB?** ChromaDB is the default choice in LangChain tutorials, which is actually a strike against it. Under the hood it's SQLite + HNSW, which means: slow full-table operations, no columnar advantage for batch processing, and an in-memory HNSW index that must rebuild on every restart. LanceDB's flat scan is faster than ChromaDB's HNSW for <50K vectors.

**Why not FAISS?** FAISS is the gold standard for raw vector search speed but offers zero metadata storage or filtering. You'd need a separate database for everything except the vectors. For a memory system where metadata (timestamps, importance, topics) matters as much as similarity, FAISS adds complexity for no benefit.

**Why not pgvector?** Overkill for an experiment. If this system graduated to production, pgvector would be a strong choice — you get full SQL, transactions, and the entire PostgreSQL ecosystem. But requiring a running PostgreSQL instance for a learning experiment is unnecessary friction.

### Retrieval: Why Time-Weighted Scoring?

Pure cosine similarity retrieval has a fundamental problem: it has no concept of time. A memory from two years ago ranks the same as one from today, even though the recent one is almost always more relevant.

The combined score formula:

```
combined = 0.5 * cosine_similarity + 0.3 * recency + 0.2 * importance
```

**Cosine similarity (weight: 0.5).** Still the primary signal — you need semantic relevance. But it only gets half the vote.

**Recency (weight: 0.3).** Exponential decay with a 30-day half-life:

```
recency = exp(-0.693 * age_days / 30)
```

A memory from today scores 1.0. From 30 days ago: 0.5. From 60 days: 0.25. This naturally surfaces recent information without completely burying old memories. The 30-day half-life is tunable — shorter for fast-moving contexts (daily standups), longer for stable knowledge (architectural decisions).

**Importance (weight: 0.2).** The LLM assigns 0.0–1.0 at extraction time. "Alice's favorite color is blue" gets ~0.3. "The production database credentials changed" gets ~0.9. This lets critical facts punch above their semantic similarity weight.

**Why these specific weights?**

They were chosen empirically. 0.5/0.3/0.2 works well for general-purpose agent memory where:
- You usually want the most semantically relevant result (hence similarity dominant)
- Recent information should be preferred when similarity is close (hence recency second)
- Truly important facts should surface even with lower similarity (hence importance as tiebreaker)

For a news-focused agent, you'd increase recency weight. For a knowledge base, increase similarity weight.

**Alternatives considered:**
- **Pure similarity**: Ignores time entirely. Bad for conversational agents.
- **BM25 + dense hybrid**: Better for keyword-heavy queries but adds complexity (sparse index, fusion). Worth it at scale, overkill here.
- **Learned ranking**: Train a small model on click/usage data to predict relevance. The right long-term answer but requires usage data you don't have yet.

### MMR: Why Diversity Matters

Without diversity re-ranking, the top-10 results for "What does Alice work on?" might be:

1. Alice is switching from Python to Rust for backend development.
2. Alice has been learning Rust for the past few months.
3. Alice prefers Rust over Python for performance-critical code.
4. Alice's backend team is migrating to Rust.
5. ...

These are all semantically similar — you're wasting 4 context window slots saying the same thing. **Maximal Marginal Relevance (MMR)** fixes this by penalizing candidates that are too similar to already-selected results:

```
MMR(d) = lambda * relevance(d) - (1 - lambda) * max_similarity(d, selected)
```

With `lambda = 0.7`, relevance still dominates (70%) but redundancy is penalized (30%). The algorithm oversamples 20 candidates, then greedily selects the top-k by MMR score.

**Why lambda = 0.7?** Lower values (0.5) produce highly diverse but less relevant results — you start getting tangentially related memories. Higher values (0.9) barely diversify at all. 0.7 is the standard default in IR literature and works well in practice.

**Why not clustering?** You could cluster results and pick one per cluster. Simpler to implement, but MMR gives finer-grained control and doesn't require choosing a cluster count.

### Consolidation: Why Summarize Old Memories?

Without consolidation, the memory store grows linearly with usage. After a year of daily use, you might have 10,000+ micro-facts, many of which overlap or are superseded by newer information.

Consolidation runs periodically (triggered manually or on schedule):

1. **Select** memories older than 30 days that haven't been consolidated yet
2. **Group** by topic
3. **Batch** into groups of up to 50
4. **Summarize** each batch via LLM into a single summary memory
5. **Link provenance**: summary stores source IDs, sources store summary ID

The summary memory gets its own embedding and participates in future searches. The source memories are marked as consolidated and excluded from search results (but not deleted — provenance is preserved).

**Why not just delete old memories?** Information loss. If someone asks about something from 6 months ago, the summary still contains the key facts. Deleting would create gaps.

**Why group by topic?** Summarizing unrelated memories produces incoherent summaries. Grouping by topic (assigned during extraction) ensures each summary is focused and useful.

**Why batch size 50?** Claude's context window can handle it comfortably, and 50 short facts typically compress into 1–3 paragraphs. Larger batches risk losing detail; smaller batches produce too many summaries.

### File Layout

```
config.py          Env-based config — all tunable parameters in one place
models.py          Pydantic schemas (MemoryRecord, SearchResult, ExtractedFact)
                   + PyArrow schema for LanceDB table
embeddings.py      Lazy-loaded sentence-transformer (loads on first call, not import)
store.py           LanceDB CRUD — the only file that touches the database
prompts.py         LLM prompt templates (system prompts + tool_use schemas)
writer.py          Write pipeline: extract -> embed -> dedup -> store
reader.py          Read pipeline: embed query -> ANN search -> score -> MMR
consolidator.py    Batch summarization of old memories
evaluator.py       Offline eval metrics (Recall@k, Precision@k, MRR)
cli.py             Typer CLI wiring
main.py            FastAPI app
static/index.html  Browser UI
```

The dependency graph is intentionally simple and acyclic:

```
config  <--  models
               ^
               |
embeddings  store
   ^          ^  ^
   |         /    \
  writer --+      reader
   ^                ^
   |                |
  cli/main --------+--- consolidator
```

No circular imports. Each module has a single responsibility. `store.py` is the only module that talks to LanceDB. `embeddings.py` is the only module that loads the model.

### Evaluation

`evaluator.py` runs three standard IR metrics against a labeled dataset:

- **Recall@k**: What fraction of known-relevant memories were retrieved?
- **Precision@k**: What fraction of retrieved memories were relevant?
- **MRR** (Mean Reciprocal Rank): How high does the first relevant result rank? (1/rank, averaged)

The eval dataset (`eval_data/basic.json`) is a list of `{"query": "...", "relevant_texts": ["..."]}` pairs. To evaluate:

1. Seed the memories (via `cli.py add` or direct store operations)
2. Run `cli.py evaluate`

This gives you a baseline to measure the impact of parameter changes (weights, thresholds, embedding model).

### What to Experiment With

**Easy knobs:**
- Scoring weights in `config.py` — try 0.7/0.2/0.1 for similarity-dominant, or 0.3/0.5/0.2 for recency-dominant
- `DEDUP_COSINE_THRESHOLD` — lower to 0.88 to allow more similar memories, raise to 0.95 for stricter dedup
- `MMR_LAMBDA` — lower for more diversity, higher for more relevance
- `RECENCY_HALF_LIFE_DAYS` — shorter half-life = more aggressive time decay

**Medium effort:**
- Swap embedding model to `all-mpnet-base-v2` (768-dim, better quality) or `BGE-M3` (multilingual)
- Add metadata filtering to search (e.g., only search `memory_type = 'fact'`)
- Implement access-count boosting (frequently accessed memories rank higher)

**Bigger projects:**
- Hybrid search: combine dense vectors with BM25 sparse retrieval (LanceDB supports full-text search)
- Multi-turn conversation memory: track conversation threads, not just isolated facts
- Automatic consolidation scheduling based on store size
- Relevance feedback: let users upvote/downvote search results to tune ranking
