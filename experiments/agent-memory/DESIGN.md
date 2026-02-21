# Agent Memory: Design Notes & Future Direction

Landscape analysis, comparison of existing systems, and a roadmap for evolving our implementation into something that takes the best ideas from the field.

---

## Landscape: Who's Doing What

### Mem0

**Architecture:** Two-phase write pipeline. Phase 1: LLM extracts discrete facts from conversation. Phase 2: LLM classifies each fact as ADD/UPDATE/DELETE/NOOP against existing memories. Vector store (default Qdrant) + optional Neo4j graph. 2-5 LLM calls per `add()`.

**Strengths:**
- LLM-based dedup with semantic merge ("Alice uses Python" can evolve into "Alice switched from Python to Rust" via UPDATE)
- Full developer control — explicit CRUD, inspectable memory objects
- Huge ecosystem (24+ vector stores, 16+ LLM providers)
- 43k GitHub stars, well-documented

**Weaknesses:**
- No built-in user profile abstraction
- No consolidation mechanism — memory store grows linearly
- No time-weighted retrieval — pure cosine similarity unless you bolt on a reranker
- No MMR or diversity in retrieval results
- Graph memory adds ~2% accuracy for 3x the LLM calls (Neo4j is overkill)
- "90% token savings" only measures read-path tokens; write-path costs 2-5 LLM calls per add

### Supermemory

**Architecture:** Managed memory API on Cloudflare Workers. Documents produce both RAG chunks AND extracted memories (dual data model). Memory Router proxy intercepts LLM API calls transparently. Static/dynamic per-user profiles.

**Strengths:**
- Memory Router (zero-code integration via HTTP proxy) — genuinely novel
- Dual data model: documents produce chunks (RAG) + memories (facts)
- Automatic static/dynamic user profiles
- Memory versioning via update/extend/derive relation chains (not a full graph, just linked lists with relation types)
- Smart decay: recency bias, frequency weighting, TTL-based expiration
- AST-aware code chunking (`code-chunk`, +28 recall points on RepoEval)
- Sub-300ms latency target

**Weaknesses:**
- Core ML pipeline is a black box (embedding model undisclosed, extraction logic proprietary)
- Contradiction resolution is vaguely documented
- "Open source" requires Cloudflare ecosystem to self-host
- Self-published benchmarks (built memorybench, then claim SOTA on it)
- Small team / early stage for reliability-critical infrastructure

### MemGPT / Letta

**Architecture:** The LLM manages its own memory. Main context = working memory. Archival storage = vector DB. The model issues function calls to page data in/out (`core_memory_append`, `archival_memory_search`). An outer loop re-invokes the model after each memory operation.

**What's novel:** The agent has metacognitive control — it decides what to store, retrieve, and evict. No external extraction pipeline.

**Weaknesses:** Every memory operation costs an LLM call. No structural representation. The OS metaphor is somewhat superficial (no LRU eviction, no memory protection). Memory quality bounded by prompt-following ability.

### Zep

**Architecture:** Async extraction pipeline producing: (1) knowledge graph of entities/relationships, (2) temporal episodes, (3) user/session summaries. Facts have `valid_from`/`valid_to` timestamps for temporal reasoning.

**What's novel:** Temporal fact tracking. Facts have validity windows, enabling "what was true when?" queries. Graph traversal + semantic search combined.

**Weaknesses:** Entity resolution across conversations is hard and brittle. Temporal reasoning is basic (when stated, not when true). Commercial with limited OSS edition.

### LangMem (LangChain)

**Architecture:** Three memory types — semantic (extracted facts), episodic (raw conversations), procedural (system prompt updates). Background "memory manager" agent processes conversations post-completion.

**What's novel:** Procedural memory — the system modifies its own system prompts based on accumulated experience. Schema-driven extraction (you define what structure memories should have).

**Weaknesses:** Tightly coupled to LangChain. Self-modifying prompts can drift. Limited relational capabilities.

### A-Mem (Agentic Memory)

**Architecture:** Each memory unit has content, embedding, activation level, and associative links. Four operations: encoding, retrieval, consolidation, forgetting. Activation model based on ACT-R from cognitive science.

**What's novel:** Spreading activation — retrieving "Rust" also activates linked memories about "systems programming" and "performance." Memory "temperature" rises with use and decays over time. This is the most cognitively-grounded approach.

**Weaknesses:** Computational overhead. ACT-R parameters need tuning. Academic prototype, not production-ready.

### MemoryBank

**Architecture:** Memories have "strength" values following Ebbinghaus forgetting curves. Stability increases on each successful retrieval (spaced repetition). Periodic consolidation strengthens important memories and lets unimportant ones decay.

**What's novel:** Direct application of cognitive science forgetting models. Memories that are recalled frequently become more stable (harder to forget).

**Weaknesses:** Forgetting curve is a simplification. Doesn't capture interference effects.

### Generative Agents (Park et al., 2023)

The foundational paper. Memory stream with recency + importance + relevance scoring (sound familiar?). The key contribution is **reflection**: periodically, the agent generates higher-level observations from its memory stream, which are stored back as memories. This creates emergent hierarchical memory without explicit architecture.

---

## Comparison: Our System vs. The Field

| Capability | Ours | Mem0 | Supermemory | Letta | Zep |
|---|---|---|---|---|---|
| **Extraction** | LLM tool_use, 1 call | LLM, 2 calls (extract + classify) | Managed, opaque | Agent self-manages | Async LLM pipeline |
| **Dedup** | Cosine threshold (0.92) | LLM ADD/UPDATE/DELETE | Automatic, opaque | None | Entity resolution |
| **Scoring** | 0.5\*sim + 0.3\*recency + 0.2\*importance | Pure similarity | Similarity + threshold | N/A (agent decides) | Graph + similarity |
| **Diversity** | MMR (lambda=0.7) | None | None documented | None | None |
| **Consolidation** | Topic-grouped LLM summaries | None | Decay-based | None | Session summaries |
| **User profiles** | None | None | Static + dynamic | Core memory block | Session-scoped |
| **Graph** | None | Optional Neo4j | Lightweight version chains | None | Entity graph |
| **Temporal** | Recency decay (exp, 30d half-life) | None | Dual timestamps + decay | None | valid_from/valid_to |
| **Versioning** | None | Overwrite | Update/extend/derive chains | Overwrite | Event log |
| **Forgetting** | None | None | TTL + decay | None | None |
| **Memory types** | fact/pref/event/relationship/summary | Flat | Hierarchical (static/dynamic) | Working/archival | Episodes/entities/facts |
| **Write cost** | 1 LLM call | 2-5 LLM calls | Opaque | 1+ LLM calls per op | 1-2 LLM calls |
| **Vector store** | LanceDB (embedded) | 24+ options | Custom (Cloudflare) | 5+ options | Custom |
| **Self-hostable** | Fully, zero infra | Yes | Requires Cloudflare | Yes | Limited OSS |

**Where we're strong:** Time-weighted scoring, MMR diversity, consolidation with provenance, low write cost (1 LLM call), zero-infra LanceDB.

**Where we're weak:** No versioning, no graph/relational structure, no user profiles, no forgetting mechanism, no contradiction detection, dedup is structural (cosine threshold) not semantic (LLM-based).

---

## Design Proposals

### 1. Memory Versioning (Version Chains + Merkle Roots)

Instead of overwriting memories or just skipping duplicates, track evolution.

**Data model extension:**

```python
class MemoryRecord:
    # ... existing fields ...

    # Version chain
    version: int = 1
    parent_id: str = ""           # Previous version of this memory
    root_id: str = ""             # Original memory in the chain
    relation: str = ""            # "updates" | "extends" | "derives"
    is_latest: bool = True

    # Content hash for integrity
    content_hash: str = ""        # SHA-256 of canonical text
```

**How it works:**

When a new fact is similar to an existing one (cosine > 0.85 but < 0.92):
1. Don't skip (current behavior) or blindly add
2. Use an NLI model or LLM to classify: contradiction, extension, or refinement
3. If **updates**: mark old as `is_latest=False`, create new with `parent_id` pointing to old
4. If **extends**: create new with `relation="extends"`, both stay `is_latest=True`
5. If **derives**: system-inferred fact, link to sources via `consolidated_from`

**Why not a full DAG?** A version chain (linked list per memory lineage) is enough for single-agent use. DAGs matter when multiple agents update concurrently — we can add merge commits later if needed. The chain subsumes into a DAG without schema changes.

**Why Merkle roots?** Content-hash each memory. Periodically compute a Merkle root over the full store. This gives you:
- Integrity verification (detect corruption)
- Efficient diffing between snapshots
- A natural "commit" abstraction for memory state

This is lightweight — just a `content_hash` field per record and a periodic root computation. No tree structure in the DB needed.

### 2. Lightweight Entity Graph

Not Neo4j. Not even a separate graph DB. Just an adjacency list in LanceDB.

**New table: `entity_edges`**

```python
EDGE_SCHEMA = pa.schema([
    pa.field("source_entity", pa.string()),   # "alice"
    pa.field("target_entity", pa.string()),   # "rust"
    pa.field("relation", pa.string()),        # "uses"
    pa.field("memory_ids", pa.string()),      # JSON list of supporting memory IDs
    pa.field("weight", pa.float32()),         # Co-occurrence strength
    pa.field("first_seen", pa.string()),
    pa.field("last_seen", pa.string()),
])
```

**Build it during extraction:** The LLM already extracts `entities` per memory. Co-occurring entities in the same memory get an edge. Weight increases with each co-occurrence.

**Use it during retrieval:** After vector search returns candidates, expand results by traversing 1-hop entity edges. If the query mentions "Alice," also pull memories connected to Alice's entity neighbors (her projects, her team, her tools).

**Why this is enough:** For a personal memory system, the entity graph is small (hundreds to low thousands of nodes). A flat adjacency list with SQL-style WHERE filters covers 95% of use cases. Multi-hop traversal can be done in application code with 2-3 queries. The remaining 5% (community detection, PageRank) can use rustworkx or igraph loaded from the same data.

**Upgrade path:** If the graph gets complex enough to justify it, PostgreSQL + Apache AGE gives full Cypher support without Neo4j operational overhead. Or DuckDB for analytical graph queries.

### 3. Memory Decay & Forgetting

Our current system has recency scoring (exponential decay at retrieval time) but no actual forgetting. Memories accumulate forever.

**Proposal: Composite decay with spaced-repetition reinforcement.**

```python
class MemoryRecord:
    # ... existing fields ...

    stability: float = 1.0       # Grows with successful retrievals
    retrievability: float = 1.0  # Current retention score (decays over time)
    forget_after: str = ""       # Optional hard TTL (ISO 8601)
    is_forgotten: bool = False   # Soft delete
```

**Decay model:**

```python
def retention(memory, now):
    elapsed_days = (now - memory.last_accessed).total_seconds() / 86400

    if elapsed_days < 1:
        # Short-term: exponential decay (context window relevance)
        return math.exp(-elapsed_days / memory.stability)
    else:
        # Long-term: power law decay (more realistic for established facts)
        # Power law has heavier tail — old important memories persist longer
        alpha = BASE_DECAY / math.log1p(memory.access_count)
        return (1 + elapsed_days / memory.stability) ** (-alpha)

def on_retrieved(memory, relevance_score):
    """Called when a memory is retrieved and used in context."""
    # Spaced repetition: stability grows more when retrieval interval was longer
    interval = (now - memory.last_accessed).total_seconds() / 86400
    memory.stability *= 1 + STABILITY_GAIN * math.log1p(interval)
    memory.last_accessed = now
    memory.access_count += 1
```

**Why power law for long-term?** Empirical evidence (Wixted & Ebbesen 1991) shows human long-term memory follows power law, not exponential. The heavy tail means well-established facts don't just vanish after a few half-lives — they persist at low but nonzero strength. This matches intuition: you remember your childhood address decades later, just not as sharply.

**Forgetting tiers:**
1. **Active** (retrievability > 0.3): In the hot retrieval index, returned by search
2. **Dormant** (0.05 < retrievability < 0.3): Still searchable but only if explicitly requested or highly relevant (similarity > 0.9)
3. **Forgotten** (retrievability < 0.05): Soft-deleted, excluded from search, preserved for provenance
4. **Expired** (past `forget_after` TTL): Same as forgotten, triggered by time

**Maintenance job:** Periodic scan (daily or on consolidation) to recompute retrievability scores and move memories between tiers.

### 4. Contradiction Detection

Our current cosine dedup (> 0.92 = skip) can't detect contradictions. "Alice uses Python" and "Alice switched to Rust" have moderate similarity (~0.6) — they'd both be stored as separate facts.

**Proposed pipeline:**

```
New fact
    |
    v
[1. Entity overlap check] -------- O(1), cheap
    Same entities + similar predicate? If no, skip to store.
    |
    v
[2. Embedding pre-filter] --------- O(log n) with ANN
    Find top-5 existing memories with same entities
    |
    v
[3. NLI classification] ----------- O(5), ~100ms total
    cross-encoder/nli-deberta-v3-base (smaller, faster)
    Classify each pair as entailment / contradiction / neutral
    |
    v
[4. Resolution]
    - Entailment: skip (already known)
    - Contradiction: version chain (mark old, store new as update)
    - Neutral: store as new memory
```

**Why NLI over LLM?** An NLI model (DeBERTa-base, ~180MB) runs in ~20ms per pair on CPU. That's 5 comparisons in 100ms. An LLM call would take 1-2 seconds and cost tokens. For contradiction detection, NLI is more accurate AND cheaper.

**Dependency:** `cross-encoder` package (~180MB model). Add to pyproject.toml as optional: `pip install agent-memory[nli]`.

**Fallback without NLI:** Use the existing cosine similarity with a lower threshold (0.75-0.85) as a soft signal for "these might conflict," then include both and let the reader's MMR handle diversity. Less accurate but zero additional dependencies.

### 5. User Profiles (Static + Dynamic)

Supermemory's static/dynamic split is a good abstraction. Implement it as a retrieval-time aggregation, not a separate store.

```python
def get_profile(user_id: str) -> dict:
    all_memories = store.get_all()  # or filtered by user_id if multi-user

    static = []   # High stability, high importance, low change rate
    dynamic = []  # Recent, moderate importance, actively changing topics

    now = datetime.now(timezone.utc)
    for mem in all_memories:
        if mem.is_forgotten or mem.consolidated_into:
            continue
        age_days = (now - mem.created_at).days

        if mem.stability > 5.0 and mem.importance > 0.6 and age_days > 7:
            static.append(mem)
        elif age_days < 14 and mem.access_count > 0:
            dynamic.append(mem)

    return {
        "static": sorted(static, key=lambda m: m.importance, reverse=True)[:20],
        "dynamic": sorted(dynamic, key=lambda m: m.last_accessed, reverse=True)[:10],
    }
```

**Why retrieval-time, not write-time?** A memory's classification (static vs dynamic) changes over time. A new preference starts dynamic and becomes static as it's confirmed repeatedly. Computing at retrieval time means the profile is always fresh.

### 6. Reflection (from Generative Agents)

The most impactful idea from the academic literature. Periodically, the agent generates higher-level observations from its memory store.

```python
def reflect(top_k: int = 20) -> list[MemoryRecord]:
    """Generate reflections from recent memories."""
    recent = reader.search("recent important events and facts", k=top_k)

    # Ask LLM: "What are the 3 most salient high-level observations?"
    texts = [r.memory.text for r in recent]
    prompt = f"""Given these memories:
{chr(10).join(f'- {t}' for t in texts)}

What are the 3 most important high-level observations or patterns you can identify?
Write each as a self-contained statement in third person."""

    # Extract via tool_use, store as memory_type="reflection"
    # with importance boosted and consolidated_from linking to source memories
```

**Why this matters:** Reflections create emergent hierarchical memory. Raw facts at the bottom, reflections on facts in the middle, reflections on reflections at the top. This is what consolidation should evolve toward — not just summarization, but insight generation.

---

## Implementation Priority

Sorted by impact/effort ratio:

| # | Feature | Effort | Impact | Dependencies |
|---|---|---|---|---|
| 1 | Memory decay & forgetting | Medium | High | None — extends existing scoring |
| 2 | Version chains | Medium | High | Schema migration |
| 3 | Entity graph (adjacency list) | Medium | Medium | Extraction already produces entities |
| 4 | User profiles | Low | Medium | Needs decay model first |
| 5 | Contradiction detection (NLI) | Medium | Medium | Optional dep (cross-encoder) |
| 6 | Reflection | Low | High | Just a new prompt + writer call |
| 7 | Merkle integrity | Low | Low | Nice-to-have, no urgency |

**Recommended order:** 1 -> 2 -> 6 -> 3 -> 4 -> 5 -> 7

Decay + versioning are foundational — everything else builds on them. Reflection is high-impact and nearly free (it's just a new prompt template + a scheduled writer call). The entity graph and profiles are medium-effort quality-of-life improvements. NLI contradiction detection is the most complex addition but can be deferred since version chains handle the "store both, let the reader sort it out" case.

---

## Concepts Worth Knowing

### Ebbinghaus Forgetting Curve

`R(t) = e^(-t/S)` where R = retention, t = time, S = stability. Each successful retrieval increases S. This is the theoretical basis for spaced repetition (Anki, SuperMemo). MemoryBank applies this directly to AI memory.

**Key insight for us:** Our exponential decay with 30-day half-life is a simplified version of this. The upgrade is making stability per-memory (not global) and increasing it on access.

### Spreading Activation (ACT-R)

From cognitive science. When a memory is activated, activation spreads to associated memories through links. Retrieving "Rust" also partially activates "systems programming," "Alice," and "performance." The activation level of a memory = base activation + sum of spreading activation from active neighbors.

**Key insight for us:** Our entity graph would enable this. During retrieval, after finding direct matches, boost scores of memories that share entities with the top results.

### Memory Interference

**Proactive interference:** Old memories make it harder to form new ones in the same domain. **Retroactive interference:** New memories degrade recall of old ones in the same domain.

**Key insight for us:** When adding a new memory about a topic, existing memories on the same topic should have their stability slightly reduced. This is the "competing memories" effect — it naturally handles the case where outdated information should fade as new information arrives, even without explicit contradiction detection.

### Dual Timestamping

Track both **document time** (when the conversation happened) and **event time** (when the described event occurred). "Yesterday I talked to Alice about her Rust migration last March."
- Document time: today
- Event time for conversation: yesterday
- Event time for migration: last March

**Key insight for us:** Our `created_at` is document time. Adding an optional `event_at` field would enable temporal queries like "what happened in Q1?" without confusing it with "what did we discuss in Q1?"

### Procedural Memory

Memory about *how to do things*, not *what is true*. Most systems ignore this entirely. LangMem's system prompt updates are the closest implementation. Reflexion stores self-generated strategy observations.

**Key insight for us:** A `memory_type="procedure"` with content like "When Marko asks about performance, he usually wants specific benchmark numbers, not general advice" would be high-value. These are meta-memories about interaction patterns.

---

## Open Research Questions

1. **How do you measure memory quality without human labels?** Round-trip fidelity (extract -> regenerate original context) and retrieval utility (was the retrieved memory actually used by the LLM?) are promising signals but untested at scale.

2. **What's the right forgetting rate?** Too aggressive and you lose useful context. Too conservative and the store bloats. The answer is probably workload-dependent — a daily standup agent needs faster decay than a knowledge management agent.

3. **When does graph structure pay for itself?** For <1K memories, flat vector search with good scoring probably outperforms graph traversal. The crossover point where graph structure adds value is unknown.

4. **Can you detect false memories?** LLM extraction can confabulate — infer plausible but incorrect facts. Multi-path verification (extract with different prompts, keep only consistent results) is theoretically sound but doubles extraction cost.

5. **What's the optimal memory granularity?** Too fine ("Alice mentioned Rust") and you need many memories to reconstruct context. Too coarse ("Alice is a senior engineer transitioning her team's backend from Python to Rust, motivated by performance requirements for their real-time data pipeline") and dedup/contradiction detection breaks. The sweet spot is probably 1-2 sentences per memory, which is what our extraction produces.
