# 5 Levels of Text Splitting

Source: Greg Kamradt (FullStackRetrieval.com)
Notebook: github.com/FullStackRetrieval-com/RetrievalTutorials

**Core Principle**: "Your goal is not to chunk for chunking sake -- our goal is to get our data in a format where it can be retrieved for value later."

---

## Level 1: Character Splitting

Fixed-size character windows with optional overlap. Simplest possible approach.

- **Parameters**: `chunk_size`, `chunk_overlap`, `separator`
- **Library**: `langchain.text_splitter.CharacterTextSplitter`
- **Trade-offs**: Trivial to implement. Completely ignores text structure -- fragments words, sentences, paragraphs arbitrarily.
- **Use case**: Quick prototyping only.

## Level 2: Recursive Character Text Splitting

Hierarchical separator list: tries paragraph boundaries (`\n\n`) first, then newlines (`\n`), then spaces, then individual characters. Falls through the hierarchy until chunks fit within `chunk_size`.

- **Library**: `langchain.text_splitter.RecursiveCharacterTextSplitter`
- **Trade-offs**: Respects document structure far better than Level 1. Still targets a fixed chunk size, so it can break semantic units. Very low overhead.
- **Recommendation**: Kamradt calls this the "Swiss army knife of splitters and my first choice when mocking up a quick application." **Default starting point.**

## Level 3: Document-Specific Splitting

Format-aware separator hierarchies tailored to specific document types.

**Variants**:
- **Markdown**: headers > code blocks > horizontal rules > paragraphs
- **Python**: classes > functions > indented functions > structural elements
- **JavaScript**: function/const/let declarations > control flow > structural breaks
- **PDFs with tables**: `unstructured.partition.pdf.partition_pdf()` extracts tables as HTML. **Summarize tables before embedding** -- raw table text embeds poorly.
- **Multi-modal (text + images)**: Extract images, generate summaries via vision models, embed summaries instead of raw images.

**Key insight**: Tables and images need derived representations (summaries) before embedding. Raw structured data does not embed well.

## Level 4: Semantic Chunking

Embedding-distance-based boundary detection ("embedding walk").

**Algorithm**:
1. Split text into individual sentences.
2. Create sliding windows of N sentences (buffer of 3 recommended).
3. Generate embeddings for each windowed group.
4. Compute cosine distance between consecutive embedding windows.
5. Identify breakpoints where distance exceeds a threshold (e.g., 95th percentile of all distances).
6. Split at those breakpoints.

**Key insight**: The buffer window (grouping 3 sentences before embedding) smooths out noise in single-sentence embeddings. Without it, too many false breakpoints.

- **Library**: `langchain_experimental.text_splitter.SemanticChunker`
- **Trade-offs**: Produces semantically coherent chunks. Significant compute overhead (one embedding call per sentence window). Requires tuning distance threshold and buffer size. Chunk sizes are variable and unpredictable -- no fixed upper bound without additional logic.
- **Sweet spot for production** when you can afford embedding compute.

## Level 5: Agentic Splitting

LLM-as-agent decides chunk membership. Based on the "Propositions" concept from academic research.

**Algorithm**:
1. Convert raw text into "propositions" (standalone factual statements).
2. For each proposition, query the LLM: "Does this belong to an existing chunk?"
3. LLM sees chunk outlines (ID, title, summary) and decides: assign to existing or create new.
4. When adding to a chunk, LLM regenerates the chunk's title and summary.
5. Summaries generalize upward (apples -> food, October -> dates and times).

- **Implementation**: GPT-4-turbo, temperature=0. Each chunk has `chunk_id`, `propositions[]`, `title`, `summary`. Uses Pydantic extraction chains for structured output.
- **Trade-offs**: Extremely expensive (multiple LLM calls per proposition). Slow. Non-deterministic. Produces the highest-quality semantic groupings. Only viable at scale if token costs trend toward zero.
- **Key insight**: The only level where chunk boundaries are fully content-aware rather than structure-aware or distance-aware.

## Bonus: Alternative Representation Chunking

Create derived representations optimized for retrieval alongside original chunks: summaries of tables/images, hypothetical questions about the content, metadata extraction. Feeds into Multi-Vector indexing strategies.

---

## Evaluation

Chunking strategies must be evaluated empirically, not assumed. Optimal strategy is domain-specific.

- **RAGAS** -- Retrieval-Augmented Generation Assessment
- **LangChain Evals**
- **Llama Index Evals**
- **ChunkViz.com** -- visual inspection of chunk boundaries

## Decision Framework

| Level | Compute | Quality | When to use |
|-------|---------|---------|-------------|
| 1 - Character | Negligible | Low | Never in production |
| 2 - Recursive | Negligible | Medium | Default starting point, prototyping |
| 3 - Doc-Specific | Low | Medium-High | Known document formats, PDFs with tables |
| 4 - Semantic | Medium (embeddings) | High | Production systems needing quality |
| 5 - Agentic | Very High (LLM calls) | Highest | Research, high-value corpora, cost-insensitive |
