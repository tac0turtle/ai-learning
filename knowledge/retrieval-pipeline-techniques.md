# Retrieval Pipeline Techniques

Source: Greg Kamradt (FullStackRetrieval.com)
Repo: github.com/FullStackRetrieval-com/RetrievalTutorials

---

## Pipeline Overview

Full retrieval pipeline (14 stages):

```
Query -> Query Transformation -> Raw Data Source -> Document Loaders ->
Documents -> Index -> Knowledge Base -> Retrieval Method -> Relevant Docs ->
Document Transform -> Context -> LLM -> Prompting Method -> Response
```

Each stage has multiple technique options. The stages that matter most for quality: **Query Transformation**, **Indexing**, **Retrieval Method**, and **Document Transform**.

---

## Query Transformation

### Multi-Query Retrieval

Generate multiple rephrased queries from a single user question. Each query retrieves its own docs, results are deduplicated and merged.

- **Why**: User queries are often suboptimal -- ambiguous, too narrow, or poorly worded. Multiple rephrasings compensate.
- **Library**: `langchain.retrievers.MultiQueryRetriever` with customizable prompt templates.
- **Trade-off**: Multiplies retrieval cost by number of generated queries. Worth it when recall matters more than latency.

---

## Indexing Strategies

### Multi-Vector Retriever

Store derived representations (summaries, hypothetical questions) alongside original chunks. Search against derived forms, return originals.

- **Key idea**: Raw chunks may not embed optimally. Derived forms (summaries, hypothetical questions) may better match anticipated queries.
- **Library**: `MultiVectorRetriever` with Chroma (vectorstore) + `InMemoryStore` (docstore).
- **Pattern**: Embed summaries for search, return full documents for context.

### Parent Document Retriever

Split large parent docs into small child chunks. Search against children (for precision), return parents (for context).

- **Example**: 1 essay -> 8 parents (4000 chars) -> 82 children (500 chars).
- **Library**: `ParentDocumentRetriever`.
- **Trade-off**: Small chunks embed more precisely but lack context. Parent retrieval restores that context. Two-level hierarchy adds indexing complexity.

---

## Retrieval Methods

### Top-K Similarity Search

Pull K most similar documents using cosine/dot-product/euclidean similarity. Foundation for everything else.

- **Pipeline**: load -> chunk -> embed -> store -> search -> generate.
- **Default K**: Typically 4-10. Tune based on context window budget and relevance dropoff.

### Maximum Marginal Relevance (MMR)

Balances relevance with diversity. Avoids returning K near-duplicate results.

- **When to use**: Complex multi-aspect queries, summarization tasks, ambiguous terms where diverse results reveal different facets.
- **Measured impact**: In tests, MMR returned 2 unique docs vs standard similarity search returning near-duplicates among 8 results.
- **Trade-off**: Slightly lower average relevance per-document, but higher information coverage across the result set.

---

## Document Transform

### Contextual Compression

Post-retrieval optimization: pass each retrieved document through an LLM that extracts only the relevant portions given the query.

- **Example**: A 3,920-char document compressed to focused excerpts containing only query-relevant information.
- **Library**: `LLMChainExtractor`.
- **Trade-off**: Adds latency and cost proportional to number of retrieved docs. Reduces noise in the final context window. Most valuable when retrieved docs are long and only partially relevant.

---

## Vocabulary Reference

| Term | Definition |
|------|-----------|
| Chunk | A segment of a larger document |
| Embedding | Dense vector representation of text |
| Vector Store | Database optimized for similarity search over embeddings |
| Cosine Similarity | Similarity metric measuring angle between vectors |
| DocStore | Storage for original documents (as opposed to their embeddings) |
| Index | Structure that organizes data for efficient retrieval |
| Knowledge Base | Collection of indexed, searchable information |
| MMR | Maximum Marginal Relevance -- relevance + diversity |
| Reranker | Model that re-scores retrieved results for final ordering |
| Retriever | Component that fetches relevant documents from an index |

---

## Practical Recommendations

1. **Start with Level 2 splitting + Top-K retrieval**. Get a baseline before adding complexity.
2. **Add Multi-Query** when recall is poor -- it's the cheapest quality improvement.
3. **Use Parent Document Retriever** when chunks lack sufficient context for LLM generation.
4. **Add Contextual Compression** when retrieved docs are long and noisy.
5. **Move to Semantic Chunking (Level 4)** when splitting quality is the bottleneck.
6. **Use Multi-Vector indexing** for heterogeneous data (tables, images, mixed formats).
7. **Always evaluate empirically** with RAGAS or equivalent. Optimal strategy is domain-specific.
