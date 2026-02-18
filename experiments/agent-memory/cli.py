"""Typer CLI: add, search, consolidate, evaluate, stats."""

import json
import sys

import typer

app = typer.Typer(help="Agent Memory System — RAG + vector DB experiment")


@app.command()
def add(text: str = typer.Argument(..., help="Conversation text to extract memories from")):
    """Extract and store memories from conversation text."""
    import writer
    typer.echo(f"Extracting memories from text ({len(text)} chars)...")
    ids = writer.write(text)
    if ids:
        typer.echo(f"Stored {len(ids)} memories: {', '.join(ids)}")
    else:
        typer.echo("No new memories extracted (empty or all duplicates).")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    k: int = typer.Option(10, "--top", "-k", help="Number of results"),
):
    """Search memories with time-weighted scoring and MMR diversity."""
    import reader
    results = reader.search(query, k=k)
    if not results:
        typer.echo("No memories found.")
        return
    typer.echo(f"Found {len(results)} results:\n")
    for i, r in enumerate(results, 1):
        typer.echo(f"  {i}. [{r.combined_score:.3f}] {r.memory.text}")
        typer.echo(f"     sim={r.raw_similarity:.3f}  recency={r.recency_score:.3f}  "
                   f"importance={r.importance_score:.2f}  type={r.memory.memory_type.value}")
        if r.memory.entities:
            typer.echo(f"     entities: {', '.join(r.memory.entities)}")
        typer.echo()


@app.command()
def consolidate():
    """Consolidate old memories into summaries."""
    import consolidator as cons
    typer.echo("Running consolidation...")
    n = cons.consolidate()
    typer.echo(f"Created {n} summary memories.")


@app.command()
def evaluate(
    data_file: str = typer.Argument("eval_data/basic.json", help="Path to evaluation dataset"),
):
    """Run evaluation metrics on a labeled dataset."""
    import evaluator
    results = evaluator.evaluate(data_file)
    typer.echo(f"\nEvaluation results ({len(results['queries'])} queries):")
    typer.echo(f"  Recall@10:    {results['recall_at_k']:.3f}")
    typer.echo(f"  Precision@10: {results['precision_at_k']:.3f}")
    typer.echo(f"  MRR:          {results['mrr']:.3f}")


@app.command()
def stats():
    """Show memory store statistics."""
    import store
    total = store.count()
    by_type = store.count_by_type()
    typer.echo(f"Total memories: {total}")
    if by_type:
        typer.echo("By type:")
        for mt, cnt in sorted(by_type.items()):
            typer.echo(f"  {mt}: {cnt}")


@app.command()
def clear(
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Delete all memories."""
    if not confirm:
        typer.confirm("Delete all memories?", abort=True)
    import store
    store.delete_all()
    typer.echo("All memories deleted.")


if __name__ == "__main__":
    app()
