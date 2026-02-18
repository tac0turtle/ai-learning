"""FastAPI app serving the agent memory UI and API."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import embeddings
import reader
import store
import writer
from models import SearchResult


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload embedding model on startup
    embeddings._get_model()
    yield


app = FastAPI(title="Agent Memory", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -- Request/Response schemas --

class AddRequest(BaseModel):
    text: str = Field(..., min_length=1)


class AddResponse(BaseModel):
    ids: list[str]
    count: int


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(default=10, ge=1, le=50)


class SearchResultItem(BaseModel):
    text: str
    memory_type: str
    importance: float
    topic: str
    entities: list[str]
    combined_score: float
    raw_similarity: float
    recency_score: float
    created_at: str


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    count: int


class StatsResponse(BaseModel):
    total: int
    by_type: dict[str, int]


# -- API endpoints --

@app.post("/api/add", response_model=AddResponse)
def api_add(req: AddRequest):
    ids = writer.write(req.text)
    return AddResponse(ids=ids, count=len(ids))


@app.post("/api/search", response_model=SearchResponse)
def api_search(req: SearchRequest):
    results = reader.search(req.query, k=req.k)
    items = [
        SearchResultItem(
            text=r.memory.text,
            memory_type=r.memory.memory_type.value,
            importance=r.memory.importance,
            topic=r.memory.topic,
            entities=r.memory.entities,
            combined_score=r.combined_score,
            raw_similarity=r.raw_similarity,
            recency_score=r.recency_score,
            created_at=r.memory.created_at.isoformat()
            if hasattr(r.memory.created_at, "isoformat")
            else str(r.memory.created_at),
        )
        for r in results
    ]
    return SearchResponse(results=items, count=len(items))


@app.get("/api/stats", response_model=StatsResponse)
def api_stats():
    return StatsResponse(total=store.count(), by_type=store.count_by_type())


@app.post("/api/consolidate")
def api_consolidate():
    import consolidator as cons
    n = cons.consolidate()
    return {"summaries_created": n}


@app.post("/api/clear")
def api_clear():
    store.delete_all()
    return {"status": "cleared"}


# Serve static files at root — mount AFTER API routes
static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    import config

    uvicorn.run(app, host=config.HOST, port=config.PORT)
