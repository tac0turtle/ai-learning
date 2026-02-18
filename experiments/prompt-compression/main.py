"""FastAPI app serving the prompt compression UI and API."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import compressor


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload model on startup
    compressor._get_compressor()
    yield


app = FastAPI(title="Prompt Compressor", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class CompressRequest(BaseModel):
    prompt: str
    rate: float = Field(default=0.33, ge=0.05, le=0.95)
    force_tokens: list[str] = Field(default_factory=lambda: ["\n", ".", "?"])


class CompressResponse(BaseModel):
    compressed: str
    original_tokens: int
    compressed_tokens: int
    ratio: float
    time_ms: float


@app.post("/api/compress", response_model=CompressResponse)
def compress_prompt(req: CompressRequest):
    result = compressor.compress(
        text=req.prompt,
        rate=req.rate,
        force_tokens=req.force_tokens,
    )
    return CompressResponse(
        compressed=result.compressed,
        original_tokens=result.original_tokens,
        compressed_tokens=result.compressed_tokens,
        ratio=result.ratio,
        time_ms=result.time_ms,
    )


# Serve static files (frontend) at root — mount AFTER API routes
static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
