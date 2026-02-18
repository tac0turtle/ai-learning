"""Environment-based configuration with sane defaults."""

import os
from pathlib import Path

# Paths
PROJECT_DIR = Path(__file__).parent
DB_PATH = os.environ.get("MEMORY_DB_PATH", str(PROJECT_DIR / "data" / "memory.lance"))
TABLE_NAME = "memories"

# Embedding model
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = 384

# LLM
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-4-20250514")

# Retrieval scoring weights (must sum to 1.0)
WEIGHT_SIMILARITY = 0.5
WEIGHT_RECENCY = 0.3
WEIGHT_IMPORTANCE = 0.2

# Recency decay
RECENCY_HALF_LIFE_DAYS = 30

# Dedup threshold
DEDUP_COSINE_THRESHOLD = 0.92

# MMR
MMR_LAMBDA = 0.7
MMR_OVERSAMPLE = 20
MMR_DEFAULT_K = 10

# Consolidation
CONSOLIDATION_BATCH_SIZE = 50
CONSOLIDATION_AGE_DAYS = 30

# Server
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8001"))
