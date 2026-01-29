# config.py - Indexer Configuration with Hybrid Search


import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

# Try to load from YAML config, fall back to defaults
try:
    from shared_config import get_config
    _config = get_config()
    
    MILVUS_URI = _config.milvus.uri
    COLLECTION_NAME = _config.milvus.collection_name
    CHUNK_SIZE = _config.chunking.chunk_size
    CHUNK_OVERLAP = _config.chunking.chunk_overlap
    DENSE_DIM = _config.embedding.dense_dim
    
except (ImportError, FileNotFoundError):
    # Fallback to defaults if no config.yaml
    MILVUS_URI = os.environ.get("MILVUS_URI", "http://127.0.0.1:19530")
    COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "audit_docs_v3")
    CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "100"))
    DENSE_DIM = int(os.environ.get("DENSE_DIM", "3072"))
