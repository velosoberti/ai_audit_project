# config.py - Configuration and Client Initialization
# ============================================================================

import os
import sys
import warnings
from pathlib import Path
from transformers import logging as transformers_logging

# Silence tokenizer warnings
transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore", message=".*XLMRobertaTokenizerFast.*")

from dotenv import load_dotenv
from pymilvus import connections, Collection
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from langchain_google_genai import ChatGoogleGenerativeAI

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load .env
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

# =============================================================================
# TRY TO LOAD FROM YAML CONFIG
# =============================================================================

try:
    from shared_config import get_config
    _config = get_config()
    
    MILVUS_URI = _config.milvus.uri
    COLLECTION_NAME = _config.milvus.collection_name
    DENSE_DIM = _config.embedding.dense_dim
    BM25_MODEL_PATH = _config.embedding.bm25_model_path
    OUTPUT_DIR = Path(_config.output.directory)
    
    # Convert criteria from config
    AUDIT_CRITERIA = [
        {"query": c.query, "confidence": c.confidence}
        for c in _config.audit_criteria
    ]
    
    # Possible answers configuration
    POSSIBLE_ANSWERS_ENABLED = _config.possible_answers.enabled
    
    _config_loaded = True
    
except (ImportError, FileNotFoundError):
    # Fallback to defaults/environment
    MILVUS_URI = os.environ.get("MILVUS_URI", "http://127.0.0.1:19530")
    COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "audit_docs_v3")
    DENSE_DIM = int(os.environ.get("DENSE_DIM", "1024"))  # BGE-M3 default
    BM25_MODEL_PATH = os.environ.get("BM25_MODEL_PATH", "./output/bm25_model.json")
    OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./output"))
    
    AUDIT_CRITERIA = [
        {"query": "Is there a registered CNPJ for the Brokerage?", "confidence": 0.8},
        {"query": "Is there a confidentiality or secrecy clause?", "confidence": 0.7},
        {"query": "Does the document mention values, fees, or compensation?", "confidence": 0.7},
        {"query": "Is there a definition of classical music process?", "confidence": 0.6},
        {"query": "Is there mention of the Settlement process?", "confidence": 0.7},
        {"query": "Does the contract mention obligations of the parties?", "confidence": 0.7},
        {"query": "Is there mention of penalties or fines?", "confidence": 0.7},
        {"query": "Does the document have any mention about A5X?", "confidence": 0.8},
    ]
    
    # Possible answers disabled by default in fallback
    POSSIBLE_ANSWERS_ENABLED = False
    
    _config_loaded = False

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# MILVUS CONNECTION
# =============================================================================

connections.connect(uri=MILVUS_URI)

# =============================================================================
# EMBEDDING MODELS
# =============================================================================

# BGE-M3 provides both sparse and dense embeddings
ef_bgem3 = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")

# For query embedding, we use the same BGE-M3 model
ef_dense = ef_bgem3

# For sparse queries, we also use BGE-M3
ef_sparse = ef_bgem3

# =============================================================================
# LANGUAGE MODEL (LLM)
# =============================================================================

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


# =============================================================================
# HELPER FUNCTIONS FOR RUNTIME CONFIG UPDATES
# =============================================================================

def update_config(
    milvus_uri: str = None,
    collection_name: str = None,
    output_dir: str = None,
    audit_criteria: list = None
):
    """
    Updates configuration at runtime.
    
    Args:
        milvus_uri: New Milvus URI
        collection_name: New collection name
        output_dir: New output directory
        audit_criteria: New audit criteria list
    """
    global MILVUS_URI, COLLECTION_NAME, OUTPUT_DIR, AUDIT_CRITERIA
    
    if milvus_uri:
        MILVUS_URI = milvus_uri
        connections.disconnect("default")
        connections.connect(uri=milvus_uri)
    
    if collection_name:
        COLLECTION_NAME = collection_name
    
    if output_dir:
        OUTPUT_DIR = Path(output_dir)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if audit_criteria:
        AUDIT_CRITERIA = audit_criteria
