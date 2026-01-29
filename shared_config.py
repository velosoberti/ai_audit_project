# shared_config.py - Shared Configuration Loader
# =============================================================================
# This module loads the YAML configuration and makes it available to all 
# other modules in the system.
# =============================================================================

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MilvusConfig:
    uri: str = "http://127.0.0.1:19530"
    collection_name: str = "audit_docs_v3"


@dataclass
class ChunkingConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 100


@dataclass
class EmbeddingConfig:
    dense_dim: int = 3072
    sparse_model: str = "BAAI/bge-m3"
    dense_model: str = "gemini-embedding-001"


@dataclass
class LLMConfig:
    model: str = "gemini-2.0-flash"
    temperature: float = 0


@dataclass
class OutputConfig:
    directory: str = "./output"
    save_json: bool = True
    save_txt: bool = True


@dataclass
class DocumentConfig:
    path: str
    doc_type: str
    skip_if_indexed: bool = True
    reset_collection: bool = False
    
    @property
    def filename(self) -> str:
        return Path(self.path).name


@dataclass
class CriterionConfig:
    query: str
    confidence: float = 0.7


@dataclass
class DeepAgentConfig:
    enabled: bool = True
    max_attempts: int = 3
    min_confidence: float = 0.7


@dataclass
class PipelineConfig:
    force_reindex: bool = False
    display_metrics: bool = True
    skip_indexing: bool = False


@dataclass
class Config:
    """Main configuration class that holds all settings."""
    milvus: MilvusConfig = field(default_factory=MilvusConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    documents: list[DocumentConfig] = field(default_factory=list)
    audit_criteria: list[CriterionConfig] = field(default_factory=list)
    deep_agent: DeepAgentConfig = field(default_factory=DeepAgentConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Loads configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml. If None, searches in:
            1. CONFIG_PATH environment variable
            2. ./config.yaml
            3. ../config.yaml
            4. ../../config.yaml
    
    Returns:
        Config object with all settings
    """
    if config_path is None:
        config_path = os.environ.get("CONFIG_PATH")
    
    if config_path is None:
        # Search for config.yaml in common locations
        search_paths = [
            Path("config.yaml"),
            Path("../config.yaml"),
            Path("../../config.yaml"),
            Path(__file__).parent / "config.yaml",
            Path(__file__).parent.parent / "config.yaml",
            Path(__file__).parent.parent.parent / "config.yaml",
        ]
        
        for path in search_paths:
            if path.exists():
                config_path = str(path)
                break
    
    if config_path is None or not Path(config_path).exists():
        raise FileNotFoundError(
            "config.yaml not found. Create one or set CONFIG_PATH environment variable."
        )
    
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    # Parse milvus config
    milvus_data = data.get("milvus", {})
    milvus = MilvusConfig(
        uri=milvus_data.get("uri", "http://127.0.0.1:19530"),
        collection_name=milvus_data.get("collection_name", "audit_docs_v3")
    )
    
    # Parse chunking config
    chunking_data = data.get("chunking", {})
    chunking = ChunkingConfig(
        chunk_size=chunking_data.get("chunk_size", 1000),
        chunk_overlap=chunking_data.get("chunk_overlap", 100)
    )
    
    # Parse embedding config
    embedding_data = data.get("embedding", {})
    embedding = EmbeddingConfig(
        dense_dim=embedding_data.get("dense_dim", 3072),
        sparse_model=embedding_data.get("sparse_model", "BAAI/bge-m3"),
        dense_model=embedding_data.get("dense_model", "gemini-embedding-001")
    )
    
    # Parse LLM config
    llm_data = data.get("llm", {})
    llm = LLMConfig(
        model=llm_data.get("model", "gemini-2.0-flash"),
        temperature=llm_data.get("temperature", 0)
    )
    
    # Parse output config
    output_data = data.get("output", {})
    output = OutputConfig(
        directory=output_data.get("directory", "./output"),
        save_json=output_data.get("save_json", True),
        save_txt=output_data.get("save_txt", True)
    )
    
    # Parse documents
    documents = []
    for doc in data.get("documents", []):
        documents.append(DocumentConfig(
            path=doc["path"],
            doc_type=doc["doc_type"],
            skip_if_indexed=doc.get("skip_if_indexed", True),
            reset_collection=doc.get("reset_collection", False)
        ))
    
    # Parse audit criteria
    audit_criteria = []
    for criterion in data.get("audit_criteria", []):
        if isinstance(criterion, str):
            audit_criteria.append(CriterionConfig(query=criterion))
        else:
            audit_criteria.append(CriterionConfig(
                query=criterion["query"],
                confidence=criterion.get("confidence", 0.7)
            ))
    
    # Parse deep agent config
    deep_agent_data = data.get("deep_agent", {})
    deep_agent = DeepAgentConfig(
        enabled=deep_agent_data.get("enabled", True),
        max_attempts=deep_agent_data.get("max_attempts", 3),
        min_confidence=deep_agent_data.get("min_confidence", 0.7)
    )
    
    # Parse pipeline config
    pipeline_data = data.get("pipeline", {})
    pipeline = PipelineConfig(
        force_reindex=pipeline_data.get("force_reindex", False),
        display_metrics=pipeline_data.get("display_metrics", True),
        skip_indexing=pipeline_data.get("skip_indexing", False)
    )
    
    return Config(
        milvus=milvus,
        chunking=chunking,
        embedding=embedding,
        llm=llm,
        output=output,
        documents=documents,
        audit_criteria=audit_criteria,
        deep_agent=deep_agent,
        pipeline=pipeline
    )


# Global config instance (lazy loaded)
_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Gets the global config instance, loading it if necessary.
    
    Args:
        config_path: Path to config.yaml (only used on first call)
    
    Returns:
        Config object
    """
    global _config
    if _config is None:
        _config = load_config(config_path)
    return _config


def reload_config(config_path: Optional[str] = None) -> Config:
    """
    Forces a reload of the configuration.
    
    Args:
        config_path: Path to config.yaml
    
    Returns:
        New Config object
    """
    global _config
    _config = load_config(config_path)
    return _config
