# models.py - Pydantic Models for the Indexer
# ============================================================================

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """
    Metadata associated with each text chunk.
    
    This metadata is stored in Milvus as dynamic fields
    and can be used to filter searches.
    """
    filename: str = Field(description="Original PDF filename")
    doc_type: str = Field(description="Document type (contract, addendum, proposal, etc)")
    page_number: int = Field(description="Source page number (1-indexed)")
    chunk_index: int = Field(description="Chunk index within the document")
    total_chunks: int = Field(description="Total chunks in the document")


class IndexedDocument(BaseModel):
    """
    Represents a processed document ready for indexing.
    """
    filename: str
    doc_type: str
    total_pages: int
    total_chunks: int
    indexed_chunks: int = 0