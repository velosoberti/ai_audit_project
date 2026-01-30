# indexer.py - Main Indexing Logic with Hybrid Search
# ==============================================================================
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from .config import COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP, MILVUS_URI, DENSE_DIM, BM25_MODEL_PATH
from .models import IndexedDocument
from .extractor import extract_text_from_pdf
from .chunker import create_chunks_by_page

console = Console()

# =============================================================================
# HYBRID EMBEDDING MODEL INITIALIZATION
# =============================================================================

# BGE-M3 provides both sparse and dense embeddings in a single model
ef_bgem3 = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")


def sparse_to_dict(sparse_array) -> dict[int, float]:
    """Converts sparse array from scipy to Milvus format."""
    coo = sparse_array.tocoo()
    return {int(i): float(v) for i, v in zip(coo.col, coo.data)}


def generate_hybrid_embeddings(texts: list[str]) -> dict:
    """
    Generates hybrid embeddings using BGE-M3:
    - Sparse: Learned sparse embeddings (lexical-like search)
    - Dense: Dense embeddings (semantic search)
    
    BGE-M3 generates both sparse and dense embeddings in a single call.
    """
    # BGE-M3 returns both sparse and dense embeddings
    embeddings = ef_bgem3.encode_documents(texts)
    
    # Extract sparse vectors and convert to dict format
    sparse_vectors = [sparse_to_dict(sp) for sp in embeddings["sparse"]]
    
    # Extract dense vectors
    dense_vectors = embeddings["dense"]
    
    # Ensure it's a list of lists
    if hasattr(dense_vectors, 'tolist'):
        dense_vectors = dense_vectors.tolist()
    
    return {
        "sparse": sparse_vectors,
        "dense": dense_vectors
    }


# =============================================================================
# COLLECTION MANAGEMENT
# =============================================================================

def initialize_collection(
    reset: bool = False,
    collection_name: str = None,
    milvus_uri: str = None,
    dense_dim: int = None
) -> Collection:
    """
    Initializes the Milvus collection with schema for hybrid search.
    
    Args:
        reset: If True, drops existing collection
        collection_name: Override for collection name
        milvus_uri: Override for Milvus URI
        dense_dim: Override for dense vector dimension
    """
    col_name = collection_name or COLLECTION_NAME
    uri = milvus_uri or MILVUS_URI
    dim = dense_dim or DENSE_DIM
    
    connections.connect(uri=uri)

    if reset and utility.has_collection(col_name):
        console.print(f"[yellow]⚠ Removing existing collection: {col_name}[/yellow]")
        Collection(col_name).drop()

    if not utility.has_collection(col_name):
        console.print(f"[cyan]Creating collection: {col_name}[/cyan]")

        fields = [
            FieldSchema(
                name="pk",
                dtype=DataType.VARCHAR,
                is_primary=True,
                auto_id=True,
                max_length=100
            ),
            FieldSchema(
                name="text",
                dtype=DataType.VARCHAR,
                max_length=8000
            ),
            FieldSchema(
                name="filename",
                dtype=DataType.VARCHAR,
                max_length=500
            ),
            FieldSchema(
                name="doc_type",
                dtype=DataType.VARCHAR,
                max_length=100
            ),
            FieldSchema(
                name="page_number",
                dtype=DataType.INT64
            ),
            FieldSchema(
                name="chunk_index",
                dtype=DataType.INT64
            ),
            FieldSchema(
                name="total_chunks",
                dtype=DataType.INT64
            ),
            FieldSchema(
                name="sparse_vector",
                dtype=DataType.SPARSE_FLOAT_VECTOR
            ),
            FieldSchema(
                name="dense_vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=dim
            ),
        ]

        schema = CollectionSchema(
            fields,
            description="Documents for auditing with hybrid search (BGE-M3 sparse + dense)"
        )
        col = Collection(col_name, schema)

        console.print("[cyan]Creating indexes...[/cyan]")
        
        sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
        col.create_index("sparse_vector", sparse_index)
        
        dense_index = {"index_type": "AUTOINDEX", "metric_type": "COSINE"}
        col.create_index("dense_vector", dense_index)

        console.print("[green]✓ Collection created successfully[/green]")
    else:
        col = Collection(col_name)

    col.load()
    return col


def check_document_exists(
    filename: str,
    doc_type: str = None,
    collection_name: str = None,
    milvus_uri: str = None
) -> bool:
    """
    Checks if a document has already been indexed.
    
    Args:
        filename: Filename to check
        doc_type: Document type filter
        collection_name: Override for collection name
        milvus_uri: Override for Milvus URI
    """
    col_name = collection_name or COLLECTION_NAME
    uri = milvus_uri or MILVUS_URI
    
    connections.connect(uri=uri)

    if not utility.has_collection(col_name):
        return False

    col = Collection(col_name)
    col.load()

    # Build expression
    expr = f'filename == "{filename}"'
    if doc_type:
        expr += f' and doc_type == "{doc_type}"'

    results = col.query(
        expr=expr,
        limit=1,
        output_fields=["pk"]
    )

    return len(results) > 0


# =============================================================================
# INDEXING
# =============================================================================

def index_document(
    pdf_path: str,
    doc_type: str = "contract",
    reset_collection: bool = False,
    skip_if_exists: bool = True,
    batch_size: int = 10,
    collection_name: str = None,
    milvus_uri: str = None,
    chunk_size: int = None,
    chunk_overlap: int = None
) -> IndexedDocument | None:
    """
    Processes and indexes a PDF document in Milvus with hybrid search.
    
    Args:
        pdf_path: Path to the PDF file
        doc_type: Document type (contract, addendum, etc)
        reset_collection: If True, clears the collection first
        skip_if_exists: If True, skips if already indexed
        batch_size: Number of chunks to process at once
        collection_name: Override for collection name
        milvus_uri: Override for Milvus URI
        chunk_size: Override for chunk size
        chunk_overlap: Override for chunk overlap
    
    Returns:
        IndexedDocument with results or None if skipped/failed
    """
    col_name = collection_name or COLLECTION_NAME
    uri = milvus_uri or MILVUS_URI
    c_size = chunk_size or CHUNK_SIZE
    c_overlap = chunk_overlap or CHUNK_OVERLAP
    
    filename = Path(pdf_path).name

    console.print(Panel.fit(
        f"[bold]Indexing:[/bold] {filename}\n"
        f"[bold]Type:[/bold] {doc_type}\n"
        f"[bold]Collection:[/bold] {col_name}",
        title="📄 Processing Document",
        border_style="cyan"
    ))

    # ----- STEP 1: Initialize collection -----
    col = initialize_collection(
        reset=reset_collection,
        collection_name=col_name,
        milvus_uri=uri
    )

    # ----- STEP 2: Check if already exists -----
    if skip_if_exists and check_document_exists(
        filename=filename,
        doc_type=doc_type,
        collection_name=col_name,
        milvus_uri=uri
    ):
        console.print(f"[yellow]⚠ Document '{filename}' (type: {doc_type}) already indexed. Skipping...[/yellow]")
        return None

    # ----- STEP 3: Extract text from PDF -----
    try:
        console.print("[bold]1/4[/bold] Extracting text from PDF...")
        pages = extract_text_from_pdf(pdf_path)

        if not pages:
            console.print("[red]✗ No text extracted from PDF[/red]")
            return None

        console.print(f"    [green]✓ {len(pages)} pages extracted[/green]")

    except FileNotFoundError as e:
        console.print(f"[red]✗ {e}[/red]")
        return None
    except Exception as e:
        console.print(f"[red]✗ Error extracting PDF: {e}[/red]")
        return None

    # ----- STEP 4: Create chunks -----
    console.print("[bold]2/4[/bold] Splitting into chunks...")
    chunks = create_chunks_by_page(pages, chunk_size=c_size, chunk_overlap=c_overlap)
    total_chunks = len(chunks)
    console.print(f"    [green]✓ {total_chunks} chunks created[/green]")

    # ----- STEP 5: Generate hybrid embeddings with BGE-M3 -----
    console.print("[bold]3/4[/bold] Generating hybrid embeddings with BGE-M3 (sparse + dense)...")

    data_to_insert = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task("Processing chunks...", total=total_chunks)

        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_texts = [c["text"] for c in batch_chunks]

            batch_embeddings = generate_hybrid_embeddings(batch_texts)

            for j, chunk_info in enumerate(batch_chunks):
                idx = i + j
                data_to_insert.append({
                    "text": chunk_info["text"],
                    "filename": filename,
                    "doc_type": doc_type,
                    "page_number": chunk_info["page_number"],
                    "chunk_index": idx,
                    "total_chunks": total_chunks,
                    "sparse_vector": batch_embeddings["sparse"][j],
                    "dense_vector": batch_embeddings["dense"][j],
                })

            progress.update(task, advance=len(batch_chunks))

    # ----- STEP 6: Insert in Milvus -----
    console.print("[bold]4/4[/bold] Inserting in Milvus...")

    try:
        col.insert(data_to_insert)
        col.flush()
        console.print(f"    [green]✓ {len(data_to_insert)} vectors inserted[/green]")

    except Exception as e:
        console.print(f"[red]✗ Error inserting in Milvus: {e}[/red]")
        return None

    # ----- RESULT -----
    result = IndexedDocument(
        filename=filename,
        doc_type=doc_type,
        total_pages=len(pages),
        total_chunks=total_chunks,
        indexed_chunks=len(data_to_insert)
    )

    console.print(Panel.fit(
        f"[green]✓ Document indexed successfully![/green]\n\n"
        f"[bold]File:[/bold] {result.filename}\n"
        f"[bold]Type:[/bold] {result.doc_type}\n"
        f"[bold]Pages:[/bold] {result.total_pages}\n"
        f"[bold]Chunks:[/bold] {result.indexed_chunks}",
        title="✅ Indexing Complete",
        border_style="green"
    ))

    return result
