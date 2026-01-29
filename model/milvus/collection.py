# collection.py - Milvus Collection Management
# =============================================================================

from rich.console import Console
from pymilvus import connections, utility, Collection

from .config import MILVUS_URI, COLLECTION_NAME

console = Console()


def _get_collection(collection_name: str = None) -> Collection | None:
    """
    Gets the Milvus collection if it exists.
    
    Args:
        collection_name: Optional override for collection name
    
    Returns:
        Collection or None if it doesn't exist
    """
    col_name = collection_name or COLLECTION_NAME
    connections.connect(uri=MILVUS_URI)
    
    if not utility.has_collection(col_name):
        return None
    
    col = Collection(col_name)
    col.load()
    return col


def check_document_exists(
    filename: str,
    doc_type: str = None,
    collection_name: str = None
) -> bool:
    """
    Checks if a document has already been indexed in the collection.
    
    Args:
        filename: Filename to check
        doc_type: Optional document type filter
        collection_name: Optional override for collection name
    
    Returns:
        True if the document already exists, False otherwise
    """
    col = _get_collection(collection_name)
    if col is None:
        return False
    
    try:
        # Build expression
        expr = f'filename == "{filename}"'
        if doc_type:
            expr += f' and doc_type == "{doc_type}"'
        
        result = col.query(
            expr=expr,
            output_fields=["pk"],
            limit=1
        )
        return len(result) > 0
    except Exception:
        return False


def list_indexed_documents(collection_name: str = None) -> list[str]:
    """
    Lists all unique documents indexed in the collection.
    
    Args:
        collection_name: Optional override for collection name
    
    Returns:
        List of unique filenames
    """
    col = _get_collection(collection_name)
    if col is None:
        return []
    
    try:
        result = col.query(
            expr="",
            output_fields=["filename"],
            limit=10000
        )
        
        filenames = set(r["filename"] for r in result)
        return sorted(list(filenames))
        
    except Exception as e:
        console.print(f"[red]Error listing documents: {e}[/red]")
        return []


def count_chunks_by_document(collection_name: str = None) -> dict[str, int]:
    """
    Counts how many chunks each document has in the collection.
    
    Args:
        collection_name: Optional override for collection name
    
    Returns:
        Dict with filename -> chunk count
    """
    col = _get_collection(collection_name)
    if col is None:
        return {}
    
    try:
        result = col.query(
            expr="",
            output_fields=["filename"],
            limit=10000
        )
        
        count = {}
        for r in result:
            filename = r["filename"]
            count[filename] = count.get(filename, 0) + 1
        
        return count
        
    except Exception as e:
        console.print(f"[red]Error counting chunks: {e}[/red]")
        return {}


def remove_document(filename: str, collection_name: str = None) -> bool:
    """
    Removes all chunks from a specific document from the collection.
    
    Args:
        filename: Filename to remove
        collection_name: Optional override for collection name
    
    Returns:
        True if removed successfully, False otherwise
    """
    col = _get_collection(collection_name)
    if col is None:
        console.print(f"[yellow]⚠ Collection does not exist[/yellow]")
        return False
    
    try:
        col.delete(expr=f'filename == "{filename}"')
        console.print(f"[green]✓ Document '{filename}' removed[/green]")
        return True
        
    except Exception as e:
        console.print(f"[red]Error removing document: {e}[/red]")
        return False


def get_collection_stats(collection_name: str = None) -> dict:
    """
    Gets general statistics from the collection.
    
    Args:
        collection_name: Optional override for collection name
    
    Returns:
        Dict with statistics (total_chunks, total_docs, etc)
    """
    col_name = collection_name or COLLECTION_NAME
    col = _get_collection(col_name)
    if col is None:
        return {"exists": False}
    
    try:
        doc_count = count_chunks_by_document(col_name)
        
        return {
            "exists": True,
            "collection_name": col_name,
            "total_chunks": col.num_entities,
            "total_documents": len(doc_count),
            "documents": doc_count
        }
        
    except Exception as e:
        console.print(f"[red]Error getting statistics: {e}[/red]")
        return {"exists": True, "error": str(e)}


def clear_collection(collection_name: str = None) -> bool:
    """
    Removes the entire collection from Milvus.
    
    Args:
        collection_name: Optional override for collection name
    
    Returns:
        True if removed successfully
    """
    col_name = collection_name or COLLECTION_NAME
    connections.connect(uri=MILVUS_URI)
    
    if utility.has_collection(col_name):
        Collection(col_name).drop()
        console.print(f"[green]✓ Collection '{col_name}' removed[/green]")
        return True
    else:
        console.print(f"[yellow]⚠ Collection '{col_name}' does not exist[/yellow]")
        return False
