#!/usr/bin/env python3
# main.py - Entry Point for the PDF Indexing System
# =============================================================================
# This script can be run standalone or via the pipeline.
#
# Usage:
#   uv run main.py                           # Uses config.yaml
#   uv run main.py --pdf /path/to/file.pdf   # Index specific file
#   uv run main.py --list                    # List indexed documents
# =============================================================================

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from indexer import index_document
from collection import count_chunks_by_document, get_collection_stats

console = Console()


def list_documents():
    """Lists all indexed documents."""
    stats = get_collection_stats()
    
    if not stats.get("exists"):
        console.print("[yellow]⚠ Collection does not exist[/yellow]")
        return
    
    table = Table(title="📚 Indexed Documents")
    table.add_column("Filename", style="cyan")
    table.add_column("Chunks", justify="right", style="green")
    
    for filename, chunks in stats.get("documents", {}).items():
        table.add_row(filename, str(chunks))
    
    console.print(table)
    console.print(f"\n[bold]Total:[/bold] {stats.get('total_documents', 0)} documents, {stats.get('total_chunks', 0)} chunks")


def main():
    """Main function with CLI support."""
    parser = argparse.ArgumentParser(description="PDF Indexing System")
    parser.add_argument("--pdf", type=str, help="Path to PDF file to index")
    parser.add_argument("--doc-type", type=str, default="contract", help="Document type")
    parser.add_argument("--reset", action="store_true", help="Reset collection before indexing")
    parser.add_argument("--list", action="store_true", help="List indexed documents")
    parser.add_argument("--config", type=str, help="Path to config.yaml")
    
    args = parser.parse_args()
    
    # List mode
    if args.list:
        list_documents()
        return
    
    # If PDF provided via CLI, use it
    if args.pdf:
        result = index_document(
            pdf_path=args.pdf,
            doc_type=args.doc_type,
            reset_collection=args.reset,
            skip_if_exists=True
        )
        if result:
            list_documents()
        return
    
    # Otherwise, try to load from config
    try:
        from shared_config import get_config
        config = get_config(args.config)
        
        if not config.documents:
            console.print("[red]✗ No documents configured in config.yaml[/red]")
            return
        
        for i, doc in enumerate(config.documents):
            console.print(f"\n[cyan]Processing document {i+1}/{len(config.documents)}[/cyan]")
            
            result = index_document(
                pdf_path=doc.path,
                doc_type=doc.doc_type,
                reset_collection=doc.reset_collection and i == 0,
                skip_if_exists=doc.skip_if_indexed,
                collection_name=config.milvus.collection_name,
                milvus_uri=config.milvus.uri,
                chunk_size=config.chunking.chunk_size,
                chunk_overlap=config.chunking.chunk_overlap
            )
        
        console.print("\n[bold]Final state:[/bold]")
        list_documents()
        
    except (ImportError, FileNotFoundError) as e:
        console.print(f"[red]✗ {e}[/red]")
        console.print("[yellow]Use --pdf to index a specific file or create a config.yaml[/yellow]")


if __name__ == "__main__":
    main()
