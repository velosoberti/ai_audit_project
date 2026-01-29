#!/usr/bin/env python3
# run_pipeline.py - Main Pipeline Orchestrator
# ============================================================================
# This script orchestrates the entire audit pipeline:
# 1. Loads configuration from YAML
# 2. Checks if documents are already indexed
# 3. Indexes documents in Milvus if necessary
# 4. Executes the audit
# 5. Saves the results
#
# Usage:
#   uv run run_pipeline.py                    # Uses default config.yaml
#   uv run run_pipeline.py --config my.yaml   # Uses specific config
#   uv run run_pipeline.py --index-only       # Index only, do not audit
#   uv run run_pipeline.py --audit-only       # Audit only, do not index
# ============================================================================

import argparse
import asyncio
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from shared_config import get_config, Config, DocumentConfig

console = Console()


def display_banner():
    """Displays the pipeline banner."""
    console.print(Panel.fit(
        "[bold]Audit Pipeline[/bold]\n"
        "Unified document indexing and auditing",
        title="🔍 AI Auditor Pipeline",
        border_style="cyan"
    ))


def display_config_summary(config: Config):
    """Displays a summary of the loaded configuration."""
    table = Table(title="📋 Configuration Summary")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Milvus URI", config.milvus.uri)
    table.add_row("Collection", config.milvus.collection_name)
    table.add_row("Documents", str(len(config.documents)))
    table.add_row("Criteria", str(len(config.audit_criteria)))
    table.add_row("Deep Agent", "Enabled" if config.deep_agent.enabled else "Disabled")
    table.add_row("Output Dir", config.output.directory)
    
    console.print(table)
    console.print()


def check_document_indexed(config: Config, doc: DocumentConfig) -> bool:
    """
    Checks if a document is already indexed in Milvus.
    
    Args:
        config: Pipeline configuration
        doc: Document configuration
    
    Returns:
        True if document is indexed, False otherwise
    """
    try:
        from pymilvus import connections, utility, Collection
        
        connections.connect(uri=config.milvus.uri)
        
        if not utility.has_collection(config.milvus.collection_name):
            return False
        
        col = Collection(config.milvus.collection_name)
        col.load()
        
        results = col.query(
            expr=f'filename == "{doc.filename}" and doc_type == "{doc.doc_type}"',
            output_fields=["pk"],
            limit=1
        )
        
        return len(results) > 0
        
    except Exception as e:
        console.print(f"[yellow]⚠ Error checking index: {e}[/yellow]")
        return False


def index_document(config: Config, doc: DocumentConfig, reset: bool = False) -> bool:
    """
    Indexes a document in Milvus.
    
    Args:
        config: Pipeline configuration
        doc: Document configuration
        reset: If True, resets the collection before indexing
    
    Returns:
        True if successful, False otherwise
    """
    from model.milvus.indexer import index_document as milvus_index
    
    try:
        result = milvus_index(
            pdf_path=doc.path,
            doc_type=doc.doc_type,
            reset_collection=reset,
            skip_if_exists=doc.skip_if_indexed,
            collection_name=config.milvus.collection_name,
            milvus_uri=config.milvus.uri,
            chunk_size=config.chunking.chunk_size,
            chunk_overlap=config.chunking.chunk_overlap
        )
        return result is not None
    except Exception as e:
        console.print(f"[red]✗ Error indexing {doc.filename}: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False


async def run_audit_for_document(config: Config, doc: DocumentConfig):
    """
    Runs the audit for a specific document.
    
    Args:
        config: Pipeline configuration
        doc: Document configuration
    
    Returns:
        AuditReport or None if failed
    """
    from model.application.auditor import run_audit
    from model.application.output import (
        display_table,
        display_json,
        save_json,
        save_table_txt
    )
    from model.application.config import update_config
    
    # Update application config
    update_config(
        milvus_uri=config.milvus.uri,
        collection_name=config.milvus.collection_name,
        output_dir=config.output.directory,
        audit_criteria=[
            {"query": c.query, "confidence": c.confidence}
            for c in config.audit_criteria
        ]
    )
    
    try:
        report = await run_audit(
            document_name=doc.filename,
            doc_type=doc.doc_type,
            use_deep_agent=config.deep_agent.enabled,
            display_metrics=config.pipeline.display_metrics,
            collection_name=config.milvus.collection_name
        )
        
        # Display results
        display_table(report)
        
        # Save outputs
        if config.output.save_json:
            filename = f"audit_{doc.filename.replace('.pdf', '')}.json"
            save_json(report, filename)
        
        if config.output.save_txt:
            filename = f"audit_{doc.filename.replace('.pdf', '')}.txt"
            save_table_txt(report, filename)
        
        return report
        
    except Exception as e:
        console.print(f"[red]✗ Error auditing {doc.filename}: {e}[/red]")
        import traceback
        traceback.print_exc()
        return None


async def run_pipeline(config_path: str = None, index_only: bool = False, audit_only: bool = False):
    """
    Main pipeline execution.
    
    Args:
        config_path: Path to config.yaml
        index_only: If True, only index documents
        audit_only: If True, only run audit (skip indexing)
    """
    display_banner()
    
    # Load configuration
    try:
        config = get_config(config_path)
        console.print(f"[green]✓ Configuration loaded[/green]")
    except FileNotFoundError as e:
        console.print(f"[red]✗ {e}[/red]")
        sys.exit(1)
    
    display_config_summary(config)
    
    if not config.documents:
        console.print("[red]✗ No documents configured in config.yaml[/red]")
        sys.exit(1)
    
    # Process each document
    for i, doc in enumerate(config.documents, 1):
        console.print(Panel.fit(
            f"[bold]Document {i}/{len(config.documents)}[/bold]\n"
            f"File: {doc.filename}\n"
            f"Type: {doc.doc_type}",
            title="📄 Processing",
            border_style="blue"
        ))
        
        # Check if file exists
        if not Path(doc.path).exists():
            console.print(f"[red]✗ File not found: {doc.path}[/red]")
            continue
        
        # ----- INDEXING PHASE -----
        if not audit_only and not config.pipeline.skip_indexing:
            is_indexed = check_document_indexed(config, doc)
            
            if is_indexed and doc.skip_if_indexed and not config.pipeline.force_reindex:
                console.print(f"[green]✓ Document already indexed, skipping...[/green]")
            else:
                console.print(f"[cyan]→ Indexing document...[/cyan]")
                
                reset = doc.reset_collection or (i == 1 and config.pipeline.force_reindex)
                success = index_document(config, doc, reset=reset)
                
                if not success:
                    console.print(f"[red]✗ Failed to index {doc.filename}[/red]")
                    continue
                
                console.print(f"[green]✓ Document indexed successfully[/green]")
        
        # ----- AUDIT PHASE -----
        if not index_only:
            console.print(f"[cyan]→ Running audit...[/cyan]")
            report = await run_audit_for_document(config, doc)
            
            if report:
                console.print(f"[green]✓ Audit completed: {report.compliance_rate}% compliance[/green]")
        
        console.print()
    
    console.print(Panel.fit(
        "[bold green]Pipeline completed![/bold green]",
        title="✅ Done",
        border_style="green"
    ))


def main():
    """Entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Audit Pipeline - Document indexing and auditing"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to config.yaml file"
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Only index documents, skip auditing"
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Only run audit, skip indexing"
    )
    
    args = parser.parse_args()
    
    if args.index_only and args.audit_only:
        console.print("[red]✗ Cannot use --index-only and --audit-only together[/red]")
        sys.exit(1)
    
    asyncio.run(run_pipeline(
        config_path=args.config,
        index_only=args.index_only,
        audit_only=args.audit_only
    ))


if __name__ == "__main__":
    main()
