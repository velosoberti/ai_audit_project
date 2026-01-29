#!/usr/bin/env python3
# main.py - Entry Point for the Audit Application
# ============================================================================
# This script can be run standalone or via the pipeline.
#
# Usage:
#   uv run main.py                                    # Uses config.yaml
#   uv run main.py --document "file.pdf"              # Audit specific file
#   uv run main.py --document "file.pdf" --simple     # Simple mode (no deep agent)
# =============================================================================

import argparse
import asyncio
import sys
from pathlib import Path

from rich.console import Console

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from auditor import run_audit
from output import (
    display_banner,
    display_table,
    display_json,
    save_json,
    save_table_txt,
    display_conclusion
)

console = Console()


async def main():
    """Main function with CLI support."""
    parser = argparse.ArgumentParser(description="Document Audit System")
    parser.add_argument("--document", "-d", type=str, help="Document filename to audit")
    parser.add_argument("--doc-type", "-t", type=str, default="contract", help="Document type")
    parser.add_argument("--simple", action="store_true", help="Use simple mode (no deep agent)")
    parser.add_argument("--no-metrics", action="store_true", help="Hide performance metrics")
    parser.add_argument("--config", "-c", type=str, help="Path to config.yaml")
    
    args = parser.parse_args()
    
    display_banner()
    
    # If document provided via CLI
    if args.document:
        report = await run_audit(
            document_name=args.document,
            doc_type=args.doc_type,
            use_deep_agent=not args.simple,
            display_metrics=not args.no_metrics
        )
        
        display_table(report)
        display_json(report)
        save_json(report)
        save_table_txt(report)
        display_conclusion()
        return
    
    # Otherwise, try to load from config
    try:
        from shared_config import get_config
        from config import update_config
        
        config = get_config(args.config)
        
        # Update runtime config
        update_config(
            milvus_uri=config.milvus.uri,
            collection_name=config.milvus.collection_name,
            output_dir=config.output.directory,
            audit_criteria=[
                {"query": c.query, "confidence": c.confidence}
                for c in config.audit_criteria
            ]
        )
        
        if not config.documents:
            console.print("[red]✗ No documents configured in config.yaml[/red]")
            return
        
        # Audit each document
        for i, doc in enumerate(config.documents, 1):
            console.print(f"\n[bold cyan]Auditing document {i}/{len(config.documents)}:[/bold cyan] {doc.filename}")
            
            report = await run_audit(
                document_name=doc.filename,
                doc_type=doc.doc_type,
                use_deep_agent=config.deep_agent.enabled,
                display_metrics=config.pipeline.display_metrics,
                collection_name=config.milvus.collection_name
            )
            
            display_table(report)
            
            # Save outputs with document-specific names
            if config.output.save_json:
                filename = f"audit_{doc.filename.replace('.pdf', '')}.json"
                save_json(report, filename)
            
            if config.output.save_txt:
                filename = f"audit_{doc.filename.replace('.pdf', '')}.txt"
                save_table_txt(report, filename)
        
        display_conclusion()
        
    except (ImportError, FileNotFoundError) as e:
        console.print(f"[red]✗ {e}[/red]")
        console.print("[yellow]Use --document to audit a specific file or create a config.yaml[/yellow]")


if __name__ == "__main__":
    asyncio.run(main())
