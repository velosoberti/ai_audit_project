# auditor.py - Main Audit Logic
# =============================================================================

import asyncio
from rich.console import Console

from .config import AUDIT_CRITERIA
from .models import AuditReport
from .deep_agent import DeepResearchAgent
from .metrics import MetricsTracker

console = Console()


async def run_audit(
    document_name: str,
    doc_type: str | None = None,
    use_deep_agent: bool = True,
    display_metrics: bool = True,
    audit_criteria: list = None,
    collection_name: str = None
) -> AuditReport:
    """
    Runs the complete audit of a document.
    
    Args:
        document_name: PDF filename
        doc_type: Document type for filtering
        use_deep_agent: If True, uses iterative deep search
        display_metrics: If True, displays performance metrics
        audit_criteria: Override for audit criteria list
        collection_name: Override for collection name
    
    Returns:
        AuditReport with all results
    """
    # Use provided criteria or fall back to config
    criteria = audit_criteria or AUDIT_CRITERIA
    
    tracker = MetricsTracker()
    tracker.start_audit()
    
    agent = DeepResearchAgent(
        filename=document_name,
        doc_type=doc_type,
        collection_name=collection_name
    )
    
    with console.status("[bold green]Running audit...") as status:
        
        if use_deep_agent:
            tasks = []
            for i, criterion_config in enumerate(criteria, 1):
                if isinstance(criterion_config, dict):
                    criterion = criterion_config["query"]
                    confidence = criterion_config.get("confidence", 0.7)
                else:
                    criterion = criterion_config
                    confidence = 0.7
                console.print(f"\n[cyan]Criterion {i}:[/cyan] {criterion[:50]}...")
                tracker.start_criterion()
                tasks.append(agent.search(criterion, min_confidence=confidence))
            
            status.update(f"[bold green]Evaluating {len(criteria)} criteria in parallel...")
            results = await asyncio.gather(*tasks)
        else:
            from .retriever import search_relevant_context
            from .evaluator import evaluate_criterion
            
            results = []
            for i, criterion_config in enumerate(criteria, 1):
                if isinstance(criterion_config, dict):
                    criterion = criterion_config["query"]
                else:
                    criterion = criterion_config
                status.update(f"[bold green]Evaluating criterion {i}/{len(criteria)}...")
                console.print(f"\n[cyan]Criterion {i}:[/cyan] {criterion[:50]}...")
                
                tracker.start_criterion()
                context, pages = await search_relevant_context(
                    criterion=criterion,
                    filename=document_name,
                    doc_type=doc_type
                )
                result = evaluate_criterion(criterion, context, pages)
                results.append(result)
                
                tracker.finish_criterion(
                    criterion=criterion,
                    attempts=1,
                    confidence=result.confidence
                )
    
    tracker.finish_audit()
    
    for i, result in enumerate(results):
        criterion_config = criteria[i]
        if isinstance(criterion_config, dict):
            criterion_query = criterion_config["query"]
        else:
            criterion_query = criterion_config
        tracker.finish_criterion(
            criterion=criterion_query,
            attempts=1,
            confidence=result.confidence
        )
    
    if display_metrics:
        tracker.get_metrics().display_summary()
    
    present = sum(1 for r in results if r.status == "PRESENT")
    absent = sum(1 for r in results if r.status == "ABSENT")
    total = len(results)
    
    total_valid = present + absent
    rate = round((present / total_valid) * 100, 2) if total_valid > 0 else 0
    
    return AuditReport(
        document=document_name,
        total_criteria=total,
        criteria_present=present,
        criteria_absent=absent,
        compliance_rate=rate,
        results=results
    )
