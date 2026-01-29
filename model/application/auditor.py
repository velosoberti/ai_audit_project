# auditor.py - Main Audit Logic
# ============================================================================

import asyncio
from pathlib import Path
from rich.console import Console

from .config import AUDIT_CRITERIA, POSSIBLE_ANSWERS_ENABLED
from .models import AuditReport
from .deep_agent import DeepResearchAgent
from .metrics import MetricsTracker
from .possible_answer_models import PossibleAnswer

console = Console()


async def run_audit(
    document_name: str,
    doc_type: str | None = None,
    use_deep_agent: bool = True,
    display_metrics: bool = True,
    audit_criteria: list = None,
    collection_name: str = None,
    pdf_path: str | None = None,
    use_possible_answers: bool | None = None
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
        pdf_path: Path to PDF file (required when use_possible_answers is True)
        use_possible_answers: Override for possible answers feature (defaults to config)
    
    Returns:
        AuditReport with all results
    """
    # Use provided criteria or fall back to config
    criteria = audit_criteria or AUDIT_CRITERIA
    
    # Determine if possible answers feature is enabled
    enable_possible_answers = (
        use_possible_answers if use_possible_answers is not None 
        else POSSIBLE_ANSWERS_ENABLED
    )
    
    tracker = MetricsTracker()
    tracker.start_audit()
    
    # Generate possible answers if feature is enabled
    possible_answers_cache: dict[str, PossibleAnswer] = {}
    
    if enable_possible_answers:
        possible_answers_cache = await _generate_possible_answers(
            criteria=criteria,
            pdf_path=pdf_path,
            document_name=document_name
        )
    
    agent = DeepResearchAgent(
        filename=document_name,
        doc_type=doc_type,
        collection_name=collection_name,
        possible_answers=possible_answers_cache if enable_possible_answers else None
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
            from .evaluator import evaluate_criterion, evaluate_criterion_enhanced
            from .enhanced_retriever import search_with_possible_answer
            
            results = []
            for i, criterion_config in enumerate(criteria, 1):
                if isinstance(criterion_config, dict):
                    criterion = criterion_config["query"]
                else:
                    criterion = criterion_config
                status.update(f"[bold green]Evaluating criterion {i}/{len(criteria)}...")
                console.print(f"\n[cyan]Criterion {i}:[/cyan] {criterion[:50]}...")
                
                tracker.start_criterion()
                
                # Get possible answer for this criterion if available
                possible_answer = possible_answers_cache.get(criterion)
                
                if enable_possible_answers and possible_answer:
                    # Use enhanced retriever with possible answer
                    context, pages = await search_with_possible_answer(
                        criterion=criterion,
                        possible_answer=possible_answer,
                        filename=document_name,
                        doc_type=doc_type
                    )
                    # Use enhanced evaluator
                    result = evaluate_criterion_enhanced(criterion, context, pages, possible_answer)
                else:
                    # Fall back to standard retrieval and evaluation
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


async def _generate_possible_answers(
    criteria: list,
    pdf_path: str | None,
    document_name: str
) -> dict[str, PossibleAnswer]:
    """
    Generates possible answers for all criteria before evaluation.
    
    Args:
        criteria: List of audit criteria (can be strings or dicts with 'query' key)
        pdf_path: Path to the PDF file
        document_name: PDF filename (used to find PDF if pdf_path not provided)
    
    Returns:
        Dict mapping criterion query to PossibleAnswer
    """
    from .raw_extractor import RawPDFExtractor, PDFExtractionError
    from .possible_answer_generator import PossibleAnswerGenerator
    
    # Resolve PDF path
    resolved_path = pdf_path
    if not resolved_path:
        # Try common locations
        possible_paths = [
            Path(f"pdfs/{document_name}"),
            Path(document_name),
            Path(f"./pdfs/{document_name}"),
        ]
        for p in possible_paths:
            if p.exists():
                resolved_path = str(p)
                break
    
    if not resolved_path:
        console.print(
            f"[yellow]Warning: Could not find PDF file for possible answer generation. "
            f"Continuing without possible answers.[/yellow]"
        )
        return {}
    
    # Extract raw PDF content
    try:
        extractor = RawPDFExtractor()
        pdf_content = extractor.extract_full_text(resolved_path)
        console.print(
            f"[green]Extracted {pdf_content.total_pages} pages from PDF for possible answer generation[/green]"
        )
    except (FileNotFoundError, PDFExtractionError) as e:
        console.print(
            f"[yellow]Warning: Failed to extract PDF content: {str(e)[:100]}. "
            f"Continuing without possible answers.[/yellow]"
        )
        return {}
    
    # Extract criterion queries from criteria list
    criterion_queries = []
    for criterion_config in criteria:
        if isinstance(criterion_config, dict):
            criterion_queries.append(criterion_config["query"])
        else:
            criterion_queries.append(criterion_config)
    
    # Generate possible answers
    try:
        generator = PossibleAnswerGenerator()
        console.print(f"[cyan]Generating possible answers for {len(criterion_queries)} criteria...[/cyan]")
        
        possible_answers = await generator.generate_answers_batch(
            criteria=criterion_queries,
            pdf_content=pdf_content
        )
        
        # Log summary
        found_count = sum(1 for pa in possible_answers.values() if pa.found)
        console.print(
            f"[green]Generated possible answers: {found_count}/{len(criterion_queries)} criteria have hints[/green]"
        )
        
        return possible_answers
        
    except Exception as e:
        console.print(
            f"[yellow]Warning: Failed to generate possible answers: {str(e)[:100]}. "
            f"Continuing without possible answers.[/yellow]"
        )
        return {}
