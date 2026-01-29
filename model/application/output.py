# output.py - Display and Export Functions
# =============================================================================

from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .models import AuditReport
from .config import OUTPUT_DIR

console = Console()


def display_banner() -> None:
    """Displays the system's initial banner."""
    console.print(Panel.fit(
        "[bold]Document Audit System[/bold]\n"
        "Using Hybrid RAG with Milvus + BGE-M3",
        title="🔍 AI Auditor",
        border_style="cyan"
    ))


def display_table(report: AuditReport) -> None:
    """Displays the report in table format."""
    
    # Summary panel
    console.print(Panel.fit(
        f"[bold]Document:[/bold] {report.document}\n"
        f"[green]✓ Present:[/green] {report.criteria_present} | "
        f"[red]✗ Absent:[/red] {report.criteria_absent} | "
        f"[blue]Compliance Rate:[/blue] {report.compliance_rate}%",
        title="📋 Audit Report",
        border_style="blue"
    ))
    
    # Results table
    table = Table(
        show_header=True,
        header_style="bold magenta",
        show_lines=True
    )
    
    table.add_column("#", style="dim", width=3, justify="center")
    table.add_column("Status", justify="center", width=12)
    table.add_column("Criterion", width=40)
    table.add_column("Evidence", width=45)
    table.add_column("Pages", justify="center", width=8)
    table.add_column("Conf.", justify="center", width=6)
    
    for i, result in enumerate(report.results, 1):
        
        # Format status
        if result.status == "PRESENT":
            status_fmt = "[green]✓ PRESENT[/green]"
        elif result.status == "ABSENT":
            status_fmt = "[red]✗ ABSENT[/red]"
        else:
            status_fmt = "[yellow]⚠ ERROR[/yellow]"
        
        # Confidence color
        if result.confidence >= 0.8:
            conf_color = "green"
        elif result.confidence >= 0.5:
            conf_color = "yellow"
        else:
            conf_color = "red"
        
        # Truncate texts
        criterion_fmt = (result.criterion[:37] + "..." 
                        if len(result.criterion) > 40 
                        else result.criterion)
        
        evidence_fmt = (result.evidence[:42] + "..." 
                       if len(result.evidence) > 45 
                       else result.evidence)
        
        # Format pages
        if result.pages:
            pages_fmt = ", ".join(str(p) for p in result.pages[:3])
            if len(result.pages) > 3:
                pages_fmt += "..."
        else:
            pages_fmt = "-"
        
        table.add_row(
            str(i),
            status_fmt,
            criterion_fmt,
            evidence_fmt,
            f"[cyan]{pages_fmt}[/cyan]",
            f"[{conf_color}]{result.confidence:.0%}[/{conf_color}]"
        )
    
    console.print(table)


def display_json(report: AuditReport) -> None:
    """Displays the report in JSON format."""
    console.print("\n[bold yellow]📄 JSON Output:[/bold yellow]")
    console.print_json(report.model_dump_json())


def save_json(report: AuditReport, filename: str = "audit_report.json") -> Path:
    """
    Saves the report to a JSON file.
    
    Args:
        report: AuditReport to save
        filename: Output filename
    
    Returns:
        Path to the saved file
    """
    output_path = OUTPUT_DIR / filename
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))
    
    console.print(f"\n[green]✓ Report saved to:[/green] {output_path}")
    return output_path


def save_table_txt(report: AuditReport, filename: str = "audit_report.txt") -> Path:
    """
    Saves the report summary to a text file.
    
    Args:
        report: AuditReport to save
        filename: Output filename
    
    Returns:
        Path to the saved file
    """
    output_path = OUTPUT_DIR / filename
    
    lines = [
        "=" * 70,
        "AUDIT REPORT",
        "=" * 70,
        f"Document: {report.document}",
        f"Total Criteria: {report.total_criteria}",
        f"Present: {report.criteria_present}",
        f"Absent: {report.criteria_absent}",
        f"Compliance Rate: {report.compliance_rate}%",
        "",
        "-" * 70,
        "DETAILED RESULTS",
        "-" * 70,
    ]
    
    for i, result in enumerate(report.results, 1):
        lines.append(f"\n{i}. {result.criterion}")
        lines.append(f"   Status: {result.status}")
        lines.append(f"   Confidence: {result.confidence:.0%}")
        lines.append(f"   Pages: {', '.join(map(str, result.pages)) if result.pages else 'N/A'}")
        lines.append(f"   Evidence: {result.evidence[:100]}{'...' if len(result.evidence) > 100 else ''}")
    
    lines.append("\n" + "=" * 70)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    console.print(f"[green]✓ Text report saved to:[/green] {output_path}")
    return output_path


def display_conclusion() -> None:
    """Displays the conclusion message."""
    console.print("\n[bold cyan]✅ Audit completed![/bold cyan]")
    console.print(f"[dim]Output files saved to: {OUTPUT_DIR}[/dim]")