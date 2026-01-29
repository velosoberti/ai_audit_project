# models.py - Pydantic Models for Data Validation
# =============================================================================

from pydantic import BaseModel, Field


class CriterionResult(BaseModel):
    """
    Represents the evaluation result of a single criterion.
    """
    criterion: str = Field(description="Description of the evaluated criterion")
    status: str = Field(description="PRESENT, ABSENT, or ERROR")
    evidence: str = Field(description="Document excerpt or justification")
    confidence: float = Field(
        default=1.0, 
        ge=0,
        le=1,
        description="Confidence level (0-1)"
    )
    pages: list[int] = Field(
        default_factory=list,
        description="Pages where evidence was found"
    )


class AuditReport(BaseModel):
    """
    Represents the complete report of an audit.
    """
    document: str = Field(description="Name of the audited document")
    total_criteria: int = Field(description="Total criteria evaluated")
    criteria_present: int = Field(description="Number of criteria found")
    criteria_absent: int = Field(description="Number of criteria not found")
    compliance_rate: float = Field(description="Compliance percentage (0-100)")
    results: list[CriterionResult] = Field(description="List of results by criterion")