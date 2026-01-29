# evaluator.py - LLM Evaluation Functions
# =============================================================================

import json

from .config import llm
from .models import CriterionResult


def evaluate_criterion(criterion: str, context: str, pages: list[int]) -> CriterionResult:
    """
    Uses the LLM to evaluate if a criterion is present in the context.
    
    Args:
        criterion: The criterion to be evaluated
        context: Document excerpts retrieved from Milvus
        pages: List of pages where the excerpts came from
    
    Returns:
        CriterionResult with status, evidence, confidence, and pages
    """
    
    prompt = f"""You are a rigorous compliance auditor analyzing a document.
    Respond only in valid JSON, without markdown and without additional text.

    CRITERION TO EVALUATE:
    {criterion}

    DOCUMENT CONTEXT (relevant retrieved excerpts):
    {context}

    CRITICAL RULES:
    1. Carefully analyze if the criterion is PRESENT or ABSENT in the context
    2. The "evidence" field MUST contain the EXACT excerpt copied from the document
    3. DO NOT paraphrase, DO NOT translate, DO NOT summarize - copy the text EXACTLY as it appears
    4. Keep the original language of the document (Portuguese) in the evidence field
    5. If the criterion is ABSENT, briefly explain why in English
    6. Evaluate your confidence in the response (0.0 = none, 1.0 = total)
    7. Indicate which pages contain the evidence (use page numbers from context)

    Respond EXACTLY in the JSON format below:

    {{
        "status": "PRESENT" or "ABSENT",
        "evidence": "EXACT QUOTE from the document in its original language, or brief explanation if absent",
        "confidence": 0.0 to 1.0,
        "relevant_pages": [list of page numbers where evidence was found]
    }}
    """
    
    response = llm.invoke(prompt)
    
    try:
        clean_content = (response.content
                         .strip()
                         .replace("```json", "")
                         .replace("```", "")
                         .strip())
        
        data = json.loads(clean_content)
        
        result_pages = data.get("relevant_pages", pages)
        result_pages = sorted(list(set(result_pages))) if result_pages else []
        
        return CriterionResult(
            criterion=criterion,
            status=data.get("status", "ABSENT"),
            evidence=data.get("evidence", "Could not determine"),
            confidence=float(data.get("confidence", 0.5)),
            pages=result_pages
        )
        
    except (json.JSONDecodeError, ValueError) as e:
        return CriterionResult(
            criterion=criterion,
            status="ERROR",
            evidence=f"Error processing LLM response: {str(e)[:100]}",
            confidence=0.0,
            pages=[]
        )