# deep_agent.py - Deep Research Agent for Auditing
# =============================================================================

from dataclasses import dataclass, field
from rich.console import Console

from .config import llm
from .retriever import search_relevant_context
from .models import CriterionResult

console = Console()


@dataclass
class SearchState:
    """Current state of the agent's search."""
    original_criterion: str
    executed_queries: list[str] = field(default_factory=list)
    found_contexts: list[dict] = field(default_factory=list)
    found_pages: set[int] = field(default_factory=set)
    attempts: int = 0
    max_attempts: int = 3
    min_confidence: float = 0.9 


class DeepResearchAgent:
    """
    Agent that performs iterative deep research.
    
    Flow:
    1. Initial search with original criterion
    2. Evaluates if sufficient evidence was found
    3. If not, generates alternative queries and searches again
    4. Repeats until found or reaches attempt limit
    """
    
    def __init__(self, filename: str, doc_type: str | None = None, collection_name: str | None = None):
        self.filename = filename
        self.doc_type = doc_type
        self.collection_name = collection_name
        self.total_chunks = self._get_total_chunks()
    
    def _get_total_chunks(self) -> int:
        """Get document size for dynamic limits."""
        try:
            from pymilvus import Collection
            from .config import COLLECTION_NAME
            col_name = self.collection_name or COLLECTION_NAME
            col = Collection(col_name)
            col.load()
            results = col.query(
                expr=f'filename == "{self.filename}"',
                output_fields=["pk"],
                limit=10000
            )
            return len(results)
        except:
            return 100
    
    def _calculate_dynamic_limit(self) -> int:
        """Calculate dynamic retrieval limit based on document size."""
        return min(10, max(3, self.total_chunks // 100))
    
    async def search(self, criterion: str, min_confidence: float | None = None) -> CriterionResult:
        """
        Executes deep search for a criterion.
        
        Args:
            criterion: Criterion to be evaluated
            min_confidence: Specific confidence threshold
        
        Returns:
            CriterionResult with the best evaluation found
        """
        state = SearchState(
            original_criterion=criterion,
            min_confidence=min_confidence
        )
        
        limit = self._calculate_dynamic_limit()
        
        while state.attempts < state.max_attempts:
            state.attempts += 1
            
            # Define query for this iteration
            if state.attempts == 1:
                query = criterion
            else:
                query = await self._generate_alternative_query(state)
                
                # If couldn't generate new query, stop
                if query in state.executed_queries:
                    break
            
            state.executed_queries.append(query)
            
            # Search context with dynamic limit
            context, pages = await search_relevant_context(
                criterion=query,
                limit=limit,
                filename=self.filename,
                doc_type=self.doc_type
            )
            
            # Store results
            state.found_contexts.append({
                "query": query,
                "context": context,
                "pages": pages
            })
            state.found_pages.update(pages)
            
            # Evaluate with all accumulated context
            result = await self._evaluate_with_accumulated_context(state)
            
            # If sufficient confidence, return
            if result.confidence >= state.min_confidence:
                console.print(f"[green]  ✓ Found on attempt {state.attempts}[/green]")
                return result
            
            console.print(f"[yellow]  ↻ Attempt {state.attempts}: confidence {result.confidence:.0%}, searching more...[/yellow]")
        
        # Return best result found
        return await self._evaluate_with_accumulated_context(state)
    
    async def _generate_alternative_query(self, state: SearchState) -> str:
        """Generates an alternative query based on history."""
        
        prompt = f"""You are searching for information in a document to verify this criterion:
"{state.original_criterion}"

Queries already tried (do not repeat):
{chr(10).join(f'- {q}' for q in state.executed_queries)}

Contexts found so far:
{chr(10).join(c['context'][:200] + '...' for c in state.found_contexts)}

Generate ONE alternative search query to find this information.
Use synonyms, related terms, or different approaches.

Respond ONLY with the query, without explanations."""

        response = llm.invoke(prompt)
        return response.content.strip()
    
    async def _evaluate_with_accumulated_context(self, state: SearchState) -> CriterionResult:
        """Evaluates using all accumulated context from searches."""
        
        # Combine all unique contexts
        full_context = "\n\n---\n\n".join(
            c["context"] for c in state.found_contexts
            if c["context"] != "No context found."
        )
        
        if not full_context:
            full_context = "No relevant context found after multiple searches."
        
         
        prompt = f"""You are a rigorous compliance auditor analyzing a document.

    CRITERION TO EVALUATE:
    {state.original_criterion}

    FULL CONTEXT (from {len(state.executed_queries)} searches):
    {full_context}

    PAGES FOUND: {sorted(state.found_pages) if state.found_pages else 'None'}

    CRITICAL RULES:
    1. Evaluate if the criterion is PRESENT or ABSENT in the document
    2. The "evidence" field MUST contain the EXACT excerpt copied from the document
    3. DO NOT paraphrase, DO NOT translate, DO NOT summarize - copy the text EXACTLY as it appears
    4. Keep the original language of the document (Portuguese) in the evidence field
    5. If the criterion is ABSENT, briefly explain why in English
    6. Be precise about which pages contain the evidence

    Respond in valid JSON:
    {{
        "status": "PRESENT" or "ABSENT",
        "evidence": "EXACT QUOTE from the document in its original language (Portuguese), or brief explanation if absent",
        "confidence": 0.0 to 1.0,
        "relevant_pages": [list of pages]
    }}"""


        response = llm.invoke(prompt)
        
        try:
            import json
            content = (response.content
                       .strip()
                       .replace("```json", "")
                       .replace("```", "")
                       .strip())
            
            data = json.loads(content)
            
            pages = data.get("relevant_pages", list(state.found_pages))
            
            return CriterionResult(
                criterion=state.original_criterion,
                status=data.get("status", "ABSENT"),
                evidence=data.get("evidence", ""),
                confidence=float(data.get("confidence", 0.5)),
                pages=sorted(set(pages)) if pages else []
            )
        except Exception as e:
            return CriterionResult(
                criterion=state.original_criterion,
                status="ERROR",
                evidence=f"Error: {str(e)[:100]}",
                confidence=0.0,
                pages=[]
            )