# possible_answer_generator.py - LLM-based Possible Answer Generation
# =============================================================================

import asyncio
import json
from typing import Optional

from rich.console import Console

from .possible_answer_models import PossibleAnswer, RawPDFContent

console = Console()


def _get_default_llm():
    """Lazy load the default LLM to avoid import-time Milvus connection."""
    from .config import llm
    return llm


class PossibleAnswerGenerator:
    """Generates possible answers from raw PDF content using LLM."""
    
    # Maximum retries for LLM calls
    MAX_RETRIES = 3
    # Base delay for exponential backoff (seconds)
    BASE_DELAY = 1.0
    
    def __init__(self, llm_client=None):
        """
        Initialize with LLM client.
        
        Args:
            llm_client: LLM client (defaults to shared llm from config.py)
        """
        self.llm = llm_client if llm_client is not None else _get_default_llm()
    
    async def generate_answer(
        self, 
        criterion: str, 
        pdf_content: RawPDFContent
    ) -> PossibleAnswer:
        """
        Generates a possible answer for a single criterion.
        
        Args:
            criterion: The audit criterion to answer
            pdf_content: Full PDF text content
            
        Returns:
            PossibleAnswer with answer text and relevant pages
        """
        # Format PDF content for the prompt
        formatted_content = self._format_pdf_content(pdf_content)
        
        prompt = self._build_prompt(criterion, formatted_content)
        
        # Try with retries and exponential backoff
        last_error: Optional[Exception] = None
        
        for attempt in range(self.MAX_RETRIES):
            try:
                response = await self._invoke_llm(prompt)
                return self._parse_response(criterion, response)
            except Exception as e:
                last_error = e
                if attempt < self.MAX_RETRIES - 1:
                    delay = self.BASE_DELAY * (2 ** attempt)
                    console.print(
                        f"[yellow]LLM call failed (attempt {attempt + 1}/{self.MAX_RETRIES}), "
                        f"retrying in {delay}s: {str(e)[:100]}[/yellow]"
                    )
                    await asyncio.sleep(delay)
        
        # All retries failed - return empty answer
        console.print(
            f"[red]Failed to generate answer for criterion after {self.MAX_RETRIES} attempts: "
            f"{str(last_error)[:100]}[/red]"
        )
        return PossibleAnswer(
            criterion=criterion,
            answer="",
            relevant_pages=[],
            found=False
        )
    
    async def generate_answers_batch(
        self, 
        criteria: list[str], 
        pdf_content: RawPDFContent
    ) -> dict[str, PossibleAnswer]:
        """
        Generates possible answers for multiple criteria concurrently.
        
        Args:
            criteria: List of audit criteria
            pdf_content: Full PDF text content
            
        Returns:
            Dict mapping criterion to PossibleAnswer
        """
        if not criteria:
            return {}
        
        # Create tasks for concurrent processing
        tasks = [
            self.generate_answer(criterion, pdf_content)
            for criterion in criteria
        ]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Build result dictionary
        answers: dict[str, PossibleAnswer] = {}
        
        for criterion, result in zip(criteria, results):
            if isinstance(result, Exception):
                console.print(
                    f"[red]Error generating answer for '{criterion[:50]}...': {str(result)[:100]}[/red]"
                )
                answers[criterion] = PossibleAnswer(
                    criterion=criterion,
                    answer="",
                    relevant_pages=[],
                    found=False
                )
            else:
                answers[criterion] = result
        
        return answers
    
    def _format_pdf_content(self, pdf_content: RawPDFContent) -> str:
        """Format PDF content for the LLM prompt."""
        if not pdf_content.pages:
            return "[No content available]"
        
        formatted_parts = []
        for page_num, text in pdf_content.pages:
            formatted_parts.append(f"[Page {page_num}]\n{text}")
        
        return "\n\n".join(formatted_parts)
    
    def _build_prompt(self, criterion: str, pdf_content: str) -> str:
        """Build the LLM prompt for generating a possible answer."""
        return f"""You are an expert document analyst. Your task is to find information in a document that answers a specific audit criterion.

AUDIT CRITERION TO FIND:
{criterion}

DOCUMENT CONTENT:
{pdf_content}

INSTRUCTIONS:
1. Carefully read the entire document content
2. Find any information that directly or indirectly answers the criterion
3. If you find relevant information, extract the key points and note which pages contain the evidence
4. If no relevant information is found, indicate that clearly

Respond ONLY in valid JSON format (no markdown, no additional text):

{{
    "found": true or false,
    "answer": "A concise summary of the relevant information found, or empty string if not found",
    "relevant_pages": [list of page numbers where the information was found, or empty list]
}}

CRITICAL RULES:
- Set "found" to true ONLY if you find information that actually addresses the criterion
- The "answer" should summarize what you found, not quote the entire text
- Include ALL page numbers where relevant information appears
- If nothing relevant is found, set "found" to false and "answer" to empty string
"""
    
    async def _invoke_llm(self, prompt: str) -> str:
        """
        Invoke the LLM asynchronously.
        
        The spelling library's Chat.invoke() is synchronous, so we run it
        in a thread pool to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, self.llm.invoke, prompt)
        return response.content
    
    def _parse_response(self, criterion: str, response: str) -> PossibleAnswer:
        """Parse the LLM response into a PossibleAnswer."""
        try:
            # Clean up response (remove markdown code blocks if present)
            clean_content = (
                response
                .strip()
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )
            
            data = json.loads(clean_content)
            
            found = data.get("found", False)
            answer = data.get("answer", "")
            relevant_pages = data.get("relevant_pages", [])
            
            # Ensure relevant_pages is a list of integers
            if relevant_pages:
                relevant_pages = sorted([int(p) for p in relevant_pages])
            
            return PossibleAnswer(
                criterion=criterion,
                answer=answer if found else "",
                relevant_pages=relevant_pages if found else [],
                found=bool(found)
            )
            
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            console.print(
                f"[yellow]Failed to parse LLM response for criterion: {str(e)[:100]}[/yellow]"
            )
            # Return empty answer on parse failure
            return PossibleAnswer(
                criterion=criterion,
                answer="",
                relevant_pages=[],
                found=False
            )
