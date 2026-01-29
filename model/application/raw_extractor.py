# raw_extractor.py - Raw PDF Text Extraction for Possible Answer Generation
# =============================================================================

import os
from rich.console import Console

from .possible_answer_models import RawPDFContent, TextSegment

try:
    import pdfplumber
    PDF_EXTRACTOR = "pdfplumber"
except ImportError:
    import PyPDF2
    PDF_EXTRACTOR = "pypdf2"

console = Console()


class PDFExtractionError(Exception):
    """Raised when PDF extraction fails."""
    pass


class RawPDFExtractor:
    """Extracts full text from PDF for possible answer generation."""
    
    # Approximate characters per token (conservative estimate)
    CHARS_PER_TOKEN = 4
    
    def extract_full_text(self, pdf_path: str) -> RawPDFContent:
        """
        Extracts all text from PDF preserving page structure.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            RawPDFContent with full text and page metadata
            
        Raises:
            FileNotFoundError: If PDF does not exist
            PDFExtractionError: If extraction fails
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        filename = os.path.basename(pdf_path)
        
        try:
            if PDF_EXTRACTOR == "pdfplumber":
                pages = self._extract_with_pdfplumber(pdf_path)
            else:
                pages = self._extract_with_pypdf2(pdf_path)
        except Exception as e:
            raise PDFExtractionError(f"Failed to extract text from {filename}: {str(e)}")
        
        total_characters = sum(len(text) for _, text in pages)
        
        return RawPDFContent(
            filename=filename,
            pages=pages,
            total_pages=len(pages),
            total_characters=total_characters
        )
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> list[tuple[int, str]]:
        """Extract text using pdfplumber (better quality)."""
        pages = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text and text.strip():
                    pages.append((i, text))
        
        return pages
    
    def _extract_with_pypdf2(self, pdf_path: str) -> list[tuple[int, str]]:
        """Extract text using PyPDF2 (fallback)."""
        pages = []
        
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text and text.strip():
                    pages.append((i, text))
        
        return pages
    
    def get_text_for_llm(
        self, 
        content: RawPDFContent, 
        max_tokens: int = 100000
    ) -> list[TextSegment]:
        """
        Splits text into LLM-manageable segments if needed.
        
        Args:
            content: Raw PDF content
            max_tokens: Maximum tokens per segment
            
        Returns:
            List of text segments with page references
        """
        if not content.pages:
            return []
        
        max_chars = max_tokens * self.CHARS_PER_TOKEN
        
        # If total content fits in one segment, return as single segment
        if content.total_characters <= max_chars:
            full_text = self._format_pages_as_text(content.pages)
            start_page = content.pages[0][0]
            end_page = content.pages[-1][0]
            estimated_tokens = content.total_characters // self.CHARS_PER_TOKEN
            
            return [TextSegment(
                text=full_text,
                start_page=start_page,
                end_page=end_page,
                estimated_tokens=estimated_tokens
            )]
        
        # Split into multiple segments
        return self._split_into_segments(content.pages, max_chars)
    
    def _format_pages_as_text(self, pages: list[tuple[int, str]]) -> str:
        """Format pages into a single text with page markers."""
        formatted_parts = []
        for page_num, text in pages:
            formatted_parts.append(f"[Page {page_num}]\n{text}")
        return "\n\n".join(formatted_parts)
    
    def _split_into_segments(
        self, 
        pages: list[tuple[int, str]], 
        max_chars: int
    ) -> list[TextSegment]:
        """Split pages into segments that fit within max_chars."""
        segments = []
        current_pages: list[tuple[int, str]] = []
        current_chars = 0
        
        for page_num, text in pages:
            page_chars = len(text) + len(f"[Page {page_num}]\n") + 2  # +2 for \n\n
            
            # If adding this page exceeds limit, finalize current segment
            if current_chars + page_chars > max_chars and current_pages:
                segment = self._create_segment(current_pages)
                segments.append(segment)
                current_pages = []
                current_chars = 0
            
            current_pages.append((page_num, text))
            current_chars += page_chars
        
        # Add remaining pages as final segment
        if current_pages:
            segment = self._create_segment(current_pages)
            segments.append(segment)
        
        return segments
    
    def _create_segment(self, pages: list[tuple[int, str]]) -> TextSegment:
        """Create a TextSegment from a list of pages."""
        text = self._format_pages_as_text(pages)
        start_page = pages[0][0]
        end_page = pages[-1][0]
        estimated_tokens = len(text) // self.CHARS_PER_TOKEN
        
        return TextSegment(
            text=text,
            start_page=start_page,
            end_page=end_page,
            estimated_tokens=estimated_tokens
        )
