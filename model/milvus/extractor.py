# extractor.py - PDF Text Extraction Functions
# =============================================================================

import os
from rich.console import Console

try:
    import pdfplumber
    PDF_EXTRACTOR = "pdfplumber"
except ImportError:
    import PyPDF2
    PDF_EXTRACTOR = "pypdf2"

console = Console()


def extract_text_pdfplumber(pdf_path: str) -> list[tuple[int, str]]:
    """
    Extracts text from a PDF using pdfplumber (better quality).
    
    Args:
        pdf_path: Path to the PDF file
    
    Returns:
        List of tuples (page_number, page_text)
    """
    pages = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if text and text.strip():
                pages.append((i, text))
    
    return pages


def extract_text_pypdf2(pdf_path: str) -> list[tuple[int, str]]:
    """
    Extracts text from a PDF using PyPDF2 (fallback).
    
    Args:
        pdf_path: Path to the PDF file
    
    Returns:
        List of tuples (page_number, page_text)
    """
    pages = []
    
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text and text.strip():
                pages.append((i, text))
    
    return pages


def extract_text_from_pdf(pdf_path: str) -> list[tuple[int, str]]:
    """
    Extracts text from a PDF using the available extractor.
    
    Args:
        pdf_path: Path to the PDF file
    
    Returns:
        List of tuples (page_number, page_text)
    
    Raises:
        FileNotFoundError: If the file does not exist
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")
    
    console.print(f"[dim]Using extractor: {PDF_EXTRACTOR}[/dim]")
    
    if PDF_EXTRACTOR == "pdfplumber":
        return extract_text_pdfplumber(pdf_path)
    else:
        return extract_text_pypdf2(pdf_path)