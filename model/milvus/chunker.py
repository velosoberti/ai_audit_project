# chunker.py - Text Chunking Functions
# =============================================================================

from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import CHUNK_SIZE, CHUNK_OVERLAP


def create_chunks_by_page(
    pages: list[dict] | list[tuple[int, str]],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> list[dict]:
    """
    Splits the text from each page into smaller chunks.
    
    Args:
        pages: List of dicts {"page_number": int, "text": str}
               OR list of tuples (page_number, text)
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of dicts with 'text' and 'page_number'
    """
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
        length_function=len,
    )
    
    all_chunks = []
    
    for page in pages:
        # Supports both dict and tuple
        if isinstance(page, dict):
            page_num = page.get("page_number") or page.get("page")
            text = page.get("text") or page.get("texto")
        else:
            page_num, text = page
        
        if not text:
            continue
            
        page_chunks = text_splitter.split_text(text)
        
        for chunk in page_chunks:
            all_chunks.append({
                "text": chunk,
                "page_number": page_num
            })
    
    return all_chunks