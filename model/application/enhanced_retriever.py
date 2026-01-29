# enhanced_retriever.py - Enhanced Retriever with Possible Answer Support
# ============================================================================
"""
Enhanced retriever that uses both criterion AND possible answer as search queries
to improve retrieval quality. Falls back to criterion-only search when no
possible answer is available.
"""

from pymilvus import Collection, AnnSearchRequest, RRFRanker

from .config import ef_sparse, ef_dense, COLLECTION_NAME
from .possible_answer_models import PossibleAnswer
from .retriever import generate_query_embeddings, sparse_to_dict


async def search_with_possible_answer(
    criterion: str,
    possible_answer: PossibleAnswer | None,
    filename: str,
    doc_type: str | None = None,
    limit: int = 5
) -> tuple[str, list[int]]:
    """
    Searches using both criterion and possible answer as queries.
    
    Uses the criterion as the primary query and the possible answer text
    as an additional query to find more relevant chunks. Results from both
    queries are merged and deduplicated.
    
    Args:
        criterion: The criterion to search for
        possible_answer: LLM-generated possible answer (optional)
        filename: Filter by filename
        doc_type: Filter by document type
        limit: Maximum results
        
    Returns:
        Tuple (formatted_context, list_of_pages)
    """
    # Get collection
    col = Collection(COLLECTION_NAME)
    col.load()
    
    # Build metadata filter
    filters = []
    if filename:
        filters.append(f'filename == "{filename}"')
    if doc_type:
        filters.append(f'doc_type == "{doc_type}"')
    
    final_filter = " and ".join(filters) if filters else None
    
    # Generate embeddings for criterion query
    criterion_embeddings = generate_query_embeddings(criterion)
    
    # Determine if we should use possible answer as additional query
    use_possible_answer = (
        possible_answer is not None 
        and possible_answer.found 
        and possible_answer.answer.strip()
    )
    
    if use_possible_answer:
        # Search with both queries and merge results
        results = await _search_with_dual_queries(
            col=col,
            criterion_embeddings=criterion_embeddings,
            possible_answer=possible_answer,
            final_filter=final_filter,
            limit=limit
        )
    else:
        # Fall back to criterion-only search
        results = _search_single_query(
            col=col,
            embeddings=criterion_embeddings,
            final_filter=final_filter,
            limit=limit
        )
    
    # Process and format results
    return _format_results(results, limit)


async def _search_with_dual_queries(
    col: Collection,
    criterion_embeddings: dict,
    possible_answer: PossibleAnswer,
    final_filter: str | None,
    limit: int
) -> list:
    """
    Executes hybrid search with both criterion and possible answer queries,
    then merges and deduplicates results.
    """
    # Generate embeddings for possible answer query
    answer_embeddings = generate_query_embeddings(possible_answer.answer)
    
    # Search with criterion query
    criterion_results = _search_single_query(
        col=col,
        embeddings=criterion_embeddings,
        final_filter=final_filter,
        limit=limit
    )
    
    # Search with possible answer query
    answer_results = _search_single_query(
        col=col,
        embeddings=answer_embeddings,
        final_filter=final_filter,
        limit=limit
    )
    
    # Merge and deduplicate results
    return _merge_and_deduplicate(criterion_results, answer_results)


def _search_single_query(
    col: Collection,
    embeddings: dict,
    final_filter: str | None,
    limit: int
) -> list:
    """
    Executes a single hybrid search query (sparse + dense).
    """
    sparse_search_params = {"metric_type": "IP"}
    sparse_req = AnnSearchRequest(
        data=[embeddings["sparse"]],
        anns_field="sparse_vector",
        param=sparse_search_params,
        limit=limit,
        expr=final_filter
    )
    
    dense_search_params = {"metric_type": "COSINE"}
    dense_req = AnnSearchRequest(
        data=[embeddings["dense"]],
        anns_field="dense_vector",
        param=dense_search_params,
        limit=limit,
        expr=final_filter
    )
    
    results = col.hybrid_search(
        reqs=[sparse_req, dense_req],
        rerank=RRFRanker(),
        limit=limit,
        output_fields=["text", "filename", "doc_type", "page_number"]
    )
    
    if not results or not results[0]:
        return []
    
    return list(results[0])


def _merge_and_deduplicate(
    criterion_results: list,
    answer_results: list
) -> list:
    """
    Merges results from criterion and possible answer queries,
    deduplicating by text content and keeping the highest score.
    """
    # Use text as key for deduplication
    seen_texts: dict[str, tuple[float, object]] = {}
    
    # Process criterion results first (primary)
    for hit in criterion_results:
        text = hit.entity.get('text', '')
        score = hit.distance
        if text not in seen_texts or score > seen_texts[text][0]:
            seen_texts[text] = (score, hit)
    
    # Process answer results (secondary)
    for hit in answer_results:
        text = hit.entity.get('text', '')
        score = hit.distance
        if text not in seen_texts or score > seen_texts[text][0]:
            seen_texts[text] = (score, hit)
    
    # Sort by score descending and return hits
    sorted_results = sorted(
        seen_texts.values(),
        key=lambda x: x[0],
        reverse=True
    )
    
    return [hit for _, hit in sorted_results]


def _format_results(results: list, limit: int) -> tuple[str, list[int]]:
    """
    Formats search results into context string and page list.
    """
    if not results:
        return "No context found.", []
    
    contexts = []
    pages = []
    
    for hit in results[:limit]:
        text = hit.entity.get('text', '')
        file = hit.entity.get('filename', 'N/A')
        type_doc = hit.entity.get('doc_type', 'N/A')
        page = hit.entity.get('page_number', 0)
        score = hit.distance
        
        pages.append(page)
        
        contexts.append(
            f"[File: {file} | Type: {type_doc} | Page: {page} | Score: {score:.3f}]\n{text}"
        )
    
    return "\n\n---\n\n".join(contexts), pages
