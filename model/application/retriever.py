# retriever.py - Hybrid Semantic Search Functions in Milvus
# =============================================================================

from pymilvus import Collection, AnnSearchRequest, RRFRanker

from .config import ef_sparse, ef_dense, COLLECTION_NAME, llm


def sparse_to_dict(sparse_array) -> dict[int, float]:
    """Converts scipy sparse array to dict format."""
    coo = sparse_array.tocoo()
    return {int(i): float(v) for i, v in zip(coo.col, coo.data)}


def expand_query(text: str) -> list[str]:
    """Expands query with synonyms and semantic variations."""
    prompt = f"""Generate 2 semantic variations of this search query:
"{text}"

Use synonyms, related terms, or reformulations.
Respond only with the queries, one per line, without numbering."""
    
    try:
        response = llm.invoke(prompt)
        variations = [v.strip() for v in response.content.strip().split('\n') if v.strip()]
        return [text] + variations[:2]
    except:
        return [text]


def generate_query_embeddings(text: str) -> dict:
    """
    Generates hybrid embeddings for a search query.
    
    - Sparse: BM25 (lexical search)
    - Dense: Gemini via spelling (semantic search)
    
    Args:
        text: Query text
    
    Returns:
        Dict with 'sparse' and 'dense' embeddings
    """
    # Sparse with BM25
    sparse_raw = ef_sparse.encode_queries([text])
    sparse_vector = sparse_to_dict(sparse_raw[0])
    
    # If BM25 returns empty vector (no matching terms), use a placeholder
    # This can happen when query terms aren't in the fitted corpus vocabulary
    if not sparse_vector:
        sparse_vector = {0: 0.0001}  # Minimal placeholder to avoid Milvus error

    dense_vector = ef_dense.embed_query(text)
    

    if hasattr(dense_vector, 'tolist'):
        dense_vector = dense_vector.tolist()
    
    return {
        "sparse": sparse_vector,
        "dense": dense_vector
    }


async def search_relevant_context(
    criterion: str, 
    limit: int = 5,
    filename: str | None = None,
    doc_type: str | None = None
) -> tuple[str, list[int]]:
    """
    Searches for relevant excerpts using hybrid search (sparse + dense).
    
    Args:
        criterion: Criterion text to search for
        limit: Maximum number of results
        filename: Filter by filename
        doc_type: Filter by document type
    
    Returns:
        Tuple (formatted_context, list_of_pages)
    """
    
    # ----- STEP 1: Generate hybrid embeddings -----
    query_embeddings = generate_query_embeddings(criterion)
    
    # ----- STEP 2: Get collection -----
    col = Collection(COLLECTION_NAME)
    col.load()
    
    # ----- STEP 3: Build metadata filter -----
    filters = []
    if filename:
        filters.append(f'filename == "{filename}"')
    if doc_type:
        filters.append(f'doc_type == "{doc_type}"')
    
    final_filter = " and ".join(filters) if filters else None
    
    # ----- STEP 4: Configure sparse and dense searches -----
    sparse_search_params = {"metric_type": "IP"}
    sparse_req = AnnSearchRequest(
        data=[query_embeddings["sparse"]],
        anns_field="sparse_vector",
        param=sparse_search_params,
        limit=limit,
        expr=final_filter
    )
    
    dense_search_params = {"metric_type": "COSINE"}
    dense_req = AnnSearchRequest(
        data=[query_embeddings["dense"]],
        anns_field="dense_vector",
        param=dense_search_params,
        limit=limit,
        expr=final_filter
    )
    
    # ----- STEP 5: Execute hybrid search -----
    results = col.hybrid_search(
        reqs=[sparse_req, dense_req],
        rerank=RRFRanker(),
        limit=limit,
        output_fields=["text", "filename", "doc_type", "page_number"]
    )
    
    # ----- STEP 6: Process results -----
    if not results or not results[0]:
        return "No context found.", []
    
    contexts = []
    pages = []
    
    for hit in results[0]:
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