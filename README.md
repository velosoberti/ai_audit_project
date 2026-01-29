# Audit Pipeline

Unified PDF document indexing and auditing system using Milvus and hybrid search.

## Structure

```
audit_pipeline/
├── config.yaml           # Centralized configuration
├── shared_config.py      # YAML configuration loader
├── run_pipeline.py       # Main orchestrator
└── model/
    ├── milvus/           # Indexing module
    │   ├── config.py
    │   ├── indexer.py
    │   ├── collection.py
    │   ├── extractor.py
    │   ├── chunker.py
    │   └── models.py
    └── application/      # Auditing module
        ├── config.py
        ├── auditor.py
        ├── retriever.py
        ├── enhanced_retriever.py  # Retrieval with possible answers
        ├── evaluator.py
        ├── deep_agent.py
        ├── output.py
        ├── metrics.py
        ├── models.py
        ├── possible_answer_generator.py  # LLM answer generation
        ├── possible_answer_models.py     # Pydantic models
        └── raw_extractor.py              # Raw PDF extraction
```

## Configuration

Edit the `config.yaml` file to configure:

### Milvus Connection

```yaml
milvus:
  uri: "http://127.0.0.1:19530"
  collection_name: "audit_docs_v3"
```

### Documents to Process

```yaml
documents:
  - path: "/path/to/document.pdf"
    doc_type: "contract"
    skip_if_indexed: true      # Skip if already indexed
    reset_collection: false    # Clear collection before (first doc)
```

### Audit Criteria

```yaml
audit_criteria:
  - query: "Does the document have a CNPJ?"
    confidence: 0.8
  
  - query: "Is there a confidentiality clause?"
    confidence: 0.7
```

### Possible Answers (Enhanced Retrieval)

```yaml
possible_answers:
  enabled: false            # Enable LLM-based possible answer generation
```

When enabled, the system generates "possible answers" for each criterion by having an LLM read the raw PDF. These are used to:
- Enhance hybrid search queries (find more relevant chunks)
- Provide hints in the evaluator prompt (alongside actual document excerpts)

### Pipeline Options

```yaml
pipeline:
  force_reindex: false      # Force reindexing even if exists
  display_metrics: true     # Display performance metrics
  skip_indexing: false      # Skip directly to auditing
```

## Usage

### Full Pipeline (indexing + auditing)

```bash
uv run run_pipeline.py
```

### With specific configuration

```bash
uv run run_pipeline.py --config my_config.yaml
```

### Indexing only

```bash
uv run run_pipeline.py --index-only
```

### Auditing only (already indexed documents)

```bash
uv run run_pipeline.py --audit-only
```

## Individual Execution

### Index specific document

```bash
cd model/milvus
uv run main.py --pdf /path/to/document.pdf --doc-type contract
```

### List indexed documents

```bash
cd model/milvus
uv run main.py --list
```

### Audit specific document

```bash
cd model/application
uv run main.py --document "document.pdf" --doc-type contract
```

## Pipeline Flow

1. **Loads configuration** from `config.yaml`
2. **For each document:**
   - Checks if already indexed (filename + doc_type)
   - If not indexed: extracts text, creates chunks, generates embeddings, inserts into Milvus
   - If indexed: skips directly to auditing
3. **Runs audit:**
   - Hybrid search (sparse + dense) for each criterion
   - Deep Agent makes multiple attempts if necessary
   - Generates report with evidence and pages
4. **Saves outputs:**
   - JSON with all data
   - TXT with formatted summary

## Environment Variables

The system also supports configuration via environment variables (fallback if no config.yaml):

```bash
export MILVUS_URI="http://127.0.0.1:19530"
export COLLECTION_NAME="audit_docs_v3"
export OUTPUT_DIR="./output"
export CONFIG_PATH="/path/to/config.yaml"
```

```