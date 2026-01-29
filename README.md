# Audit Pipeline

Unified PDF document auditing system using Hybrid RAG (Retrieval-Augmented Generation) with Milvus vector database.

## Overview

This system automates document compliance auditing by combining:
- **Hybrid Search**: BGE-M3 (sparse/lexical) + Gemini (dense/semantic) embeddings
- **Deep Research Agent**: Iterative search with alternative query generation
- **Possible Answers**: LLM pre-analysis of raw PDF for enhanced retrieval (optional)

## How It Works

### Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           AUDIT PIPELINE                                 │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌────────────────────────────┐  │
│  │   PDF File   │───▶│   Indexer    │───▶│   Milvus Vector DB         │  │
│  └──────────────┘    │  - Extract   │    │  - Sparse vectors (BGE)    │  │
│                      │  - Chunk     │    │  - Dense vectors (Spelling)│  │
│                      │  - Embed     │    └────────────────────────────┘  │
│                      └──────────────┘               │                    │
│                                                     │                    │
│  ┌──────────────┐    ┌──────────────┐              │                     │
│  │   Criteria   │───▶│  Deep Agent  │◀─────────────┘                     │
│  │  (config)    │    │  - Search    │                                    │
│  └──────────────┘    │  - Evaluate  │    ┌──────────────────────────┐    │
│                      │  - Retry     │───▶│   Audit Report           │    │
│                      └──────────────┘    │  - JSON + TXT output     │    │
│                             ▲            └──────────────────────────┘    │
│                             │                                            │
│                      ┌──────────────┐                                    │
│                      │  Possible    │  (Optional)                        │
│                      │  Answers     │  LLM pre-analyzes PDF              │
│                      └──────────────┘                                    │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Pipeline Flow

1. **Indexing Phase**
   - Extracts text from PDF (using pdfplumber)
   - Splits into chunks (1000 chars, 100 overlap)
   - Generates hybrid embeddings (sparse + dense)
   - Stores in Milvus with metadata (filename, doc_type, page_number)

2. **Audit Phase**
   - For each criterion in config:
     - Deep Agent performs hybrid search in Milvus
     - Retrieves relevant chunks with scores
     - LLM evaluates if criterion is PRESENT or ABSENT
     - If confidence < threshold, generates alternative queries and retries
   - Generates final report with evidence and page references

3. **Possible Answers (Optional Enhancement)**
   - Before evaluation, LLM reads the entire raw PDF
   - Generates "hints" for each criterion
   - Uses hints as additional search queries for better retrieval
   - Provides context to the evaluation LLM

## Project Structure

```
autoreg/
├── config.yaml              # Main configuration file
├── shared_config.py         # YAML configuration loader
├── run_pipeline.py          # Main orchestrator
├── docker-compose.yml       # Milvus infrastructure
├── model/
│   ├── milvus/              # Indexing module
│   │   ├── config.py
│   │   ├── indexer.py       # PDF → Milvus indexing
│   │   ├── collection.py    # Collection management
│   │   ├── extractor.py     # PDF text extraction
│   │   ├── chunker.py       # Text chunking
│   │   └── models.py
│   └── application/         # Auditing module
│       ├── config.py
│       ├── auditor.py       # Main audit logic
│       ├── retriever.py     # Hybrid search
│       ├── enhanced_retriever.py  # Search with possible answers
│       ├── evaluator.py     # LLM evaluation
│       ├── deep_agent.py    # Iterative search agent
│       ├── possible_answer_generator.py  # LLM pre-analysis
│       ├── raw_extractor.py # Raw PDF extraction
│       ├── output.py        # Report generation
│       ├── metrics.py       # Performance tracking
│       └── models.py
├── pdfs/                    # PDF documents to audit
└── output/                  # Generated reports
```

## Installation

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### 1. Clone and Setup

```bash
cd ~/autoreg

# Install dependencies with uv
uv sync

# Or with pip
pip install -r requirements.txt
```

### 2. Start Milvus

```bash
# Start Milvus vector database
docker-compose up -d

# Wait ~30 seconds for initialization
sleep 30

# Verify it's running
curl http://127.0.0.1:9091/healthz
# Should return: OK
```

### 3. Configure Environment

Create a `.env` file with your Spelling keys:

```bash
SPELLING_API_KEY="..."
SPELLING_API_BASE="..."
```

## Configuration

Edit `config.yaml` to customize the pipeline:

### Milvus Connection

```yaml
milvus:
  uri: "http://127.0.0.1:19530"
  collection_name: "collection_name" #if want to use another collection change the name
```

### Documents to Process

```yaml
documents:
  - path: "/home/user/pdfs/contract.pdf"
    doc_type: "contract"
    skip_if_indexed: true      # Skip if already in Milvus
    reset_collection: false    # Clear collection first (use for first doc)
```

### Audit Criteria

```yaml
audit_criteria:
  - query: "Is there a registered CNPJ for the Brokerage?"
    confidence: 0.8    # Minimum confidence threshold

  - query: "Is there a confidentiality clause?"
    confidence: 0.7

  - query: "Does the document mention penalties or fines?"
    confidence: 0.7
```

### Possible Answers Feature

```yaml
possible_answers:
  enabled: false    # Set to true to enable LLM pre-analysis
```

When enabled:
- LLM reads the entire PDF before evaluation
- Generates hints for each criterion
- Improves retrieval by using hints as additional queries
- Increases accuracy but adds LLM cost

### Deep Agent Settings

```yaml
deep_agent:
  enabled: true
  max_attempts: 3       # Max search iterations per criterion
  min_confidence: 0.7   # Confidence threshold to stop searching
```

### Pipeline Options

```yaml
pipeline:
  force_reindex: false     # Re-index even if document exists
  display_metrics: true    # Show performance metrics
  skip_indexing: false     # Skip to audit (assumes indexed)
```

## Usage

### Full Pipeline (Index + Audit)

```bash
uv run run_pipeline.py
```

### Index Only

```bash
uv run run_pipeline.py --index-only
```

### Audit Only (Documents Already Indexed)

```bash
uv run run_pipeline.py --audit-only
```

### Custom Configuration

```bash
uv run run_pipeline.py --config my_config.yaml
```

## Output

Reports are saved to the configured output directory:

### JSON Report (`audit_*.json`)

```json
{
  "document": "contract.pdf",
  "total_criteria": 8,
  "criteria_present": 6,
  "criteria_absent": 2,
  "compliance_rate": 75.0,
  "results": [
    {
      "criterion": "Is there a registered CNPJ?",
      "status": "PRESENT",
      "evidence": "CNPJ: 12.345.678/0001-90...",
      "confidence": 0.95,
      "pages": [1, 2]
    }
  ]
}
```

### Text Report (`audit_*.txt`)

```
======================================================================
AUDIT REPORT
======================================================================
Document: contract.pdf
Total Criteria: 8
Present: 6
Absent: 2
Compliance Rate: 75.0%

----------------------------------------------------------------------
DETAILED RESULTS
----------------------------------------------------------------------

1. Is there a registered CNPJ?
   Status: PRESENT
   Confidence: 95%
   Pages: 1, 2
   Evidence: CNPJ: 12.345.678/0001-90...
```

## How the AI Works

### 1. Hybrid Search

The system uses two types of embeddings for retrieval:

| Type | Model | Purpose |
|------|-------|---------|
| Sparse | BGE-M3 | Lexical matching (exact terms) |
| Dense | Spelling (gemini-embedding-001) | Semantic understanding (meaning) |

Results are combined using RRF (Reciprocal Rank Fusion) for best accuracy.

### 2. Deep Research Agent

For each criterion, the agent:

1. Searches Milvus with the criterion as query
2. LLM evaluates if evidence is sufficient
3. If confidence < threshold:
   - Generates alternative queries (synonyms, related terms)
   - Searches again with new queries
   - Accumulates context from all searches
4. Returns best result after max_attempts

### 3. Possible Answers (Optional)

When enabled, before the main evaluation:

1. Raw PDF text is extracted (preserving page numbers)
2. For each criterion, LLM is asked:
   > "Given this document, is there information about [criterion]?"
3. LLM returns: `{found: true/false, answer: "...", pages: [...]}`
4. These "hints" are used to:
   - Improve search queries (dual-query search)
   - Provide context to the evaluation LLM

This increases accuracy for complex documents but adds LLM API costs.

## Docker Commands

```bash
# Start Milvus
docker-compose up -d

# Stop Milvus
docker-compose down

# View logs
docker-compose logs -f milvus

# Check status
docker-compose ps

# Reset everything (delete all data)
docker-compose down
sudo rm -rf volumes/
docker-compose up -d
```

## Environment Variables

Alternative to `config.yaml`:

```bash
export MILVUS_URI="http://127.0.0.1:19530"
export COLLECTION_NAME="audit_docs_v4"
export OUTPUT_DIR="./output"
export CONFIG_PATH="/path/to/config.yaml"
```

## Troubleshooting

### Milvus Connection Error

```
MilvusException: Fail connecting to server on 127.0.0.1:19530
```

**Solution:**
```bash
docker-compose down
docker-compose up -d
sleep 30
curl http://127.0.0.1:9091/healthz
```

### Permission Denied on volumes/

```bash
sudo rm -rf volumes/
docker-compose up -d
```

### Import Errors

Ensure you're running from the project root:
```bash
cd ~/autoreg
uv run run_pipeline.py
```

## License

MIT