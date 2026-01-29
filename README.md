# Audit Pipeline

Sistema unificado de indexação e auditoria de documentos PDF usando Milvus e busca híbrida.

## Estrutura

```
audit_pipeline/
├── config.yaml           # Configuração centralizada
├── shared_config.py      # Carregador de configuração YAML
├── run_pipeline.py       # Orquestrador principal
└── model/
    ├── milvus/           # Módulo de indexação
    │   ├── config.py
    │   ├── indexer.py
    │   ├── collection.py
    │   ├── extractor.py
    │   ├── chunker.py
    │   └── models.py
    └── application/      # Módulo de auditoria
        ├── config.py
        ├── auditor.py
        ├── retriever.py
        ├── evaluator.py
        ├── deep_agent.py
        ├── output.py
        ├── metrics.py
        └── models.py
```

## Configuração

Edite o arquivo `config.yaml` para configurar:

### Conexão Milvus
```yaml
milvus:
  uri: "http://127.0.0.1:19530"
  collection_name: "audit_docs_v3"
```

### Documentos a Processar
```yaml
documents:
  - path: "/caminho/para/documento.pdf"
    doc_type: "contract"
    skip_if_indexed: true      # Pula se já estiver indexado
    reset_collection: false    # Limpa collection antes (primeiro doc)
```

### Critérios de Auditoria
```yaml
audit_criteria:
  - query: "O documento possui CNPJ?"
    confidence: 0.8
  
  - query: "Existe cláusula de confidencialidade?"
    confidence: 0.7
```

### Opções do Pipeline
```yaml
pipeline:
  force_reindex: false      # Força reindexação mesmo se existir
  display_metrics: true     # Exibe métricas de performance
  skip_indexing: false      # Pula direto para auditoria
```

## Uso

### Pipeline Completo (indexação + auditoria)
```bash
uv run run_pipeline.py
```

### Com configuração específica
```bash
uv run run_pipeline.py --config minha_config.yaml
```

### Somente indexação
```bash
uv run run_pipeline.py --index-only
```

### Somente auditoria (documentos já indexados)
```bash
uv run run_pipeline.py --audit-only
```

## Execução Individual

### Indexar documento específico
```bash
cd model/milvus
uv run main.py --pdf /caminho/documento.pdf --doc-type contract
```

### Listar documentos indexados
```bash
cd model/milvus
uv run main.py --list
```

### Auditar documento específico
```bash
cd model/application
uv run main.py --document "documento.pdf" --doc-type contract
```

## Fluxo do Pipeline

1. **Carrega configuração** do `config.yaml`
2. **Para cada documento:**
   - Verifica se já está indexado (filename + doc_type)
   - Se não indexado: extrai texto, cria chunks, gera embeddings, insere no Milvus
   - Se indexado: pula direto para auditoria
3. **Executa auditoria:**
   - Busca híbrida (sparse + dense) para cada critério
   - Deep Agent faz múltiplas tentativas se necessário
   - Gera relatório com evidências e páginas
4. **Salva outputs:**
   - JSON com todos os dados
   - TXT com resumo formatado

## Variáveis de Ambiente

O sistema também suporta configuração via variáveis de ambiente (fallback se não houver config.yaml):

```bash
export MILVUS_URI="http://127.0.0.1:19530"
export COLLECTION_NAME="audit_docs_v3"
export OUTPUT_DIR="./output"
export CONFIG_PATH="/caminho/config.yaml"
```

## Dependências

Adicione ao seu `pyproject.toml`:

```toml
[project]
dependencies = [
    "pymilvus",
    "pdfplumber",
    "langchain-text-splitters",
    "pydantic",
    "python-dotenv",
    "rich",
    "pyyaml",
    "spelling",  # Biblioteca interna para embeddings
]
```
