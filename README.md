# ADG Assistant

A multi-agent RAG system for querying government policy documents. Built with CrewAI, LlamaIndex, and Ollama for fully local inference.

## Overview

This system processes PDF policy, HR, Info Security documents and provides accurate, cited answers to user questions through a chat interface. It uses a team of specialized AI agents to route queries, optimize searches, generate answers, and validate responses.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Web Interface                          │
│                    (Open WebUI - Port 3000)                 │
└─────────────────────────────────────────────────────────────┘
                              │
                    POST /v1/chat/completions
                              │
┌─────────────────────────────────────────────────────────────┐
│                      Backend API                            │
│                   (FastAPI - Port 8000)                     │
│                                                             │
│   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐     │
│   │ Router  │ → │Rewriter │ → │ Answer  │ → │Validator│     │
│   │  Agent  │   │  Agent  │   │  Agent  │   │  Agent  │     │
│   └─────────┘   └─────────┘   └─────────┘   └─────────┘     │
│                       │                                     │
│                       ▼                                     │
│   ┌─────────────────────────────────────────────────────┐   │
│   │      Hybrid Retrieval (Vector + BM25)               │   │
│   │                    ↓                                │   │
│   │      Cross-Encoder Reranking                        │   │
│   └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
          │                    │                    │
          ▼                    ▼                    ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│    PostgreSQL    │  │      Ollama      │  │     Phoenix      │
│   (pgvector)     │  │    (llama3.1)    │  │   (Tracing)      │
│   Port 5432      │  │   Port 11434     │  │   Port 6006      │
└──────────────────┘  └──────────────────┘  └──────────────────┘
```

## Features

- **Multi-Agent Pipeline**: Specialized agents for routing, query optimization, answer generation, and validation
- **Hybrid Retrieval**: Combines semantic vector search with BM25 keyword matching
- **Cross-Encoder Reranking**: Uses BGE reranker for accurate result ordering
- **Local Inference**: Runs entirely on local hardware via Ollama (no API costs)
- **Conversation Memory**: PostgreSQL-backed session history for contextual follow-ups
- **Full Observability**: Phoenix tracing for debugging and performance monitoring
- **Citation Support**: Answers include numbered references to source documents

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Ollama (llama3.1) |
| Embeddings | BGE-base-en-v1.5 |
| Vector Store | PostgreSQL + pgvector |
| Reranker | BGE-reranker-base |
| Agents | CrewAI |
| Backend | FastAPI |
| Frontend | Open WebUI |
| Observability | Arize Phoenix |
| Evaluation | RAGAs |

## Quick Start

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support (recommended)
- 16GB+ RAM

### Setup

1. Clone the repository:
```bash
git clone <repo-url>
cd abu-dhabi-rag
```

2. Create environment file:
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY (only needed for evaluation)
```

3. Add your PDF documents to the `documents/` folder

4. Start the services:
```bash
docker-compose up -d
```

5. Pull the LLM model:
```bash
docker-compose exec ollama ollama pull llama3.1
```

6. Index the documents:
```bash
docker-compose exec backend python indexer.py
```

7. Access the interfaces:
   - Chat UI: http://localhost:3000
   - API Docs: http://localhost:8000/docs
   - Phoenix Dashboard: http://localhost:6006

## Project Structure

```
.
├── backend/
│   ├── config.py           # Configuration management
│   ├── prompts.py          # System prompts
│   ├── phoenix_prompts.py  # Prompt lifecycle management
│   ├── tracing.py          # Phoenix observability
│   ├── rag.py              # Document processing and retrieval
│   ├── memory.py           # Conversation memory
│   ├── agents.py           # CrewAI agent pipeline
│   └── main.py             # FastAPI application
├── scripts/
│   └── init-db.sql         # Database initialization
├── documents/              # Place PDF documents here
├── indexer.py              # Document indexing script
├── evaluator.py            # RAGAs evaluation script
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/models` | GET | List available models |
| `/v1/chat/completions` | POST | Chat endpoint (OpenAI-compatible) |
| `/v1/prompts` | GET | List registered prompts |

## Evaluation

Run RAGAs evaluation to measure system performance:

```bash
# Set OpenAI API key (used as judge model)
export OPENAI_API_KEY=OPENAI_API_KEY

# Run evaluation
docker-compose exec backend python evaluator.py
```

Metrics evaluated:
- **Faithfulness**: Is the answer grounded in the retrieved context?
- **Answer Relevancy**: Does the answer address the question?
- **Context Precision**: Are the retrieved documents relevant?

## Configuration

Key environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | llama3.1 | LLM model to use |
| `EMBED_MODEL` | BAAI/bge-base-en-v1.5 | Embedding model |
| `PHOENIX_ENDPOINT` | http://phoenix:4317 | Tracing endpoint |
| `OPENAI_API_KEY` | - | For evaluation only |

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Ollama | Local inference, no API costs, data privacy |
| pgvector | Integrated with PostgreSQL (already used for memory) |
| CrewAI | Clean agent abstraction, good Ollama integration |
| Hybrid retrieval | Combines semantic understanding with keyword precision |
| BGE models | High quality open-source embeddings and reranking |

## Troubleshooting

**Ollama not responding:**
```bash
docker-compose logs ollama
docker-compose exec ollama ollama list
```

**Documents not indexed:**
```bash
docker-compose exec backend ls -la /app/documents
docker-compose exec backend python indexer.py
```

**Check traces:**
Open http://localhost:6006 to view request traces and latencies.

## License

MIT
