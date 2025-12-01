# HR Chatbot Evaluation

RAGAS-based evaluation framework for the HR Chatbot.

## Quick Start

### 1. Ensure services are running

```bash
docker-compose up -d postgres qdrant api
```

### 2. Create evaluation dataset

Copy the example and customize:

```bash
cp datasets/eval_dataset.example.json datasets/eval_dataset.json
# Edit datasets/eval_dataset.json with your test cases
```

### 3. Run evaluation locally (with uv)

```bash
cd evaluation
uv run python -m hr_chatbot_eval \
  --api-url http://localhost:8000/api/v1 \
  --dataset datasets/eval_dataset.json
```

### 4. Run evaluation via Docker

```bash
# From project root
docker-compose --profile eval run --rm evaluation
```

Or with custom options:

```bash
docker-compose --profile eval run --rm evaluation \
  python -m hr_chatbot_eval \
  --dataset datasets/eval_dataset.json
```

## Understanding Evaluation Metrics

This evaluation uses a **dual-purpose approach** to separately test retrieval and generation:

### Retrieval Metrics

Test whether RAG finds the right documents:

| Metric | What it measures |
|--------|------------------|
| **Source Precision** | % of retrieved URLs that match expected sources |
| **Source Recall** | % of expected sources that were retrieved |
| **Source F1** | Balanced measure of retrieval quality |

**How it works**: Compares API `sources[].url` against dataset `expected_sources`

### Generation Metrics (RAGAS)

Test whether the agent produces good answers given ideal contexts:

| Metric | What it measures |
|--------|------------------|
| **Faithfulness** | Is answer grounded in the provided contexts? |
| **Answer Relevancy** | Does answer address the question? |
| **Context Precision** | Are relevant contexts ranked higher? |
| **Context Recall** | Were all relevant contexts found? (requires ground_truth) |

**How it works**: RAGAS evaluates API answer against dataset `contexts` (golden references)

### Why Two Separate Metrics?

The API doesn't return full context content, only source metadata. Therefore:

1. **Retrieval metrics** validate RAG pipeline effectiveness
2. **Generation metrics** validate agent reasoning given ideal contexts
3. Dataset `contexts` serve as "golden references" for what perfect retrieval would return

This approach is valid because we're testing two distinct capabilities of the system.

### Interpreting Results

- **Low retrieval, high generation**: RAG isn't finding the right docs, but agent reasons well
- **High retrieval, low generation**: RAG works, but agent produces unfaithful/irrelevant answers
- **Both high**: System working well end-to-end
- **Both low**: Multiple issues to address

## Output

Results are saved to `results/` as JSON files with the following structure:

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "duration_seconds": 45.2,
  "num_samples": 10,
  "retrieval_metrics": {
    "avg_source_precision": 0.85,
    "avg_source_recall": 0.92,
    "avg_source_f1": 0.88
  },
  "generation_metrics": {
    "faithfulness": 0.78,
    "answer_relevancy": 0.89,
    "context_precision": 0.95,
    "context_recall": 0.88
  },
  "by_category": {
    "policy": {
      "num_samples": 8,
      "retrieval_metrics": {...},
      "generation_metrics": {...}
    },
    "procedure": {
      "num_samples": 6,
      "retrieval_metrics": {...},
      "generation_metrics": {...}
    }
  },
  "per_sample": [
    {
      "question": "...",
      "answer": "...",
      "category": "policy",
      "retrieval": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
      "generation": {"faithfulness": 0.85, "answer_relevancy": 0.92, ...}
    }
  ]
}
```

## Dataset Format

See `datasets/eval_dataset.example.json` for the expected format.

### Required Fields & Their Purpose

| Field | Purpose | Used By |
|-------|---------|---------|
| `question` | Question to ask API | All metrics |
| `contexts` | Golden reference contexts | Generation metrics (RAGAS) |
| `expected_sources` | URLs that should be retrieved | Retrieval metrics |
| `ground_truth` | Expected answer | Context recall metric |

**Important**: `contexts` are curated ideal contexts, NOT expected to match API retrieval exactly. They represent what a perfect RAG would retrieve and are used as golden references for testing generation quality.

### Optional Fields

- `category`: For grouping results (e.g., "policy", "procedure")

### Example

```json
{
  "version": "1.0",
  "description": "HR Chatbot evaluation dataset",
  "samples": [
    {
      "question": "What is the vacation policy?",
      "ground_truth": "Employees receive 15 days of paid vacation annually.",
      "contexts": [
        "Vacation Policy: All full-time employees are entitled to 15 days..."
      ],
      "category": "policy"
    }
  ]
}
```

## Configuration

The evaluation framework uses the same configuration as the main application:

- **Development**: Uses Ollama for LLM/embeddings
- **Production**: Uses OpenAI

Environment variables:
- `ENVIRONMENT`: Set to `development` or `production`
- `OLLAMA_BASE_URL`: Ollama server URL (default: http://localhost:11434)
- `OLLAMA_MODEL`: Ollama chat model (default: mistral)
- `OLLAMA_EMBEDDING_MODEL`: Ollama embedding model (default: nomic-embed-text)

## Directory Structure

```
evaluation/
├── pyproject.toml          # Package dependencies
├── Dockerfile              # Docker build for eval service
├── README.md               # This file
├── src/
│   └── hr_chatbot_eval/
│       ├── __init__.py             # Package exports
│       ├── __main__.py             # CLI entry point
│       ├── config.py               # RAGAS + Ollama configuration
│       ├── runner.py               # Evaluation execution
│       ├── retrieval_metrics.py    # URL-based precision/recall/F1
│       └── dataset_validator.py    # Dataset validation checks
├── datasets/
│   ├── .gitkeep
│   ├── eval_dataset.json           # Your evaluation dataset
│   └── eval_dataset.example.json  # Example dataset
└── results/
    └── .gitkeep            # Evaluation results output
```
