# Multi-Model Review Experiments

Scripts for running comparative experiments across multiple LLM review generators.

## Overview

1. **`run_multi_model_reviews.py`**: Generate reviews for the same papers using multiple models
2. **`eval_topic_coverage.py`**: Compare topic coverage between model outputs

## Quick Start

### 1. Generate Reviews with Multiple Models

```bash
# Set API key for cloud models
export TOGETHER_API_KEY="your-key"

# Run mock + Mixtral on 20 papers
python experiments/run_multi_model_reviews.py \
  --input data/processed/review_subset.jsonl \
  --output-dir experiments/outputs/run_001 \
  --models "mock,mistralai/Mixtral-8x7B-Instruct-v0.1" \
  --limit 20 \
  --seed 42 \
  --temperature 0
```

### 2. Evaluate Topic Coverage

```bash
# Using keyword extraction (fast, no API needed)
python experiments/eval_topic_coverage.py \
  --input experiments/outputs/run_001/reviews.jsonl \
  --output-dir experiments/outputs/run_001 \
  --extractor keywords

# Using LLM extraction (better quality, requires API)
python experiments/eval_topic_coverage.py \
  --input experiments/outputs/run_001/reviews.jsonl \
  --output-dir experiments/outputs/run_001 \
  --extractor llm
```

## Model Specification Formats

| Format | Provider | Example |
|--------|----------|---------|
| `mock` | Mock generator | `--models "mock"` |
| `org/model-name` | Together AI | `--models "mistralai/Mixtral-8x7B-Instruct-v0.1"` |
| `llamacpp:/path/to/model.gguf` | Local GGUF | `--models "llamacpp:models/llama-3-8b.gguf"` |

### Multiple Models

Comma-separate to compare multiple:

```bash
--models "mock,mistralai/Mixtral-8x7B-Instruct-v0.1,meta-llama/Llama-3-8b-chat-hf"
```

## Output Files

### From `run_multi_model_reviews.py`:

```
experiments/outputs/run_001/
├── reviews.jsonl      # All generated reviews
└── run_meta.json      # Run configuration
```

**reviews.jsonl schema:**
```json
{
  "paper_id": "abc123",
  "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
  "review_text": "Summary: This paper...",
  "score": 7,
  "meta": {
    "provider": "together",
    "prompt_version": "review-v1",
    "temperature": 0,
    "latency_ms": 2345,
    "token_usage": {"prompt_tokens": 800, "completion_tokens": 400}
  }
}
```

### From `eval_topic_coverage.py`:

```
experiments/outputs/run_001/
├── topics_extracted.jsonl   # Raw extracted topics per review
├── topic_eval.csv           # Per-paper-pair metrics
└── topic_eval_summary.md    # Aggregated summary
```

**topic_eval.csv columns:**
- `paper_id`: Paper identifier
- `model_a`, `model_b`: Models being compared
- `overlap_count`: Topics in both reviews
- `union_count`: Total unique topics
- `jaccard`: Overlap/Union ratio (0-1)
- `topics_only_a`: Topics unique to model A
- `topics_only_b`: Topics unique to model B

## Full Example Workflow

```bash
# 1. Set up environment
export TOGETHER_API_KEY="your-key"
export PYTHONPATH=src

# 2. Run experiment with 3 models
python experiments/run_multi_model_reviews.py \
  --input data/processed/review_subset.jsonl \
  --output-dir experiments/outputs/comparison_001 \
  --models "mock,mistralai/Mixtral-8x7B-Instruct-v0.1,meta-llama/Llama-3-8b-chat-hf" \
  --limit 50 \
  --temperature 0

# 3. Evaluate with keyword extraction (fast)
python experiments/eval_topic_coverage.py \
  --input experiments/outputs/comparison_001/reviews.jsonl \
  --output-dir experiments/outputs/comparison_001 \
  --extractor keywords

# 4. View results
cat experiments/outputs/comparison_001/topic_eval_summary.md
```

## CLI Reference

### run_multi_model_reviews.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | required | Input papers JSONL |
| `--output-dir` | required | Output directory |
| `--models` | required | Comma-separated model list |
| `--limit` | 0 (all) | Limit papers processed |
| `--seed` | 42 | Random seed |
| `--temperature` | 0 | Sampling temperature |
| `--max-concurrency` | 4 | Concurrent API requests |

### eval_topic_coverage.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | required | reviews.jsonl from step 1 |
| `--output-dir` | required | Output directory |
| `--extractor` | keywords | `keywords` or `llm` |
| `--model-a` | None | Compare specific pair (optional) |
| `--model-b` | None | Compare specific pair (optional) |
| `--max-concurrency` | 8 | Concurrent API requests (for llm) |

## Error Handling

- Both scripts continue on per-paper errors
- Failed reviews have `meta.error` field
- Summary reports success/failure counts

## Determinism

With `--temperature 0` and `--seed 42`:
- Mock generator: fully deterministic
- Together AI: mostly deterministic (some variance possible)
- LlamaCpp: mostly deterministic

## Resource Usage

| Model Type | Latency (per review) | API Cost |
|------------|---------------------|----------|
| Mock | <1ms | Free |
| Together AI | 2-5s | ~$0.001 |
| LlamaCpp (8B) | 10-30s | Free (local GPU) |
