# 3-Phase LLM Review Evaluation Framework

This framework evaluates whether LLMs can identify the same salient technical points as human reviewers when reviewing papers.

## Evaluation Design

### Phase 1: Build Ground Truth (Human Keywords)
- **Input**: Human review text (NOT the abstract)
- **Process**: Extract canonical keywords/claims from human reviews using DeepSeek
- **Output**: Keywords representing what human reviewers found important

### Phase 2: Generate LLM Reviews (Blind)
- **Input**: Paper abstract only (no access to human reviews)
- **Process**: Each model generates a structured review
- **Output**: LLM reviews with strengths, weaknesses, questions, keywords

### Phase 3: Compare Coverage
- **Input**: Human keywords (Phase 1) + LLM reviews (Phase 2)
- **Process**: Measure keyword recall, precision, F1
- **Output**: Per-model coverage scores + visualizations

---

## Quick Start

```bash
export TOGETHER_API_KEY="your-key-here"

# Phase 1: Extract keywords from human reviews
python evaluation/phase1_extract_human_keywords.py \
  --input data/processed/review_subset_enriched.jsonl \
  --output evaluation/outputs/human_keywords.jsonl

# Phase 2: Generate LLM reviews from abstracts
python evaluation/phase2_generate_llm_reviews.py \
  --input data/processed/review_subset_enriched.jsonl \
  --output-dir evaluation/outputs

# Phase 3: Compare coverage
python evaluation/phase3_compare_coverage.py \
  --human-keywords evaluation/outputs/human_keywords.jsonl \
  --llm-reviews evaluation/outputs/llm_reviews.jsonl \
  --output-dir evaluation/outputs
```

---

## Scripts

### Phase 1: `phase1_extract_human_keywords.py`

Extract canonical keywords from human reviews.

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--input` | Yes | - | Input JSONL with papers and human reviews |
| `--output` | Yes | - | Output JSONL for keywords |
| `--model` | No | `deepseek-ai/DeepSeek-V3` | Model for extraction |
| `--max-concurrency` | No | 5 | Concurrent API requests |
| `--limit` | No | - | Limit papers to process |

**Output format:**
```json
{
  "paper_id": "abc123",
  "keywords": ["ablation study", "baseline comparison", ...],
  "strengths": ["novel architecture", ...],
  "weaknesses": ["missing experiments", ...]
}
```

### Phase 2: `phase2_generate_llm_reviews.py`

Generate reviews from abstracts using multiple models.

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--input` | Yes | - | Input JSONL with papers |
| `--output-dir` | Yes | - | Output directory |
| `--models` | No | (predefined) | Comma-separated model IDs |
| `--temperature` | No | 0.3 | Sampling temperature |
| `--limit` | No | - | Limit papers to process |

**Default models:**
- `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo`
- `mistralai/Mixtral-8x7B-Instruct-v0.1`
- `meta-llama/Llama-3.3-70B-Instruct-Turbo`
- `Qwen/Qwen2.5-7B-Instruct-Turbo`
- `deepseek-ai/DeepSeek-V3`

### Phase 3: `phase3_compare_coverage.py`

Compare LLM reviews against human keywords.

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--human-keywords` | Yes | - | Phase 1 output |
| `--llm-reviews` | Yes | - | Phase 2 output |
| `--output-dir` | Yes | - | Output directory |

---

## Metrics

### Exact Recall
Proportion of human keywords that appear exactly in LLM keywords.

### Fuzzy Recall  
Proportion of human keywords mentioned anywhere in the LLM review (with fuzzy matching for synonyms/variants).

### Precision
Proportion of LLM keywords that match human keywords.

### F1 Score
Harmonic mean of precision and fuzzy recall.

---

## Output Files

```
evaluation/outputs/
├── human_keywords.jsonl        # Phase 1: Ground truth
├── human_keywords.meta.json
├── llm_reviews.jsonl           # Phase 2: LLM reviews
├── phase2_meta.json
├── coverage_results.jsonl      # Phase 3: Detailed results
├── coverage_report.md          # Summary report
├── phase3_meta.json
└── charts/
    ├── coverage_by_model.png
    ├── metrics_comparison.png
    ├── missed_keywords.png
    └── model_ranking.png
```

---

## Research Questions

This evaluation helps answer:

1. **Can LLMs identify salient points?** Do LLMs focus on the same technical aspects as human reviewers?

2. **Which models perform best?** How do different model sizes/families compare?

3. **What do LLMs miss?** Which human-identified concerns are LLMs blind to?

4. **Do LLMs hallucinate?** Do LLMs raise concerns not mentioned by humans?
