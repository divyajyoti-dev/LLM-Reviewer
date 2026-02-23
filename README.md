# LLM-Reviewer

Research prototype for LLM-based peer review simulation. Enables comparison between human reviews and AI-generated reviews using local GGUF models or cloud APIs.

---

## Project Structure

```
LLM-Reviewer/
├── data/
│   ├── gen_review.db                    # Source database (~1.5GB, not committed)
│   └── processed/
│       ├── review_subset.jsonl          # Clean export (200 papers, 2021+)
│       └── review_subset_enriched.jsonl # + LLM-classified primary_area
├── models/
│   └── *.gguf                           # GGUF models (not committed)
├── outputs/                             # Generated results (not committed)
├── scripts/
│   └── summarize_results.py             # Results summary tool
├── src/reviewer_sim/
│   ├── __init__.py
│   ├── run.py                           # Pipeline entry point
│   ├── ingest/
│   │   ├── export_review_subset.py      # SQLite → JSONL exporter
│   │   ├── enrich_primary_area.py       # LLM classification enrichment
│   │   └── load_jsonl.py                # JSONL loader utility
│   ├── generate/
│   │   └── providers.py                 # Mock + LlamaCpp generators
│   ├── evaluate/
│   │   └── metrics.py                   # Evaluation metrics
│   └── utils/
│       └── config.py                    # ModelConfig + validation
├── tests/
│   └── test_export_review_subset.py     # Pytest tests
├── .gitignore
├── Dockerfile
├── Makefile
├── README.md
└── requirements.txt
```

---

## Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt
```

### For Apple Silicon (M1/M2/M3) with Metal GPU
```bash
CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install llama-cpp-python
```

---

## Data Pipeline

### 1. Export from SQLite
Export a clean subset of papers with reviews:

```bash
make export
```

Or with custom options:
```bash
PYTHONPATH=src python -m reviewer_sim.ingest.export_review_subset \
  --db-path data/gen_review.db \
  --out-path data/processed/review_subset.jsonl \
  --n 200 --seed 42 --min-year 2021 --min-review-chars 50
```

**Output schema:**
```json
{
  "paper_id": "abc123",
  "title": "Paper Title",
  "abstract": "...",
  "primary_area": "general",
  "year": 2022,
  "review": {"main_review": "..."},
  "meta": {
    "source": "gen_review_sqlite",
    "filters": {"min_year": 2021, "min_review_chars": 50},
    "exported_at": "2024-01-01T00:00:00+00:00"
  }
}
```

### 2. Enrich with LLM Classification
Classify papers by research area using Together AI:

```bash
export TOGETHER_API_KEY="your-key"
make enrich
```

Or with custom options:
```bash
PYTHONPATH=src python -m reviewer_sim.ingest.enrich_primary_area \
  --in-path data/processed/review_subset.jsonl \
  --out-path data/processed/review_subset_enriched.jsonl \
  --model "mistralai/Mixtral-8x7B-Instruct-v0.1" \
  --max-concurrency 8
```

**Added fields:**
```json
{
  "primary_area_llm": "ml",
  "primary_area_llm_confidence": 0.9,
  "meta": {
    "llm_labeling": {
      "provider": "together",
      "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
      "prompt_version": "area-v1",
      "temperature": 0,
      "timestamp_utc": "..."
    }
  }
}
```

**Label distribution (200 papers):**
| Label | Count | % |
|-------|-------|---|
| ml | 128 | 64% |
| computer_vision | 25 | 12.5% |
| nlp | 23 | 11.5% |
| theory | 9 | 4.5% |
| robotics | 7 | 3.5% |
| bio_medical | 3 | 1.5% |
| graphics | 2 | 1% |
| other | 2 | 1% |
| multi_modal | 1 | 0.5% |

---

## Review Generation

### Run with Mock Generator
```bash
make run-mock
```

### Run with Local LLM (GGUF)
```bash
MODEL_PATH=models/llama-3-8b-instruct-q4_k_m.gguf make run-llamacpp
```

### Full Custom Run
```bash
PYTHONPATH=src \
MODEL_PROVIDER=llamacpp \
MODEL_PATH=models/your-model.gguf \
INPUT_JSONL=data/processed/review_subset.jsonl \
OUTPUT_JSONL=outputs/results.jsonl \
python -m reviewer_sim.run
```

---

## Provider Interface

### MockGenerator
- Deterministic output based on `reviewer_profile`
- Score adjustment: `critical` (-2), `neutral` (0), `positive` (+2) from base 6
- Fast iteration and testing

### LlamaCppGenerator
- Local GGUF model inference via llama-cpp-python
- Metal GPU acceleration on Apple Silicon
- Model loaded once, reused for all examples
- Robust JSON parsing with fallback

---

## Configuration (Environment Variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PROVIDER` | `mock` | `mock` or `llamacpp` |
| `MODEL_PATH` | None | Path to GGUF file (required for llamacpp) |
| `TEMPERATURE` | `0.2` | Sampling temperature |
| `TOP_P` | `0.95` | Nucleus sampling threshold |
| `MAX_TOKENS` | `600` | Maximum generation length |
| `N_CTX` | `4096` | Context window size |
| `N_GPU_LAYERS` | `-1` | GPU layers (-1 = all) |
| `INPUT_JSONL` | `data/processed/review_subset.jsonl` | Input file |
| `OUTPUT_JSONL` | `outputs/results.jsonl` | Output file |
| `TOGETHER_API_KEY` | None | API key for enrichment |

---

## Evaluation Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| `tfidf_cosine` | Cosine similarity of TF-IDF vectors | 0-1 (higher = more similar) |
| `keyword_jaccard` | Jaccard index of token sets | 0-1 (higher = more overlap) |
| `score_abs_diff` | Absolute difference between ratings | 0+ (lower = better) |

### Summarize Results
```bash
make summarize OUTPUT_JSONL=outputs/results.jsonl
```

---

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make export` | Export 200 papers from SQLite (2021+) |
| `make enrich` | Enrich with LLM-classified primary areas |
| `make run` | Run review simulation pipeline |
| `make run-mock` | Run with mock generator |
| `make run-llamacpp` | Run with GGUF model (requires `MODEL_PATH`) |
| `make test` | Run pytest tests |
| `make test-imports` | Verify all modules import |
| `make summarize` | Summarize results JSONL |

---

## Database Statistics

| Table | Count |
|-------|-------|
| Papers (SUBMISSION) | 32,652 |
| Human Reviews (REVIEW) | 124,615 |
| Papers with non-empty reviews | 9,766 |
| GenAI Reviews | 81,850 |

**Processed subset:** 200 papers from 2021+ with reviews ≥50 chars

---

## Tests

```bash
make test
```

5 tests covering:
- Year filtering
- Deterministic sampling
- Empty review filtering
- Short review filtering
- Output schema validation

---

## Docker

```bash
docker build -t reviewer_sim .
docker run -v $(pwd)/outputs:/app/outputs reviewer_sim
```

For llamacpp:
```bash
docker run -v $(pwd)/models:/app/models \
           -v $(pwd)/outputs:/app/outputs \
           -e MODEL_PROVIDER=llamacpp \
           -e MODEL_PATH=/app/models/your-model.gguf \
           reviewer_sim
```

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Provider pattern | Swap mock↔LLM with one env var |
| Model loaded once | Avoid 30+ second reload per example |
| Deterministic export | Same seed = identical output |
| LLM enrichment separate | Original data unchanged, derived datasets |
| Per-example error handling | Pipeline continues on failures |
| Config via env vars | Easy Docker/CI integration |
| models/ in .gitignore | Don't commit multi-GB model files |

---

## Experimental Results

### Mock vs LlamaCpp (Llama 3 8B Q4_K_M)

| Metric | Mock | LlamaCpp |
|--------|------|----------|
| tfidf_cosine (mean) | 0.136 | **0.155** |
| keyword_jaccard (mean) | 0.087 | 0.080 |

LLM reviews show ~14% higher semantic similarity to human reviews.

---

## Future Work

1. Prompt engineering for review quality
2. Model comparison (Mistral, Phi, etc.)
3. Additional metrics (BLEU, ROUGE, BERTScore)
4. Reviewer profile impact analysis
5. Human evaluation study
6. Domain-specific fine-tuning
