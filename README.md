# reviewer_sim (Exploratory Version)

Research prototype for LLM-based peer review simulation. This exploratory version enables comparison between human reviews and AI-generated reviews using local GGUF models without relying on external API services.

---

## Key Capabilities

- Export real peer review data from SQLite database to JSONL format
- Generate synthetic reviews using either a deterministic mock or a local Llama model
- Evaluate generated reviews against human reviews using text similarity metrics
- Run entirely offline with GPU acceleration on Apple Silicon

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         reviewer_sim Pipeline                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │    INGEST    │───▶│   GENERATE   │───▶│   EVALUATE   │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│         │                   │                   │                   │
│         ▼                   ▼                   ▼                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │  SQLite DB   │    │   Provider   │    │   Metrics    │          │
│  │  → JSONL     │    │   Interface  │    │  Comparison  │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│                             │                                        │
│              ┌──────────────┴──────────────┐                        │
│              ▼                              ▼                        │
│       ┌────────────┐              ┌─────────────────┐               │
│       │    Mock    │              │   LlamaCpp      │               │
│       │  Generator │              │   Generator     │               │
│       │(deterministic)            │  (GGUF + Metal) │               │
│       └────────────┘              └─────────────────┘               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
LLM-Reviewer/
├── data/
│   ├── gen_review.db              # Source database (1.5GB)
│   └── processed/
│       └── gen_review_sample.jsonl # Exported samples
├── models/
│   └── *.gguf                     # GGUF models (not committed)
├── outputs/
│   ├── results_mock.jsonl
│   └── results_llamacpp.jsonl
├── scripts/
│   └── summarize_results.py
├── src/reviewer_sim/
│   ├── __init__.py
│   ├── run.py                     # Entry point
│   ├── ingest/
│   │   ├── gen_review_sqlite.py   # SQLite exporter
│   │   └── load_jsonl.py          # JSONL loader
│   ├── generate/
│   │   ├── generator.py           # Thin wrapper
│   │   └── providers.py           # Mock + LlamaCpp generators
│   ├── evaluate/
│   │   └── metrics.py             # Evaluation functions
│   └── utils/
│       └── config.py              # ModelConfig + validation
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

### Run with Mock Generator
```bash
make run-mock
```

### Run with Local LLM
```bash
MODEL_PATH=models/llama-3-8b-instruct-q4_k_m.gguf make run-llamacpp
```

---

## Data Flow

### Input: Gen-Review Database
- **Source**: `data/gen_review.db` (SQLite)
- **Tables**: `SUBMISSION`, `REVIEW`, `GENAI_REVIEW`
- **Content**: Real peer reviews from academic conferences

### Export to JSONL
```bash
make export
```

### Per-Record Schema
```json
{
  "paper_id": "...",
  "title": "...",
  "abstract": "...",
  "metadata": { ... },
  "reviewer_profile": {
    "expertise": "representation learning",
    "tone": "neutral",
    "seniority": "senior"
  },
  "human_review": {
    "text": "...",
    "rating": 7,
    "confidence": 4
  }
}
```

### Output Schema
```json
{
  "paper_id": "...",
  "generated_review": {
    "text": "...",
    "score": 8
  },
  "metrics": {
    "tfidf_cosine": 0.155,
    "keyword_jaccard": 0.079,
    "score_abs_diff": 1.0
  }
}
```

---

## Reviewer Profile Augmentation

Each exported example includes a synthesized reviewer persona:

| Field | Source | Values |
|-------|--------|--------|
| `expertise` | `SUBMISSION.primary_area` | e.g., "representation learning", "NLP" |
| `tone` | Deterministic hash of `paper_id` | `"critical"` / `"neutral"` / `"positive"` |
| `seniority` | Deterministic hash of `paper_id` | `"junior"` / `"senior"` |

Deterministic assignment ensures reproducibility across runs with the same seed.

---

## Provider Interface

### MockGenerator
- **Purpose**: Fast iteration, testing, baseline comparison
- **Behavior**: Deterministic output based on `reviewer_profile`
- **Score adjustment**: `critical` (-2), `neutral` (0), `positive` (+2) from base score of 6

### LlamaCppGenerator
- **Purpose**: Real LLM inference with local GGUF models
- **Model loading**: Once in `__init__`, reused for all examples
- **GPU acceleration**: Metal on Apple Silicon (`n_gpu_layers=-1`)
- **JSON output**: Strict format requested, robust parsing with fallback

---

## Configuration (Environment Variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PROVIDER` | `"mock"` | `mock` or `llamacpp` |
| `MODEL_PATH` | None | Path to GGUF file (required for llamacpp) |
| `TEMPERATURE` | `0.2` | Sampling temperature |
| `TOP_P` | `0.95` | Nucleus sampling threshold |
| `MAX_TOKENS` | `600` | Maximum generation length |
| `N_CTX` | `4096` | Context window size |
| `N_GPU_LAYERS` | `-1` | GPU layers (-1 = all) |
| `SYSTEM_PROMPT` | (built-in) | LLM system instruction |
| `INPUT_JSONL` | `data/sample/examples.jsonl` | Input file path |
| `OUTPUT_JSONL` | `outputs/results.jsonl` | Output file path |

---

## Evaluation Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| `tfidf_cosine` | Cosine similarity of TF-IDF vectors | 0-1 (higher = more similar) |
| `keyword_jaccard` | Jaccard index of token sets | 0-1 (higher = more overlap) |
| `score_abs_diff` | Absolute difference between ratings | 0+ (lower = better agreement) |

### Summarize Results
```bash
python scripts/summarize_results.py outputs/results_llamacpp.jsonl
```

---

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make export` | SQLite → JSONL export |
| `make run` | Run pipeline with defaults |
| `make run-mock` | Run with mock generator |
| `make run-llamacpp` | Run with GGUF model (requires `MODEL_PATH`) |
| `make smoke` | Quick 1-example test |
| `make test-imports` | Verify all modules import |

---

## Experimental Results

### Mock Generator Baseline
| Metric | Mean | Median |
|--------|------|--------|
| tfidf_cosine | 0.1361 | 0.1395 |
| keyword_jaccard | 0.0874 | 0.0769 |

### LlamaCpp (Llama 3 8B Q4_K_M)
| Metric | Mean | Median |
|--------|------|--------|
| tfidf_cosine | **0.1551** | **0.1652** |
| keyword_jaccard | 0.0795 | 0.0762 |
| score_abs_diff | 100% non-null | - |

**Key Observation**: LLM reviews show ~14% higher semantic similarity (tfidf_cosine) vs mock.

---

## Sample Generated Review (LlamaCpp)

**Paper**: "Online Learning Rate Adaptation with Hypergradient Descent"

```json
{
  "text": "The authors propose a novel method for adapting the learning rate 
           in gradient-based optimizers, which is easy to implement and shows 
           promising results in various optimization problems. The concept of 
           hypergradient descent is well-explained, and the additional 
           computational cost is minimal. However, the paper could benefit 
           from more detailed analysis of the method's performance in different 
           scenarios and a more comprehensive comparison with existing methods. 
           Overall, the paper is well-written and presents an interesting 
           contribution to the field. Score: 8",
  "score": 8
}
```

---

## Error Handling

- **Per-example try/except**: Pipeline continues even if individual examples fail
- **JSON parsing fallback**: If LLM output isn't valid JSON, returns `{"text": raw_output, "score": None}`
- **Config validation**: Clear error messages for missing `MODEL_PATH` when using llamacpp

---

## Docker

```bash
docker build -t reviewer_sim .
docker run -v $(pwd)/outputs:/app/outputs reviewer_sim
```

For llamacpp, mount the models directory:
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
| Deterministic mock | Reproducible baselines for testing |
| JSON with fallback | Pipeline never crashes on bad output |
| Config via env vars | Easy Docker/CI integration |
| Makefile targets | One-command reproducible runs |
| models/ in .gitignore | Don't commit multi-GB model files |

---

## Future Work

1. **Prompt engineering**: Experiment with different system prompts and review templates
2. **Model comparison**: Test other GGUF models (Mistral, Phi, etc.)
3. **Evaluation expansion**: Add BLEU, ROUGE, BERTScore metrics
4. **Reviewer profile impact**: Analyze how tone/expertise affect generated reviews
5. **Human evaluation**: Blind comparison study of mock vs LLM vs human reviews
6. **Fine-tuning**: Create domain-specific adapter for peer review style

---

## Full Custom Run Example

```bash
MODEL_PROVIDER=llamacpp \
MODEL_PATH=models/llama-3-8b-instruct-q4_k_m.gguf \
N_CTX=4096 \
MAX_TOKENS=400 \
TEMPERATURE=0.2 \
TOP_P=0.95 \
INPUT_JSONL=data/processed/gen_review_sample.jsonl \
OUTPUT_JSONL=outputs/results.jsonl \
PYTHONPATH=src python -m reviewer_sim.run
```

---

*Exploratory Version - Research Prototype*
