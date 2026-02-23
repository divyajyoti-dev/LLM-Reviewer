# LLM Peer Review Evaluation — Semantic Coverage Analysis

**Branch**: `evaluation_v3`
**Date**: February 23, 2026

---

## Research Question

> If an LLM reviews a paper using only its abstract, how much of what human reviewers care about does it actually capture?

We measure this by extracting the **top 10 keywords** from both human and LLM reviews, then checking how many human keywords are **semantically present** in the LLM output. If 5 out of 10 match, that's **50% coverage**.

---

## Approach

### What Changed from the Previous Evaluation

| Previous Approach | New Approach |
|---|---|
| Used a single human review per paper | **All human reviews per paper consolidated** (avg 3.9 reviews/paper) |
| String-based matching (exact + fuzzy) | **Semantic matching** via sentence embeddings |
| Complex F1/precision/recall metrics | **Simple coverage: X out of 10 = X0%** |
| No sentence-level analysis | **Sentence-level mapping** between human and LLM reviews |

### 3-Step Pipeline

```
Step 1: Consolidate Reviews    Step 2: Generate LLM Reviews    Step 3: Semantic Evaluation
┌─────────────────────┐       ┌──────────────────────────┐    ┌───────────────────────────┐
│ SQLite DB            │       │ Paper abstract only      │    │ Extract top 10 keywords   │
│ (124K human reviews) │──────>│ (blind — no human review │───>│ from BOTH sides via LLM   │
│                      │       │  access)                 │    │                           │
│ Group ALL reviews    │       │ 5 models via Together AI │    │ Embed keywords with       │
│ per paper            │       │ 200 papers each          │    │ sentence-transformers     │
│                      │       │                          │    │                           │
│ Output: 200 papers   │       │ Output: 1000 reviews     │    │ Cosine similarity > 0.55  │
│ with merged reviews  │       │                          │    │ = semantic match          │
└─────────────────────┘       └──────────────────────────┘    │                           │
                                                               │ Coverage = matched / 10   │
                                                               └───────────────────────────┘
```

### Step 1: Consolidate Human Reviews

Instead of using a single review per paper, we go back to the source database and **merge all human reviews** for each paper into one combined document.

- **Source**: SQLite database with 32,652 papers and 124,615 reviews
- **Filter**: Papers from 2021+, reviews >= 50 characters
- **Result**: 200 papers, each with **3–5 consolidated reviews** (mean: 3.88)

This gives a more complete picture of what humans collectively found important.

### Step 2: Generate LLM Reviews (Blind)

Each LLM receives **only the paper title and abstract** — no access to human reviews. It generates a structured review with summary, strengths, weaknesses, and questions.

**Models evaluated** (all via Together AI):

| Model | Parameters | Family |
|-------|-----------|--------|
| Meta-Llama-3.1-8B-Instruct-Turbo | 8B | Llama 3.1 |
| Mixtral-8x7B-Instruct-v0.1 | 8x7B (MoE) | Mixtral |
| Llama-3.3-70B-Instruct-Turbo | 70B | Llama 3.3 |
| Qwen2.5-7B-Instruct-Turbo | 7B | Qwen 2.5 |
| DeepSeek-V3 | Large MoE | DeepSeek |

- **200 papers x 5 models = 1,000 LLM reviews generated**
- **1,000/1,000 success rate** (100%)

### Step 3: Semantic Keyword Evaluation

This is the core evaluation step:

1. **Keyword Extraction**: Use DeepSeek-V3 to extract exactly **10 keywords** from each human review and each LLM review. Keywords represent what the reviewer found most important (methods, concerns, contributions, gaps).

2. **Semantic Matching**: Embed all keywords using `sentence-transformers` (`all-MiniLM-L6-v2`). Compute pairwise cosine similarity between human and LLM keywords. If the best match for a human keyword exceeds **0.55 similarity**, it counts as a match.

3. **Coverage Score**: `matched_keywords / 10 = coverage %`

4. **Sentence Mapping**: Additionally, map individual sentences between human and LLM reviews to show qualitative examples of same concepts expressed differently.

---

## Dataset

| Statistic | Value |
|-----------|-------|
| Papers evaluated | 200 |
| Year range | 2021+ |
| Human reviews per paper | 3–5 (mean: 3.88) |
| Total human reviews used | ~776 |
| LLM reviews generated | 1,000 (5 models x 200 papers) |
| Keywords per review | 10 |
| Total keyword comparisons | 10,000 |

---

## Results

### Overall Finding

**LLMs cover 40.8% of human reviewer keywords on average.**

In other words, out of every 10 things human reviewers emphasize, LLMs independently identify about 4 of them from the abstract alone. The remaining ~59% of human concerns are missed.

### Per-Model Performance

| Rank | Model | Mean Coverage | Std Dev |
|------|-------|:------------:|:-------:|
| 1 | **DeepSeek-V3** | **41.7%** | 15.9% |
| 2 | Llama-3.3-70B | 41.4% | 16.5% |
| 3 | Mixtral-8x7B | 41.1% | 17.4% |
| 4 | Qwen 2.5-7B | 40.5% | 16.3% |
| 5 | Meta-Llama 3.1-8B | 39.2% | 16.2% |

**Key observation**: The performance gap between the best and worst model is only **2.5 percentage points**. All models cluster around 40%, suggesting a shared structural limitation rather than a model-specific one.

### Coverage Distribution (across all 1,000 comparisons)

| Coverage Range | Count | % |
|:---:|:---:|:---:|
| 0–20% | 45 | 4.5% |
| 20–40% | 326 | 32.6% |
| **40–60%** | **444** | **44.4%** |
| 60–80% | 163 | 16.3% |
| 80–100% | 22 | 2.2% |

The distribution is roughly normal, centered around 40%. Most comparisons fall in the 30–60% range. Near-perfect coverage (80%+) is rare (2.2%).

### Most Frequently Missed Keywords

These human-reviewer keywords were most often **not covered** by any LLM:

| Keyword | Times Missed (out of 1000) | Interpretation |
|---------|:-:|---|
| baseline comparison | 160 | Humans demand comparison to prior work |
| ablation study | 116 | Humans want component-level analysis |
| ablation studies | 48 | (variant of above) |
| theoretical analysis | 42 | Humans want mathematical rigor |
| experimental evaluation | 40 | Humans focus on empirical validation |
| generalization | 39 | Concern about out-of-distribution performance |
| novelty concerns | 35 | Humans question originality |
| training stability | 30 | Practical training concerns |
| hyperparameter sensitivity | 23 | Humans worry about tuning fragility |

**Pattern**: The most missed keywords are all about **methodological rigor** — "did you prove this actually works?" Human reviewers ask for ablations, baselines, and statistical tests. LLMs, seeing only the abstract, cannot know what experiments are missing from the full paper.

### Most Successfully Matched Keywords

| Keyword | Times Matched | Avg Similarity |
|---------|:---:|:---:|
| baseline comparison | 50 | 0.933 |
| generalization | 46 | 0.893 |
| computational complexity | 32 | 0.862 |
| knowledge distillation | 31 | 0.962 |
| sample efficiency | 28 | 0.902 |
| contrastive learning | 27 | 0.914 |
| adversarial training | 25 | 0.848 |

**Pattern**: LLMs reliably identify **domain-specific technical terms** and **what the paper is about**. When a paper discusses contrastive learning, the LLM catches it. The strong matches are topical keywords, not evaluative ones.

---

## Concrete Example

### Paper: "Accurately Solving Rod Dynamics with Graph Learning" (40% coverage)

**Human keywords** (from consolidated reviews):
> position-based dynamics, constraint projection step, initial guess, nonlinear optimization, convergence iterations, runtime reduction, state-of-the-art comparison, CG solver, Newton iterations, training data baseline

**LLM keywords** (from abstract-only review):
> graph networks, iterative solvers, rod dynamics, long-term stability, computational efficiency, physics-informed ML, initial guess prediction, generalization, scalability, architecture details

**Matching results**:

| Human Keyword | Best LLM Match | Similarity | Result |
|---|---|:---:|:---:|
| initial guess | initial guess prediction | 0.704 | MATCH |
| cg solver | iterative solvers | 0.591 | MATCH |
| convergence iterations | iterative solvers | 0.565 | MATCH |
| newton iterations | iterative solvers | 0.561 | MATCH |
| runtime reduction | computational efficiency | 0.519 | MISS (close) |
| position-based dynamics | rod dynamics | 0.412 | MISS |
| state-of-the-art comparison | — | 0.307 | MISS |
| training data baseline | — | 0.261 | MISS |

**Coverage: 4/10 = 40%**

The LLM correctly identifies the general domain (solvers, dynamics, efficiency) but misses the specific techniques the human reviewers critiqued (PBD, CG solver specifics, baseline demands).

---

## Key Findings

### 1. LLMs Cover ~41% of Human Reviewer Concerns

This is consistent across all 5 models tested. The remaining 59% represents concepts, critiques, and experimental demands that LLMs do not raise independently.

### 2. Larger Models Do Not Guarantee Better Coverage

Llama 3.3 70B (largest) scores 41.4% while the smaller Llama 3.1 8B scores 39.2% — a marginal difference. DeepSeek-V3 leads at 41.7%. Model size is not the bottleneck.

### 3. LLMs Are Good at "What" but Weak at "What's Missing"

- **Strong at**: Identifying the paper's topic, techniques, and contributions (the "what")
- **Weak at**: Identifying missing experiments, ablation demands, baseline comparisons (the "what's not there")

This makes sense — human reviewers read the full paper and can identify gaps. LLMs see only the abstract and can describe what's present but cannot assess what's absent.

### 4. The Limitation Is Structural, Not Linguistic

The narrow 2.5% gap between models (39.2%–41.7%) suggests the bottleneck is **information access** (abstract vs full paper), not **language understanding capability**. All models hit the same ceiling because they all face the same information constraint.

### 5. Methodological Rigor Keywords Are Systematically Missed

"Ablation study," "baseline comparison," "statistical significance," and "hyperparameter sensitivity" are the most frequently missed keywords. These represent the empirical validation demands that define rigorous peer review — and that LLMs consistently overlook.

---

## Visualizations

All charts are in `evaluation_v3/outputs/charts/`:

| File | Description |
|------|-------------|
| `evaluation_dashboard.png` | 6-panel overview: boxplots, ranking, distribution, missed/matched keywords, per-paper coverage |
| `similarity_analysis.png` | Similarity score heatmap by model + matched vs missed keyword counts |
| `summary_infographic.png` | One-page summary with key findings |

---

## Reproducibility

### Requirements
```bash
pip install -r requirements.txt
```

### Running the Pipeline
```bash
export TOGETHER_API_KEY="your-key"

# Step 1: Consolidate human reviews (no API needed)
python evaluation_v3/step1_consolidate_reviews.py \
  --db-path data/gen_review.db \
  --output evaluation_v3/outputs/consolidated_reviews.jsonl \
  --n 200 --seed 42 --min-year 2021

# Step 2: Generate LLM reviews (~20 min, 1000 API calls)
python evaluation_v3/step2_generate_llm_reviews.py \
  --input evaluation_v3/outputs/consolidated_reviews.jsonl \
  --output evaluation_v3/outputs/llm_reviews.jsonl

# Step 3: Semantic evaluation (~25 min, 1200 API calls + local embeddings)
python evaluation_v3/step3_semantic_evaluation.py \
  --human-reviews evaluation_v3/outputs/consolidated_reviews.jsonl \
  --llm-reviews evaluation_v3/outputs/llm_reviews.jsonl \
  --output-dir evaluation_v3/outputs

# Visualizations
python evaluation_v3/visualize_results.py
```

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n` | 200 | Number of papers to sample |
| `--seed` | 42 | Random seed (deterministic) |
| `--threshold` | 0.55 | Cosine similarity threshold for keyword match |
| `--extraction-model` | DeepSeek-V3 | LLM used for keyword extraction |
| `--models` | 5 predefined | LLM models for review generation |

### Output Files

```
evaluation_v3/outputs/
├── consolidated_reviews.jsonl       # Step 1: 200 papers with merged human reviews
├── llm_reviews.jsonl                # Step 2: 1000 LLM-generated reviews
├── semantic_coverage_results.jsonl  # Step 3: Per paper-model detailed results
├── coverage_summary.json            # Aggregate statistics
├── coverage_report.md               # Auto-generated summary report
└── charts/
    ├── evaluation_dashboard.png     # 6-panel overview
    ├── similarity_analysis.png      # Similarity heatmap + matched vs missed
    └── summary_infographic.png      # One-page findings summary
```

---

## Technical Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| LLM inference | Together AI API | Review generation + keyword extraction |
| Keyword extraction | DeepSeek-V3 | Extract top 10 from each review |
| Semantic matching | sentence-transformers (`all-MiniLM-L6-v2`) | Embed keywords, compute cosine similarity |
| Data source | SQLite (1.5GB, 32K papers) | Human reviews from ICLR/NeurIPS-style venues |
| Visualization | matplotlib | Charts and infographics |

---

## Limitations and Future Work

1. **Abstract-only reviews**: LLMs review from abstracts, while humans review full papers. Providing full paper text could significantly improve coverage.

2. **Keyword extraction bias**: Using an LLM (DeepSeek-V3) to extract keywords from both sides could introduce systematic bias in what gets identified as "important."

3. **Threshold sensitivity**: The 0.55 cosine similarity threshold is a design choice. Lowering it increases coverage scores but risks false matches. Sensitivity analysis across thresholds (0.4–0.7) would strengthen the findings.

4. **Single embedding model**: Results depend on `all-MiniLM-L6-v2`. Cross-validating with a second embedding model (e.g., `all-mpnet-base-v2`) would add robustness.

5. **Prompt sensitivity**: LLM review quality depends on the system prompt. Iterating on prompts (e.g., explicitly asking for ablation demands) could improve coverage.
