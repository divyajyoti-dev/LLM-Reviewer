# LLM Review Evaluation - Progress Update

**Date**: February 17, 2026  
**Branch**: `evaluation_v2`

---


## Methodology (3-Phase Evaluation)

### Phase 1: Ground Truth Extraction
- **Input**: Human review text (NOT abstract)
- **Process**: Used DeepSeek-V3 to extract canonical keywords from human reviews
- **Output**: 3,146 keywords from 200 papers representing what humans found important

### Phase 2: Blind LLM Review Generation
- **Input**: Paper abstract only (no access to human reviews)
- **Process**: Each model generates structured review (strengths, weaknesses, questions, keywords)
- **Models tested**: 5 LLMs via Together AI
- **Output**: 1,000 LLM-generated reviews (5 models × 200 papers)

### Phase 3: Coverage Analysis
- **Input**: Human keywords + LLM reviews
- **Process**: Measure keyword recall, precision, F1 score
- **Output**: Per-model performance metrics

---

## Results

### Model Performance Rankings

| Rank | Model | Fuzzy Recall | Precision | F1 Score |
|------|-------|--------------|-----------|----------|
| 1 | **DeepSeek-V3** | 0.517 | 0.505 | **0.492** |
| 2 | Mixtral 8x7B | 0.450 | 0.441 | 0.425 |
| 3 | Qwen 2.5 7B | 0.426 | 0.443 | 0.418 |
| 4 | Llama 3.1 8B | 0.438 | 0.411 | 0.409 |
| 5 | Llama 3.3 70B | 0.272 | 0.262 | **0.257** |

### Overall Statistics
- **Mean keyword recall**: 42.0% (LLMs miss 58% of human concerns)
- **Mean F1 score**: 0.400
- **Performance gap**: 0.235 between best and worst model

---

## Key Findings

### 1. LLMs Miss Significant Human Concerns
On average, LLMs only cover **42%** of the keywords/concerns that human reviewers emphasized. This suggests LLMs may focus on different aspects of papers than experienced reviewers.

### 2. Larger Models ≠ Better Coverage
**Surprising result**: Llama 3.3 70B (largest model tested) performed **worst** with F1=0.257, while smaller models like DeepSeek-V3 performed best (F1=0.492).

### 3. Most Frequently Missed Keywords
LLMs consistently fail to mention:
- "Ablation study" (missed 191 times)
- "Experimental validation"
- "Theoretical justification"
- "Hyperparameter sensitivity"
- "Statistical significance"

This suggests LLMs may not emphasize methodological rigor as much as human reviewers.

### 4. DeepSeek Outperforms Others
DeepSeek-V3 achieves 51.7% recall vs 27.2% for Llama 70B - nearly **2x better** at identifying human-salient points.

---

## Implications

1. **LLMs as reviewer assistants**: Current models miss ~60% of what humans care about, limiting their utility as standalone reviewers.

2. **Model selection matters**: Smaller, well-tuned models may outperform larger ones for this task.

3. **Methodological blind spots**: LLMs appear less focused on experimental rigor (ablations, statistical tests) than humans.

---

## Visualizations

See attached:
- `evaluation_dashboard.png` - 6-panel overview
- `summary_infographic.png` - Key findings at a glance


